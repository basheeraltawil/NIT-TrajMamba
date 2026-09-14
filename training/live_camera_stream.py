#!/usr/bin/env python3
"""
human_traj_estimation.py
-------------------------
ROS2 node that detects people from synchronized RGB-D images, tracks each
person's 3-D hip position over time, and predicts their future trajectory
with a TrajMamba model. Structured like `nit_pose_estimation`'s
`pose_estimation.py`: parameters come from `declare_parameter`/config.yaml
(and therefore the launch file), rather than argparse.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger, SetBool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge
import message_filters

from ament_index_python.packages import get_package_share_directory

try:
    from .helper import (
        POSE_LEFT_HIP, POSE_RIGHT_HIP,
        track_colors,
        MultiPersonTracker, PersonTrack,
        sample_depth_patch, deproject_pixel, project_point, velocity_normalize,
        draw_skeleton, draw_trail, draw_uncertainty_cone, draw_prediction,
        draw_current_marker, draw_id_badge, draw_depth_bar, draw_legend, draw_hud,
    )
    from .model_utils import load_trajmamba_checkpoint
except ImportError:
    from helper import (
        POSE_LEFT_HIP, POSE_RIGHT_HIP,
        track_colors,
        MultiPersonTracker, PersonTrack,
        sample_depth_patch, deproject_pixel, project_point, velocity_normalize,
        draw_skeleton, draw_trail, draw_uncertainty_cone, draw_prediction,
        draw_current_marker, draw_id_badge, draw_depth_bar, draw_legend, draw_hud,
    )
    from model_utils import load_trajmamba_checkpoint

PACKAGE_NAME = 'nit_human_traj_estimation'


class HumanTrajEstimation(Node):
    def __init__(self):
        super().__init__('nit_human_traj_estimation')

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.intrinsics: Optional[Dict[str, float]] = None
        self.intrinsics_ready = False
        self.last_camera_info_stamp = None
        self.latest_fps = 0.0
        self.prev_time = time.time()
        self._shot_count = 0

        # ----- Diagnostics: warn loudly instead of silently doing nothing -----
        self.last_rgb_raw_time = None
        self.last_depth_raw_time = None
        self.last_camera_info_time = None
        self.last_synced_time = None
        self.node_start_time = self.get_clock().now()
        self.warning_period = 5.0
        self.has_warned_no_rgb = False
        self.has_warned_no_depth = False
        self.has_warned_no_camera_info = False
        self.has_warned_no_sync = False

        self.screenshot_dir = self._resolve_share_path(self.screenshot_dir_param)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.get_logger().info(
            f"[torch-diag] executable={sys.executable} "
            f"torch={torch.__version__} torch_file={torch.__file__} "
            f"cuda_available={torch.cuda.is_available()} "
            f"cuda_version={torch.version.cuda} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

        checkpoint_path = self._resolve_share_path(self.checkpoint_path_param)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model, self.config = load_trajmamba_checkpoint(
            str(checkpoint_path), self.device, log=self.get_logger().info)
        self.global_vel_mean = np.array(self.config.get("vel_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.global_vel_std = np.array(self.config.get("vel_std", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.pos_dim = int(self.config.get("pos_dim", 3))
        self.obs_len = int(self.config.get("obs_len", 20))
        self.pred_len = int(self.config.get("pred_len", 30))

        buffer_maxlen = max(60, self.obs_len * 3)
        self.tracker = MultiPersonTracker(
            buffer_maxlen=buffer_maxlen,
            max_match_dist=self.max_match_dist,
            max_misses=self.max_misses,
            max_tracks=self.max_people,
        )

        task_path = self._resolve_share_path(self.pose_landmarker_task_param)
        if not task_path.exists():
            raise FileNotFoundError(f"pose_landmarker task not found: {task_path}")
        self.get_logger().info(f"Using pose landmarker task: {task_path}")

        base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.max_people,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

        # ----- Publishers -----
        self.vis_pub = self.create_publisher(Image, self.output_image_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.pose_pub = self.create_publisher(PoseArray, self.pose_array_topic, 1)

        # ----- Subscriptions (created lazily via activation) -----
        self.rgb_sub = None
        self.depth_sub = None
        self.sync = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10,
        )

        # ----- Activation control (mirrors nit_pose_estimation) -----
        self.status_service = self.create_service(
            Trigger, self.status_service_name, self.status_service_callback)
        self.activate_service = self.create_service(
            SetBool, self.activation_service_name, self.activate_service_callback)

        self.active = False
        if self.activation:
            self.activate_subscriptions()

        self.diagnostic_timer = self.create_timer(1.0, self.check_diagnostics)

        if self.enable_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("Human Trajectory Estimation Node started.")
        self.get_logger().info(f"RGB topic:            {self.rgb_topic}")
        self.get_logger().info(f"Depth topic:          {self.depth_topic}")
        self.get_logger().info(f"Camera info topic:    {self.camera_info_topic}")
        self.get_logger().info(f"Output image topic:   {self.output_image_topic}")
        self.get_logger().info(f"Marker topic:         {self.marker_topic}")
        self.get_logger().info(f"Predicted poses topic:{self.pose_array_topic}")
        self.get_logger().info(f"Activation status:    {self.activation}")
        self.get_logger().info(f"Activate service:      {self.activation_service_name}")
        self.get_logger().info(f"Status service:        {self.status_service_name}")
        self.get_logger().info(f"Checkpoint:           {checkpoint_path}")
        self.get_logger().info(f"Device:               {self.device}")
        if self.enable_window:
            self.get_logger().info("Keys:  q=quit   s=save screenshot   h=toggle HUD   k=toggle skeleton")
            self.get_logger().info(f"Screenshots -> {self.screenshot_dir}")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self):
        self.declare_parameter('checkpoint_path', 'checkpoints/best_model.pth')
        self.declare_parameter('pose_landmarker_task', 'checkpoints/pose_landmarker.task')

        self.declare_parameter('rgb_topic', 'rgb/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('frame_id', 'camera_link')

        self.declare_parameter('output_image_topic', 'nit_human_traj_estimation/image')
        self.declare_parameter('marker_topic', 'nit_human_traj_estimation/markers')
        self.declare_parameter('pose_array_topic', 'nit_human_traj_estimation/predicted_trajectories')

        self.declare_parameter('activation', True)

        self.declare_parameter('enable_window', True)
        self.declare_parameter('window_name', 'TrajMamba ROS Live')
        self.declare_parameter('show_hud', True)
        self.declare_parameter('show_skeleton', True)
        self.declare_parameter('screenshot_dir', 'figures')

        self.declare_parameter('queue_size', 10)
        self.declare_parameter('sync_slop', 0.08)

        self.declare_parameter('activation_service_name', '/nit_human_traj_estimation/activate')
        self.declare_parameter('status_service_name', '/nit_human_traj_estimation/status')

        self.declare_parameter('max_people', 4)
        self.declare_parameter('max_match_dist', 0.8)
        self.declare_parameter('max_misses', 15)

    def _read_parameters(self):
        gp = self.get_parameter
        self.checkpoint_path_param = gp('checkpoint_path').get_parameter_value().string_value
        self.pose_landmarker_task_param = gp('pose_landmarker_task').get_parameter_value().string_value

        self.rgb_topic = gp('rgb_topic').get_parameter_value().string_value
        self.depth_topic = gp('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = gp('camera_info_topic').get_parameter_value().string_value
        self.frame_id = gp('frame_id').get_parameter_value().string_value

        self.output_image_topic = gp('output_image_topic').get_parameter_value().string_value
        self.marker_topic = gp('marker_topic').get_parameter_value().string_value
        self.pose_array_topic = gp('pose_array_topic').get_parameter_value().string_value

        self.activation = gp('activation').get_parameter_value().bool_value

        self.enable_window = gp('enable_window').get_parameter_value().bool_value
        self.window_name = gp('window_name').get_parameter_value().string_value
        self.show_hud = gp('show_hud').get_parameter_value().bool_value
        self.show_skeleton = gp('show_skeleton').get_parameter_value().bool_value
        self.screenshot_dir_param = gp('screenshot_dir').get_parameter_value().string_value

        self.queue_size = gp('queue_size').get_parameter_value().integer_value
        self.sync_slop = gp('sync_slop').get_parameter_value().double_value

        self.activation_service_name = gp('activation_service_name').get_parameter_value().string_value
        self.status_service_name = gp('status_service_name').get_parameter_value().string_value

        self.max_people = gp('max_people').get_parameter_value().integer_value
        self.max_match_dist = gp('max_match_dist').get_parameter_value().double_value
        self.max_misses = gp('max_misses').get_parameter_value().integer_value

    def _resolve_share_path(self, path_str: str) -> Path:
        """Relative paths resolve against the installed package share
        directory; absolute paths are used as-is."""
        p = Path(path_str).expanduser()
        if p.is_absolute():
            return p
        try:
            share_dir = Path(get_package_share_directory(PACKAGE_NAME))
        except Exception:
            share_dir = Path(__file__).resolve().parents[1]
        return share_dir / p

    # ------------------------------------------------------------------
    # Activation control
    # ------------------------------------------------------------------
    def activate_subscriptions(self):
        if self.active:
            return
        self.rgb_is_compressed = self.rgb_topic.endswith('/compressed')
        rgb_msg_type = CompressedImage if self.rgb_is_compressed else Image

        self.rgb_sub = message_filters.Subscriber(self, rgb_msg_type, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        # Lightweight raw subscriptions purely for diagnostics: message_filters
        # only tells us about MATCHED pairs, so these let us tell the user
        # whether each individual topic is publishing at all, even if the
        # synchronizer never finds a match.
        rgb_msg_type_diag = CompressedImage if self.rgb_is_compressed else Image
        self.rgb_diag_sub = self.create_subscription(
            rgb_msg_type_diag, self.rgb_topic, self._rgb_diag_callback, 10)
        self.depth_diag_sub = self.create_subscription(
            Image, self.depth_topic, self._depth_diag_callback, 10)

        self.active = True
        self.get_logger().info("Human trajectory estimation activated.")

    def deactivate_subscriptions(self):
        if not self.active:
            return
        self.sync = None
        self.rgb_sub = None
        self.depth_sub = None
        if getattr(self, 'rgb_diag_sub', None):
            self.destroy_subscription(self.rgb_diag_sub)
            self.rgb_diag_sub = None
        if getattr(self, 'depth_diag_sub', None):
            self.destroy_subscription(self.depth_diag_sub)
            self.depth_diag_sub = None
        self.active = False
        self.get_logger().info("Human trajectory estimation deactivated.")

    def _rgb_diag_callback(self, msg):
        self.last_rgb_raw_time = self.get_clock().now()

    def _depth_diag_callback(self, msg):
        self.last_depth_raw_time = self.get_clock().now()

    def check_diagnostics(self):
        """Warn loudly (once) if RGB, depth, camera_info, or the sync
        haven't produced anything within warning_period seconds, instead of
        the node silently doing nothing forever."""
        if not self.active:
            return
        elapsed = (self.get_clock().now() - self.node_start_time).nanoseconds / 1e9
        if elapsed < self.warning_period:
            return

        if self.last_rgb_raw_time is None and not self.has_warned_no_rgb:
            self.get_logger().warn(
                f"No messages received on rgb_topic='{self.rgb_topic}' after "
                f"{self.warning_period:.0f}s. Check 'ros2 topic hz {self.rgb_topic}' "
                "and that config.yaml points at the right topic."
            )
            self.has_warned_no_rgb = True
        elif self.last_rgb_raw_time is not None and self.has_warned_no_rgb:
            self.get_logger().info(f"RGB data now being received on '{self.rgb_topic}'.")
            self.has_warned_no_rgb = False

        if self.last_depth_raw_time is None and not self.has_warned_no_depth:
            self.get_logger().warn(
                f"No messages received on depth_topic='{self.depth_topic}' after "
                f"{self.warning_period:.0f}s. Check 'ros2 topic hz {self.depth_topic}' "
                "and that config.yaml points at the right topic."
            )
            self.has_warned_no_depth = True
        elif self.last_depth_raw_time is not None and self.has_warned_no_depth:
            self.get_logger().info(f"Depth data now being received on '{self.depth_topic}'.")
            self.has_warned_no_depth = False

        if not self.intrinsics_ready and not self.has_warned_no_camera_info:
            self.get_logger().warn(
                f"No CameraInfo received on camera_info_topic='{self.camera_info_topic}' "
                f"after {self.warning_period:.0f}s. The node will not process any frames "
                "until this arrives. Check 'ros2 topic hz " + self.camera_info_topic + "'."
            )
            self.has_warned_no_camera_info = True
        elif self.intrinsics_ready and self.has_warned_no_camera_info:
            self.get_logger().info(f"CameraInfo now being received on '{self.camera_info_topic}'.")
            self.has_warned_no_camera_info = False

        # Only meaningful to check sync once both raw topics are flowing.
        have_both_raw = self.last_rgb_raw_time is not None and self.last_depth_raw_time is not None
        if have_both_raw and self.last_synced_time is None and not self.has_warned_no_sync:
            self.get_logger().warn(
                "RGB and depth are both publishing individually, but the "
                "ApproximateTimeSynchronizer has never matched a pair. Their "
                "timestamps may be too far apart for sync_slop="
                f"{self.sync_slop}s, or one of the streams may have stalled "
                "since it started publishing. Try increasing sync_slop in "
                "config.yaml."
            )
            self.has_warned_no_sync = True
        elif self.last_synced_time is not None and self.has_warned_no_sync:
            self.get_logger().info("RGB/depth synchronization recovered.")
            self.has_warned_no_sync = False

    def status_service_callback(self, request, response):
        response.success = True
        response.message = str(self.active)
        return response

    def activate_service_callback(self, request, response):
        if request.data and not self.active:
            self.activate_subscriptions()
            response.message = "Human trajectory estimation activated."
        elif request.data and self.active:
            response.message = "Human trajectory estimation remains activated."
        elif not request.data and self.active:
            self.deactivate_subscriptions()
            response.message = "Human trajectory estimation deactivated."
        else:
            response.message = "Human trajectory estimation remains deactivated."
        response.success = True
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Camera / image decoding
    # ------------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo):
        self.last_camera_info_time = self.get_clock().now()
        if self.intrinsics_ready and self.last_camera_info_stamp == msg.header.stamp:
            return
        self.intrinsics = {
            "fx": float(msg.k[0]),
            "fy": float(msg.k[4]),
            "cx": float(msg.k[2]),
            "cy": float(msg.k[5]),
            "width": float(msg.width),
            "height": float(msg.height),
        }
        self.intrinsics_ready = True
        self.last_camera_info_stamp = msg.header.stamp

    def decode_rgb(self, msg):
        if self.rgb_is_compressed:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Failed to decode compressed RGB frame")
            return frame
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def decode_depth(self, msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        return depth

    def hip_center_3d(self, landmarks, width: int, height: int, depth_image: np.ndarray
                       ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Deproject the hip centre of ONE pose to a 3-D point. Returns
        (xyz, pixel) or (None, pixel/None) when depth is unavailable."""
        if landmarks is None or len(landmarks) <= max(POSE_LEFT_HIP, POSE_RIGHT_HIP):
            return None, None

        left = landmarks[POSE_LEFT_HIP]
        right = landmarks[POSE_RIGHT_HIP]
        u = int(((left.x + right.x) * 0.5) * width)
        v = int(((left.y + right.y) * 0.5) * height)

        depth_m = sample_depth_patch(depth_image, u, v, patch=7)
        if depth_m <= 0 or self.intrinsics is None:
            return None, (u, v)

        point = deproject_pixel(u, v, depth_m, self.intrinsics)
        return point, (u, v)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_track(self, track: PersonTrack):
        """Run TrajMamba for one person. Returns (pred_positions or None,
        predicting flag)."""
        if len(track.position_buffer) < self.obs_len:
            return None, False

        obs_positions = np.stack(list(track.position_buffer)[-self.obs_len:], axis=0)
        obs_vel_n, _ = velocity_normalize(obs_positions, self.global_vel_mean, self.global_vel_std)

        obs_vel_t = torch.tensor(obs_vel_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pred_vel_n = self.model(obs_vel_t).squeeze(0).cpu().numpy()

        pred_vel = (pred_vel_n * self.global_vel_std) + self.global_vel_mean
        pred_positions = [obs_positions[-1]]
        for t in range(pred_vel.shape[0]):
            pred_positions.append(pred_positions[-1] + pred_vel[t])
        return np.stack(pred_positions, axis=0), True

    # ------------------------------------------------------------------
    # Main synchronized callback
    # ------------------------------------------------------------------
    def synced_callback(self, rgb_msg, depth_msg):
        self.last_synced_time = self.get_clock().now()
        if not self.intrinsics_ready or self.intrinsics is None:
            return

        try:
            frame = self.decode_rgb(rgb_msg)
            depth_image = self.decode_depth(depth_msg)
        except Exception as exc:
            self.get_logger().warn(f"Frame decode failed: {exc}")
            return

        h, w = frame.shape[:2]
        rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_mp)
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        all_poses = result.pose_landmarks if result.pose_landmarks else []

        # ----- Detect every person's hip centre this frame -----
        detections: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        for landmarks in all_poses:
            if self.show_skeleton:
                draw_skeleton(frame, landmarks, w, h, self.connections)
            xyz, px = self.hip_center_3d(landmarks, w, h, depth_image)
            if xyz is not None:
                detections.append((xyz, px))
            elif px is not None:
                cv2.circle(frame, px, 5, (0, 0, 255), -1, cv2.LINE_AA)

        # ----- Associate detections to persistent tracks -----
        tracks = self.tracker.update(detections)

        any_predicting = False
        legend_tracks = []
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id
        marker_array = MarkerArray()
        pose_array = PoseArray()
        pose_array.header = hdr

        draw_order = sorted(
            tracks,
            key=lambda t: (t.last_position[2] if t.last_position is not None else 0.0),
            reverse=True,
        )

        for track in draw_order:
            obs_col, obs_dark, pred_col, pred_dark = track_colors(track.id)
            legend_tracks.append((track.id, obs_col, pred_col))

            obs_pixels = [project_point(p, self.intrinsics) for p in track.position_buffer]
            pred_positions, predicting = self.predict_track(track)

            if pred_positions is not None:
                pred_pixels = [project_point(p, self.intrinsics) for p in pred_positions]
                draw_uncertainty_cone(frame, pred_pixels, col_fill=pred_col)
                if len(obs_pixels) > 1:
                    draw_trail(frame, obs_pixels, col_main=obs_col, col_dark=obs_dark)
                draw_prediction(frame, pred_pixels, col_main=pred_col, col_dark=pred_dark)
                draw_current_marker(frame, pred_pixels[0], ring_col=pred_col)
                draw_id_badge(frame, pred_pixels[0], track.id, pred_col)
                any_predicting = True

                m = Marker()
                m.header = hdr
                m.ns = f"human_traj_pred_{track.id}"
                m.id = track.id
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.02
                m.color.r = pred_col[2] / 255.0
                m.color.g = pred_col[1] / 255.0
                m.color.b = pred_col[0] / 255.0
                m.color.a = 0.9
                for p in pred_positions:
                    pt = Point()
                    pt.x = float(p[0]); pt.y = float(p[1]); pt.z = float(p[2])
                    m.points.append(pt)
                marker_array.markers.append(m)

                for p in pred_positions:
                    pose = Pose()
                    pose.position.x = float(p[0])
                    pose.position.y = float(p[1])
                    pose.position.z = float(p[2])
                    pose.orientation.w = 1.0
                    pose_array.poses.append(pose)
            else:
                if len(obs_pixels) > 1:
                    draw_trail(frame, obs_pixels, col_main=obs_col, col_dark=obs_dark)
                    draw_current_marker(frame, obs_pixels[-1], ring_col=obs_col)
                    draw_id_badge(frame, obs_pixels[-1], track.id, obs_col)

        # Depth gauge: show the closest predicting person (most relevant target).
        closest = None
        for track in tracks:
            if track.last_position is None:
                continue
            if closest is None or track.last_position[2] < closest.last_position[2]:
                closest = track
        if closest is not None:
            cz = float(closest.last_position[2])
            cz_pred, _ = self.predict_track(closest)
            future_z = [float(p[2]) for p in cz_pred] if cz_pred is not None else [cz]
            draw_depth_bar(frame, cz, future_z)

        # ----- Publish -----
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            vis_msg.header = hdr
            self.vis_pub.publish(vis_msg)
        except Exception:
            pass
        try:
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
            if pose_array.poses:
                self.pose_pub.publish(pose_array)
        except Exception:
            pass

        predicting = any_predicting
        n_tracks = len(tracks)

        now = time.time()
        self.latest_fps = 0.9 * self.latest_fps + 0.1 * (1.0 / max(now - self.prev_time, 1e-6))
        self.prev_time = now

        if self.enable_window:
            draw_hud(frame, self.latest_fps, n_tracks, predicting,
                     self.obs_len, self.pred_len, self.show_hud)
            draw_legend(frame, active_tracks=legend_tracks if legend_tracks else None)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("Quit requested from window.")
                rclpy.shutdown()
            elif key == ord("s"):
                self._save_screenshot(frame)
            elif key == ord("h"):
                self.show_hud = not self.show_hud
                self.get_logger().info(f"HUD {'on' if self.show_hud else 'off'}")
            elif key == ord("k"):
                self.show_skeleton = not self.show_skeleton
                self.get_logger().info(f"Skeleton {'on' if self.show_skeleton else 'off'}")

    def _save_screenshot(self, frame: np.ndarray):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._shot_count += 1
        path = self.screenshot_dir / f"human_traj_{ts}_{self._shot_count:03d}.png"
        try:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            self.get_logger().info(f"Saved screenshot: {path}")
        except Exception as exc:
            self.get_logger().warn(f"Screenshot failed: {exc}")

    def destroy_node(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
        try:
            if self.enable_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HumanTrajEstimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()