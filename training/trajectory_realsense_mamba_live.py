#!/usr/bin/env python3

import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import pyrealsense2 as rs

from mamba_model import TrajMamba


POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
TRAIL_COLOR = (0, 255, 120)
PRED_COLOR = (0, 180, 255)


class Kalman1D:
    def __init__(self, q: float = 1e-3, r: float = 1e-2, p: float = 1.0):
        self.q = q
        self.r = r
        self.p = p
        self.x = 0.0
        self.initialized = False

    def update(self, z: float) -> float:
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x


class EMAFilter:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = value.astype(np.float32)
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


def sample_depth_patch(depth_frame, u: int, v: int, patch: int = 7) -> float:
    half = patch // 2
    values = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            du = int(np.clip(u + dx, 0, depth_frame.get_width() - 1))
            dv = int(np.clip(v + dy, 0, depth_frame.get_height() - 1))
            d = depth_frame.get_distance(du, dv)
            if d > 0:
                values.append(d)
    if not values:
        return 0.0
    return float(np.median(values))


def get_center_3d(
    landmarks,
    width: int,
    height: int,
    depth_frame,
    intrinsics,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    if landmarks is None or len(landmarks) <= max(POSE_LEFT_HIP, POSE_RIGHT_HIP):
        return None, None

    left = landmarks[POSE_LEFT_HIP]
    right = landmarks[POSE_RIGHT_HIP]
    u = int(((left.x + right.x) * 0.5) * width)
    v = int(((left.y + right.y) * 0.5) * height)

    depth = sample_depth_patch(depth_frame, u, v, patch=7)
    if depth <= 0:
        return None, (u, v)

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
    return np.array(point, dtype=np.float32), (u, v)

def velocity_normalize(obs_positions: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray) -> np.ndarray:
    obs_vel = np.diff(obs_positions, axis=0)
    # Use the GLOBAL stats from training, not local stats!
    obs_vel_n = (obs_vel - global_mean) / global_std
    return obs_vel_n, obs_vel

def draw_skeleton(frame: np.ndarray, landmarks, width: int, height: int, connections):
    if landmarks is None:
        return
    points = []
    for lm in landmarks:
        points.append((int(lm.x * width), int(lm.y * height)))

    for conn in connections:
        a = conn.start
        b = conn.end
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, points: List[Tuple[int, int]]):
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        t = i / max(1, len(points) - 1)
        color = (
            int(TRAIL_COLOR[0] * t),
            int(TRAIL_COLOR[1] * t),
            int(TRAIL_COLOR[2] * t),
        )
        cv2.line(frame, points[i - 1], points[i], color, 2, cv2.LINE_AA)
        cv2.circle(frame, points[i], 3, color, -1, cv2.LINE_AA)


def draw_prediction(frame: np.ndarray, points: List[Tuple[int, int]]):
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        if i % 2 == 0:
            cv2.line(frame, points[i - 1], points[i], PRED_COLOR, 2, cv2.LINE_AA)
    cv2.circle(frame, points[-1], 5, PRED_COLOR, -1, cv2.LINE_AA)
    if len(points) >= 3:
        cv2.arrowedLine(frame, points[-2], points[-1], PRED_COLOR, 2, cv2.LINE_AA, tipLength=0.25)


def draw_depth_bar(frame: np.ndarray, current_z: float, future_z: List[float]):
    h, w = frame.shape[:2]
    bar_x = w - 50
    cv2.rectangle(frame, (bar_x, 20), (w - 10, h - 20), (50, 50, 50), 2)

    if current_z <= 0:
        return

    z_min = max(0.1, min([current_z] + future_z))
    z_max = max([current_z] + future_z) + 1e-6

    def z_to_y(z):
        t = (z - z_min) / (z_max - z_min)
        return int((h - 30) - t * (h - 60))

    cy = z_to_y(current_z)
    cv2.circle(frame, (bar_x + 20, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

    for z in future_z[::3]:
        py = z_to_y(z)
        cv2.circle(frame, (bar_x + 20, py), 3, (0, 180, 255), -1, cv2.LINE_AA)

    cv2.putText(frame, f"Z: {current_z:.2f}m", (bar_x - 10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_hud(frame: np.ndarray, fps: float, history_len: int, max_len: int, predicting: bool):
    status = "Predicting via Mamba" if predicting else "Collecting" 
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"History: {history_len}/{max_len}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, status, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    checkpoint_path = "/home/basheer/basheer_ws/src/n-stac/trajectory_estimator/cl/checkpoints/best_model.pth"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # Load the global stats
    global_vel_mean = np.array(config.get("vel_mean", [0,0,0]), dtype=np.float32)
    global_vel_std  = np.array(config.get("vel_std",  [1,1,1]), dtype=np.float32)

    model = TrajMamba(
        pos_dim=int(config.get("pos_dim", 3)),
        obs_len=int(config.get("obs_len", 20)),
        pred_len=int(config.get("pred_len", 30)),
        d_model=int(config.get("d_model", 64)),
        d_state=int(config.get("d_state", 16)),
        d_conv=int(config.get("d_conv", 4)),
        expand=int(config.get("expand", 2)),
        n_layers=int(config.get("n_layers", 3)),
        dropout=float(config.get("dropout", 0.1)),
       
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pos_dim = int(config.get("pos_dim", 3))
    obs_len = int(config.get("obs_len", 20))
    pred_len = int(config.get("pred_len", 30))

    task_path = Path(__file__).resolve().parents[2] / "pose_landmarker.task"
    if not task_path.exists():
        task_path = Path(__file__).resolve().parents[3] / "pose_landmarker.task"
    if not task_path.exists():
        raise FileNotFoundError("pose_landmarker.task not found.")

    base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config_rs)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    position_buffer: Deque[np.ndarray] = deque(maxlen=60)
    kalman_filters = [Kalman1D() for _ in range(3)]
    ema_filter = EMAFilter(alpha=0.35)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h, w = frame.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            draw_skeleton(frame, landmarks, w, h, connections)

            center_3d, center_px = get_center_3d(landmarks, w, h, depth_frame, intrinsics)

            if center_3d is not None:
                filtered = np.array([
                    kalman_filters[i].update(center_3d[i]) for i in range(3)
                ], dtype=np.float32)
                filtered = ema_filter.update(filtered)
                position_buffer.append(filtered)

            predicting = False
            pred_positions = None

            if len(position_buffer) >= obs_len:
                obs_positions = np.stack(list(position_buffer)[-obs_len:], axis=0)
                
                # Use the new global normalization
                obs_vel_n, raw_obs_vel = velocity_normalize(obs_positions, global_vel_mean, global_vel_std)

                obs_vel_t = torch.tensor(obs_vel_n, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    pred_vel_n = model(obs_vel_t).squeeze(0).cpu().numpy()

                # Denormalize using the GLOBAL stats
                pred_vel = (pred_vel_n * global_vel_std) + global_vel_mean
                
                pred_positions = [obs_positions[-1]]
                for t in range(pred_vel.shape[0]):
                    pred_positions.append(pred_positions[-1] + pred_vel[t])
                pred_positions = np.stack(pred_positions, axis=0)
                predicting = True

            if len(position_buffer) > 1:
                obs_pixels = []
                for p in position_buffer:
                    px = rs.rs2_project_point_to_pixel(intrinsics, p.tolist())
                    obs_pixels.append((int(px[0]), int(px[1])))
                draw_trail(frame, obs_pixels)

            if pred_positions is not None:
                pred_pixels = []
                for p in pred_positions:
                    px = rs.rs2_project_point_to_pixel(intrinsics, p.tolist())
                    pred_pixels.append((int(px[0]), int(px[1])))
                draw_prediction(frame, pred_pixels)

                future_z = [float(p[2]) for p in pred_positions]
                draw_depth_bar(frame, float(pred_positions[0][2]), future_z)
            elif len(position_buffer) > 0:
                draw_depth_bar(frame, float(position_buffer[-1][2]), [float(position_buffer[-1][2])])

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            draw_hud(frame, fps, len(position_buffer), position_buffer.maxlen, predicting)

            cv2.imshow("TrajMamba RealSense Live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
