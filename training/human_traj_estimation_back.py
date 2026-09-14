#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge
import message_filters


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from mamba_model import TrajMamba  # noqa: E402


POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

COL_OBSERVED      = (200, 160, 0)     # teal/cyan  -> observed (past) trajectory
COL_OBSERVED_DARK = (120, 95, 0)
COL_PRED          = (40, 120, 255)    # orange     -> predicted (future) trajectory
COL_PRED_DARK     = (20, 70, 170)
COL_CONE          = (60, 140, 255)    # uncertainty cone fill (orange-ish)
COL_CURRENT       = (255, 255, 255)   # current position marker (white core)
COL_CURRENT_RING  = (40, 120, 255)
COL_SKELETON      = (235, 235, 235)
COL_TEXT          = (255, 255, 255)
COL_PANEL_BG      = (28, 28, 28)

# Backwards-compatible aliases (used elsewhere in the file).
TRAIL_COLOR = COL_OBSERVED
PRED_COLOR = COL_PRED
TRACK_PALETTE = [
    # observed,           observed_dark,     predicted,          predicted_dark
    ((200, 160, 0),       (120, 95, 0),      (40, 120, 255),     (20, 70, 170)),    # teal / orange
    ((80, 200, 80),       (40, 110, 40),     (200, 80, 220),     (120, 40, 130)),   # green / magenta
    ((230, 120, 60),      (140, 70, 30),     (40, 200, 255),     (25, 120, 160)),   # blue / amber
    ((90, 90, 230),       (45, 45, 140),     (220, 200, 60),     (130, 120, 35)),   # red / cyan
    ((180, 160, 120),     (100, 90, 70),     (60, 220, 180),     (35, 130, 105)),   # slate / lime
    ((200, 100, 180),     (110, 55, 100),    (90, 220, 240),     (55, 130, 145)),   # purple / yellow
]


def track_colors(track_id: int):
    """Return the (observed, observed_dark, pred, pred_dark) BGR tuple for a
    given integer track id, cycling through the palette."""
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


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


class PersonTrack:
    """Per-person state: a smoothed 3-D position history plus its own filters.

    Each tracked person keeps an independent observation buffer so TrajMamba can
    be run separately for every individual in the scene.
    """

    def __init__(self, track_id: int, buffer_maxlen: int):
        self.id = track_id
        self.position_buffer: Deque[np.ndarray] = deque(maxlen=buffer_maxlen)
        self.kalman_filters = [Kalman1D() for _ in range(3)]
        self.ema_filter = EMAFilter(alpha=0.35)
        self.misses = 0                     # consecutive frames without a match
        self.age = 0                        # total frames seen
        self.last_position: Optional[np.ndarray] = None
        self.last_pixel: Optional[Tuple[int, int]] = None

    def update_with(self, raw_xyz: np.ndarray, pixel: Tuple[int, int]):
        filtered = np.array(
            [self.kalman_filters[i].update(float(raw_xyz[i])) for i in range(3)],
            dtype=np.float32,
        )
        filtered = self.ema_filter.update(filtered)
        self.position_buffer.append(filtered)
        self.last_position = filtered
        self.last_pixel = pixel
        self.misses = 0
        self.age += 1

    def mark_missed(self):
        self.misses += 1


class MultiPersonTracker:
    def __init__(self, buffer_maxlen: int, max_match_dist: float = 0.8,
                 max_misses: int = 15, max_tracks: int = 6):
        self.buffer_maxlen = buffer_maxlen
        self.max_match_dist = max_match_dist
        self.max_misses = max_misses
        self.max_tracks = max_tracks
        self.tracks: List[PersonTrack] = []
        self._next_id = 0

    def update(self, detections: List[Tuple[np.ndarray, Tuple[int, int]]]) -> List[PersonTrack]:
        """detections: list of (xyz, pixel). Returns the list of live tracks."""
        unmatched_dets = list(range(len(detections)))

        # Build cost pairs (track, det) by Euclidean distance in 3-D.
        pairs = []
        for ti, track in enumerate(self.tracks):
            if track.last_position is None:
                continue
            for di in unmatched_dets:
                d = float(np.linalg.norm(detections[di][0] - track.last_position))
                pairs.append((d, ti, di))
        pairs.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_dets = set()
        for dist, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            if dist > self.max_match_dist:
                continue
            xyz, px = detections[di]
            self.tracks[ti].update_with(xyz, px)
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Unmatched existing tracks -> count a miss.
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.mark_missed()

        # Unmatched detections -> spawn new tracks (respecting the cap).
        for di in unmatched_dets:
            if di in matched_dets:
                continue
            if len(self.tracks) >= self.max_tracks:
                break
            xyz, px = detections[di]
            track = PersonTrack(self._next_id, self.buffer_maxlen)
            self._next_id += 1
            track.update_with(xyz, px)
            self.tracks.append(track)

        # Retire stale tracks.
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks


def _lerp_color(c1, c2, t: float):
    """Linear interpolate between two BGR colours."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _draw_polyline_aa(frame, pts, color, thickness):
    """Anti-aliased polyline that ignores invalid (0,0) projected points."""
    valid = [p for p in pts if not (p[0] == 0 and p[1] == 0)]
    if len(valid) < 2:
        return
    cv2.polylines(frame, [np.array(valid, dtype=np.int32)], False, color,
                  thickness, cv2.LINE_AA)


def _draw_glow_polyline(frame, pts, color, thickness):
    """Draw a soft outer 'glow' then the crisp core line, for a polished look
    that still reproduces well in print."""
    valid = [p for p in pts if not (p[0] == 0 and p[1] == 0)]
    if len(valid) < 2:
        return
    arr = [np.array(valid, dtype=np.int32)]
    # Dark halo underneath improves contrast over busy backgrounds.
    cv2.polylines(frame, arr, False, (15, 15, 15), thickness + 6, cv2.LINE_AA)
    cv2.polylines(frame, arr, False, color, thickness, cv2.LINE_AA)


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
            cv2.line(frame, points[a], points[b], COL_SKELETON, 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 2, COL_SKELETON, -1, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, points: List[Tuple[int, int]], base_thickness: int = 6,
               col_main=COL_OBSERVED, col_dark=COL_OBSERVED_DARK):
    """Observed/past trajectory: thick tapered line that grows toward the
    present, with a dark halo for contrast and small node dots."""
    valid = [p for p in points if not (p[0] == 0 and p[1] == 0)]
    if len(valid) < 2:
        return

    n = len(valid)
    for i in range(1, n):
        t = i / max(1, n - 1)                       # 0 (oldest) -> 1 (newest)
        thickness = max(2, int(base_thickness * (0.35 + 0.65 * t)))
        color = _lerp_color(col_dark, col_main, t)
        # halo
        cv2.line(frame, valid[i - 1], valid[i], (15, 15, 15), thickness + 4, cv2.LINE_AA)
        cv2.line(frame, valid[i - 1], valid[i], color, thickness, cv2.LINE_AA)

    # node markers (sparse so it stays clean)
    for i in range(0, n, max(1, n // 12)):
        cv2.circle(frame, valid[i], 3, col_main, -1, cv2.LINE_AA)


def draw_uncertainty_cone(frame: np.ndarray, points: List[Tuple[int, int]],
                          col_fill=COL_CONE):
    """A translucent cone that widens along the predicted path, conveying that
    uncertainty grows with the prediction horizon. Purely illustrative."""
    valid = [p for p in points if not (p[0] == 0 and p[1] == 0)]
    if len(valid) < 3:
        return

    overlay = frame.copy()
    n = len(valid)
    top, bot = [], []
    for i, (x, y) in enumerate(valid):
        t = i / max(1, n - 1)
        spread = int(2 + 26 * t)        # widens toward the horizon
        top.append((x, y - spread))
        bot.append((x, y + spread))
    poly = np.array(top + bot[::-1], dtype=np.int32)
    cv2.fillPoly(overlay, [poly], col_fill)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def draw_prediction(frame: np.ndarray, points: List[Tuple[int, int]], base_thickness: int = 6,
                    col_main=COL_PRED, col_dark=COL_PRED_DARK):
    """Predicted/future trajectory: bold dashed line in the contrast colour,
    ending in a clear directional arrow."""
    valid = [p for p in points if not (p[0] == 0 and p[1] == 0)]
    if len(valid) < 2:
        return

    # Dashed segments (draw every other segment) with halo for contrast.
    n = len(valid)
    for i in range(1, n):
        if (i // 1) % 2 == 0:                 # dashed pattern
            cv2.line(frame, valid[i - 1], valid[i], (15, 15, 15),
                     base_thickness + 4, cv2.LINE_AA)
            cv2.line(frame, valid[i - 1], valid[i], col_main,
                     base_thickness, cv2.LINE_AA)
        # waypoint dots regardless, so the path is readable even where dashed
        cv2.circle(frame, valid[i], 3, col_dark, -1, cv2.LINE_AA)

    # Directional arrow head at the end.
    if n >= 2:
        cv2.arrowedLine(frame, valid[-2], valid[-1], (15, 15, 15),
                        base_thickness + 4, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(frame, valid[-2], valid[-1], col_main,
                        base_thickness, cv2.LINE_AA, tipLength=0.35)


def draw_current_marker(frame: np.ndarray, point: Tuple[int, int], ring_col=COL_CURRENT_RING):
    """Crisp marker at the present position where observed meets predicted."""
    if point is None or (point[0] == 0 and point[1] == 0):
        return
    cv2.circle(frame, point, 9, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(frame, point, 8, ring_col, 2, cv2.LINE_AA)
    cv2.circle(frame, point, 4, COL_CURRENT, -1, cv2.LINE_AA)


def draw_id_badge(frame: np.ndarray, point: Tuple[int, int], track_id: int, col):
    """Small numbered badge above the current-position marker so each person's
    paths can be matched to a legend entry."""
    if point is None or (point[0] == 0 and point[1] == 0):
        return
    label = f"ID {track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    bx, by = point[0] + 10, point[1] - 14
    cv2.rectangle(frame, (bx - 3, by - th - 5), (bx + tw + 3, by + 4), (15, 15, 15), -1)
    cv2.rectangle(frame, (bx - 3, by - th - 5), (bx + tw + 3, by + 4), col, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def draw_depth_bar(frame: np.ndarray, current_z: float, future_z: List[float]):
    """Vertical depth gauge showing current depth and the predicted depth
    profile. Rendered on a subtle panel so it reads cleanly in a figure."""
    h, w = frame.shape[:2]
    bar_x = w - 64
    x0, y0, x1, y1 = bar_x, 28, w - 14, h - 28

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 16), (x1 + 4, y1 + 6), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1, cv2.LINE_AA)

    if current_z <= 0:
        return

    z_all = [current_z] + [z for z in future_z if z > 0]
    z_min = max(0.1, min(z_all))
    z_max = max(z_all) + 1e-6

    def z_to_y(z):
        t = (z - z_min) / (z_max - z_min)
        return int((y1 - 6) - t * (y1 - y0 - 12))

    cx = (x0 + x1) // 2

    # predicted depth profile as a small connected line
    prof = [(cx, z_to_y(z)) for z in future_z if z > 0]
    if len(prof) >= 2:
        cv2.polylines(frame, [np.array(prof, dtype=np.int32)], False,
                      COL_PRED, 2, cv2.LINE_AA)
    for (px, py) in prof[::3]:
        cv2.circle(frame, (px, py), 2, COL_PRED, -1, cv2.LINE_AA)

    cy = z_to_y(current_z)
    cv2.circle(frame, (cx, cy), 5, COL_OBSERVED, -1, cv2.LINE_AA)

    cv2.putText(frame, "depth", (x0 - 2, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{current_z:.2f} m", (x0 - 2, y1 + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)


def draw_legend(frame: np.ndarray, active_tracks=None):
    """Compact legend. The top block explains the observed/predicted/current
    convention; if active_tracks is provided it also lists one colour-coded row
    per tracked person so each ID in the scene can be matched to its paths.

    active_tracks: optional list of (track_id, observed_col, predicted_col).
    """
    h, w = frame.shape[:2]
    pad = 12
    n_extra = len(active_tracks) if active_tracks else 0
    box_w = 270
    box_h = 92 + (22 + 18 * n_extra if n_extra else 0)
    x0, y0 = pad, h - box_h - pad
    x1, y1 = x0 + box_w, y0 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1, cv2.LINE_AA)

    lx = x0 + 16
    ty = y0 + 26
    # observed swatch (uses person-1 colour if available, else default)
    obs_demo = active_tracks[0][1] if n_extra else COL_OBSERVED
    pred_demo = active_tracks[0][2] if n_extra else COL_PRED
    cv2.line(frame, (lx, ty), (lx + 34, ty), obs_demo, 6, cv2.LINE_AA)
    cv2.putText(frame, "Observed (past)", (lx + 46, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    # predicted swatch (dashed)
    ty += 30
    for seg in range(0, 34, 10):
        cv2.line(frame, (lx + seg, ty), (lx + seg + 6, ty), pred_demo, 6, cv2.LINE_AA)
    cv2.arrowedLine(frame, (lx + 26, ty), (lx + 36, ty), pred_demo, 4,
                    cv2.LINE_AA, tipLength=0.7)
    cv2.putText(frame, "Predicted (TrajMamba)", (lx + 46, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    # current marker
    ty += 30
    cv2.circle(frame, (lx + 17, ty), 6, COL_CURRENT_RING, 2, cv2.LINE_AA)
    cv2.circle(frame, (lx + 17, ty), 3, COL_CURRENT, -1, cv2.LINE_AA)
    cv2.putText(frame, "Current position", (lx + 46, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)

    # per-track colour key
    if n_extra:
        ty += 26
        cv2.putText(frame, "Tracked people:", (lx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        for (tid, ocol, pcol) in active_tracks:
            ty += 18
            cv2.line(frame, (lx, ty - 3), (lx + 16, ty - 3), ocol, 5, cv2.LINE_AA)
            cv2.line(frame, (lx + 20, ty - 3), (lx + 36, ty - 3), pcol, 5, cv2.LINE_AA)
            cv2.putText(frame, f"ID {tid}", (lx + 46, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, n_tracks: int,
             predicting: bool, obs_len: int, pred_len: int, show_hud: bool = True):
    if not show_hud:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (308, 96), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (8, 8), (308, 96), (90, 90, 90), 1, cv2.LINE_AA)

    status = f"Predicting  ({obs_len} -> {pred_len})" if predicting else "Collecting observations"
    status_col = COL_PRED if predicting else COL_OBSERVED
    cv2.putText(frame, "TrajMamba", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"People tracked: {n_tracks}", (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (18, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1, cv2.LINE_AA)


def sample_depth_patch(depth_image: np.ndarray, u: int, v: int, patch: int = 7) -> float:
    half = patch // 2
    values = []
    h, w = depth_image.shape[:2]
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            du = int(np.clip(u + dx, 0, w - 1))
            dv = int(np.clip(v + dy, 0, h - 1))
            d = float(depth_image[dv, du])
            if d > 0:
                values.append(d)
    if not values:
        return 0.0
    return float(np.median(values))


def deproject_pixel(u: int, v: int, depth_m: float, intrinsics: Dict[str, float]) -> np.ndarray:
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return np.array([x, y, depth_m], dtype=np.float32)


def velocity_normalize(obs_positions: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obs_vel = np.diff(obs_positions, axis=0)
    obs_vel_n = (obs_vel - global_mean) / global_std
    return obs_vel_n, obs_vel


def project_point(point: np.ndarray, intrinsics: Dict[str, float]) -> Tuple[int, int]:
    z = float(point[2])
    if z <= 0:
        return 0, 0
    u = int((float(point[0]) * intrinsics["fx"] / z) + intrinsics["cx"])
    v = int((float(point[1]) * intrinsics["fy"] / z) + intrinsics["cy"])
    return u, v


def model_uses_fast_kernels(model: TrajMamba) -> bool:
    """Return True if the instantiated model stores its Mamba parameters under
    the fast ('._fast.') sub-module, which happens when mamba-ssm is installed."""
    for key in model.state_dict().keys():
        if "._fast." in key:
            return True
    return False


def remap_slow_to_fast_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # Already in fast layout -> leave alone.
        if "._fast." in key:
            remapped[key] = value
            continue

        if "._ssm." in key:
            base, suffix = key.split("._ssm.", 1)
            # A_log / D live directly under _fast, x_proj / dt_proj stay nested.
            if suffix.startswith("A_log") or suffix.startswith("D"):
                remapped[f"{base}._fast.{suffix}"] = value
            else:
                remapped[f"{base}._fast.{suffix}"] = value
            continue

        matched = False
        for slow_name, fast_name in (
            (".in_proj.", "._fast.in_proj."),
            (".conv.", "._fast.conv1d."),
            (".out_proj.", "._fast.out_proj."),
        ):
            if slow_name in key:
                remapped[key.replace(slow_name, fast_name, 1)] = value
                matched = True
                break

        if not matched:
            remapped[key] = value

    return remapped


def remap_fast_to_slow_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "._fast." not in key:
            remapped[key] = value
            continue

        base, suffix = key.split("._fast.", 1)
        if suffix.startswith("A_log"):
            remapped[f"{base}._ssm.A_log"] = value
        elif suffix.startswith("D"):
            remapped[f"{base}._ssm.D"] = value
        elif suffix.startswith("x_proj."):
            remapped[f"{base}._ssm.x_proj.{suffix.split('.', 1)[1]}"] = value
        elif suffix.startswith("dt_proj."):
            remapped[f"{base}._ssm.dt_proj.{suffix.split('.', 1)[1]}"] = value
        elif suffix.startswith("in_proj."):
            remapped[f"{base}.in_proj.{suffix.split('.', 1)[1]}"] = value
        elif suffix.startswith("conv1d."):
            remapped[f"{base}.conv.{suffix.split('.', 1)[1]}"] = value
        elif suffix.startswith("out_proj."):
            remapped[f"{base}.out_proj.{suffix.split('.', 1)[1]}"] = value
        else:
            remapped[key] = value
    return remapped


def align_checkpoint_to_model(
    model: TrajMamba,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rewrite checkpoint keys so they match whichever Mamba implementation the
    instantiated model is actually using (fast vs slow)."""
    model_wants_fast = model_uses_fast_kernels(model)
    ckpt_is_fast = any("._fast." in key for key in state_dict.keys())

    if model_wants_fast and not ckpt_is_fast:
        # Checkpoint was saved slow, but mamba-ssm is installed -> convert up.
        return remap_slow_to_fast_keys(state_dict)
    if (not model_wants_fast) and ckpt_is_fast:
        # Checkpoint was saved fast, but we are running slow path -> convert down.
        return remap_fast_to_slow_keys(state_dict)

    # Layouts already agree.
    return state_dict


def _first_tensor_shape(state_dict: Dict[str, torch.Tensor], candidates: List[str]) -> Optional[Tuple[int, ...]]:
    for key in candidates:
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape"):
            return tuple(int(dim) for dim in tensor.shape)
    return None


def filter_state_dict_by_shape(
    model: TrajMamba,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """Keep only checkpoint tensors whose names and shapes match the model."""
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped_missing: List[str] = []
    skipped_mismatch: List[str] = []

    for key, tensor in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(tensor.shape) != tuple(model_state[key].shape):
            skipped_mismatch.append(
                f"{key}: ckpt={tuple(tensor.shape)} model={tuple(model_state[key].shape)}"
            )
            continue
        filtered[key] = tensor

    return filtered, skipped_missing, skipped_mismatch


def load_trajmamba_checkpoint(checkpoint_path: str, device: torch.device):
    # weights_only=False is required because the checkpoint stores a config dict
    # alongside the tensors. The file is produced by our own training run, so
    # this is trusted input.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    state_dict = checkpoint.get("model_state_dict", checkpoint)

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

    if config:
        print(
            "[TrajMamba] Checkpoint config: "
            f"pos_dim={config.get('pos_dim', 3)} d_model={config.get('d_model', 64)} "
            f"d_state={config.get('d_state', 16)} d_conv={config.get('d_conv', 4)} "
            f"expand={config.get('expand', 2)} n_layers={config.get('n_layers', 3)}"
        )
    using_fast = model_uses_fast_kernels(model)
    ckpt_is_fast = any("._fast." in key for key in state_dict.keys())
    print(
        f"[TrajMamba] Model uses {'fast (mamba-ssm)' if using_fast else 'slow (fallback)'} kernels; "
        f"checkpoint stored in {'fast' if ckpt_is_fast else 'slow'} layout."
    )
    state_dict = align_checkpoint_to_model(model, state_dict)

    state_dict, skipped_missing, skipped_mismatch = filter_state_dict_by_shape(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if skipped_mismatch:
        print(f"[TrajMamba] Skipped mismatched tensors: {len(skipped_mismatch)}")
        for item in skipped_mismatch[:12]:
            print(f"  - {item}")
        if len(skipped_mismatch) > 12:
            print("  - ...")
    if skipped_missing:
        print(f"[TrajMamba] Skipped unknown tensors: {len(skipped_missing)}")
        for item in skipped_missing[:8]:
            print(f"  - {item}")
        if len(skipped_missing) > 8:
            print("  - ...")
    if missing:
        print(f"[TrajMamba] Missing keys after load: {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[TrajMamba] Unexpected keys after load: {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")

    n_loaded = len(state_dict)
    n_total = len(model.state_dict())
    print(f"[TrajMamba] Loaded {n_loaded}/{n_total} tensors into the model.")
    if n_loaded == 0:
        print("[TrajMamba] WARNING: no tensors were loaded — predictions will be from random weights!")

    model.eval()
    return model, config


def resolve_task_path(cli_path: Optional[str]) -> Path:
    candidates: List[Path] = []

    if cli_path:
        candidates.append(Path(cli_path).expanduser())

    here = Path(__file__).resolve()
    candidates.extend([
        here.parent / "pose_landmarker.task",          # alongside the script
        here.parents[1] / "pose_landmarker.task",       # repo root (trajectory_estimator/)
        here.parents[2] / "pose_landmarker.task",
        here.parents[3] / "pose_landmarker.task",
        REPO_ROOT / "pose_landmarker.task",
    ])

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            return cand

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "pose_landmarker.task not found. Searched:\n  " + searched +
        "\nPass the correct path with --pose-landmarker-task."
    )


class RosTrajMambaNode(Node):
    def __init__(self, args):
        super().__init__("mamba_traj_node")
        self.args = args
        self.frame_id = args.frame_id
        self.window_name = args.window_name
        self.show_hud = True
        self.show_skeleton = True
        self.screenshot_dir = Path(args.screenshot_dir).expanduser()
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._shot_count = 0
        self.intrinsics: Optional[Dict[str, float]] = None
        self.intrinsics_ready = False
        self.last_camera_info_stamp = None
        self.latest_fps = 0.0
        self.prev_time = time.time()
        self.bridge = CvBridge()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.model, self.config = load_trajmamba_checkpoint(args.checkpoint, self.device)
        self.global_vel_mean = np.array(self.config.get("vel_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.global_vel_std = np.array(self.config.get("vel_std", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.pos_dim = int(self.config.get("pos_dim", 3))
        self.obs_len = int(self.config.get("obs_len", 20))
        self.pred_len = int(self.config.get("pred_len", 30))

        buffer_maxlen = max(60, self.obs_len * 3)
        self.tracker = MultiPersonTracker(
            buffer_maxlen=buffer_maxlen,
            max_match_dist=float(args.max_match_dist),
            max_misses=int(args.max_misses),
            max_tracks=int(args.max_people),
        )
        task_path = resolve_task_path(args.pose_landmarker_task)
        self.get_logger().info(f"Using pose landmarker task: {task_path}")

        base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=int(args.max_people),     # detect multiple people per frame
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

        self.rgb_is_compressed = args.rgb_topic.endswith("/compressed")
        rgb_msg_type = CompressedImage if self.rgb_is_compressed else Image

        self.rgb_sub = message_filters.Subscriber(self, rgb_msg_type, args.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, args.depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=args.queue_size,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        # Publishers for RViz / visualization
        self.vis_pub = self.create_publisher(Image, "/humans/visualization", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/humans/markers", 1)
        self.pose_pub = self.create_publisher(PoseArray, "/humans/predicted_trajectories", 1)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"Subscribed to RGB={args.rgb_topic}, depth={args.depth_topic}, camera_info={args.camera_info_topic}"
        )
        self.get_logger().info(f"Checkpoint: {args.checkpoint}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(
            "Keys:  q=quit   s=save screenshot   h=toggle HUD   k=toggle skeleton")
        self.get_logger().info(f"Screenshots -> {self.screenshot_dir}")

    def camera_info_callback(self, msg: CameraInfo):
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

    def synced_callback(self, rgb_msg, depth_msg):
        if not self.intrinsics_ready or self.intrinsics is None:
            return

        try:
            frame = self.decode_rgb(rgb_msg)
            depth_image = self.decode_depth(depth_msg)
        except Exception as exc:
            self.get_logger().warn(f"Frame decode failed: {exc}")
            return

        h, w = frame.shape[:2]
        # MediaPipe expects RGB; the decoded OpenCV frame is BGR.
        rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_mp)
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        all_poses = result.pose_landmarks if result.pose_landmarks else []

        # ----- Detect every person's hip centre this frame -----
        detections: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        pose_for_pixel: List[Tuple[Optional[Tuple[int, int]], object]] = []
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
        legend_tracks = []                      # (id, observed_col, predicted_col)
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id
        marker_array = MarkerArray()
        pose_array = PoseArray()
        pose_array.header = hdr

        # ----- Per-person prediction + drawing -----
        # Draw far-to-near (largest depth first) so closer people overlay nicely.
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

                # ---- RViz messages (namespaced per track) ----
                m = Marker()
                m.header = hdr
                m.ns = f"mamba_pred_{track.id}"
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
                # Not enough history yet: just show the building observed trail.
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

        # Publish aggregated visualization / markers / poses.
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
        draw_hud(frame, self.latest_fps, n_tracks, predicting,
                 self.obs_len, self.pred_len, self.show_hud)
        draw_legend(frame, active_tracks=legend_tracks if legend_tracks else None)

        if self.frame_id and self.show_hud:
            cv2.putText(frame, "", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

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
        path = self.screenshot_dir / f"trajmamba_{ts}_{self._shot_count:03d}.png"
        try:
            # Save at full quality, no compression artifacts, for print figures.
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
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrajMamba node subscribing to RGB-D ROS topics")
    parser.add_argument("--checkpoint", default=str(THIS_DIR.parent / "checkpoints" / "best_model.pth"))
    parser.add_argument("--pose-landmarker-task", default=str(THIS_DIR.parent / "checkpoints" / "pose_landmarker.task"))
    parser.add_argument("--rgb-topic", default="/head_front_camera/rgb/image_raw")
    parser.add_argument("--depth-topic", default="/head_front_camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/head_front_camera/rgb/camera_info")
    parser.add_argument("--frame-id", default="head_front_camera_rgb_frame")
    parser.add_argument("--window-name", default="TrajMamba ROS Live")
    parser.add_argument("--screenshot-dir", default=str(THIS_DIR.parent / "figures"),
                        help="Directory where 's' key saves publication screenshots.")
    parser.add_argument("--queue-size", type=int, default=10)
    parser.add_argument("--sync-slop", type=float, default=0.08)
    parser.add_argument("--max-people", type=int, default=4,
                        help="Maximum number of people to detect and track at once.")
    parser.add_argument("--max-match-dist", type=float, default=0.8,
                        help="Max 3-D distance (m) to associate a detection to an existing track.")
    parser.add_argument("--max-misses", type=int, default=15,
                        help="Drop a track after this many consecutive frames without a match.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    rclpy.init()
    node = RosTrajMambaNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()