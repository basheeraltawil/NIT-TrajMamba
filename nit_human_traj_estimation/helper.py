#!/usr/bin/env python3
"""
helper.py
---------
Drawing, filtering, tracking and 3-D geometry utilities used by the
`human_traj_estimation` node. Kept separate from the node itself so the ROS
class stays focused on wiring topics/parameters, mirroring the
`nit_pose_estimation` package's helper.py pattern.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ANSI colors for important log lines — matches nit_pose_estimation's
# colors.GREEN / colors.YELLOW / colors.RESET usage exactly.
# ---------------------------------------------------------------------------
class colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


# ---------------------------------------------------------------------------
# Checkpoint auto-download — mirrors nit_pose_estimation's behavior of
# fetching model weights automatically on first run instead of requiring the
# user to place files manually. MMPoseInferencer does this itself against
# OpenMMLab's mirror; TrajMamba/MediaPipe have no such built-in mechanism, so
# we replicate it here against an OVGU Nextcloud share.
#
# Handles both possible share shapes since this can't be verified from the
# sandbox that generated this file (cloud.ovgu.de blocks automated access):
#   - a single-file share  -> downloaded bytes are the file itself
#   - a folder share       -> downloaded bytes are a .zip; every required
#                              filename is searched for anywhere inside it
# ---------------------------------------------------------------------------
def ensure_checkpoints_downloaded(
    required_files: Dict[str, Path],
    share_url: str,
    log=print,
) -> None:
    """
    required_files: {filename: destination_path} — e.g.
        {'best_model.pth': Path('.../checkpoints/best_model.pth'),
         'pose_landmarker.task': Path('.../checkpoints/pose_landmarker.task')}
    share_url: Nextcloud public share URL for the FOLDER containing these
        files, e.g. 'https://cloud.ovgu.de/s/JLQQa6f2e8qb5gn' — no trailing
        '/download' or filename; each file is fetched individually as
        f'{share_url}/download/{filename}' (Nextcloud's per-file-in-a-shared-
        folder download convention).

    No-ops per-file if it already exists locally. Only touches the network
    for files that are actually missing.
    """
    missing = {name: dest for name, dest in required_files.items() if not dest.exists()}
    if not missing:
        return

    log(f"{colors.YELLOW}Missing checkpoint file(s): {', '.join(missing)} — "
        f"downloading from {share_url} ...{colors.RESET}")

    try:
        import requests
    except ImportError:
        log(f"{colors.RED}Cannot auto-download checkpoints: the 'requests' package "
            f"is not installed. Run `pip install requests`, or place the file(s) "
            f"manually: {list(missing.values())}{colors.RESET}")
        return

    base = share_url.rstrip('/')
    for filename, dest in missing.items():
        file_url = f"{base}/download/{filename}"
        try:
            with requests.get(file_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, 'wb') as out:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        out.write(chunk)
            log(f"{colors.GREEN}Downloaded {filename} -> {dest}{colors.RESET}")
        except Exception as exc:
            log(f"{colors.RED}Failed to download {filename} from {file_url}: {exc}. "
                f"Place it manually at {dest}.{colors.RESET}")

# ---------------------------------------------------------------------------
# Pose landmark indices (MediaPipe Pose / BlazePose topology)
# ---------------------------------------------------------------------------
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COL_OBSERVED = (200, 160, 0)      # observed (past) trajectory
COL_OBSERVED_DARK = (120, 95, 0)
COL_PRED = (40, 120, 255)         # predicted (future) trajectory
COL_PRED_DARK = (20, 70, 170)
COL_CONE = (60, 140, 255)         # uncertainty cone fill
COL_CURRENT = (255, 255, 255)     # current position marker (core)
COL_CURRENT_RING = (40, 120, 255)
COL_SKELETON = (235, 235, 235)
COL_TEXT = (255, 255, 255)
COL_PANEL_BG = (28, 28, 28)

TRAIL_COLOR = COL_OBSERVED
PRED_COLOR = COL_PRED

TRACK_PALETTE = [
    # observed,           observed_dark,     predicted,          predicted_dark
    ((200, 160, 0),       (120, 95, 0),      (40, 120, 255),     (20, 70, 170)),
    ((80, 200, 80),       (40, 110, 40),     (200, 80, 220),     (120, 40, 130)),
    ((230, 120, 60),      (140, 70, 30),     (40, 200, 255),     (25, 120, 160)),
    ((90, 90, 230),       (45, 45, 140),     (220, 200, 60),     (130, 120, 35)),
    ((180, 160, 120),     (100, 90, 70),     (60, 220, 180),     (35, 130, 105)),
    ((200, 100, 180),     (110, 55, 100),    (90, 220, 240),     (55, 130, 145)),
]


def track_colors(track_id: int):
    """Return the (observed, observed_dark, pred, pred_dark) BGR tuple for a
    given integer track id, cycling through the palette."""
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Multi-person tracking
# ---------------------------------------------------------------------------
class PersonTrack:
    """Per-person state: a smoothed 3-D position history plus its own filters.

    Each tracked person keeps an independent observation buffer so TrajMamba
    can be run separately for every individual in the scene.
    """

    def __init__(self, track_id: int, buffer_maxlen: int):
        self.id = track_id
        self.position_buffer: Deque[np.ndarray] = deque(maxlen=buffer_maxlen)
        self.kalman_filters = [Kalman1D() for _ in range(3)]
        self.ema_filter = EMAFilter(alpha=0.35)
        self.misses = 0
        self.age = 0
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

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.mark_missed()

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

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks


# ---------------------------------------------------------------------------
# 3-D geometry helpers
# ---------------------------------------------------------------------------
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


def project_point(point: np.ndarray, intrinsics: Dict[str, float]) -> Tuple[int, int]:
    z = float(point[2])
    if z <= 0:
        return 0, 0
    u = int((float(point[0]) * intrinsics["fx"] / z) + intrinsics["cx"])
    v = int((float(point[1]) * intrinsics["fy"] / z) + intrinsics["cy"])
    return u, v


def velocity_normalize(obs_positions: np.ndarray, global_mean: np.ndarray,
                        global_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obs_vel = np.diff(obs_positions, axis=0)
    obs_vel_n = (obs_vel - global_mean) / global_std
    return obs_vel_n, obs_vel


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _lerp_color(c1, c2, t: float):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


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
        t = i / max(1, n - 1)
        thickness = max(2, int(base_thickness * (0.35 + 0.65 * t)))
        color = _lerp_color(col_dark, col_main, t)
        cv2.line(frame, valid[i - 1], valid[i], (15, 15, 15), thickness + 4, cv2.LINE_AA)
        cv2.line(frame, valid[i - 1], valid[i], color, thickness, cv2.LINE_AA)

    for i in range(0, n, max(1, n // 12)):
        cv2.circle(frame, valid[i], 3, col_main, -1, cv2.LINE_AA)


def draw_uncertainty_cone(frame: np.ndarray, points: List[Tuple[int, int]], col_fill=COL_CONE):
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
        spread = int(2 + 26 * t)
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

    n = len(valid)
    for i in range(1, n):
        if (i // 1) % 2 == 0:
            cv2.line(frame, valid[i - 1], valid[i], (15, 15, 15),
                      base_thickness + 4, cv2.LINE_AA)
            cv2.line(frame, valid[i - 1], valid[i], col_main,
                      base_thickness, cv2.LINE_AA)
        cv2.circle(frame, valid[i], 3, col_dark, -1, cv2.LINE_AA)

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
    convention; if active_tracks is provided it also lists one colour-coded
    row per tracked person so each ID in the scene can be matched to its
    paths.

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
    obs_demo = active_tracks[0][1] if n_extra else COL_OBSERVED
    pred_demo = active_tracks[0][2] if n_extra else COL_PRED
    cv2.line(frame, (lx, ty), (lx + 34, ty), obs_demo, 6, cv2.LINE_AA)
    cv2.putText(frame, "Observed (past)", (lx + 46, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    ty += 30
    for seg in range(0, 34, 10):
        cv2.line(frame, (lx + seg, ty), (lx + seg + 6, ty), pred_demo, 6, cv2.LINE_AA)
    cv2.arrowedLine(frame, (lx + 26, ty), (lx + 36, ty), pred_demo, 4,
                     cv2.LINE_AA, tipLength=0.7)
    cv2.putText(frame, "Predicted (TrajMamba)", (lx + 46, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    ty += 30
    cv2.circle(frame, (lx + 17, ty), 6, COL_CURRENT_RING, 2, cv2.LINE_AA)
    cv2.circle(frame, (lx + 17, ty), 3, COL_CURRENT, -1, cv2.LINE_AA)
    cv2.putText(frame, "Current position", (lx + 46, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)

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