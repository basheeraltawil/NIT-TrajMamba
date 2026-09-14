#!/usr/bin/env python3

import copy
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Quaternion → yaw (optional, not used in position-only mode)
# ---------------------------------------------------------------------------
def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw angle (rotation around Z) from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BonnTrajectoryDataset(Dataset):
    """
    Bonn RGBD person-tracking trajectory dataset.

    Supports multiple scenario folders (rgbd_bonn_person_tracking,
    rgbd_bonn_person_tracking2, etc.) each containing a groundtruth.txt.

    Args:
        data_dirs   : list of paths, each containing a groundtruth.txt
        obs_len     : number of observed time steps
        pred_len    : number of predicted time steps
        downsample  : keep every N-th frame (raw ~30 Hz → 10 Hz with N=3)
        use_3d      : if True use (tx, ty, tz); if False use only (tx, ty)
        normalize   : subtract first-position so all seqs start at origin
        augment     : random horizontal flip & speed perturbation (train only)
    """

    def __init__(
        self,
        data_dirs: List[str],
        obs_len: int = 20,
        pred_len: int = 30,
        downsample: int = 3,
        use_3d: bool = True,
        normalize: bool = True,
        augment: bool = False,
    ):
        self.obs_len   = obs_len
        self.pred_len  = pred_len
        self.downsample = downsample
        self.use_3d    = use_3d
        self.normalize = normalize
        self.augment   = augment
        self.pos_dim   = 3 if use_3d else 2

        self.sequences: List[np.ndarray] = []   # each: (obs_len+pred_len, pos_dim)
        self._vel_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None

        for d in data_dirs:
            self._load_dir(Path(d))

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No trajectory windows found. Check data_dirs: {data_dirs}"
            )

        # Compute global velocity statistics for normalisation
        self._compute_vel_stats()

        print(
            f"[BonnDataset] dirs={len(data_dirs)}  "
            f"windows={len(self.sequences)}  "
            f"obs={obs_len}  pred={pred_len}  "
            f"dim={self.pos_dim}  downsample=1/{downsample}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_dir(self, root: Path):
        gt_file = root / "groundtruth.txt"
        if not gt_file.exists():
            print(f"[BonnDataset] WARNING: groundtruth.txt not found in {root}")
            return

        timestamps, positions = [], []

        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    ts = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue

                timestamps.append(ts)
                if self.use_3d:
                    positions.append([tx, ty, tz])
                else:
                    positions.append([tx, ty])

        if len(positions) < 10:
            print(f"[BonnDataset] WARNING: too few points in {gt_file} ({len(positions)})")
            return

        positions = np.array(positions, dtype=np.float32)

        # Downsample
        positions = positions[::self.downsample]

        seq_len = self.obs_len + self.pred_len

        # Sliding-window extraction
        for start in range(0, len(positions) - seq_len + 1):
            window = positions[start : start + seq_len]

            if self.normalize:
                window = window - window[0]          # relative to start

            self.sequences.append(window)

    # ------------------------------------------------------------------
    # Velocity statistics (computed over training set only in practice)
    # ------------------------------------------------------------------
    def _compute_vel_stats(self):
        all_vel = []
        for seq in self.sequences:
            obs_vel = np.diff(seq[: self.obs_len], axis=0)
            all_vel.append(obs_vel)
        all_vel = np.concatenate(all_vel, axis=0)
        self._vel_mean = all_vel.mean(axis=0).astype(np.float32)
        self._vel_std  = (all_vel.std(axis=0) + 1e-6).astype(np.float32)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Random speed perturbation ±10 % and axis flip."""
        if np.random.rand() < 0.5:
            seq = seq * np.random.uniform(0.9, 1.1)
        if np.random.rand() < 0.5:
            seq[:, 0] = -seq[:, 0]          # flip X
        if self.use_3d and np.random.rand() < 0.3:
            seq[:, 1] = -seq[:, 1]          # flip Y
        return seq

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx].copy()  # (obs_len + pred_len, pos_dim)

        if self.augment:
            seq = self._augment(seq)

        obs  = seq[: self.obs_len]          # (obs_len, pos_dim)
        pred = seq[self.obs_len :]          # (pred_len, pos_dim)

        # Velocities
        obs_vel  = np.diff(obs,  axis=0)   # (obs_len-1, pos_dim)
        pred_vel = np.diff(pred, axis=0)   # (pred_len-1, pos_dim)

        # Normalise with training-set statistics
        obs_vel_n  = (obs_vel  - self._vel_mean) / self._vel_std
        pred_vel_n = (pred_vel - self._vel_mean) / self._vel_std

        return (
            torch.tensor(obs_vel_n,  dtype=torch.float32),   # (obs_len-1,  dim)
            torch.tensor(pred_vel_n, dtype=torch.float32),   # (pred_len-1, dim)
            torch.tensor(obs,        dtype=torch.float32),   # raw positions (for metric)
            torch.tensor(pred,       dtype=torch.float32),   # raw positions (for metric)
            torch.tensor(self._vel_mean, dtype=torch.float32),
            torch.tensor(self._vel_std,  dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Helper: discover all scenario sub-folders inside a root
# ---------------------------------------------------------------------------
def find_scenario_dirs(root: str) -> List[str]:
    """
    Walk root and return every directory that contains a groundtruth.txt.
    Works whether root IS the scenario dir or a parent of many scenarios.
    """
    root = Path(root)
    dirs = []

    # root itself?
    if (root / "groundtruth.txt").exists():
        dirs.append(str(root))

    # sub-directories?
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "groundtruth.txt").exists():
            dirs.append(str(sub))

    return dirs


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def create_dataloaders(
    dataset_roots: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    obs_len: int = 20,
    pred_len: int = 30,
    downsample: int = 3,
    use_3d: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train / val DataLoaders for the Bonn RGBD dataset.

    Args:
        dataset_roots : list of dataset roots (or scenario dirs) to pool
        batch_size    : mini-batch size
        num_workers   : DataLoader worker processes
        val_fraction  : fraction of windows held out for validation
        obs_len       : observed trajectory length (frames after downsampling)
        pred_len      : predicted trajectory length (frames after downsampling)
        downsample    : keep every N-th raw frame
        use_3d        : True → 3-D, False → 2-D
        seed          : random seed for the train/val split

    Returns:
        train_loader, val_loader
    """
    scenario_dirs: List[str] = []
    for root in dataset_roots:
        scenario_dirs.extend(find_scenario_dirs(root))

    scenario_dirs = list(dict.fromkeys(scenario_dirs))
    if not scenario_dirs:
        roots_str = ", ".join(dataset_roots)
        raise FileNotFoundError(
            f"No groundtruth.txt found under {roots_str}. "
            "Please check the dataset path(s)."
        )

    print(f"[DataLoader] Found {len(scenario_dirs)} scenario(s):")
    for d in scenario_dirs:
        print(f"  {d}")

    full_dataset = BonnTrajectoryDataset(
        data_dirs=scenario_dirs,
        obs_len=obs_len,
        pred_len=pred_len,
        downsample=downsample,
        use_3d=use_3d,
        normalize=True,
        augment=False,          # will be set per-split below
    )

    # Split
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    train_dataset = copy.deepcopy(full_dataset)
    val_dataset = copy.deepcopy(full_dataset)

    train_dataset.sequences = [full_dataset.sequences[i] for i in train_set.indices]
    val_dataset.sequences = [full_dataset.sequences[i] for i in val_set.indices]

    train_dataset._compute_vel_stats()
    val_dataset._vel_mean = train_dataset._vel_mean.copy()
    val_dataset._vel_std = train_dataset._vel_std.copy()

    train_dataset.augment = True
    val_dataset.augment = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"[DataLoader] train={n_train}  val={n_val}  "
        f"train_batches={len(train_loader)}  val_batches={len(val_loader)}"
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    root = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/media/basheer/OVGU/Datasets/rgbd_bonn_dataset"
    )

    train_loader, val_loader = create_dataloaders(
        [root],
        batch_size=16,
        obs_len=20,
        pred_len=30,
        downsample=3,
        use_3d=True,
    )

    obs_v, pred_v, obs_pos, pred_pos, v_mean, v_std = next(iter(train_loader))
    print(f"obs_vel   shape : {obs_v.shape}")      # (B, obs_len-1, 3)
    print(f"pred_vel  shape : {pred_v.shape}")     # (B, pred_len-1, 3)
    print(f"obs_pos   shape : {obs_pos.shape}")    # (B, obs_len,   3)
    print(f"pred_pos  shape : {pred_pos.shape}")   # (B, pred_len,  3)
    print("Smoke test passed ✓")
