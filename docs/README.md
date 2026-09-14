# TrajMamba — Selective SSM for 3-D Person Trajectory Prediction

Bidirectional Mamba (Selective State-Space Model) for person trajectory prediction,
trained on the **RGBD Bonn Person-Tracking Dataset**.

---

## Architecture Overview

```
Input  (B, obs_len-1, D)   ← normalised velocity sequence
  │
InputEmbed  Linear(D → d_model) + LayerNorm
  │
  ┌────┴─────────────────────────────────────────────┐
  │   BidirectionalMambaBlock  ×  n_layers           │
  │                                                   │
  │   ┌── Forward  MambaBlock ──────────────────┐    │
  │   │  norm → in_proj → depthwise-conv        │    │
  │   │       → Selective SSM (S6)              │    │
  │   │       → gate(z) → out_proj → residual   │    │
  │   └─────────────────────────────────────────┘    │
  │   ┌── Backward MambaBlock (flipped) ────────┐    │
  │   └─────────────────────────────────────────┘    │
  │   Learned soft fusion gate                        │
  └──────────────────────────────────────────────────┘
  │
AsymmetricDecoder  (final state → GRU → MLP)
  │
Output (B, pred_len-1, D)  ← predicted future velocities
```

**Key design choices** (from CVPR 2025 & 2026 literature):
- **Selective SSM (S6)**: input-dependent B, C, Δ allow the model to "select"
  which parts of the past to remember — critical for trajectory prediction where
  sudden direction changes must be captured.
- **Bidirectional encoding**: forward + backward scans fused by a learned gate,
  substantially improving encoder quality (see Bi-Mamba ablations).
- **Velocity-space training**: the model predicts velocity increments, not
  absolute positions. This makes the task translation-invariant and stabilises
  training.
- **Pure-PyTorch fallback**: the selective scan is implemented in standard
  PyTorch so the code runs on any device. Install `mamba-ssm` for 4–5× faster
  CUDA kernels (optional).

---

## Dataset

**RGBD Bonn Person Tracking** — indoor RGB-D sequences with centimetre-accurate
groundtruth poses from an external motion-capture system.

```
rgbd_bonn_person_tracking/
├── rgb/          ← colour frames (not used for training)
├── depth/        ← depth maps   (not used for training)
├── rgb.txt       ← frame timestamps
├── depth.txt
└── groundtruth.txt  ← used for training
    # timestamp  tx  ty  tz  qx  qy  qz  qw
    1548265882.52459  0.4652  -1.1971  1.9601  ...
```

The loader uses only `(tx, ty, tz)` — the 3-D position of the person in metres.

---

## File Structure

```
trajectory_estimator/
├── bonn_dataset_loader.py   Dataset loader for RGBD Bonn
├── mamba_model.py           TrajMamba architecture (pure-PyTorch + optional CUDA)
├── metrics.py               ADE / FDE metrics + TrajLoss
├── train_mamba.py           Training script (main entry point)
├── evaluate.py              Evaluation + trajectory visualisation
├── checkpoint_loader.py     Load / inspect / list checkpoints
├── requirements.txt         Dependencies
└── README.md                This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy matplotlib
# Optional (fast CUDA kernels, Linux + NVIDIA GPU):
pip install mamba-ssm causal-conv1d --no-build-isolation
```
INSTALLING MAMBA on pc 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install mamba-ssm --no-build-isolation


on jetson 

1. Stash your edits and switch to v1.2.2:
bashcd ~/basheer_ws/src/n-stac/trajectory_estimator/cl/mamba
git stash
git checkout v1.2.2
2. Downgrade transformers (critical — 5.9.0 breaks on Jetson PyTorch):
bashpip install "transformers>=4.36.0,<5.0.0"
3. Rebuild mamba_ssm v1.2.2 from scratch:
bashpip uninstall mamba-ssm -y
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 pip install --no-build-isolation .
4. Also install causal-conv1d (required by v1.x):
bashcd ~/basheer_ws/src/n-stac/trajectory_estimator/cl/causal-conv1d
git checkout main   # or latest tag
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 pip install --no-build-isolation .
5. Verify:
bashpython -c "from mamba_ssm import Mamba; print('Mamba OK')"

### 2. Smoke-test the dataset loader

```bash
python bonn_dataset_loader.py \
    /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_person_tracking
```

### 3. Train

```bash
# Single scenario
python train_mamba.py \
  --dataset_roots /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_person_tracking \
  --checkpoint_dir ./checkpoints \
  --epochs 150 --batch_size 32 --lr 3e-4

# Multiple datasets
python train_mamba.py \
  --dataset_roots \
    /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_person_tracking \
    /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_crowd \
  --checkpoint_dir ./checkpoints
```
python train_mamba.py --dataset_roots /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_person_tracking /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_crowd --epochs 150 --batch_size 32 --d_model 64 --n_layers 3 --dropout 0.3 --lr 5e-4
### 4. Evaluate

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --dataset_roots /media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_person_tracking \
  --plot
```

### 5. Resume interrupted training

```bash
python train_mamba.py \
  --dataset_roots ... \
  --checkpoint_dir ./checkpoints \
  --resume
```

### 6. Live camera demo (RealSense or webcam)

```bash
# RealSense (for 3-D checkpoints)
python live_camera_stream.py \
  --checkpoint ./checkpoints/best_model.pth \
  --camera realsense

# Webcam (for 2-D checkpoints)
python live_camera_stream.py \
  --checkpython /home/basheer/projects/trajectory_estimator/cl/mamba_traj_node.pypoint ./checkpoints/best_model.pth \
  --camera webcam --webcam_id 0
```
for ros2 node streaming 
python /home/basheer/basheer_ws/src/nit_human_traj_estimation/nit_human_traj_estimation/human_traj_estimation.py
Controls: press `r` to reselect ROI, `q` to quit.

---

## Hyperparameter Guide

| Arg | Default | Notes |
|---|---|---|
| `--obs_len` | 20 | Observed steps after downsampling. At 10 Hz → 2 s |
| `--pred_len` | 30 | Predicted steps. At 10 Hz → 3 s |
| `--downsample` | 3 | Raw ~30 Hz → 10 Hz. Use 1 for full rate |
| `--d_model` | 64 | Mamba hidden dim. Try 128 for more capacity |
| `--d_state` | 16 | SSM state size N. 16 is standard |
| `--n_lapython /home/basheer/projects/trajectory_estimator/cl/mamba_traj_node.pyyers` | 3 | BiMamba encoder depth |
| `--lr` | 3e-4 | AdamW initial LR. Cosine-annealed |
| `--T0` | 30 | Cosine restart period (epochs) |
| `--patience` | 25 | Early-stopping patience on val ADE |

### Tips for the Bonn dataset (small dataset)

- Start wpython /home/basheer/projects/trajectory_estimator/cl/mamba_traj_node.pyith `--d_model 64 --n_layers 3` (≈ 200 k params) — avoids overfitting.
- Use `--downsample 3` (10 Hz) — reduces noise, more regular step size.
- If you have multiple Bonn scenarios, pass the **parent folder**; the loader
  will pool all sequences, giving more training data.
- `--augment` (speed ±10%, axis flip) is implemented in the dataset — enable
  it by setting `augment=True` in the dataset constructor for the training split.

---

## Metrics

| Metric | Definition |
|---|---|
| **ADE** | Mean Euclidean distance between predicted and GT positions over all future steps |
| **FDE** | Euclidean distance at the final predicted step only |

Both are reported in **metres** in the Bonn coordinate system.

---

## References

1. Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv 2312.00752, 2023.
2. Huang et al., *Trajectory Mamba: Efficient Attention-Mamba Forecasting Model*, **CVPR 2025**.
3. He et al., *TrajMamba: Ego-Motion-Guided Mamba for Pedestrian Trajectory Prediction*, arXiv 2603.14739, 2026.
4. Lai et al., *iDMaTraj: Improved Diffusion Mamba for Stochastic Trajectory Prediction*, Computers 2026.
5. Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012 (Bonn dataset).
