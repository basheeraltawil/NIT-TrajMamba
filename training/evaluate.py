#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe
import matplotlib.pyplot as plt
from pathlib import Path

from checkpoint_loader import load_checkpoint
from bonn_dataset_loader import create_dataloaders
from metrics import ade_fde


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device=str(device))

    obs_len  = config["obs_len"]
    pred_len = config["pred_len"]
    use_3d   = config.get("use_3d", True)
    ds       = config.get("downsample", 3)

    _, val_loader = create_dataloaders(
        dataset_roots=args.dataset_roots,
        batch_size=args.batch_size,
        obs_len=obs_len,
        pred_len=pred_len,
        downsample=ds,
        use_3d=use_3d,
        num_workers=2,
    )

    total_ade = total_fde = total_batches = 0.0

    all_obs_pos   = []
    all_pred_pos  = []
    all_gt_pos    = []

    for batch in val_loader:
        obs_vel, pred_vel, obs_pos, pred_pos, vel_mean, vel_std = [
            b.to(device) for b in batch
        ]

        pred_hat = model(obs_vel)                     # (B, pred_len-1, D)

        # Denormalise
        vm = vel_mean[:, None, :]
        vs = vel_std[:,  None, :]
        pred_vel_m = pred_hat * vs + vm               # metres/step

        # Integrate
        B, T, D = pred_vel_m.shape
        obs_last = obs_pos[:, -1]
        pos_seq = [obs_last.unsqueeze(1)]
        for t in range(T):
            pos_seq.append(pos_seq[-1] + pred_vel_m[:, t:t+1])
        pred_positions = torch.cat(pos_seq[1:], dim=1)

        gt_positions = pred_pos[:, :T]

        ade, fde = ade_fde(pred_positions, gt_positions)
        total_ade    += ade
        total_fde    += fde
        total_batches += 1

        # Collect for plotting
        if len(all_pred_pos) < args.num_samples:
            all_obs_pos.append(obs_pos.cpu().numpy())
            all_pred_pos.append(pred_positions.cpu().numpy())
            all_gt_pos.append(gt_positions.cpu().numpy())

    avg_ade = total_ade / total_batches
    avg_fde = total_fde / total_batches

    print(f"\n{'='*40}")
    print(f"  ADE : {avg_ade:.4f} m")
    print(f"  FDE : {avg_fde:.4f} m")
    print(f"{'='*40}\n")

    # ---- Optional plot --------------------------------------------------
    if args.plot:
        obs_arr  = np.concatenate(all_obs_pos,  axis=0)   # (N, obs_len, D)
        pred_arr = np.concatenate(all_pred_pos, axis=0)   # (N, T, D)
        gt_arr   = np.concatenate(all_gt_pos,   axis=0)   # (N, T, D)

        n_show = min(args.num_samples, len(obs_arr))
        pos_dim = obs_arr.shape[-1]

        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
        if n_show == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            obs  = obs_arr[i]
            pred = pred_arr[i]
            gt   = gt_arr[i]

            ax.plot(obs[:, 0],  obs[:, 1],  "b.-",  label="Observed", linewidth=1.5)
            ax.plot(gt[:, 0],   gt[:, 1],   "g.-",  label="Ground Truth", linewidth=1.5)
            ax.plot(pred[:, 0], pred[:, 1], "r.--", label="Predicted", linewidth=1.5)
            ax.set_title(f"Sample {i+1}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend(fontsize=7)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"TrajMamba  |  ADE={avg_ade:.4f} m  FDE={avg_fde:.4f} m", fontsize=12)
        plt.tight_layout()

        out_path = Path(args.checkpoint).parent / "eval_predictions.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Plot saved to {out_path}")

    return avg_ade, avg_fde

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate TrajMamba checkpoint")
    p.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint")
    p.add_argument(
        "--dataset_roots",
        nargs="+",
        default=["/media/basheer/OVGU/Datasets/rgbd_bonn_dataset/rgbd_bonn_person_tracking"],
    )
    p.add_argument("--batch_size",   type=int,  default=64)
    p.add_argument("--num_samples",  type=int,  default=6,
                   help="Number of trajectories to visualise")
    p.add_argument("--plot",         action="store_true",
                   help="Save a prediction plot as eval_predictions.png")
    args = p.parse_args()

    evaluate(args)
