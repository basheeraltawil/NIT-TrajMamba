#!/usr/bin/env python3

import os
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from bonn_dataset_loader import create_dataloaders
from mamba_model import TrajMamba, count_parameters
from metrics import TrajLoss, compute_metrics_from_batch


def train_epoch(
    model:      nn.Module,
    loader,
    optimizer:  torch.optim.Optimizer,
    loss_fn:    nn.Module,
    device:     torch.device,
    epoch:      int,
    log_every:  int = 20,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        obs_vel, pred_vel = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        pred_hat = model(obs_vel)
        n = min(pred_hat.shape[1], pred_vel.shape[1])
        loss = loss_fn(pred_hat[:, :n], pred_vel[:, :n])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_every == 0:
            print(
                f"  Ep {epoch:>4}  Batch {batch_idx+1:>4}/{len(loader)}  "
                f"Loss {loss.item():.6f}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(
    model:   nn.Module,
    loader,
    device:  torch.device,
) -> Tuple[float, float, float]:
    """Returns (avg_loss, avg_ade_m, avg_fde_m)."""
    model.eval()
    tot_loss = tot_ade = tot_fde = 0.0

    for batch in loader:
        loss, ade, fde = compute_metrics_from_batch(model, batch, device)
        tot_loss += loss
        tot_ade  += ade
        tot_fde  += fde

    n = len(loader)
    return tot_loss / n, tot_ade / n, tot_fde / n


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, metrics: dict, config: dict):
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics":              metrics,
            "config":               config,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def train(args):
    # ---- Device --------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f" TrajMamba Training")
    print(f"{'='*60}")
    print(f" Device        : {device}")
    print(f" Dataset roots : {', '.join(args.dataset_roots)}")
    print(f" Checkpoints   : {args.checkpoint_dir}")
    print(f" obs/pred      : {args.obs_len}/{args.pred_len}")
    print(f" 3-D           : {args.use_3d}")
    print(f" Downsample    : 1/{args.downsample}")
    print(f"{'='*60}\n")

    # ---- Checkpoint dir ------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----------------------------------------------------------
    train_loader, val_loader = create_dataloaders(
        dataset_roots=args.dataset_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        downsample=args.downsample,
        use_3d=args.use_3d,
        seed=args.seed,
    )
    # Extract the global velocity stats from the dataset
    global_vel_mean = train_loader.dataset._vel_mean.tolist()
    global_vel_std  = train_loader.dataset._vel_std.tolist()
    pos_dim = 3 if args.use_3d else 2

    # ---- Model ---------------------------------------------------------
    model = TrajMamba(
        pos_dim=pos_dim,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(f"[Model] TrajMamba  params={n_params:,}")

    # ---- Config (persisted with every checkpoint) ----------------------
    config = vars(args)
    config["pos_dim"]   = pos_dim
    config["n_params"]  = n_params
    config["device"]    = str(device)
    config["timestamp"] = datetime.now().isoformat()
    # ADD THESE TWO LINES:
    config["vel_mean"]  = global_vel_mean 
    config["vel_std"]   = global_vel_std

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Optimiser / Scheduler / Loss ----------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98),
    )

    # Cosine annealing with warm restarts — helps escape flat minima
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T0, T_mult=args.T_mult, eta_min=1e-6
    )

    loss_fn = TrajLoss(huber_delta=0.1, l1_weight=0.1)

    # ---- Resume from checkpoint ----------------------------------------
    start_epoch = 1
    if args.resume and (ckpt_dir / "latest_model.pth").exists():
        resume_path = str(ckpt_dir / "latest_model.pth")
        start_epoch, prev_metrics = load_checkpoint(
            resume_path, model, optimizer, scheduler, device=str(device)
        )
        start_epoch += 1
        print(f"[Resume] Continuing from epoch {start_epoch}")
        print(f"         prev metrics: {prev_metrics}")

    # ---- CSV log -------------------------------------------------------
    log_path = ckpt_dir / "training_log.csv"
    csv_header = ["epoch", "train_loss", "val_loss", "val_ade_m", "val_fde_m", "lr", "time_s"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    # ---- Training loop -------------------------------------------------
    best_val_ade   = float("inf")
    patience_count = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, args.log_every
        )

        val_loss, val_ade, val_fde = val_epoch(model, val_loader, device)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:>4}/{args.epochs} | "
            f"TrainLoss {train_loss:.6f} | "
            f"ValLoss {val_loss:.6f} | "
            f"ADE {val_ade:.4f} m | "
            f"FDE {val_fde:.4f} m | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.1f}s\n"
        )

        # ---- CSV log ---------------------------------------------------
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss, val_ade, val_fde, current_lr, elapsed
            ])

        metrics = {
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_ade":    val_ade,
            "val_fde":    val_fde,
        }

        # ---- Always save latest ----------------------------------------
        save_checkpoint(
            str(ckpt_dir / "latest_model.pth"),
            model, optimizer, scheduler, epoch, metrics, config
        )

        # ---- Save best (on ADE) ----------------------------------------
        if val_ade < best_val_ade:
            improvement = best_val_ade - val_ade
            best_val_ade   = val_ade
            patience_count = 0
            save_checkpoint(
                str(ckpt_dir / "best_model.pth"),
                model, optimizer, scheduler, epoch, metrics, config
            )
            print(
                f"  ✓ New best! ADE {val_ade:.4f} m  "
                f"(↓{improvement:.4f})  saved to best_model.pth"
            )
        else:
            patience_count += 1
            print(
                f"  No improvement for {patience_count}/{args.patience} epoch(s) "
                f"(best ADE {best_val_ade:.4f} m)"
            )

        # ---- Early stopping --------------------------------------------
        if patience_count >= args.patience:
            print(f"\n[EarlyStop] No ADE improvement for {args.patience} epochs. Stopping.")
            break

    print(f"\n{'='*60}")
    print(f" Training complete!")
    print(f" Best Val ADE : {best_val_ade:.4f} m")
    print(f" Checkpoints  : {ckpt_dir}")
    print(f"{'='*60}\n")


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train TrajMamba on RGBD Bonn Person Tracking Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument(
    "--dataset_roots",
    nargs="+",
    default=["./rgbd_bonn_person_tracking"],
    help="One or more dataset roots (containing groundtruth.txt or parents of scenarios)",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest_model.pth if it exists",
    )

    # Data
    p.add_argument("--obs_len",      type=int,   default=20,   help="Observed frames (after downsample)")
    p.add_argument("--pred_len",     type=int,   default=30,   help="Predicted frames (after downsample)")
    p.add_argument("--downsample",   type=int,   default=3,    help="Keep every N-th raw frame (~30Hz/3 = 10Hz)")
    p.add_argument("--use_3d",       action="store_true", default=True, help="Use 3-D positions (tx,ty,tz)")
    p.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of data for validation")
    p.add_argument("--seed",         type=int,   default=42)

    # Model
    p.add_argument("--d_model",  type=int,   default=64,  help="Mamba hidden dimension")
    p.add_argument("--d_state",  type=int,   default=16,  help="SSM state expansion factor")
    p.add_argument("--d_conv",   type=int,   default=4,   help="Local conv width in Mamba block")
    p.add_argument("--expand",   type=int,   default=2,   help="Inner expansion factor (d_inner = expand*d_model)")
    p.add_argument("--n_layers", type=int,   default=3,   help="Number of stacked BiMamba blocks")
    p.add_argument("--dropout",  type=float, default=0.1, help="Dropout in prediction head")

    # Training
    p.add_argument("--epochs",      type=int,   default=150,  help="Max training epochs")
    p.add_argument("--batch_size",  type=int,   default=32,   help="Mini-batch size")
    p.add_argument("--lr",          type=float, default=3e-4, help="Initial learning rate")
    p.add_argument("--T0",          type=int,   default=30,   help="CosineAnnealingWarmRestarts T0")
    p.add_argument("--T_mult",      type=int,   default=2,    help="CosineAnnealingWarmRestarts T_mult")
    p.add_argument("--patience",    type=int,   default=25,   help="Early-stopping patience (epochs)")
    p.add_argument("--num_workers", type=int,   default=4,    help="DataLoader worker processes")
    p.add_argument("--log_every",   type=int,   default=20,   help="Print batch loss every N batches")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate dataset directory
    from pathlib import Path
    for dataset_root in args.dataset_roots:
        if not Path(dataset_root).exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {dataset_root}\n"
                "Please check the --dataset_roots argument."
            )

    train(args)
