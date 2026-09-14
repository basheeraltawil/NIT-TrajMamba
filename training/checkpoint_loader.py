#!/usr/bin/env python3

import os
import json
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from mamba_model import TrajMamba


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    arch_override: Optional[Dict[str, Any]] = None,
) -> Tuple[TrajMamba, Dict[str, Any]]:


    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    if arch_override:
        config.update(arch_override)

    pos_dim  = config.get("pos_dim",  3)
    obs_len  = config.get("obs_len",  20)
    pred_len = config.get("pred_len", 30)
    d_model  = config.get("d_model",  64)
    d_state  = config.get("d_state",  16)
    d_conv   = config.get("d_conv",   4)
    expand   = config.get("expand",   2)
    n_layers = config.get("n_layers", 3)
    dropout  = config.get("dropout",  0.1)

    model = TrajMamba(
        pos_dim=pos_dim,
        obs_len=obs_len,
        pred_len=pred_len,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics = ckpt.get("metrics", {})
    epoch   = ckpt.get("epoch",   "?")

    print(f"[Checkpoint] Loaded : {checkpoint_path}")
    print(f"[Checkpoint] Epoch  : {epoch}")
    print(f"[Checkpoint] ADE    : {metrics.get('val_ade', 'N/A'):.4f} m")
    print(f"[Checkpoint] FDE    : {metrics.get('val_fde', 'N/A'):.4f} m")
    print(f"[Checkpoint] Config : pos_dim={pos_dim} obs={obs_len} pred={pred_len} "
          f"d_model={d_model} n_layers={n_layers}")

    return model, config


def list_checkpoints(checkpoint_dir: str) -> list:
    """Return sorted list of .pth files in checkpoint_dir."""
    d = Path(checkpoint_dir)
    return sorted(d.glob("*.pth")) if d.exists() else []


def get_best_checkpoint(checkpoint_dir: str) -> str:
    """Return path to best_model.pth; raises if not found."""
    best = Path(checkpoint_dir) / "best_model.pth"
    if best.exists():
        return str(best)
    raise FileNotFoundError(f"No best_model.pth in {checkpoint_dir}")


def inspect_checkpoint(checkpoint_path: str):
    """Print all metadata stored in a checkpoint without loading the model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"\n{'='*50}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epoch      : {ckpt.get('epoch', 'N/A')}")
    print(f"Metrics    : {ckpt.get('metrics', {})}")
    print(f"\nConfig:")
    for k, v in ckpt.get("config", {}).items():
        print(f"  {k:<18} : {v}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints"

    checkpoints = list_checkpoints(ckpt_dir)
    print(f"Found {len(checkpoints)} checkpoint(s) in {ckpt_dir}:")
    for cp in checkpoints:
        print(f"  {cp}")

    try:
        best = get_best_checkpoint(ckpt_dir)
        inspect_checkpoint(best)
        model, cfg = load_checkpoint(best)
        print("✓ Model loaded successfully")

        # Quick forward pass
        pos_dim = cfg.get("pos_dim", 3)
        obs_len = cfg.get("obs_len", 20)
        x = torch.randn(1, obs_len - 1, pos_dim)
        with torch.no_grad():
            y = model(x)
        print(f"✓ Forward pass: {x.shape} → {y.shape}")

    except Exception as e:
        print(f"✗ Error: {e}")
