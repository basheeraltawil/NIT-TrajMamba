#!/usr/bin/env python3
"""
model_utils.py
---------------
TrajMamba checkpoint loading and key-remapping utilities. Split out of the
node itself, mirroring how `nit_pose_estimation` keeps its helpers separate
from the ROS node class.

Note: this module depends on `mamba_model.TrajMamba`, which defines the
actual network architecture. That file was not part of the original
node-organisation request and must be copied into this package
(`nit_human_traj_estimation/mamba_model.py`) from your training repo, since
it is required to load the checkpoint. A placeholder is included so the
package installs cleanly; replace it with the real definition before running
the node.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from .mamba_model import TrajMamba


def model_uses_fast_kernels(model: TrajMamba) -> bool:
    """Return True if the instantiated model stores its Mamba parameters under
    the fast ('._fast.') sub-module, which happens when mamba-ssm is
    installed."""
    for key in model.state_dict().keys():
        if "._fast." in key:
            return True
    return False


def remap_slow_to_fast_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "._fast." in key:
            remapped[key] = value
            continue

        if "._ssm." in key:
            base, suffix = key.split("._ssm.", 1)
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
        return remap_slow_to_fast_keys(state_dict)
    if (not model_wants_fast) and ckpt_is_fast:
        return remap_fast_to_slow_keys(state_dict)

    return state_dict


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


def load_trajmamba_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    log: Optional[Callable[[str], None]] = None,
):
    """Load a TrajMamba checkpoint, adapting fast/slow Mamba key layouts and
    filtering out any shape-mismatched tensors. `log` defaults to `print` but
    can be a ROS logger method (e.g. `node.get_logger().info`)."""
    log = log or print

    # weights_only=False is required because the checkpoint stores a config
    # dict alongside the tensors. The file is produced by our own training
    # run, so this is trusted input.
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
        log(
            "[TrajMamba] Checkpoint config: "
            f"pos_dim={config.get('pos_dim', 3)} d_model={config.get('d_model', 64)} "
            f"d_state={config.get('d_state', 16)} d_conv={config.get('d_conv', 4)} "
            f"expand={config.get('expand', 2)} n_layers={config.get('n_layers', 3)}"
        )

    using_fast = model_uses_fast_kernels(model)
    ckpt_is_fast = any("._fast." in key for key in state_dict.keys())
    log(
        f"[TrajMamba] Model uses {'fast (mamba-ssm)' if using_fast else 'slow (fallback)'} kernels; "
        f"checkpoint stored in {'fast' if ckpt_is_fast else 'slow'} layout."
    )
    state_dict = align_checkpoint_to_model(model, state_dict)

    state_dict, skipped_missing, skipped_mismatch = filter_state_dict_by_shape(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if skipped_mismatch:
        log(f"[TrajMamba] Skipped mismatched tensors: {len(skipped_mismatch)}")
        for item in skipped_mismatch[:12]:
            log(f"  - {item}")
        if len(skipped_mismatch) > 12:
            log("  - ...")
    if skipped_missing:
        log(f"[TrajMamba] Skipped unknown tensors: {len(skipped_missing)}")
        for item in skipped_missing[:8]:
            log(f"  - {item}")
        if len(skipped_missing) > 8:
            log("  - ...")
    if missing:
        log(f"[TrajMamba] Missing keys after load: {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        log(f"[TrajMamba] Unexpected keys after load: {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")

    n_loaded = len(state_dict)
    n_total = len(model.state_dict())
    log(f"[TrajMamba] Loaded {n_loaded}/{n_total} tensors into the model.")
    if n_loaded == 0:
        log("[TrajMamba] WARNING: no tensors were loaded - predictions will be from random weights!")

    model.eval()
    return model, config
