#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class TrajLoss(nn.Module):

    def __init__(self, huber_delta: float = 0.1, l1_weight: float = 0.1):
        super().__init__()
        self.huber   = nn.HuberLoss(delta=huber_delta, reduction="mean")
        self.l1_w    = l1_weight

    def forward(
        self,
        pred_vel: torch.Tensor,
        target_vel: torch.Tensor,
    ) -> torch.Tensor:
        n = min(pred_vel.shape[1], target_vel.shape[1])
        p, t = pred_vel[:, :n], target_vel[:, :n]
        return self.huber(p, t) + self.l1_w * torch.abs(p - t).mean()

@torch.no_grad()
def ade_fde(
    pred_pos: torch.Tensor,
    gt_pos:   torch.Tensor,
) -> Tuple[float, float]:
    """
    Args:
        pred_pos : (B, T, D)  predicted positions
        gt_pos   : (B, T, D)  ground-truth positions
    Returns:
        ade, fde  (in the same units as input, e.g. metres)
    """
    diff = pred_pos - gt_pos                              # (B, T, D)
    dist = torch.norm(diff, dim=-1)                       # (B, T)
    ade  = dist.mean().item()
    fde  = dist[:, -1].mean().item()
    return ade, fde


@torch.no_grad()
def compute_metrics_from_batch(
    model,
    batch,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Compute loss + ADE + FDE for a single batch.

    Returns: (loss, ade, fde)
    """
    obs_vel, pred_vel, obs_pos, pred_pos, vel_mean, vel_std = [
        b.to(device) for b in batch
    ]

    loss_fn = TrajLoss()

    pred_vel_hat = model(obs_vel)
    n = min(pred_vel_hat.shape[1], pred_vel.shape[1])
    loss = loss_fn(pred_vel_hat[:, :n], pred_vel[:, :n]).item()

    # Denormalise and integrate to get positions
    vm = vel_mean[:, None, :]       # (B, 1, D)
    vs = vel_std[:, None, :]        # (B, 1, D)
    pred_vel_denorm = pred_vel_hat * vs + vm

    # Integrate: start from last observed position
    obs_last = obs_pos[:, -1]       # (B, D)
    B, T, D  = pred_vel_denorm.shape
    pos_seq  = [obs_last.unsqueeze(1)]
    for t in range(T):
        pos_seq.append(pos_seq[-1] + pred_vel_denorm[:, t:t+1])
    pred_positions = torch.cat(pos_seq[1:], dim=1)        # (B, T, D)

    # Ground-truth positions
    gt_positions = pred_pos[:, :T]

    ade, fde = ade_fde(pred_positions, gt_positions)
    return loss, ade, fde
