#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional: use fast CUDA kernels when mamba-ssm is installed
# ---------------------------------------------------------------------------
FAST_MAMBA = False
try:
    from mamba_ssm import Mamba as _FastMamba
    FAST_MAMBA = True
    print("[TrajMamba] mamba-ssm found — fast CUDA kernels enabled.")
except ImportError:
    print("[TrajMamba] mamba-ssm not installed — using pure-PyTorch SSM (correct results, slightly slower).")

class SelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = -1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        if dt_rank < 0:
            dt_rank = max(1, d_inner // 16)
        self.dt_rank = dt_rank

        # x_proj: maps x → (Δ_rank, B, C) per token
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # dt_proj: maps Δ_rank → d_inner (low-rank factorisation of Δ)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(
            self.dt_proj.bias,
            -math.log(dt_rank), math.log(dt_rank)
        )

        # Fixed diagonal A (log-parameterised for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, N)
        A = A.expand(d_inner, -1)                                             # (d, N)
        self.A_log = nn.Parameter(torch.log(A))                               # learnt
        self.D     = nn.Parameter(torch.ones(d_inner))                        # skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_inner)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, d = x.shape
        assert d == self.d_inner

        # ---- Compute input-dependent parameters ----------------------------
        xbc   = self.x_proj(x)                              # (B, L, r+2N)
        delta = xbc[..., : self.dt_rank]                    # (B, L, r)
        B_ssm = xbc[..., self.dt_rank : self.dt_rank + self.d_state]  # (B, L, N)
        C_ssm = xbc[..., self.dt_rank + self.d_state :]               # (B, L, N)

        delta = F.softplus(self.dt_proj(delta))              # (B, L, d_inner)

        # ---- Discretise ---------------------------------------------------
        A = -torch.exp(self.A_log.float())                   # (d, N)  always negative
        # dA: (B, L, d, N)
        dA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        # dBu: (B, L, d, N)
        dBu = torch.einsum("bld,bln,bld->bldn", delta, B_ssm, x)

        # ---- Sequential scan ----------------------------------------------
        h = torch.zeros(B, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h  = dA[:, t] * h + dBu[:, t]                   # (B, d, N)
            yt = torch.einsum("bdn,bn->bd", h, C_ssm[:, t]) # (B, d)
            ys.append(yt)

        y = torch.stack(ys, dim=1)                           # (B, L, d)
        y = y + x * self.D                                   # skip connection
        return y


class MambaBlock(nn.Module):
    """
    Full Mamba residual block wrapping a SelectiveSSM (or fast mamba-ssm).

    Layout (follows the original paper):
        x → LayerNorm → split(z, x̃) → conv → SSM → gate(z) → proj → + residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int  = 4,
        expand: int  = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv (groups = d_inner → channelwise)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        if FAST_MAMBA:
            # Use official mamba-ssm kernel (requires CUDA)
            self._fast = _FastMamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,
            )
            self._ssm = None
        else:
            self._fast = None
            self._ssm  = SelectiveSSM(self.d_inner, d_state)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # Project → split gating branch
        xz   = self.in_proj(x)                               # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)                          # each (B, L, d_inner)

        # Causal depthwise conv
        x_ = x_.transpose(1, 2)                              # (B, d_inner, L)
        x_ = self.conv(x_)[..., : x.shape[1]]               # trim causal padding
        x_ = x_.transpose(1, 2)                              # (B, L, d_inner)
        x_ = F.silu(x_)

        # Selective SSM
        if FAST_MAMBA and self._fast is not None:
            y = self._fast(x_)
        else:
            y = self._ssm(x_)

        # Gate + output
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block: forward + backward SSM, fused by a learned gate.

    Bidirectionality substantially improves trajectory encoder quality
    (see S-Mamba, Bi-Mamba literature and ablations in TrajMamba).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand)
        # Fusion gate — learned scalar per channel
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)

        fused   = torch.cat([fwd_out, bwd_out], dim=-1)      # (B, L, 2*d)
        g       = self.gate(fused)                            # (B, L, d)
        out     = g * fwd_out + (1 - g) * bwd_out            # soft fusion
        return self.norm(out)


# ===========================================================================
# TrajMamba — full model
# ===========================================================================

class TrajMamba(nn.Module):
    def __init__(
        self,
        pos_dim:  int = 3,
        obs_len:  int = 20,
        pred_len: int = 30,
        d_model:  int = 64,
        d_state:  int = 16,
        d_conv:   int = 4,
        expand:   int = 2,
        n_layers: int = 3,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.pos_dim  = pos_dim
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.d_model  = d_model

        vel_len = obs_len - 1          # number of velocity vectors

        # Input projection
        self.input_embed = nn.Sequential(
            nn.Linear(pos_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Bidirectional Mamba encoder
        self.encoder = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        # Asymmetric decoder (Trajectory Mamba-style)
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.decoder_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pos_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_embed(x)                     # (B, L, d_model)
        for block in self.encoder:
            h = block(h)
        context = h[:, -1, :]                        # (B, d_model)
        dec_in = context.unsqueeze(1).repeat(1, self.pred_len - 1, 1)
        dec_init = torch.tanh(self.context_proj(context)).unsqueeze(0)

        dec_out, _ = self.decoder_gru(dec_in, dec_init)
        return self.decoder_out(dec_out)

    # ------------------------------------------------------------------
    # Inference helper: decode velocities → absolute positions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_positions(
        self,
        obs_vel_norm: torch.Tensor,
        obs_last_pos: torch.Tensor,
        vel_mean: torch.Tensor,
        vel_std:  torch.Tensor,
    ) -> torch.Tensor:
        pred_vel_norm = self(obs_vel_norm)                           # (B, L-1, D)

        # Denormalise
        vm = vel_mean.to(pred_vel_norm.device)
        vs = vel_std.to(pred_vel_norm.device)
        pred_vel = pred_vel_norm * vs + vm                           # (B, L-1, D)
        B, T, D = pred_vel.shape
        positions = [obs_last_pos.unsqueeze(1)]
        for t in range(T):
            positions.append(positions[-1] + pred_vel[:, t:t+1])
        positions = torch.cat(positions, dim=1)                      # (B, pred_len, D)
        return positions

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    B, obs, pred, D = 4, 20, 30, 3

    model = TrajMamba(
        pos_dim=D, obs_len=obs, pred_len=pred,
        d_model=64, d_state=16, n_layers=3,
    )

    x = torch.randn(B, obs - 1, D)
    y = model(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {y.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
    assert y.shape == (B, pred - 1, D), "Shape mismatch!"
    print("Model smoke-test passed ✓")
