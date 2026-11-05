"""Gated multi-head attention MIL regressor (the histological clock).

The pan-organ clock and every per-organ clock use this same architecture:

    patch embeddings (N, 1536)
      -> 4 gated attention heads  a_h = softmax_i( w_h^T [tanh(V_h x_i) * sigmoid(U_h x_i)] )
      -> per-head pooled vectors  z_h = sum_i a_h,i x_i
      -> condensing layer         concat(z_1..z_4) (6144) -> 1536
      -> regressor                1536 -> 256 -> LayerNorm -> GELU -> dropout -> 1
      -> standardized age

The model receives no organ, tissue or sex input.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hera.config import FEATURE_DIM


def masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """scores (B, N, 1); mask (B, N) bool with True = real patch (False = zero padding)."""
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
    return F.softmax(scores, dim=1)


class GatedABMIL(nn.Module):
    """Attention module: (B, N, D) patches -> (B, D) slide embedding and (B, H, N) attention."""

    def __init__(self, feature_dim: int = FEATURE_DIM, head_dim: int = 256, n_heads: int = 4,
                 dropout: float = 0.25):
        super().__init__()
        self.n_heads = n_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, head_dim), nn.Tanh(), nn.Dropout(dropout))
            for _ in range(n_heads)])
        self.gating_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, head_dim), nn.Sigmoid(), nn.Dropout(dropout))
            for _ in range(n_heads)])
        self.branching_layers = nn.ModuleList([nn.Linear(head_dim, 1) for _ in range(n_heads)])
        self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        pooled, attns = [], []
        for i in range(self.n_heads):
            h = self.attention_heads[i](x) * self.gating_layers[i](x)
            a = masked_softmax(self.branching_layers[i](h), mask)          # (B, N, 1)
            pooled.append(torch.einsum('bnr,bnf->brf', a, x).squeeze(1))   # (B, D)
            attns.append(a.squeeze(-1))
        z = self.condensing_layer(torch.cat(pooled, dim=-1))
        return z, torch.stack(attns, dim=1)


class AgeClock(nn.Module):
    """forward(x (B, N, D), mask (B, N)) -> (standardized age (B,), attention (B, H, N))"""

    def __init__(self, feature_dim: int = FEATURE_DIM, head_dim: int = 256, n_heads: int = 4,
                 dropout: float = 0.25):
        super().__init__()
        self.abmil = GatedABMIL(feature_dim, head_dim, n_heads, dropout)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        z, attn = self.abmil(x, mask)
        return self.regressor(z).squeeze(-1), attn


def load_clock(path, device='cpu') -> AgeClock:
    """Load a trained clock. Keys outside abmil./regressor. are ignored, so checkpoints saved by
    the original research code (which carried an unused tissue-embedding table) load as well."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k: v for k, v in sd.items() if k.startswith(('abmil.', 'regressor.'))}
    model = AgeClock()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
