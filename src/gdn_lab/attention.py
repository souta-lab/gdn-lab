"""Causal attention baseline with the same residual MLP as GDN."""
import torch
from torch import Tensor, nn


class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        out, _ = self.attention(z, z, z, attn_mask=mask, need_weights=False)
        x = x + out
        return x + self.mlp(self.mlp_norm(x))
