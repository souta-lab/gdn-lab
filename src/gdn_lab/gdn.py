"""Sequential GDN reference; chunkwise/WY execution is not shipped.

State uses the key-by-value orientation (dk, dv). The paper's value-by-key
Householder expression is its transpose.
"""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gated_delta_recurrent(
    q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply GDN to (B,H,T,D) inputs and (B,H,T) scalar gates.

    q and k are accepted as given, without normalization. This function
    always scales q by 1/sqrt(dk); callers must not pre-apply that scale.
    initial_state has shape (B,H,dk,dv) and is never modified in place.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    if k.shape != q.shape or v.shape[:3] != (b, h, t):
        raise ValueError("Incompatible q, k, v shapes")
    if alpha.shape != (b, h, t) or beta.shape != (b, h, t):
        raise ValueError("Gates must have shape (B,H,T)")
    state = q.new_zeros(b, h, dk, dv) if initial_state is None else initial_state
    if state.shape != (b, h, dk, dv):
        raise ValueError("initial_state must have shape (B,H,dk,dv)")
    outputs = []
    for i in range(t):
        decay = alpha[:, :, i, None, None]
        key = k[:, :, i]
        prediction = torch.einsum("bhkd,bhk->bhd", state, key)
        error = v[:, :, i] - alpha[:, :, i, None] * prediction
        state = decay * state + beta[:, :, i, None, None] * (
            key.unsqueeze(-1) * error.unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhkd,bhk->bhd", state, q[:, :, i] / math.sqrt(dk)))
    out = torch.stack(outputs, dim=2) if outputs else v.new_empty(b, h, 0, dv)
    return out, state


class GatedDeltaNetBlock(nn.Module):
    """Pre-norm gated recurrent mixer and pre-norm residual MLP."""

    def __init__(self, d_model: int = 64, heads: int = 4) -> None:
        super().__init__()
        if heads <= 0 or d_model % heads:
            raise ValueError("d_model must be divisible by positive heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.alpha_proj = nn.Linear(d_model, heads)
        self.beta_proj = nn.Linear(d_model, heads)
        self.output_gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def compute_gates(self, normalized_x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute (B,H,T) gates from the same pre-normalized mixer input."""
        return (self.alpha_proj(normalized_x).sigmoid().transpose(1, 2),
                self.beta_proj(normalized_x).sigmoid().transpose(1, 2))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        b, t, d = z.shape
        def split(y: Tensor) -> Tensor:
            return y.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        q = F.normalize(split(self.q_proj(z)), dim=-1, eps=1e-6)
        k = F.normalize(split(self.k_proj(z)), dim=-1, eps=1e-6)
        alpha, beta = self.compute_gates(z)
        out, _ = gated_delta_recurrent(q, k, split(self.v_proj(z)), alpha, beta)
        mixed = out.transpose(1, 2).reshape(b, t, d)
        x = x + self.out_proj(mixed * self.output_gate(z).sigmoid())
        return x + self.mlp(self.mlp_norm(x))
