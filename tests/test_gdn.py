import math

import pytest
import torch

import gdn_lab.gdn as gdn
from gdn_lab.attention import CausalAttentionBlock
from gdn_lab.recall import RecallConfig, make_batch


def inputs():
    gen = torch.Generator().manual_seed(7)
    shapes = [(2, 3, 5, 4), (2, 3, 5, 4), (2, 3, 5, 6),
              (2, 3, 5), (2, 3, 5), (2, 3, 4, 6)]
    return [torch.randn(s, generator=gen, dtype=torch.float64) for s in shapes]


def test_frozen_state():
    q, k, v, a, b, s = inputs()
    original = s.clone()
    out, state = gdn.gated_delta_recurrent(q, k, v, torch.ones_like(a), torch.zeros_like(b), s)
    assert torch.equal(state, s)
    assert torch.equal(s, original)
    expected = torch.einsum("bhkd,bhtk->bhtd", s, q / math.sqrt(q.shape[-1]))
    torch.testing.assert_close(out, expected, atol=1e-14, rtol=1e-14)


def test_decay():
    q, k, v, a, b, s = inputs()
    _, state = gdn.gated_delta_recurrent(q, k, v, torch.full_like(a, 0.5), torch.zeros_like(b), s)
    assert torch.equal(state, 0.5 ** q.shape[2] * s)
    torch.testing.assert_close(state.norm(), s.norm() * 0.5 ** q.shape[2])


def test_shapes():
    q, k, v, a, b, _ = inputs()
    out, state = gdn.gated_delta_recurrent(q, k, v, a.sigmoid(), b.sigmoid())
    assert out.shape == (2, 3, 5, 6)
    assert state.shape == (2, 3, 4, 6)


def test_householder_equivalence():
    q, k, v, a, b, initial = inputs()
    a, b = a.sigmoid(), b.sigmoid()
    actual, final = gdn.gated_delta_recurrent(q, k, v, a, b, initial)
    # Independent literal loop: paper orientation is (dv, dk).
    expected = torch.empty_like(actual)
    reference = torch.empty_like(final)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            state = initial[batch, head].T.clone()
            for t in range(q.shape[2]):
                key, value = k[batch, head, t], v[batch, head, t]
                state = (a[batch, head, t] * state @
                         (torch.eye(key.numel(), dtype=q.dtype) -
                          b[batch, head, t] * torch.outer(key, key)) +
                         b[batch, head, t] * torch.outer(value, key))
                expected[batch, head, t] = state @ q[batch, head, t] / math.sqrt(key.numel())
            reference[batch, head] = state.T
    assert (actual - expected).abs().max() < 1e-10
    assert (final - reference).abs().max() < 1e-10


@pytest.mark.skipif(not hasattr(gdn, "gated_delta_chunkwise"), reason="Sequential reference shipped")
def test_chunkwise():
    q, k, v, a, b, s = inputs()
    expected = gdn.gated_delta_recurrent(q, k, v, a.sigmoid(), b.sigmoid(), s)
    actual = gdn.gated_delta_chunkwise(q, k, v, a.sigmoid(), b.sigmoid(), s)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def test_gradient_flow():
    block = gdn.GatedDeltaNetBlock(16, 4)
    out = block(torch.randn(2, 7, 16))
    assert out.shape == (2, 7, 16)
    out.square().mean().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_module_gates():
    block = gdn.GatedDeltaNetBlock(16, 4)
    captured = []
    handles = [proj.register_forward_hook(lambda module, args, output: captured.append(output))
               for proj in (block.alpha_proj, block.beta_proj)]
    try:
        block(torch.randn(2, 7, 16))
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == 2
    for logits in captured:
        assert logits.shape == (2, 7, 4)
        assert ((logits.sigmoid() > 0) & (logits.sigmoid() < 1)).all()


def test_recall_data():
    cfg = RecallConfig()
    tokens, target = make_batch(128, cfg, torch.Generator().manual_seed(cfg.seed))
    assert tokens.dtype == target.dtype == torch.long
    assert tokens.shape == (128, 17)
    keys, values = tokens[:, :-1:2], tokens[:, 1::2] - cfg.n_keys
    assert ((keys >= 0) & (keys < cfg.n_keys)).all()
    assert ((values >= 0) & (values < cfg.n_values)).all()
    assert (keys.sort(dim=1).values.diff(dim=1) > 0).all()
    matches = keys == tokens[:, -1:]
    assert (matches.sum(dim=1) == 1).all()
    assert torch.equal(values[matches], target)


def test_attention_is_causal():
    block = CausalAttentionBlock(16, 4).eval()
    x = torch.randn(2, 7, 16)
    altered = x.clone()
    altered[:, 4:] = torch.randn_like(altered[:, 4:])
    torch.testing.assert_close(block(x)[:, :4], block(altered)[:, :4])


def test_empty_sequence():
    q = torch.empty(2, 3, 0, 4)
    v = torch.empty(2, 3, 0, 6)
    out, state = gdn.gated_delta_recurrent(q, q, v, torch.empty(2, 3, 0), torch.empty(2, 3, 0))
    assert out.shape == (2, 3, 0, 6)
    assert torch.equal(state, torch.zeros(2, 3, 4, 6))
