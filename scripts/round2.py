"""Round 2: hypothesis sweep for the GDN generalization gap.

Each run trains a GDN model (or the attention baseline) on the same seeded
MQAR-lite stream as scripts/train_recall.py and reports final held-out
accuracy. Gate initialisation is controlled by setting the gate-projection
biases so that sigmoid(bias) equals the requested initial gate value; the
shipped modules in src/ are untouched.

Usage:
    .venv/bin/python scripts/round2.py --runs a95,a99 --steps 1200
    .venv/bin/python scripts/round2.py --runs all
"""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import Tensor, nn
from gdn_lab.attention import CausalAttentionBlock
from gdn_lab.gdn import GatedDeltaNetBlock
from gdn_lab.recall import RecallConfig, make_batch


class RecallModel(nn.Module):
    """Same model shell as scripts/train_recall.py (shared embed/head)."""

    def __init__(self, block: type[nn.Module], d_model: int, heads: int,
                 layers: int, cfg: RecallConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.n_keys + cfg.n_values, d_model)
        self.position = nn.Parameter(torch.randn(2 * cfg.n_pairs + 1, d_model) * 0.02)
        self.blocks = nn.Sequential(*(block(d_model, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, cfg.n_values)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens) + self.position[:tokens.shape[1]]
        return self.head(self.norm(self.blocks(x))[:, -1])


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def set_gate_init(model: nn.Module, alpha0: float, beta0: float) -> None:
    """Bias the gate projections so gates start near (alpha0, beta0)."""
    for block in model.blocks:
        if isinstance(block, GatedDeltaNetBlock):
            with torch.no_grad():
                block.alpha_proj.bias.fill_(logit(alpha0))
                block.beta_proj.bias.fill_(logit(beta0))


def gate_stats(model: nn.Module, tokens: Tensor) -> str:
    """Mean learned alpha/beta over a batch, per layer (diagnostic)."""
    parts = []
    with torch.no_grad():
        x = model.embedding(tokens) + model.position[:tokens.shape[1]]
        for block in model.blocks:
            if isinstance(block, GatedDeltaNetBlock):
                alpha, beta = block.compute_gates(block.norm(x))
                parts.append(f"a={alpha.mean():.3f} b={beta.mean():.3f}")
            x = block(x)
    return " ".join(parts) if parts else "-"


# name -> overrides on top of the defaults in main()
RUNS: dict[str, dict] = {
    "base-1L":    dict(),
    "base-2L":    dict(layers=2),
    "a95-1L":     dict(alpha0=0.95),
    "a99-1L":     dict(alpha0=0.99),
    "a999-1L":    dict(alpha0=0.999),
    "a99b90-1L":  dict(alpha0=0.99, beta0=0.90),
    "a95-lr1e3":  dict(alpha0=0.95, lr=1e-3),
    "a95-lr1e2":  dict(alpha0=0.95, lr=1e-2),
    "a99-lr1e3":  dict(alpha0=0.99, lr=1e-3),
    "a99-lr1e2":  dict(alpha0=0.99, lr=1e-2),
    "a95-2L":     dict(alpha0=0.95, layers=2),
    "a99-2L":     dict(alpha0=0.99, layers=2),
    "a99-h2-1L":  dict(alpha0=0.99, heads=2),
    "a99-h1-1L":  dict(alpha0=0.99, heads=1),
    "big-1L":     dict(alpha0=0.99, d_model=128),
    "big-attn":   dict(block="attn", d_model=128),
    "base-cos4k": dict(steps=4000, schedule="cosine"),
    "base-2L-cos4k": dict(layers=2, steps=4000, schedule="cosine"),
    "a99-cos4k":  dict(alpha0=0.99, steps=4000, schedule="cosine"),
    "a99-p2":     dict(alpha0=0.99, n_pairs=2),
    "a99-p4":     dict(alpha0=0.99, n_pairs=4),
    "base-p4":    dict(n_pairs=4),
}


def run_one(name: str, cfgd: dict) -> dict:
    seed = cfgd.get("seed", 0)
    steps = cfgd.get("steps", 1200)
    lr = cfgd.get("lr", 3e-3)
    d_model = cfgd.get("d_model", 64)
    heads = cfgd.get("heads", 4)
    layers = cfgd.get("layers", 1)
    alpha0 = cfgd.get("alpha0", 0.5)
    beta0 = cfgd.get("beta0", 0.5)
    n_pairs = cfgd.get("n_pairs", 8)
    schedule = cfgd.get("schedule", "constant")
    block_cls = CausalAttentionBlock if cfgd.get("block") == "attn" else GatedDeltaNetBlock

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cfg = RecallConfig(n_pairs=n_pairs, seed=seed)
    held_tokens, held_target = make_batch(
        512, cfg, torch.Generator().manual_seed(cfg.seed + 1))
    model = RecallModel(block_cls, d_model, heads, layers, cfg)
    if block_cls is GatedDeltaNetBlock and (alpha0 != 0.5 or beta0 != 0.5):
        set_gate_init(model, alpha0, beta0)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = None
    if schedule == "cosine":
        warmup = min(50, steps // 10)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    generator = torch.Generator().manual_seed(cfg.seed)
    start = time.perf_counter()
    loss_value = float("nan")
    for step in range(1, steps + 1):
        model.train()
        tokens, target = make_batch(64, cfg, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(tokens), target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_value = loss.item()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        accuracy = (model(held_tokens).argmax(-1) == held_target).float().mean().item()
    result = dict(name=name, params=params, steps=steps, lr=lr, layers=layers,
                  d_model=d_model, heads=heads, alpha0=alpha0, beta0=beta0,
                  n_pairs=n_pairs,
                  schedule=schedule, loss=loss_value, acc=accuracy,
                  seconds=elapsed, gates=gate_stats(model, held_tokens))
    print("RESULT {name}: params={params:,} steps={steps} lr={lr:g} L={layers} "
          "d={d_model} h={heads} a0={alpha0} b0={beta0} pairs={n_pairs} sched={schedule} "
          "loss={loss:.4f} acc={acc:.2%} gates[{gates}] ({seconds:.0f}s)".format(**result),
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="base-1L",
                        help="comma-separated run names, or 'all'")
    parser.add_argument("--steps", type=int, default=None,
                        help="override step count for all selected runs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    names = list(RUNS) if args.runs == "all" else [n.strip() for n in args.runs.split(",")]
    for name in names:
        cfgd = dict(RUNS[name])
        if args.steps is not None and "steps" not in cfgd:
            cfgd["steps"] = args.steps
        run_one(name, cfgd)


if __name__ == "__main__":
    main()
