"""Train two small CPU models on identical seeded associative recall streams."""
import argparse
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
    def __init__(self, block: type[nn.Module], d_model: int, heads: int,
                 layers: int, cfg: RecallConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.n_keys + cfg.n_values, d_model)
        # Both mixers receive the same learned positional information.
        self.position = nn.Parameter(torch.randn(2 * cfg.n_pairs + 1, d_model) * 0.02)
        self.blocks = nn.Sequential(*(block(d_model, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, cfg.n_values)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens) + self.position[:tokens.shape[1]]
        return self.head(self.norm(self.blocks(x))[:, -1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("steps", 400), ("batch-size", 64), ("d-model", 64),
                          ("heads", 4), ("layers", 1), ("n-pairs", 8), ("seed", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.layers, args.d_model, args.heads) < 1:
        parser.error("steps, batch-size, layers, d-model and heads must be positive")
    torch.set_num_threads(1)  # Small CPU operations otherwise suffer thread overhead.
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    cfg = RecallConfig(n_pairs=args.n_pairs, seed=args.seed)
    held_tokens, held_target = make_batch(
        512, cfg, torch.Generator().manual_seed(cfg.seed + 1))
    models = {name: RecallModel(block, args.d_model, args.heads, args.layers, cfg)
              for name, block in [("GDN", GatedDeltaNetBlock),
                                  ("Attention", CausalAttentionBlock)]}
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    for name, count in counts.items():
        print(f"{name}: {count:,} parameters", flush=True)
    print(f"Parameter ratio: {max(counts.values()) / min(counts.values()):.4f}", flush=True)
    results = []
    for name, model in models.items():
        generator = torch.Generator().manual_seed(cfg.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        start = time.perf_counter()
        for step in range(1, args.steps + 1):
            model.train()
            tokens, target = make_batch(args.batch_size, cfg, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(tokens), target)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    accuracy = (model(held_tokens).argmax(-1) == held_target).float().mean().item()
                print(f"{name} step {step}: loss={loss.item():.4f} held-out={accuracy:.2%}", flush=True)
        elapsed = time.perf_counter() - start
        print(f"{name} wall-clock: {elapsed:.2f}s", flush=True)
        results.append((name, counts[name], loss.item(), accuracy))
    print("\nmodel | params | final train loss | held-out accuracy")
    for name, count, loss_value, accuracy in results:
        print(f"{name} | {count} | {loss_value:.4f} | {accuracy:.2%}")


if __name__ == "__main__":
    main()
