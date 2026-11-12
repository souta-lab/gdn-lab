"""MQAR-lite: distinct keys followed by offset value tokens and a query."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RecallConfig:
    n_pairs: int = 8
    n_keys: int = 32
    n_values: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.n_pairs <= self.n_keys or self.n_values < 1:
            raise ValueError("Require 1 <= n_pairs <= n_keys and n_values >= 1")


def make_batch(
    batch_size: int, cfg: RecallConfig, generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Draw CPU examples using only the supplied generator's random stream.

    Initialize that generator with cfg.seed when creating a training stream.
    """
    keys = torch.rand(batch_size, cfg.n_keys, generator=generator).argsort(dim=1)[:, :cfg.n_pairs]
    values = torch.randint(cfg.n_values, (batch_size, cfg.n_pairs), generator=generator)
    query = torch.randint(cfg.n_pairs, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    tokens = torch.empty(batch_size, 2 * cfg.n_pairs + 1, dtype=torch.long)
    tokens[:, :-1:2] = keys
    tokens[:, 1::2] = values + cfg.n_keys
    tokens[:, -1] = keys[rows, query]
    return tokens, values[rows, query]
