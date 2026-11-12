"""Overfit probe: can 1-layer GDN memorize a fixed batch of 256 recall examples?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
from torch import nn
from train_recall import RecallModel
from gdn_lab.gdn import GatedDeltaNetBlock
from gdn_lab.recall import RecallConfig, make_batch

torch.set_num_threads(1)
torch.manual_seed(0)
cfg = RecallConfig(n_pairs=8, seed=0)
tokens, target = make_batch(256, cfg, torch.Generator().manual_seed(123))
model = RecallModel(GatedDeltaNetBlock, 64, 4, 1, cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(1, 301):
    opt.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    opt.step()
    if step % 60 == 0:
        acc = (logits.argmax(-1) == target).float().mean().item()
        print(f"step {step}: loss={loss.item():.4f} train-acc={acc:.2%}", flush=True)
