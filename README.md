<h1 align="center">🧪 gdn-lab</h1>

<p align="center">
  A small, clean <strong>reference implementation of Gated DeltaNet (GDN)</strong> in PyTorch —<br>
  plus a controlled experiment comparing GDN against a standard attention block<br>
  on synthetic associative recall (MQAR-lite). CPU-only, readable, tested.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >= 3.10">
  <img src="https://img.shields.io/badge/pytorch-CPU-orange" alt="PyTorch CPU">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/tests-9%20passed-brightgreen" alt="9 tests passed">
</p>

---

## 📑 Table of Contents

- [What is GDN?](#-what-is-gdn)
- [The math implemented here](#-the-math-implemented-here)
- [How this maps to the paper](#-how-this-maps-to-the-paper)
- [Layout](#-layout)
- [Quick start](#-quick-start)
- [Experiment: GDN vs attention on associative recall](#-experiment-gdn-vs-attention-on-associative-recall)
- [Round 2: hypothesis sweep for the generalization gap](#-round-2-hypothesis-sweep-for-the-generalization-gap)
- [Limitations](#️-limitations)
- [References](#-references)

---

## 🔍 What is GDN?

**Gated DeltaNet** (Yang et al., 2024) is a linear-attention / recurrent architecture.
Instead of a softmax attention matrix, each head maintains a **matrix-valued state**
$S \in \mathbb{R}^{d_k \times d_v}$ that acts as an associative memory:

- a **delta-rule update** writes each new key–value pair by first erasing what the
  memory currently predicts for that key, then writing the correction — so repeated
  keys *overwrite* instead of accumulating;
- a **data-dependent decay gate** $\alpha_t \in (0,1)$ lets the memory forget;
- a **write-strength gate** $\beta_t \in (0,1)$ controls how aggressively each
  token rewrites the memory.

The result is a sequence mixer with $O(T)$ time and fixed-size state, which in
hybrid models (Qwen3-Next, Kimi Linear) replaces most full-attention layers.

This repo implements the recurrence in its naive sequential form, on purpose:
every step is a few lines of plain PyTorch that you can diff against the equations.

---

## 📐 The math implemented here

Per head, per step $t$, with $q_t, k_t \in \mathbb{R}^{d_k}$ (both L2-normalized,
$q$ additionally scaled by $1/\sqrt{d_k}$), $v_t \in \mathbb{R}^{d_v}$,
and scalar gates $\alpha_t, \beta_t \in (0,1)$:

$$S_t = \alpha_t \, S_{t-1} \;+\; \beta_t \, k_t \left( v_t - \alpha_t \, S_{t-1}^{\top} k_t \right)^{\!\top},
\qquad o_t = S_t^{\top} q_t$$

The term $S_{t-1}^{\top} k_t$ is what the memory currently recalls for key $k_t$;
the update writes only the *error* against the (decayed) target $v_t$.
This delta form is algebraically identical to the paper's Householder-like form

$$S_t = \alpha_t \, S_{t-1} \left( I - \beta_t \, k_t k_t^{\top} \right) + \beta_t \, v_t k_t^{\top}$$

and a unit test proves the two forms agree to $< 10^{-10}$ in float64.

---

## 🗺️ How this maps to the paper

| Paper component | This repo |
| --- | --- |
| Delta-rule recurrence (background) | `gated_delta_recurrent` in `src/gdn_lab/gdn.py` |
| Gating: data-dependent decay $\alpha_t$ and write strength $\beta_t$ | sigmoid projections per head in `GatedDeltaNetBlock` |
| L2-normalized $q, k$; $q$ scaled by $1/\sqrt{d_k}$ | in the block / recurrent function respectively |
| Gated output (sigmoid gate on the mixer output) | `output_gate` in `GatedDeltaNetBlock` |
| Chunkwise parallel form (WY representation) | ❌ **not implemented** — sequential reference only |
| Short depthwise convolutions on $q, k, v$ | ❌ not implemented (noted as a limitation) |
| Fused Triton kernels | ❌ out of scope |

State-update invariants are pinned down by tests: with $\beta=0$ the state is
pure decay ($\alpha^T S_0$), with $\alpha=1, \beta=0$ it is exactly unchanged,
and gradients flow to every projection.

---

## 📂 Layout

```
src/gdn_lab/
  gdn.py        # gated_delta_recurrent + GatedDeltaNetBlock (mixer + MLP)
  attention.py  # CausalAttentionBlock baseline (same MLP / norms)
  recall.py     # MQAR-lite data: interleaved key/value pairs + query
scripts/
  train_recall.py   # trains both models on identical streams, prints a table
  overfit_probe.py  # can GDN memorize a fixed 256-example set? (yes — see below)
  round2.py         # round-2 hypothesis sweep (gate init / LR / state size / length)
tests/
  test_gdn.py       # invariants, form equivalence, gradients, gates, data sanity
```

---

## 🚀 Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/python -m pytest -q          # 9 passed, 1 skipped (chunkwise not shipped)
.venv/bin/python scripts/train_recall.py
```

The training script is deterministic for a fixed `--seed` and finishes in
well under a minute on CPU with default settings.

---

## 📊 Experiment: GDN vs attention on associative recall

**Task (MQAR-lite).** Each example interleaves 8 distinct keys with random
values — `[k1, v1, k2, v2, …, k8, v8, k_query]` — and the model must output
the value paired with the query key (32 key and 32 value classes;
chance accuracy = 3.125%). Loss is applied at the last position only.

**Setup.** Both models share the embedding, learned positional embeddings,
final norm and value head. One block each: the GDN block vs a causal-attention
block with the *same* MLP and norms. Parameter counts are within 10%
(GDN 62,056 vs attention 57,376 — GDN's gate/output projections account for
the difference). Both train on identical fresh-batch streams
(AdamW, lr 3e-3, batch 64), and are evaluated on the same fixed held-out
batch of 512 examples.

### Headline run (default settings: 1 layer, 400 steps)

```bash
.venv/bin/python scripts/train_recall.py
```

| Model | Params | Final train loss | Held-out accuracy | Wall-clock |
| --- | ---: | ---: | ---: | ---: |
| GDN | 62,056 | 2.7148 | **16.60%** | ~22 s |
| Attention | 57,376 | 2.5999 | **16.41%** | ~7 s |

Both mixers land far above chance but **neither solves 8-pair recall in a
single layer** at this budget.

### Follow-up probes (same script / repo)

| Probe | Command | GDN | Attention |
| --- | --- | ---: | ---: |
| Longer training, 1 layer | `--steps 2000` | 19.14% | 19.92% |
| Two layers, 1200 steps | `--layers 2 --steps 1200` | 18.75% | **98.63%** |
| Overfit sanity (fixed 256 examples) | `scripts/overfit_probe.py` | **100% train acc. in 60 steps** | — |

### Reading the results

- More steps barely move either 1-layer model: the plateau is structural,
  not a budget artifact.
- A second layer lets attention build the classic two-step induction
  circuit and solve the task outright. The GDN block gains nothing from
  depth here — its recall is supposed to live in the *first* layer's state.
- The overfit probe shows the delta-rule state can express and memorize
  recall perfectly when the key→value mappings repeat. Together with the
  unit-test invariants, this says the streaming gap above is about
  **generalization dynamics of the learned addressing scheme** (projections
  + gates trained from scratch on ever-fresh random pairs), not about the
  correctness of the recurrence.
- These are tiny-config results (d_model 64, head_dim 16, one seed). The
  paper's models — short convolutions, larger state, tuned schedules — are
  a different regime; nothing here contradicts them.

---

## 🔁 Round 2: hypothesis sweep for the generalization gap

Round 1 left a sharp question: why does a second layer take attention to
98.63% while GDN stays at 18.75%, when the recurrence itself is provably
correct (overfit probe: 100%)? Round 2 (`scripts/round2.py`; seed 0, the
same held-out batch of 512 — at this size ±~1.7 pt is one standard
deviation) tests the four standard explanations. First, a control:
**base-2L reproduces the round-1 number exactly (18.75%)**, so the
sweep is directly comparable to round 1.

### (a) Residuals + LayerNorm — already present, not the missing piece

`GatedDeltaNetBlock` is a pre-norm block: LayerNorm before the mixer and
before the MLP, with residual connections around both. Round 1's numbers
were already produced *with* residuals and norm in place, so there was
nothing to add — the gap exists in spite of them.

### (b) Gate initialisation and learning rate — refuted

| Run | Change vs base | Held-out |
| --- | --- | ---: |
| base-1L | — (α₀=β₀=0.5, lr 3e-3, 1 layer) | 16.80% |
| a95-1L | α₀=0.95 | 18.36% |
| a99-1L | α₀=0.99 | 16.41% |
| a999-1L | α₀=0.999 | 15.82% |
| a99b90-1L | α₀=0.99, β₀=0.90 | 15.23% |
| a95-lr1e3 | α₀=0.95, lr 1e-3 | 18.55% |
| a95-lr1e2 | α₀=0.95, lr 1e-2 | 18.75% |
| a99-lr1e3 | α₀=0.99, lr 1e-3 | 18.16% |
| a99-lr1e2 | α₀=0.99, lr 1e-2 | 16.99% |
| a95-2L | α₀=0.95, 2 layers | 16.41% |
| a99-2L | α₀=0.99, 2 layers | 15.82% |

Every variant lands in the same 15–19% band as the base runs — the
differences are within noise. A diagnostic explains why initialisation
cannot matter much here: after training, the learned mean decay
ᾱ ≈ 0.999–1.000 in *every* GDN run, whatever it started from. Training
saturates the decay gate to ≈1 on its own; where it starts is irrelevant
by the end.

### (c) State dimension at matched parameters — refuted

State per layer is H·d_k·d_v, so fewer heads at fixed d_model give a
larger state for almost identical parameter counts:

| Run | State/head | Total state | Params | Held-out |
| --- | ---: | ---: | ---: | ---: |
| base (h=4) | 16×16 | 1,024 | 62,056 | 16.80% |
| a99-h2-1L | 32×32 | 2,048 | 61,796 | 16.60% |
| a99-h1-1L | 64×64 | 4,096 | 61,666 | 16.99% |
| big-1L (d=128, h=4) | 32×32 | 4,096 | 230,568 | 17.38% |
| big-attn (d=128) | — | — | 213,024 | 12.89% |

2×–4× the state — and even ~4× the parameters overall — moves nothing.
(One-layer attention at d=128 also fails: for attention on this task it
is depth, not width, that matters.)

### (d) Longer training — marginal, refuted as a fix

| Run | Steps / schedule | Final loss | Held-out |
| --- | --- | ---: | ---: |
| base-cos4k (1L) | 4,000, cosine | 1.9776 | 20.51% |
| base-2L-cos4k | 4,000, cosine | 1.9464 | 19.73% |
| a99-cos4k (1L) | 4,000, cosine | 1.9961 | 17.77% |

3.3× the steps lowers the loss and buys +2–4 pt — and still lands ~78 pt
short of 2-layer attention at 1,200 steps. The plateau is not a budget
problem.

### Difficulty scaling (diagnostic)

| Pairs | 2 | 4 | 8 |
| --- | ---: | ---: | ---: |
| GDN held-out | 51.37% | 29.69–30.47% | 15.23–18.75% |

Accuracy degrades smoothly with the number of pairs instead of falling
off a capacity cliff — and the state can in principle hold 8 rank-1
writes (the overfit probe memorises fixed mappings perfectly). This is
consistent with round 1's reading: the bottleneck is the learned
addressing — which slot a fresh random key maps to, and how the query
finds it — not raw state capacity.

### Round 2 conclusion

None of the four standard knobs closes the gap: residuals/norm were
already present, and gate initialisation, learning rate, state size and
training length are all refuted above. What round 2 did *not* test —
and where the paper's models differ from this reference — is the honest
remaining candidate list: short convolutions on q/k/v, per-channel
(vector) decay instead of a scalar per head, and multi-seed statistics
(all round-2 numbers are seed 0). The gap itself survives round 2
intact.

---

## ⚠️ Limitations

- **Reference implementation.** The recurrent form is $O(T)$ sequential
  Python/PyTorch steps — fine for study and short sequences, not a kernel.
- **No chunkwise/WY parallel form**, no fused kernels, no short convolutions,
  scalar (not per-channel) decay — for per-channel gating see Kimi Linear's KDA.
- **One seed, one small configuration** for the experiment; treat the numbers
  as a reproducible smoke comparison, not a benchmark verdict.
- Exact bitwise reproducibility across PyTorch versions/hardware is not
  guaranteed, though runs are seeded and deterministic on a given setup.

---

## 📚 References

- Songlin Yang, Jan Kautz, Ali Hatamizadeh, et al. **Gated Delta Networks:
  Improving Mamba2 with Delta Rule.** arXiv:2412.06464 (2024).
  https://arxiv.org/abs/2412.06464
- Songlin Yang, Bailin Wang, Yikang Shen, et al. **Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length.** arXiv:2406.06484
  (2024) — the DeltaNet predecessor.
- **Qwen3-Next** — hybrid architecture using Gated DeltaNet layers for most
  of its depth. https://qwen.ai/blog?id=qwen3-next *(Qwen team blog)*
- **Kimi Linear** — Kimi Delta Attention (KDA), a per-channel-gated extension
  of GDN. arXiv:2510.26692 (2025). https://arxiv.org/abs/2510.26692

## 📄 License

MIT — see [LICENSE](LICENSE).
