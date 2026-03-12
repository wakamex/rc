# Track A Results

Reservoir computing sidecar (ESN) bolted onto frozen Qwen3.5-0.8B via LoRA + cross-attention.

## Setup

- **Base model:** Qwen3.5-0.8B-Base (frozen, 753M params)
- **LoRA:** rank=16, alpha=32, q_proj+v_proj (639K trainable params, 0.085%)
- **ESN reservoir:** 10,000 nodes, spectral_radius=0.9, leak_rate=0.5, sparsity=0.01
- **Sidecar:** CrossAttentionSidecar at layers [3, 7, 11, 15, 19, 23], Flamingo-style tanh(alpha) gating
- **Training:** 5000 steps, batch=1, grad_accum=16, seq_len=2048, lr=2e-4, 30% synthetic memory tasks + 70% FineWeb
- **Hardware:** 1x RTX 3090 (24 GB), ~7h training, ~3h eval

## Definitive Benchmark Results (v3, n=200)

| Task | Vanilla | LoRA-only | LoRA+Sidecar | Reservoir delta |
|------|---------|-----------|-------------|-----------------|
| PasskeyRetrieval | 0.000 | 1.000 | 1.000 | +0.000 |
| AssociativeRecall | 0.855 | 1.000 | 1.000 | +0.000 |
| VariableTracking | 0.000 | 0.680 | 0.660 | -0.020 |
| MultiDigitArithmetic | 0.625 | 0.965 | 0.970 | +0.005 |
| ProgramTrace | 0.000 | 0.355 | 0.285 | -0.070 |
| AlgorithmicTransfer | 0.000 | 0.105 | 0.090 | -0.015 |
| **ModularArithmetic** | 0.005 | 0.090 | **0.380** | **+0.290** |
| **LengthExtrapolation** | 0.000 | 0.005 | **0.290** | **+0.285** |
| **DyckLanguage** | 0.085 | 0.000 | **0.125** | **+0.125** |
| CompositionalGen. | 0.005 | 0.070 | 0.015 | -0.055 |

- **LoRA-only** = same data, same steps, no sidecar (isolates training data effect)
- **Reservoir delta** = LoRA+Sidecar minus LoRA-only (isolates reservoir contribution)
- **Perplexity:** Vanilla 6.82 → LoRA-only 6.78 → LoRA+Sidecar 7.72

## Durable Conclusions

1. **Most gains come from training data, not the reservoir.** 7/10 benchmarks are fully explained by LoRA fine-tuning on memory task data. PasskeyRetrieval, AR, VT, MultiDigitArithmetic all learned from data alone.

2. **The reservoir uniquely helps 3 structured reasoning tasks:** ModularArithmetic (+0.290), LengthExtrapolation (+0.285), DyckLanguage (+0.125). These require tracking structured/recursive state beyond what LoRA can internalize.

3. **Perplexity cost is significant.** +0.94 over LoRA-only, +0.90 over vanilla. The proposal's <2% degradation target (Gate A) is not met at ~13%.

4. **Gate warmup target=0.1 is optimal.** 0.2 and 0.5 are too aggressive (AR collapses). Without warmup, gates never open (chicken-and-egg problem with tanh gating).

5. **30% memory task ratio prevents sidecar degradation.** Pure FineWeb: VT collapses 9→3 over 3000 steps. 10% mix: non-monotonic. 30% mix: stable or improving over 5000 steps.

6. **Gates self-regulate with mixed training.** They grow to ~0.093, then decrease to ~0.044-0.059 as the memory task gradient finds the optimal operating point. Per-layer decoupling emerges naturally.

7. **Minimum 500 steps for VT signal.** VT jumps from 0 to 9 between steps 300-500. Shorter runs are unreliable for evaluating memory tasks.

8. **GPU ESN gives 38.7x speedup** (18.3s → 0.47s per 2048-step sequence). Essential for practical eval.

## Gate A Assessment

| Criterion | Target | Result | |
|-----------|--------|--------|-|
| Long-context recall | ≥10% | +100% (Passkey), +14.5% (AR) | PASS — but from training data |
| Algorithmic memory | ≥15% | +66% (VT) | PASS — but from training data |
| Compositional gen | ≥10% | +37.5% (ModArith), +4% (Dyck) | PARTIAL — ModArith from reservoir |
| Latency overhead | ≤20% | ~15-20% | PASS |
| Perplexity degradation | <2% | +13% | **FAIL** |

**Verdict:** Gate A fails on perplexity. The reservoir's unique contribution is narrow (3 tasks). Most value comes from fine-tuning on synthetic memory data, which works without a reservoir at all.

## Key Architectural Lessons

- **Ungated sidecar injection destroys the base model.** LayerNorm(hidden + random_sidecar) re-normalizes hidden states, causing format collapse within 1000 steps.
- **Flamingo-style tanh(alpha) gating is essential.** Raw scalar gates don't open (gradient ∝ gate_value ≈ 0). tanh gives gradient = 1 at alpha=0 regardless of gate value.
- **Don't combine tanh gate with zero-init projections.** Both zeroing mechanisms together kill all gradient flow (double-zero trap).
- **Gradient checkpointing is incompatible with sidecar hooks.** Recomputation doesn't replay hooks consistently. Use batch_size reduction instead.
- **PYTORCH_ALLOC_CONF=expandable_segments:True** required to prevent CUDA memory fragmentation OOM on long runs.
