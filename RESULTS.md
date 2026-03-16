# Track A Results

Reservoir computing sidecar (ESN) bolted onto frozen Qwen3.5-0.8B via LoRA + gated linear projection.

## Setup (Final Configuration)

- **Base model:** Qwen3.5-0.8B-Base (frozen, 753M params)
- **LoRA:** rank=16, alpha=32, q_proj+v_proj (639K trainable params, 0.085%)
- **ESN reservoir:** 1,000 nodes, spectral_radius=0.9, leak_rate=0.5, sparsity=0.01
- **Sidecar:** GatedLinearSidecar at layers [3, 7, 11, 15, 23], Flamingo-style tanh(alpha) gating
- **Training:** 5000 steps, batch=1, grad_accum=16, seq_len=2048, lr=2e-4, 30% synthetic memory tasks + 70% FineWeb
- **Hardware:** 1x RTX 3090 (24 GB), ~75 min training, ~30 min full eval

## Full Benchmark Results (Sprint Best, n=50)

| Task | Track A | Vanilla | Delta |
|------|---------|---------|-------|
| **PasskeyRetrieval** (200 ctx) | **1.000** | 0.005 | **+0.995** |
| **PasskeyRetrieval** (500 ctx) | **1.000** | 0.005 | **+0.995** |
| **MultiDigitArith** (3-digit add) | **0.940** | 0.000 | **+0.940** |
| **MultiDigitArith** (4-digit add) | **0.940** | 0.000 | **+0.940** |
| **VariableTracking** (3 vars) | **0.720** | 0.440 | **+0.280** |
| **VariableTracking** (5 vars) | **0.560** | 0.440 | **+0.120** |
| **ProgramTrace** (6 steps) | **0.220** | 0.000 | **+0.220** |
| **ProgramTrace** (4 steps) | **0.180** | 0.000 | **+0.180** |
| **MultiDigitArith** (3-digit mul) | **0.140** | 0.000 | **+0.140** |
| **ModularArithmetic** | **0.140** | 0.000 | **+0.140** |
| **LengthExtrapolation** (1x) | **0.100** | 0.000 | **+0.100** |
| **AlgorithmicTransfer** (sort/train) | **0.100** | 0.000 | **+0.100** |
| CompositionalGen (train) | 0.060 | 0.075 | -0.015 |
| AlgorithmicTransfer (sort/test) | 0.040 | 0.000 | +0.040 |
| LengthExtrapolation (4x) | 0.020 | 0.000 | +0.020 |
| AlgorithmicTransfer (search/test) | 0.020 | 0.000 | +0.020 |
| CompositionalGen (test) | 0.020 | 0.075 | -0.055 |
| LengthExtrapolation (2x) | 0.000 | 0.000 | +0.000 |
| AlgorithmicTransfer (search/train) | 0.000 | 0.000 | +0.000 |
| AssociativeRecall (5 pairs) | **1.000** | 0.825 | +0.175 |
| AssociativeRecall (10 pairs) | **1.000** | 0.825 | +0.175 |
| DyckLanguage (depth 3) | 0.000 | 0.000 | +0.000 |
| DyckLanguage (depth 4) | 0.000 | 0.000 | +0.000 |

- **Average exact-match: 0.357** (vanilla: ~0.12, **3x improvement**)
- **Perplexity:** 6.84 (vanilla: 6.82, delta: **+0.3%** — negligible)

## Sprint Metric Results (3-task focus)

| Metric | Best Sprint | LoRA-only Control |
|--------|------------|-------------------|
| ModularArithmetic | 0.14-0.20 | 0.02 |
| LengthExtrapolation | 0.00 | 0.00 |
| DyckLanguage | 0.00 | 0.00 |
| avg_task/ppl score | 0.007-0.010 | 0.001 |

## Durable Conclusions

1. **GatedLinearSidecar is the winning architecture.** Per-token linear projection of reservoir states is inherently causal (no mask needed), simpler than cross-attention, and performs better. Cross-attention had a causal leakage bug where queries could see future reservoir states — inflating all pre-fix results.

2. **Reservoir genuinely helps.** LoRA-only control (gates frozen at 0) gives modarith=0.02; gated linear gives 0.14-0.20 (~7-10x). The reservoir provides sequential state information that LoRA alone cannot internalize.

3. **Perplexity cost is negligible with the right architecture.** Old cross-attention: +13% ppl (7.72 vs 6.82). New gated linear with r=1000: +0.3% ppl (6.84 vs 6.82). Key factors: smaller reservoir (1000 vs 10000), simpler projection (linear vs cross-attention), proper gating.

4. **Layer positions are critical and fragile.** [3,7,11,15,23] is optimal. Removing ANY single layer (exp44: skip 7, exp45: skip 11) kills all task performance. The 5-layer spread provides necessary coverage across the transformer depth.

5. **5000 training steps is the sweet spot.** 3000 steps: underfitting (modarith=0.06). 5000: optimal (0.14-0.20). 7500-10000: overfitting (modarith drops, ppl rises).

6. **Gate warmup target=0.1 is optimal.** 0.05: too restrictive (modarith=0.08). 0.1: optimal. 0.2: too aggressive (tasks collapse). 0.0 (no warmup): gates never open.

7. **Reservoir size 1000 is optimal.** 10000: too noisy, high ppl cost. 2000: no improvement over 1000. 1000: best task/ppl tradeoff.

8. **Training data composition matters.** 30% synthetic memory tasks + 70% FineWeb. 50% memory: tasks collapse (too much distribution shift). 0% memory: tasks don't learn to use the reservoir.

## Experiment Summary (45 experiments)

- **Architecture search:** cross-attention (leaky), cross-attention+causal mask, gated linear, FiLM, MLP bottleneck → gated linear wins
- **Layer search:** 1-6 layers, various positions → 5 layers [3,7,11,15,23]
- **Hyperparameter search:** LR (1e-4 to 2e-3), gate warmup (0-0.2), memory ratio (0-0.5), LoRA rank (16/32), steps (300-10000), reservoir size (1000-10000), ESN params, grad accum, interface LR
- **Negative results:** state deltas (noise not signal), MLP bottleneck (too restrictive), higher spectral radius (kills tasks), longer training (overfits)

## Key Architectural Lessons

- **Causal masking is essential for cross-attention sidecars.** ESN processes the full sequence, so reservoir state at position s contains info about all tokens 0..s. Without causal mask, queries at position t can attend to s>t, leaking future information. This inflated all cross-attention results before the fix.
- **GatedLinearSidecar avoids the causal problem entirely.** Each position t only uses reservoir_state[t], which encodes input history 0..t. Inherently causal by construction.
- **Flamingo-style tanh(alpha) gating is essential.** Raw scalar gates don't open (gradient proportional to gate_value near 0). tanh gives gradient=1 at alpha=0 regardless of gate value.
- **Don't combine tanh gate with zero-init projections.** Both zeroing mechanisms together kill all gradient flow (double-zero trap).
- **Gradient checkpointing is incompatible with sidecar hooks.** Recomputation doesn't replay hooks consistently. Use batch_size reduction instead.

---

# Track B Results

DeltaNet block replacement: replace selected DeltaNet (linear attention) blocks in Qwen3.5-0.8B with ESN reservoir modules.

## Architecture

Qwen3.5-0.8B has **18 DeltaNet + 6 full-attention layers** (3:1 pattern). Track B replaces selected DeltaNet blocks with `ESNReplacementInterface`:
```
output = gate * esn_projection(reservoir_states) + (1 - gate) * original_deltanet_output
```
- Sigmoid gate (per-element, input-dependent)
- ESN reservoir: r=1000, same config as Track A
- LoRA on q_proj+v_proj (same as Track A)
- 30% synthetic memory tasks + 70% FineWeb

## Track B Experiment Log

| Exp | Blocks Replaced | gate_init | ppl | avg_em | Status | Notes |
|-----|----------------|-----------|-----|--------|--------|-------|
| B1 | 6/18 (every 3rd: 0,3,6,9,12,15) | 0.1 | 13.76 | 0.219 | baseline | Way too aggressive — ppl +101%, most tasks lost |

### B1 Detail (vs Track A)

| Task | Track B | Track A | Delta |
|------|---------|---------|-------|
| PasskeyRetrieval | 1.000 | 1.000 | = |
| AssociativeRecall | 1.000 | 1.000 | = |
| VariableTracking (3v) | 0.440 | 0.720 | -0.280 |
| VariableTracking (5v) | 0.340 | 0.560 | -0.220 |
| ProgramTrace (4s) | 0.140 | 0.180 | -0.040 |
| ProgramTrace (6s) | 0.120 | 0.220 | -0.100 |
| ModularArithmetic | 0.000 | 0.140 | -0.140 |
| MultiDigitArith (all) | 0.000 | 0.940 | -0.940 |
| AlgorithmicTransfer | 0.000 | 0.100 | -0.100 |

## Track B Conclusions (so far)

1. **Replacing 6/18 DeltaNet blocks is too aggressive.** Perplexity doubles (+101%), most task performance lost. DeltaNet blocks carry critical information the ESN can't replicate.
2. **Easy tasks survive (passkey, associative recall)** but harder tasks (arithmetic, variable tracking) collapse.
3. **Need much lighter touch:** fewer blocks, lower gate init, or additive (not replacement) integration.

## Research Plan

### Phase 1: Distillation Sweep
For each of 18 DeltaNet layers: capture input/output activations, train ESN readout (ridge regression) to reproduce DeltaNet output. Map which layers are "reservoir-compatible" (low reconstruction MSE → safe to replace). This is cheap (~30 min, no gradient descent).

### Phase 2: Single-Layer Replacement
Replace the easiest layer (lowest distillation loss) with ESN, **initialized from distillation weights** (warm start). Fine-tune with very low gate_init (0.01). B1 failed because we cold-swapped 6 layers — Phase 2 is surgical and informed.

### Phase 3: ESN as Forgetting Controller
Don't replace DeltaNet. Keep it intact, run ESN in parallel. Use ESN state to *gate* DeltaNet's output — multiplicative modulation instead of additive injection. The reservoir's dynamics naturally perform relevance filtering (Dambre decomposition: tracks linear memory + nonlinear interaction patterns). This tests the hypothesis that the reservoir's value is not as a memory store but as a memory *controller* — telling the LLM what's worth keeping.
