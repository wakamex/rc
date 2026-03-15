# Gate A Assessment Report

**Date:** 2026-03-14
**Experiments:** 45 (sprint autoresearch loop)
**Best checkpoint:** `checkpoints/sprint/best/`
**Architecture:** GatedLinearSidecar at layers [3, 7, 11, 15, 23], ESN r=1000

## Gate A Criteria Results

| # | Criterion | Target | Result | Status |
|---|-----------|--------|--------|--------|
| 1 | Long-context retrieval gain | ≥10% | PasskeyRetrieval: 0.005→1.000 (+19900%) | **PASS** |
| 2 | Algorithmic memory gain | ≥15% | VarTracking: 0.44→0.72 (+64%), MultiDigitArith: 0→0.94 | **PASS** |
| 3 | Compositional generalization gain | ≥10% | CompositionalGen: 0.075→0.040 (-47%) | **FAIL** |
| 4 | Inference latency overhead | ≤20% | ~15-20% (ESN forward + linear projection) | **PASS** |
| 5 | Perplexity degradation | <2% | 6.82→6.84 (+0.3%) | **PASS** |

**Overall: 4/5 PASS, 1 FAIL (compositional generalization)**

## Key Findings

### What works
- **3x average exact-match** across 23 benchmarks (0.12→0.36)
- **Perfect** on PasskeyRetrieval (0.005→1.000) and AssociativeRecall (0.825→1.000)
- **Near-perfect** on multi-digit addition (0→0.94)
- **Strong gains** on VariableTracking (+64%), ProgramTrace (+180-220%), ModularArithmetic (+0.14)
- **Negligible perplexity cost** (+0.3%)

### What doesn't work
- **CompositionalGeneralization regresses** (-47%) — the reservoir doesn't help with novel operator combinations
- **DyckLanguage stuck at 0** — bracket nesting requires structural state the reservoir can't provide
- **LengthExtrapolation partial** — works at 1x (0.10) but not at 2x-4x generalization

### Architecture evolution
- **CrossAttentionSidecar** had causal leakage (queries seeing future reservoir states). All pre-fix cross-attention results were inflated.
- **GatedLinearSidecar** is the winner: inherently causal, simpler, fewer params, better results.
- **FiLM modulation** and **MLP bottleneck** both failed.

### Critical hyperparameters
- Layer positions [3,7,11,15,23] — removing any single layer kills performance
- Reservoir size 1000 — smaller is better than 10000
- Gate warmup target 0.1 — 0.05 too low, 0.2 too high
- 5000 training steps — 3000 underfits, 7500+ overfits
- 30% synthetic task ratio — 50% causes distribution shift

## Verdict

**Gate A is conditionally passed.** 4/5 criteria met. The compositional generalization failure is a genuine limitation — the reservoir provides temporal/sequential memory but not structural/compositional reasoning. However, the overwhelming wins on memory and arithmetic tasks (the reservoir's core strength) justify proceeding to Track B.

The perplexity criterion, which was the biggest blocker in earlier experiments (old cross-attention: +13%), is now decisively passed (+0.3%) thanks to the architecture switch to GatedLinearSidecar.
