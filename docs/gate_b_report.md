# Gate B Evaluation Report

**Generated:** 2026-03-05 06:25 UTC  
**Project:** Latent Reservoir Scratchpads for LLMs (LRS)  
**Purpose:** Pass/fail decision point before Track C (From-Scratch Architecture)

---

## Overall Decision

> **PENDING — all GPU evaluations outstanding**

**Recommendation:** Suspend judgment; all Track B GPU evaluations must complete before Gate B can be assessed.

---

## Data Availability

| Status | Tasks |
| --- | --- |
| Available | None |
| Missing / pending | T7, T8, T14, T15, T16, T19, T20, T21 |

All missing tasks require GPU training/evaluation to be run first.
See the relevant scripts in `scripts/` for each task.

---

## Gate B Criteria

| # | Criterion | Threshold | T7 (ref) | Best model (value) | Δ vs T7/T8 | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Exact-match gain on long program-trace tasks | >= 20% gain vs T7 | N/A | — (N/A) | N/A | **⧖ PENDING** |
| 2 | Memory-quality-per-byte vs RoPE/YaRN (T8) | Track B quality/byte >= T8 quality/byte | N/A | — (N/A) | N/A | **⧖ PENDING** |
| 3 | Perplexity degradation vs T7 | < 2% degradation vs T7 | N/A | — (N/A) | N/A | **⧖ PENDING** |

### Criterion definitions

1. **Long program-trace exact-match gain** — averaged exact-match accuracy on
   ProgramTrace benchmarks (synthetic program execution traces).  Falls back to
   VariableTracking + AssociativeRecall average if ProgramTrace not available.
   Must show ≥ 20% improvement vs T7 (stricter than Gate A's 15% on algorithmic memory).
2. **Memory-quality-per-byte vs T8** — ratio of memory-task accuracy to additional
   parameter bytes.  YaRN/RoPE (T8) adds near-zero extra parameters; if T8 quality
   is available, Track B must achieve ≥ T8 absolute quality to pass this criterion
   (since T8's quality/byte ratio is effectively infinite).  When T8 result is
   unavailable, Track B quality/byte ratio is compared against T8's estimated ratio.
3. **Perplexity degradation** — best Track B perplexity must be < 2% worse than T7.
   Ensures reservoir insertion does not degrade language modelling.

---

## Track A vs Track B Comparison

| Track | Best model | Quality | Quality gain vs T7 |
| --- | --- | --- | --- |
| Track A | **T14** Qwen3.5 + RC read-only (Track A) | N/A | N/A |
| Track B | **—** PENDING | N/A | N/A |

**Track B gain over Track A:** N/A

**Verdict:** PENDING

---

## Track B Strategy Comparison

| Config | Quality gain vs T7 | Passkey gain vs T7 | Perplexity delta vs T7 | Extra params |
| --- | --- | --- | --- | --- |
| **T19 RIL** | N/A | N/A | N/A | N/A |
| **T20 DeltaNet** | N/A | N/A | N/A | N/A |
| **T21 Multi-res** | N/A | N/A | N/A | N/A |

**Best strategy by quality gain:** PENDING

**Single vs multi-reservoir:** PENDING
  - Quality gain delta (multi - single RIL): N/A

---

## Full Comparison Table

Delta values in parentheses are percentage change relative to T7.
Positive delta = improvement for accuracy metrics.

| Model | Passkey (↑) | Algo Mem (↑) | Prog Trace (↑) | Perplexity (↓) | Extra Params |
| --- | --- | --- | --- | --- | --- |
| **T7** Qwen3.5-0.8B vanilla | N/A | N/A | N/A | N/A | N/A |
| **T8** Qwen3.5-0.8B + YaRN | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |
| **T14** Qwen3.5 + RC read-only (Track A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |
| **T15** Qwen3.5 + RC read/write (Track A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |
| **T16** Best HP sweep config (Track A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |
| **T19** Track B: RIL insertion | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |
| **T20** Track B: DeltaNet replacement | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |
| **T21** Track B: Multi-reservoir RIL | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A (N/A) | N/A |

---

## Efficiency Analysis

Quality-per-byte is computed as accuracy / (extra_parameters × 2 bytes).
A higher value is better.  T8 (YaRN) has near-zero extra params → ratio ≈ ∞.

| Model | Quality | Quality gain vs T7 | Extra params | Quality/byte | Gain/byte |
| --- | --- | --- | --- | --- | --- |
| **T8** Qwen3.5-0.8B + YaRN | N/A | N/A | N/A | N/A | N/A |
| **T14** Qwen3.5 + RC read-only (Track A) | N/A | N/A | N/A | N/A | N/A |
| **T15** Qwen3.5 + RC read/write (Track A) | N/A | N/A | N/A | N/A | N/A |
| **T19** Track B: RIL insertion | N/A | N/A | N/A | N/A | N/A |
| **T20** Track B: DeltaNet replacement | N/A | N/A | N/A | N/A | N/A |
| **T21** Track B: Multi-reservoir RIL | N/A | N/A | N/A | N/A | N/A |

**Recommended for Track C:** PENDING

**Rationale:** No Track B results available.

---

## Architecture Recommendation for Track C

**Gate B status:** No Gate B criteria met (may be PENDING).

**Gate B not passed: recommend revising Track B before Track C.**

### Recommended Track C Architecture

**PENDING**: Track B GPU evaluations must complete before a specific
Track C architecture recommendation can be made.

**Provisional guidance (based on architecture design):**

1. If multi-reservoir (T21) outperforms single-reservoir (T19): use dual-timescale
   reservoirs in Track C (fast scratch-space + slow long-term memory)
2. If RIL insertion (T19) outperforms DeltaNet replacement (T20): use RIL-style
   cross-attention injection in the from-scratch architecture
3. Reserve DeltaNet replacement as a backup if RIL introduces too much overhead
4. From-scratch training (T26) should use the T25 curriculum with Stage 2
   procedural objectives for program-trace optimization

---

## Reproducibility

All numbers in this report are derived from JSON artifacts in `results/`.
Re-run the analysis at any time:

```bash
python scripts/gate_b_analysis.py
```

To regenerate after running all evaluations:

| Task | Script |
| --- | --- |
| T7 (Qwen vanilla)   | `python scripts/eval_qwen_vanilla.py` |
| T8 (YaRN)           | `python scripts/eval_qwen35_yarn.py` |
| T14 (read-only)     | `python scripts/train_track_a_readonly.py` |
| T15 (read/write)    | *(see T15 training script)* |
| T16 (HP sweep)      | `python scripts/sweep_reservoir_hp.py` |
| T19 (Track B RIL)   | `python scripts/train_track_b_ril.py` |
| T20 (Track B DeltaNet) | `python scripts/train_track_b_deltanet.py` |
| T21 (Track B Multi) | `python scripts/train_track_b_multi.py` |

After each evaluation completes, rerun this script to update the report.

---

*Report auto-generated by `scripts/gate_b_analysis.py`.*
