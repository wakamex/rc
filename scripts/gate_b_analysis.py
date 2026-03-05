#!/usr/bin/env python3
"""Gate B evaluation analysis script (rc-wwh.23).

Reads result JSONs produced by T22 (Track B training & evaluation) and
prior track results, then generates a comparison table plus pass/fail
assessment for each Gate B criterion.  Also writes docs/gate_b_report.md.

Expected result file paths
--------------------------
T7  (Qwen3.5 vanilla)   : results/baselines/qwen35_vanilla.json
T8  (YaRN)              : results/baselines/qwen35_yarn.json
T14 (read-only sidecar) : results/track_a/readonly.json
T15 (read/write sidecar): results/track_a/readwrite.json
T16 (best HP sweep)     : results/track_a/sweep/pareto_frontier.json
T19 (Track B RIL)       : results/track_b/ril.json
T20 (Track B DeltaNet)  : results/track_b/deltanet.json
T21 (Track B Multi-res) : results/track_b/multi.json

Gate B Thresholds (ALL must pass for full pass)
----------------------------------------------
1. >= 20% exact-match gain on long program-trace tasks vs T7
2. Better memory-quality-per-byte than RoPE/YaRN (T8)
3. < 2% perplexity degradation vs T7

Usage::

    python scripts/gate_b_analysis.py
    python scripts/gate_b_analysis.py --results_dir results --report_out docs/gate_b_report.md
    python scripts/gate_b_analysis.py --no_report   # stdout only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.resolve()

_RESULT_PATHS: dict[str, Path] = {
    "T7":  _REPO_ROOT / "results" / "baselines" / "qwen35_vanilla.json",
    "T8":  _REPO_ROOT / "results" / "baselines" / "qwen35_yarn.json",
    "T14": _REPO_ROOT / "results" / "track_a" / "readonly.json",
    "T15": _REPO_ROOT / "results" / "track_a" / "readwrite.json",
    "T16": _REPO_ROOT / "results" / "track_a" / "sweep" / "pareto_frontier.json",
    "T19": _REPO_ROOT / "results" / "track_b" / "ril.json",
    "T20": _REPO_ROOT / "results" / "track_b" / "deltanet.json",
    "T21": _REPO_ROOT / "results" / "track_b" / "multi.json",
}

_MODEL_LABELS: dict[str, str] = {
    "T7":  "Qwen3.5-0.8B vanilla",
    "T8":  "Qwen3.5-0.8B + YaRN",
    "T14": "Qwen3.5 + RC read-only (Track A)",
    "T15": "Qwen3.5 + RC read/write (Track A)",
    "T16": "Best HP sweep config (Track A)",
    "T19": "Track B: RIL insertion",
    "T20": "Track B: DeltaNet replacement",
    "T21": "Track B: Multi-reservoir RIL",
}

# Gate B thresholds
_GATE_THRESHOLDS = {
    "program_trace_gain":          0.20,  # >=20% gain vs T7
    "memory_quality_per_byte_vs_t8": None,  # Better ratio than T8 (dynamic)
    "perplexity_degradation_max":  0.02,  # <2% degradation vs T7
}

# ---------------------------------------------------------------------------
# Parameter estimation constants
# ---------------------------------------------------------------------------
# Qwen3.5-0.8B architecture parameters
_QWEN_HIDDEN_DIM = 896
_QWEN_NUM_LAYERS = 28
_QWEN_BYTES_PER_PARAM = 2  # bfloat16 / float16

# YaRN adds negligible extra parameters (rope scaling coefficients only)
_YARN_EXTRA_PARAMS = 128  # ~128 scalar coefficients for RoPE scaling

# LoRA rank 16 on Q/V projections for Qwen3.5-0.8B
# Q_proj: (hidden_dim, hidden_dim), V_proj: (hidden_dim, hidden_dim)
# LoRA: 2 matrices per projection (A: rank×hidden, B: hidden×rank)
_LORA_RANK = 16
_LORA_TARGET_COUNT = 2  # q_proj + v_proj
_LORA_EXTRA_PARAMS = (
    _LORA_TARGET_COUNT * _QWEN_NUM_LAYERS * 2 * _LORA_RANK * _QWEN_HIDDEN_DIM
)  # 2 LoRA matrices (A, B) per target, per layer


def _estimate_extra_params(data: dict[str, Any]) -> int:
    """Estimate additional parameters introduced by a Track B configuration."""
    arch = data.get("architecture", {})
    config = data.get("config", "")
    track = data.get("track", "")

    if track != "B":
        return 0

    reservoir_size = arch.get("reservoir_size", 10_000)
    sparsity = arch.get("reservoir_sparsity", arch.get("sparsity", 0.01))
    hidden_dim = _QWEN_HIDDEN_DIM

    # ESN reservoir matrix (sparse): ~reservoir_size^2 * sparsity non-zeros
    esn_matrix_params = int(reservoir_size * reservoir_size * sparsity)
    # Input projection: embed_dim -> reservoir_size
    esn_input_params = hidden_dim * reservoir_size

    esn_total = esn_matrix_params + esn_input_params

    if config == "multi":
        # Two reservoirs of equal size
        fast_size = arch.get("fast_reservoir_size", reservoir_size // 2)
        slow_size = arch.get("slow_reservoir_size", reservoir_size // 2)
        fast_matrix = int(fast_size * fast_size * sparsity)
        slow_matrix = int(slow_size * slow_size * sparsity)
        fast_input = hidden_dim * fast_size
        slow_input = hidden_dim * slow_size
        esn_total = fast_matrix + slow_matrix + fast_input + slow_input
        combined_reservoir_size = fast_size + slow_size
    else:
        combined_reservoir_size = reservoir_size

    # RIL layers (if present)
    ril_params = 0
    ril_layers = arch.get("ril_layers", [])
    if ril_layers:
        num_heads = 8
        # Per RIL: Q_proj + K_proj + V_proj + out_proj + gate_proj + norms
        q_proj = hidden_dim * hidden_dim
        k_proj = combined_reservoir_size * hidden_dim
        v_proj = combined_reservoir_size * hidden_dim
        out_proj = hidden_dim * hidden_dim
        gate_proj = (hidden_dim + combined_reservoir_size) * hidden_dim
        norms = 2 * hidden_dim  # pre-norm + post-norm
        per_ril = q_proj + k_proj + v_proj + out_proj + gate_proj + norms
        ril_params = len(ril_layers) * per_ril

    # DeltaNet replacement (no RIL in this config — ESN replaces recurrent blocks)
    deltanet_replacement_params = 0
    if config == "deltanet":
        replaced_blocks = arch.get("replaced_deltanet_blocks", [])
        # Each replacement adds a gate parameter (scalar per hidden dim)
        # and an output projection
        deltanet_replacement_params = len(replaced_blocks) * (hidden_dim * hidden_dim + hidden_dim)

    # LoRA params (always added in Track B)
    lora_rank = arch.get("lora_rank", _LORA_RANK)
    lora_targets = arch.get("lora_targets", ["q_proj", "v_proj"])
    lora_params = (
        len(lora_targets) * _QWEN_NUM_LAYERS * 2 * lora_rank * hidden_dim
    )

    return esn_total + ril_params + deltanet_replacement_params + lora_params


def _estimate_yarn_extra_params(data: dict[str, Any]) -> int:
    """YaRN adds only rope scaling parameters — essentially zero."""
    return _YARN_EXTRA_PARAMS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file; return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _avg_metric(
    results: list[dict[str, Any]],
    task_prefix: str,
    metric: str = "exact_match",
) -> float | None:
    """Average EvalResult values where task name starts with task_prefix."""
    vals = [
        r["value"]
        for r in results
        if r.get("task", "").startswith(task_prefix) and r.get("metric") == metric
    ]
    return sum(vals) / len(vals) if vals else None


def _first_metric(
    results: list[dict[str, Any]],
    task_prefix: str,
    metric: str = "exact_match",
) -> float | None:
    """Return the first matching EvalResult value."""
    for r in results:
        if r.get("task", "").startswith(task_prefix) and r.get("metric") == metric:
            return r["value"]
    return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_metrics(data: dict[str, Any], task_id: str) -> dict[str, float | None]:
    """Extract Gate B relevant metrics from a result dict.

    Returns a dict with keys:
        passkey_acc            – PasskeyRetrieval exact match
        variable_tracking_acc  – VariableTracking exact match
        associative_recall_acc – AssociativeRecall exact match
        algorithmic_memory_acc – Average of variable_tracking + associative_recall
        program_trace_acc      – ProgramTrace / synthetic execution traces exact match
        comp_gen_acc           – CompositionalGeneralization test-split exact match
        latency_p50_s          – p50 inference latency in seconds
        tokens_per_sec         – generation throughput
        perplexity             – held-out text perplexity
        extra_params           – additional parameters vs T7 (estimated)
    """
    status = data.get("status", "")
    if status == "pending_gpu_training":
        return {k: None for k in _metric_keys()}

    results_list: list[dict[str, Any]] = data.get("results", [])

    if task_id == "T16":
        return _extract_t16_metrics(data)

    if task_id in ("T14", "T15"):
        return _extract_training_metrics(data, results_list, task_id)

    if task_id in ("T19", "T20", "T21"):
        return _extract_track_b_metrics(data, results_list, task_id)

    # Standard baseline eval format (T7, T8)
    passkey = _avg_metric(results_list, "PasskeyRetrieval")
    vt      = _avg_metric(results_list, "VariableTracking")
    ar      = _avg_metric(results_list, "AssociativeRecall")
    pt      = _avg_metric(results_list, "ProgramTrace")
    alg_mem = _combine_memory(vt, ar)
    cg_test = _avg_metric(results_list, "CompositionalGeneralization")

    latency = data.get("latency", {})
    throughput = data.get("throughput", {})
    perplexity = data.get("perplexity")
    if perplexity is None:
        ppl_result = _first_metric(results_list, "", metric="perplexity")
        perplexity = ppl_result

    extra_params = 0
    if task_id == "T8":
        extra_params = _estimate_yarn_extra_params(data)

    return {
        "passkey_acc":            passkey,
        "variable_tracking_acc":  vt,
        "associative_recall_acc": ar,
        "algorithmic_memory_acc": alg_mem,
        "program_trace_acc":      pt,
        "comp_gen_acc":           cg_test,
        "latency_p50_s":          latency.get("p50_s"),
        "tokens_per_sec":         throughput.get("tokens_per_sec"),
        "perplexity":             perplexity,
        "extra_params":           extra_params,
    }


def _metric_keys() -> list[str]:
    return [
        "passkey_acc", "variable_tracking_acc", "associative_recall_acc",
        "algorithmic_memory_acc", "program_trace_acc", "comp_gen_acc",
        "latency_p50_s", "tokens_per_sec", "perplexity", "extra_params",
    ]


def _combine_memory(vt: float | None, ar: float | None) -> float | None:
    if vt is not None and ar is not None:
        return (vt + ar) / 2
    return vt if vt is not None else ar


def _extract_t16_metrics(data: dict[str, Any]) -> dict[str, float | None]:
    """Extract metrics from T16 pareto frontier JSON."""
    pareto_list: list[dict[str, Any]] = data.get("pareto_frontier", [])
    base: dict[str, float | None] = {k: None for k in _metric_keys()}
    if not pareto_list:
        return base
    best = pareto_list[0]
    base.update({
        "passkey_acc":   best.get("passkey_acc"),
        "latency_p50_s": (best.get("step_latency_ms") or 0) / 1000 or None,
        "_quality_score": best.get("quality_score"),
    })
    return base


def _extract_training_metrics(
    data: dict[str, Any],
    results_list: list[dict[str, Any]],
    task_id: str,
) -> dict[str, float | None]:
    """Extract from training-format JSON (T14/T15)."""
    passkey = _avg_metric(results_list, "PasskeyRetrieval") if results_list else None
    vt      = _avg_metric(results_list, "VariableTracking") if results_list else None
    ar      = _avg_metric(results_list, "AssociativeRecall") if results_list else None
    pt      = _avg_metric(results_list, "ProgramTrace") if results_list else None
    alg_mem = _combine_memory(vt, ar)
    cg_test = _avg_metric(results_list, "CompositionalGeneralization") if results_list else None

    metrics = data.get("metrics", {})
    perplexity = (
        metrics.get("final_train_perplexity")
        or data.get("perplexity")
    )
    latency = data.get("latency", {})
    throughput = data.get("throughput", {})

    return {
        "passkey_acc":            passkey,
        "variable_tracking_acc":  vt,
        "associative_recall_acc": ar,
        "algorithmic_memory_acc": alg_mem,
        "program_trace_acc":      pt,
        "comp_gen_acc":           cg_test,
        "latency_p50_s":          latency.get("p50_s"),
        "tokens_per_sec":         throughput.get("tokens_per_sec"),
        "perplexity":             perplexity,
        "extra_params":           0,
    }


def _extract_track_b_metrics(
    data: dict[str, Any],
    results_list: list[dict[str, Any]],
    task_id: str,
) -> dict[str, float | None]:
    """Extract from Track B training-format JSON (T19/T20/T21)."""
    passkey = _avg_metric(results_list, "PasskeyRetrieval") if results_list else None
    vt      = _avg_metric(results_list, "VariableTracking") if results_list else None
    ar      = _avg_metric(results_list, "AssociativeRecall") if results_list else None
    pt      = _avg_metric(results_list, "ProgramTrace") if results_list else None
    alg_mem = _combine_memory(vt, ar)
    cg_test = _avg_metric(results_list, "CompositionalGeneralization") if results_list else None

    metrics = data.get("metrics", {})
    perplexity = (
        metrics.get("final_train_perplexity")
        or data.get("perplexity")
    )
    latency = data.get("latency", {})
    throughput = data.get("throughput", {})

    extra_params = _estimate_extra_params(data)

    return {
        "passkey_acc":            passkey,
        "variable_tracking_acc":  vt,
        "associative_recall_acc": ar,
        "algorithmic_memory_acc": alg_mem,
        "program_trace_acc":      pt,
        "comp_gen_acc":           cg_test,
        "latency_p50_s":          latency.get("p50_s"),
        "tokens_per_sec":         throughput.get("tokens_per_sec"),
        "perplexity":             perplexity,
        "extra_params":           extra_params,
    }


# ---------------------------------------------------------------------------
# Gate B criterion evaluation
# ---------------------------------------------------------------------------


def _pct_gain(val: float | None, ref: float | None) -> float | None:
    """Return (val - ref) / ref as a fraction."""
    if val is None or ref is None or ref == 0.0:
        return None
    return (val - ref) / ref


def _quality_per_byte(
    quality: float | None,
    extra_params: int | None,
    bytes_per_param: int = _QWEN_BYTES_PER_PARAM,
) -> float | None:
    """Compute quality / extra_param_bytes ratio.

    Returns None if quality or extra_params is missing.
    If extra_params == 0, returns None (division by zero / undefined).
    """
    if quality is None or extra_params is None or extra_params == 0:
        return None
    return quality / (extra_params * bytes_per_param)


def evaluate_gate_criteria(
    all_metrics: dict[str, dict[str, float | None]],
    reference: str = "T7",
    yarn_ref: str = "T8",
) -> list[dict[str, Any]]:
    """Evaluate each Gate B criterion.

    Returns a list of criterion dicts with keys:
        id, name, threshold_str, best_model, best_value, ref_value,
        gain_or_overhead, pass_fail
    """
    ref = all_metrics.get(reference, {})
    yarn = all_metrics.get(yarn_ref, {})

    track_b_ids = ("T19", "T20", "T21")

    def best_b(key: str, higher_is_better: bool = True) -> tuple[str | None, float | None]:
        candidates = {
            tid: m[key]
            for tid, m in all_metrics.items()
            if tid in track_b_ids and m.get(key) is not None
        }
        if not candidates:
            return None, None
        best_tid = max(candidates, key=lambda t: candidates[t]) if higher_is_better \
            else min(candidates, key=lambda t: candidates[t])
        return best_tid, candidates[best_tid]

    # ------------------------------------------------------------------
    # Criterion 1: >= 20% exact-match gain on long program-trace tasks vs T7
    # ------------------------------------------------------------------
    # Use ProgramTrace if available, else fall back to algorithmic_memory (VT+AR avg)
    pt_key = "program_trace_acc"
    mem_key = "algorithmic_memory_acc"

    best_tid1, best_val1 = best_b(pt_key, higher_is_better=True)
    ref_pt = ref.get(pt_key)

    # Fall back to algorithmic_memory if program_trace not available
    if best_val1 is None:
        best_tid1, best_val1 = best_b(mem_key, higher_is_better=True)
        ref_pt = ref.get(mem_key)
        metric_used = "algorithmic_memory_acc (proxy for program-trace)"
    else:
        metric_used = "program_trace_acc"

    gain1 = _pct_gain(best_val1, ref_pt)
    c1 = {
        "id": 1,
        "name": "Exact-match gain on long program-trace tasks",
        "threshold_str": ">= 20% gain vs T7",
        "threshold": _GATE_THRESHOLDS["program_trace_gain"],
        "direction": "higher",
        "metric_used": metric_used,
        "best_model": best_tid1,
        "best_value": best_val1,
        "ref_value": ref_pt,
        "gain_or_overhead": gain1,
        "pass_fail": (
            "PASS" if gain1 is not None and gain1 >= _GATE_THRESHOLDS["program_trace_gain"]
            else ("FAIL" if gain1 is not None else "PENDING")
        ),
    }

    # ------------------------------------------------------------------
    # Criterion 2: Better memory-quality-per-byte than T8 (YaRN)
    # ------------------------------------------------------------------
    # Memory quality proxy: use program_trace_acc, else algorithmic_memory_acc
    q_key = pt_key if any(
        all_metrics.get(t, {}).get(pt_key) is not None for t in track_b_ids
    ) else mem_key

    # Track B best quality / extra_bytes
    tb_quality_per_byte: dict[str, float | None] = {}
    for tid in track_b_ids:
        m = all_metrics.get(tid, {})
        q = m.get(q_key)
        ep = m.get("extra_params")
        if isinstance(ep, float):
            ep = int(ep)
        tb_quality_per_byte[tid] = _quality_per_byte(q, ep)

    best_tb_qpb_tid = max(
        (t for t in track_b_ids if tb_quality_per_byte.get(t) is not None),
        key=lambda t: tb_quality_per_byte[t] or 0.0,
        default=None,
    )
    best_tb_qpb = tb_quality_per_byte.get(best_tb_qpb_tid) if best_tb_qpb_tid else None

    # T8 reference quality / extra_bytes
    yarn_quality = yarn.get(q_key)
    yarn_extra = yarn.get("extra_params")
    if isinstance(yarn_extra, float):
        yarn_extra = int(yarn_extra)
    yarn_qpb = _quality_per_byte(yarn_quality, yarn_extra)

    # Comparison: if T8 has quality >= T7 with near-zero extra params,
    # we use absolute quality gain from T7 as the quality metric instead
    # (since dividing by near-zero extra params is undefined/infinite)
    if yarn_qpb is None and yarn_quality is not None and ref.get(q_key) is not None:
        # T8 has near-zero extra params → use absolute quality for comparison
        # Track B must achieve higher absolute quality than T8 to be competitive
        ref_qpb_note = "T8 quality/byte undefined (near-zero extra params); comparing absolute quality"
        t8_comparison_val = yarn_quality
        tb_comparison_val = (
            all_metrics.get(best_tb_qpb_tid, {}).get(q_key) if best_tb_qpb_tid else None
        )
        c2_pass = (
            "PASS" if (tb_comparison_val is not None and
                       t8_comparison_val is not None and
                       tb_comparison_val >= t8_comparison_val)
            else ("FAIL" if tb_comparison_val is not None else "PENDING")
        )
        c2_gain = _pct_gain(tb_comparison_val, t8_comparison_val)
    else:
        ref_qpb_note = "T8 quality/byte ratio computed"
        tb_comparison_val = best_tb_qpb
        t8_comparison_val = yarn_qpb
        if best_tb_qpb is not None and yarn_qpb is not None:
            c2_pass = "PASS" if best_tb_qpb >= yarn_qpb else "FAIL"
            c2_gain = _pct_gain(best_tb_qpb, yarn_qpb)
        elif best_tb_qpb is not None:
            c2_pass = "FAIL"
            c2_gain = None
        else:
            c2_pass = "PENDING"
            c2_gain = None

    c2 = {
        "id": 2,
        "name": "Memory-quality-per-byte vs RoPE/YaRN (T8)",
        "threshold_str": "Track B quality/byte >= T8 quality/byte",
        "threshold": None,
        "direction": "higher",
        "note": ref_qpb_note,
        "best_model": best_tb_qpb_tid,
        "best_value": best_tb_qpb,
        "ref_value": yarn_qpb,
        "t8_quality": t8_comparison_val,
        "tb_quality": tb_comparison_val,
        "gain_or_overhead": c2_gain,
        "pass_fail": c2_pass,
    }

    # ------------------------------------------------------------------
    # Criterion 3: < 2% perplexity degradation vs T7
    # ------------------------------------------------------------------
    # Best (lowest) perplexity among Track B models
    track_b_ppls = {
        tid: m.get("perplexity")
        for tid, m in all_metrics.items()
        if tid in track_b_ids and m.get("perplexity") is not None
    }
    if track_b_ppls:
        best_ppl_tid = min(track_b_ppls, key=lambda t: track_b_ppls[t])  # type: ignore[arg-type]
        best_ppl_val = track_b_ppls[best_ppl_tid]
        ppl_degradation = _pct_gain(best_ppl_val, ref.get("perplexity"))
    else:
        best_ppl_tid = None
        best_ppl_val = None
        ppl_degradation = None

    c3 = {
        "id": 3,
        "name": "Perplexity degradation vs T7",
        "threshold_str": "< 2% degradation vs T7",
        "threshold": _GATE_THRESHOLDS["perplexity_degradation_max"],
        "direction": "lower",
        "best_model": best_ppl_tid,
        "best_value": best_ppl_val,
        "ref_value": ref.get("perplexity"),
        "gain_or_overhead": ppl_degradation,
        "pass_fail": (
            "PASS" if ppl_degradation is not None and ppl_degradation < _GATE_THRESHOLDS["perplexity_degradation_max"]
            else ("FAIL" if ppl_degradation is not None else "PENDING")
        ),
    }

    return [c1, c2, c3]


# ---------------------------------------------------------------------------
# Strategy comparison analysis
# ---------------------------------------------------------------------------


def strategy_comparison(
    all_metrics: dict[str, dict[str, float | None]],
    t7_metrics: dict[str, float | None],
) -> dict[str, Any]:
    """Compare Track B insertion strategies.

    Compares RIL (T19), DeltaNet replacement (T20), and Multi-reservoir (T21).
    Also checks single (T19/T20) vs multi-reservoir (T21) performance.
    """
    configs = {
        "ril":      all_metrics.get("T19", {}),
        "deltanet": all_metrics.get("T20", {}),
        "multi":    all_metrics.get("T21", {}),
    }

    q_key = "program_trace_acc"
    fallback_key = "algorithmic_memory_acc"

    def _pick_quality(m: dict[str, float | None]) -> float | None:
        return m.get(q_key) if m.get(q_key) is not None else m.get(fallback_key)

    def _gain(val: float | None, ref: float | None) -> float | None:
        return _pct_gain(val, ref)

    t7_q = _pick_quality(t7_metrics)
    t7_ppl = t7_metrics.get("perplexity")

    results = {}
    for name, m in configs.items():
        q = _pick_quality(m)
        ppl = m.get("perplexity")
        ep = m.get("extra_params")
        results[name] = {
            "quality": q,
            "quality_gain_vs_t7": _gain(q, t7_q),
            "perplexity": ppl,
            "ppl_delta_vs_t7": _gain(ppl, t7_ppl),
            "extra_params": ep,
            "passkey_acc": m.get("passkey_acc"),
            "passkey_gain_vs_t7": _gain(m.get("passkey_acc"), t7_metrics.get("passkey_acc")),
        }

    # Best strategy by quality gain
    best_by_quality = max(
        (n for n in ("ril", "deltanet", "multi") if results[n]["quality_gain_vs_t7"] is not None),
        key=lambda n: results[n]["quality_gain_vs_t7"] or -1,
        default=None,
    )

    # Single vs multi-reservoir comparison (RIL single vs multi)
    ril_q_gain = results["ril"]["quality_gain_vs_t7"]
    multi_q_gain = results["multi"]["quality_gain_vs_t7"]
    if ril_q_gain is not None and multi_q_gain is not None:
        multi_advantage = multi_q_gain - ril_q_gain
        if multi_advantage > 0.05:
            multi_verdict = "SUPERIOR (multi outperforms single by >5%)"
        elif multi_advantage > 0.0:
            multi_verdict = "MARGINALLY_BETTER (multi slightly outperforms single)"
        elif multi_advantage < -0.05:
            multi_verdict = "INFERIOR (single outperforms multi by >5%)"
        else:
            multi_verdict = "COMPARABLE (no significant difference)"
    else:
        multi_verdict = "PENDING"

    return {
        "per_strategy": results,
        "best_by_quality": best_by_quality,
        "single_vs_multi_quality_delta": (
            multi_q_gain - ril_q_gain
            if ril_q_gain is not None and multi_q_gain is not None
            else None
        ),
        "multi_reservoir_verdict": multi_verdict,
    }


# ---------------------------------------------------------------------------
# Efficiency analysis
# ---------------------------------------------------------------------------


def efficiency_analysis(
    all_metrics: dict[str, dict[str, float | None]],
    t7_metrics: dict[str, float | None],
    t8_metrics: dict[str, float | None],
    track_a_best_id: str = "T15",
) -> dict[str, Any]:
    """Efficiency analysis: quality gain per additional parameter/byte.

    Compares Track B configs against T8 (YaRN) and Track A best.
    """
    q_key = "program_trace_acc"
    fallback_key = "algorithmic_memory_acc"

    def _pick_quality(m: dict[str, float | None]) -> float | None:
        return m.get(q_key) if m.get(q_key) is not None else m.get(fallback_key)

    t7_q = _pick_quality(t7_metrics)
    entries: dict[str, dict[str, Any]] = {}

    for tid in ("T8", "T14", "T15", "T19", "T20", "T21"):
        m = all_metrics.get(tid, {})
        q = _pick_quality(m)
        ep = m.get("extra_params")
        if isinstance(ep, float):
            ep = int(ep)
        q_gain = _pct_gain(q, t7_q)
        q_per_byte = _quality_per_byte(q, ep)
        gain_per_byte = (
            (q_gain / (ep * _QWEN_BYTES_PER_PARAM))
            if q_gain is not None and ep and ep > 0
            else None
        )
        entries[tid] = {
            "label": _MODEL_LABELS.get(tid, tid),
            "quality": q,
            "quality_gain_vs_t7": q_gain,
            "extra_params": ep,
            "extra_bytes": ep * _QWEN_BYTES_PER_PARAM if ep else None,
            "quality_per_byte": q_per_byte,
            "gain_per_byte": gain_per_byte,
        }

    # Recommended architecture for Track C
    track_b_candidates = {
        tid: entries[tid]
        for tid in ("T19", "T20", "T21")
        if entries[tid].get("quality_gain_vs_t7") is not None
    }
    if track_b_candidates:
        # Prefer high quality gain with reasonable parameter overhead
        # Score = quality_gain / log(extra_bytes + 1) if gain > 0
        def _score(e: dict[str, Any]) -> float:
            g = e.get("quality_gain_vs_t7") or 0.0
            b = e.get("extra_bytes") or 1
            import math
            return g / math.log(b + 1) if g > 0 else g

        recommended_tid = max(track_b_candidates, key=lambda t: _score(track_b_candidates[t]))
        recommendation = {
            "task_id": recommended_tid,
            "label": _MODEL_LABELS.get(recommended_tid, recommended_tid),
            "rationale": (
                f"Best quality-gain per log-byte ratio among Track B configs. "
                f"Quality gain: {_fmt_pct(track_b_candidates[recommended_tid].get('quality_gain_vs_t7'))}. "
                f"Extra params: {track_b_candidates[recommended_tid].get('extra_params', 'N/A'):,}."
            ) if track_b_candidates[recommended_tid].get("extra_params") else "PENDING",
        }
    else:
        recommended_tid = None
        recommendation = {"task_id": None, "label": "PENDING", "rationale": "No Track B results available."}

    return {
        "per_model": entries,
        "recommended_for_track_c": recommendation,
    }


# ---------------------------------------------------------------------------
# Track A vs Track B comparison
# ---------------------------------------------------------------------------


def track_a_vs_b(
    all_metrics: dict[str, dict[str, float | None]],
    t7_metrics: dict[str, float | None],
) -> dict[str, Any]:
    """Compare best Track A model vs best Track B model."""
    q_key = "program_trace_acc"
    fallback_key = "algorithmic_memory_acc"

    def _pick_quality(m: dict[str, float | None]) -> float | None:
        return m.get(q_key) if m.get(q_key) is not None else m.get(fallback_key)

    t7_q = _pick_quality(t7_metrics)

    # Best Track A: prefer T15 (read/write), else T14
    track_a_best_id = "T15" if all_metrics.get("T15", {}).get("algorithmic_memory_acc") is not None else "T14"
    track_a_best = all_metrics.get(track_a_best_id, {})
    ta_q = _pick_quality(track_a_best)

    # Best Track B
    track_b_ids = ["T19", "T20", "T21"]
    track_b_qualities = {
        tid: _pick_quality(all_metrics.get(tid, {}))
        for tid in track_b_ids
    }
    valid_tb = {tid: q for tid, q in track_b_qualities.items() if q is not None}
    if valid_tb:
        best_tb_id = max(valid_tb, key=lambda t: valid_tb[t])
        best_tb_q = valid_tb[best_tb_id]
    else:
        best_tb_id = None
        best_tb_q = None

    ta_gain = _pct_gain(ta_q, t7_q)
    tb_gain = _pct_gain(best_tb_q, t7_q)
    tb_over_ta = _pct_gain(best_tb_q, ta_q)

    if tb_gain is not None and ta_gain is not None:
        if tb_gain > ta_gain + 0.05:
            verdict = "TRACK_B_BETTER (significantly exceeds Track A)"
        elif tb_gain > ta_gain:
            verdict = "TRACK_B_BETTER (marginally exceeds Track A)"
        elif tb_gain >= ta_gain - 0.01:
            verdict = "COMPARABLE (Track B matches Track A)"
        else:
            verdict = "TRACK_A_BETTER (Track B underperforms Track A)"
    else:
        verdict = "PENDING"

    return {
        "track_a_best_id": track_a_best_id,
        "track_a_quality": ta_q,
        "track_a_gain_vs_t7": ta_gain,
        "track_b_best_id": best_tb_id,
        "track_b_quality": best_tb_q,
        "track_b_gain_vs_t7": tb_gain,
        "track_b_gain_over_track_a": tb_over_ta,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val*100:.1f}%"


def _fmt_params(val: int | float | None) -> str:
    if val is None:
        return "N/A"
    v = int(val)
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(v)


def _pf_symbol(pass_fail: str) -> str:
    return {"PASS": "✓ PASS", "FAIL": "✗ FAIL", "PENDING": "⧖ PENDING"}.get(pass_fail, pass_fail)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def build_comparison_table(
    all_metrics: dict[str, dict[str, float | None]],
    t7_ref: dict[str, float | None],
) -> list[str]:
    """Return lines of a markdown table comparing all models."""
    lines: list[str] = []
    cols = [
        ("Model", "model"),
        ("Passkey (↑)", "passkey_acc"),
        ("Algo Mem (↑)", "algorithmic_memory_acc"),
        ("Prog Trace (↑)", "program_trace_acc"),
        ("Perplexity (↓)", "perplexity"),
        ("Extra Params", "extra_params"),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines.append(header)
    lines.append(sep)

    order = ("T7", "T8", "T14", "T15", "T16", "T19", "T20", "T21")
    for tid in order:
        m = all_metrics.get(tid, {})
        if not m:
            continue
        label = _MODEL_LABELS.get(tid, tid)

        passkey  = _fmt(m.get("passkey_acc"))
        alg_mem  = _fmt(m.get("algorithmic_memory_acc"))
        prog_tr  = _fmt(m.get("program_trace_acc"))
        ppl      = _fmt(m.get("perplexity"), ".2f")
        ep       = _fmt_params(m.get("extra_params"))

        if tid != "T7":
            g1 = _fmt_pct(_pct_gain(m.get("passkey_acc"), t7_ref.get("passkey_acc")))
            g2 = _fmt_pct(_pct_gain(m.get("algorithmic_memory_acc"), t7_ref.get("algorithmic_memory_acc")))
            g3 = _fmt_pct(_pct_gain(m.get("program_trace_acc"), t7_ref.get("program_trace_acc")))
            g5 = _fmt_pct(_pct_gain(m.get("perplexity"), t7_ref.get("perplexity")))
            passkey  = f"{passkey} ({g1})"
            alg_mem  = f"{alg_mem} ({g2})"
            prog_tr  = f"{prog_tr} ({g3})"
            ppl      = f"{ppl} ({g5})"

        row = (
            f"| **{tid}** {label} | {passkey} | {alg_mem} | {prog_tr} | {ppl} | {ep} |"
        )
        lines.append(row)

    return lines


# ---------------------------------------------------------------------------
# Architecture recommendation for Track C
# ---------------------------------------------------------------------------


def build_track_c_recommendation(
    efficiency: dict[str, Any],
    strategy: dict[str, Any],
    gate_criteria: list[dict[str, Any]],
) -> str:
    """Build narrative recommendation for Track C architecture."""
    rec = efficiency.get("recommended_for_track_c", {})
    rec_id = rec.get("task_id")
    rec_label = rec.get("label", "PENDING")
    best_strategy = strategy.get("best_by_quality")
    multi_verdict = strategy.get("multi_reservoir_verdict", "PENDING")

    pass_count = sum(1 for c in gate_criteria if c["pass_fail"] == "PASS")
    total = len(gate_criteria)

    if pass_count == total:
        proceed = "Gate B passed: proceed to Track C."
        status_note = "All Gate B criteria met."
    elif pass_count > 0:
        proceed = f"Gate B partially passed ({pass_count}/{total} criteria): "
        proceed += "conditional proceed to Track C with adjusted expectations."
        status_note = f"{pass_count}/{total} criteria met."
    else:
        proceed = "Gate B not passed: recommend revising Track B before Track C."
        status_note = "No Gate B criteria met (may be PENDING)."

    lines = [
        f"**Gate B status:** {status_note}",
        "",
        f"**{proceed}**",
        "",
        "### Recommended Track C Architecture",
        "",
    ]

    if rec_id and rec_id != "PENDING":
        lines += [
            f"Based on Gate B analysis, **{rec_label}** ({rec_id}) shows the best",
            "quality-efficiency trade-off among Track B configurations.",
            "",
            f"- Best insertion strategy: **{best_strategy or 'PENDING'}**",
            f"- Multi-reservoir verdict: **{multi_verdict}**",
            "",
            "**Track C should build on the best Track B configuration by:**",
            "",
            "1. Using the winning insertion strategy (RIL or DeltaNet replacement) as the base",
            "2. Extending to a multi-reservoir (fast/slow) scheme if multi-reservoir showed advantage",
            "3. Integrating reservoir interaction deeper into the from-scratch architecture (T24)",
            "4. Applying the training curriculum (T25) with Stage 2 procedural objectives",
            "   to directly optimize for program-trace tasks",
            "5. Scaling the reservoir size based on efficiency analysis",
            "   (larger reservoir may be justified for from-scratch training)",
        ]
    else:
        lines += [
            "**PENDING**: Track B GPU evaluations must complete before a specific",
            "Track C architecture recommendation can be made.",
            "",
            "**Provisional guidance (based on architecture design):**",
            "",
            "1. If multi-reservoir (T21) outperforms single-reservoir (T19): use dual-timescale",
            "   reservoirs in Track C (fast scratch-space + slow long-term memory)",
            "2. If RIL insertion (T19) outperforms DeltaNet replacement (T20): use RIL-style",
            "   cross-attention injection in the from-scratch architecture",
            "3. Reserve DeltaNet replacement as a backup if RIL introduces too much overhead",
            "4. From-scratch training (T26) should use the T25 curriculum with Stage 2",
            "   procedural objectives for program-trace optimization",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_report(
    all_metrics: dict[str, dict[str, float | None]],
    gate_criteria: list[dict[str, Any]],
    strategy: dict[str, Any],
    efficiency: dict[str, Any],
    ab_comparison: dict[str, Any],
    available_tasks: list[str],
    missing_tasks: list[str],
    generated_at: str,
) -> str:
    t7_ref = all_metrics.get("T7", {})

    pass_count   = sum(1 for c in gate_criteria if c["pass_fail"] == "PASS")
    fail_count   = sum(1 for c in gate_criteria if c["pass_fail"] == "FAIL")
    pending_count = sum(1 for c in gate_criteria if c["pass_fail"] == "PENDING")
    total = len(gate_criteria)

    if pending_count == total:
        overall = "PENDING — all GPU evaluations outstanding"
        recommendation = (
            "Suspend judgment; all Track B GPU evaluations must complete "
            "before Gate B can be assessed."
        )
    elif fail_count > 0 and pass_count < total:
        overall = f"FAIL — {fail_count}/{total} criteria failed"
        recommendation = (
            "Review failing criteria and revise Track B approach before "
            "proceeding to Track C."
        )
    elif pass_count == total:
        overall = "PASS — all criteria met"
        recommendation = "Proceed to Track C (T24: RW-Transformer architecture definition)."
    else:
        pf_str = f"{pass_count} PASS / {fail_count} FAIL / {pending_count} PENDING"
        overall = f"PARTIAL — {pf_str}"
        recommendation = (
            "Review partial results; proceed to Track C only if all criteria "
            "pass upon completion of pending evaluations."
        )

    table_lines = build_comparison_table(all_metrics, t7_ref)
    table_md = "\n".join(table_lines)

    # Gate criteria table
    gate_rows = []
    for c in gate_criteria:
        pf = _pf_symbol(c["pass_fail"])
        best = c.get("best_model") or "—"
        go   = _fmt_pct(c.get("gain_or_overhead"))
        ref_v = _fmt(c.get("ref_value"))
        best_v = _fmt(c.get("best_value"))
        gate_rows.append(
            f"| {c['id']} | {c['name']} | {c['threshold_str']} | "
            f"{ref_v} | {best} ({best_v}) | {go} | **{pf}** |"
        )
    gate_table = (
        "| # | Criterion | Threshold | T7 (ref) | Best model (value) | Δ vs T7/T8 | Result |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(gate_rows)
    )

    # Strategy comparison
    strategy_rows = []
    for name in ("ril", "deltanet", "multi"):
        sr = strategy["per_strategy"].get(name, {})
        label = {"ril": "T19 RIL", "deltanet": "T20 DeltaNet", "multi": "T21 Multi-res"}[name]
        q_gain = _fmt_pct(sr.get("quality_gain_vs_t7"))
        pk_gain = _fmt_pct(sr.get("passkey_gain_vs_t7"))
        ppl_delta = _fmt_pct(sr.get("ppl_delta_vs_t7"))
        ep = _fmt_params(sr.get("extra_params"))
        strategy_rows.append(
            f"| **{label}** | {q_gain} | {pk_gain} | {ppl_delta} | {ep} |"
        )

    best_strat = strategy.get("best_by_quality") or "PENDING"
    multi_verdict = strategy.get("multi_reservoir_verdict", "PENDING")
    single_multi_delta = _fmt_pct(strategy.get("single_vs_multi_quality_delta"))

    strategy_md = "\n".join([
        "| Config | Quality gain vs T7 | Passkey gain vs T7 | Perplexity delta vs T7 | Extra params |",
        "| --- | --- | --- | --- | --- |",
        *strategy_rows,
        "",
        f"**Best strategy by quality gain:** {best_strat}",
        "",
        f"**Single vs multi-reservoir:** {multi_verdict}",
        f"  - Quality gain delta (multi - single RIL): {single_multi_delta}",
    ])

    # Efficiency table
    eff_rows = []
    for tid in ("T8", "T14", "T15", "T19", "T20", "T21"):
        e = efficiency["per_model"].get(tid, {})
        if not e:
            continue
        label = e.get("label", tid)
        q = _fmt(e.get("quality"), ".4f")
        q_gain = _fmt_pct(e.get("quality_gain_vs_t7"))
        ep = _fmt_params(e.get("extra_params"))
        ep_val = e.get("extra_params")
        qpb_val = e.get("quality_per_byte")
        if qpb_val is not None:
            qpb = _fmt(qpb_val, ".2e")
        elif ep_val is not None and ep_val <= 1000 and e.get("quality") is not None:
            qpb = "∞ (near-zero params)"
        else:
            qpb = "N/A"
        gpb = _fmt(e.get("gain_per_byte"), ".2e") if e.get("gain_per_byte") is not None else "N/A"
        eff_rows.append(f"| **{tid}** {label} | {q} | {q_gain} | {ep} | {qpb} | {gpb} |")

    rec = efficiency.get("recommended_for_track_c", {})
    rec_label = rec.get("label", "PENDING")
    rec_rationale = rec.get("rationale", "No results available.")

    efficiency_md = "\n".join([
        "| Model | Quality | Quality gain vs T7 | Extra params | Quality/byte | Gain/byte |",
        "| --- | --- | --- | --- | --- | --- |",
        *eff_rows,
        "",
        f"**Recommended for Track C:** {rec_label}",
        "",
        f"**Rationale:** {rec_rationale}",
    ])

    # Track A vs B comparison
    ta_label = _MODEL_LABELS.get(ab_comparison.get("track_a_best_id") or "", "")
    tb_label = _MODEL_LABELS.get(ab_comparison.get("track_b_best_id") or "", "PENDING")
    ab_md = "\n".join([
        "| Track | Best model | Quality | Quality gain vs T7 |",
        "| --- | --- | --- | --- |",
        f"| Track A | **{ab_comparison.get('track_a_best_id', '—')}** {ta_label} | "
        f"{_fmt(ab_comparison.get('track_a_quality'))} | "
        f"{_fmt_pct(ab_comparison.get('track_a_gain_vs_t7'))} |",
        f"| Track B | **{ab_comparison.get('track_b_best_id', '—') or '—'}** {tb_label} | "
        f"{_fmt(ab_comparison.get('track_b_quality'))} | "
        f"{_fmt_pct(ab_comparison.get('track_b_gain_vs_t7'))} |",
        "",
        f"**Track B gain over Track A:** {_fmt_pct(ab_comparison.get('track_b_gain_over_track_a'))}",
        "",
        f"**Verdict:** {ab_comparison.get('verdict', 'PENDING')}",
    ])

    # Track C recommendation
    track_c_md = build_track_c_recommendation(efficiency, strategy, gate_criteria)

    available_str = ", ".join(available_tasks) if available_tasks else "None"
    missing_str   = ", ".join(missing_tasks)   if missing_tasks   else "None"

    sections = [
        "# Gate B Evaluation Report\n",
        f"**Generated:** {generated_at}  ",
        f"**Project:** Latent Reservoir Scratchpads for LLMs (LRS)  ",
        f"**Purpose:** Pass/fail decision point before Track C (From-Scratch Architecture)",
        "",
        "---",
        "",
        "## Overall Decision",
        "",
        f"> **{overall}**",
        "",
        f"**Recommendation:** {recommendation}",
        "",
        "---",
        "",
        "## Data Availability",
        "",
        "| Status | Tasks |",
        "| --- | --- |",
        f"| Available | {available_str} |",
        f"| Missing / pending | {missing_str} |",
        "",
        "All missing tasks require GPU training/evaluation to be run first.",
        "See the relevant scripts in `scripts/` for each task.",
        "",
        "---",
        "",
        "## Gate B Criteria",
        "",
        gate_table,
        "",
        "### Criterion definitions",
        "",
        "1. **Long program-trace exact-match gain** — averaged exact-match accuracy on",
        "   ProgramTrace benchmarks (synthetic program execution traces).  Falls back to",
        "   VariableTracking + AssociativeRecall average if ProgramTrace not available.",
        "   Must show ≥ 20% improvement vs T7 (stricter than Gate A's 15% on algorithmic memory).",
        "2. **Memory-quality-per-byte vs T8** — ratio of memory-task accuracy to additional",
        "   parameter bytes.  YaRN/RoPE (T8) adds near-zero extra parameters; if T8 quality",
        "   is available, Track B must achieve ≥ T8 absolute quality to pass this criterion",
        "   (since T8's quality/byte ratio is effectively infinite).  When T8 result is",
        "   unavailable, Track B quality/byte ratio is compared against T8's estimated ratio.",
        "3. **Perplexity degradation** — best Track B perplexity must be < 2% worse than T7.",
        "   Ensures reservoir insertion does not degrade language modelling.",
        "",
        "---",
        "",
        "## Track A vs Track B Comparison",
        "",
        ab_md,
        "",
        "---",
        "",
        "## Track B Strategy Comparison",
        "",
        strategy_md,
        "",
        "---",
        "",
        "## Full Comparison Table",
        "",
        "Delta values in parentheses are percentage change relative to T7.",
        "Positive delta = improvement for accuracy metrics.",
        "",
        table_md,
        "",
        "---",
        "",
        "## Efficiency Analysis",
        "",
        "Quality-per-byte is computed as accuracy / (extra_parameters × 2 bytes).",
        "A higher value is better.  T8 (YaRN) has near-zero extra params → ratio ≈ ∞.",
        "",
        efficiency_md,
        "",
        "---",
        "",
        "## Architecture Recommendation for Track C",
        "",
        track_c_md,
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "All numbers in this report are derived from JSON artifacts in `results/`.",
        "Re-run the analysis at any time:",
        "",
        "```bash",
        "python scripts/gate_b_analysis.py",
        "```",
        "",
        "To regenerate after running all evaluations:",
        "",
        "| Task | Script |",
        "| --- | --- |",
        "| T7 (Qwen vanilla)   | `python scripts/eval_qwen_vanilla.py` |",
        "| T8 (YaRN)           | `python scripts/eval_qwen35_yarn.py` |",
        "| T14 (read-only)     | `python scripts/train_track_a_readonly.py` |",
        "| T15 (read/write)    | *(see T15 training script)* |",
        "| T16 (HP sweep)      | `python scripts/sweep_reservoir_hp.py` |",
        "| T19 (Track B RIL)   | `python scripts/train_track_b_ril.py` |",
        "| T20 (Track B DeltaNet) | `python scripts/train_track_b_deltanet.py` |",
        "| T21 (Track B Multi) | `python scripts/train_track_b_multi.py` |",
        "",
        "After each evaluation completes, rerun this script to update the report.",
        "",
        "---",
        "",
        "*Report auto-generated by `scripts/gate_b_analysis.py`.*",
    ]
    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate B evaluation: compile Track B results and assess pass/fail.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Root results directory.",
    )
    p.add_argument(
        "--report_out",
        type=Path,
        default=_REPO_ROOT / "docs" / "gate_b_report.md",
        help="Output path for the markdown report.",
    )
    p.add_argument(
        "--no_report",
        action="store_true",
        help="Skip writing the markdown report; print summary to stdout only.",
    )
    p.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional: also write full results as JSON to this path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths relative to results_dir if provided
    paths = dict(_RESULT_PATHS)
    if args.results_dir != (_REPO_ROOT / "results"):
        rd = args.results_dir
        paths = {
            "T7":  rd / "baselines" / "qwen35_vanilla.json",
            "T8":  rd / "baselines" / "qwen35_yarn.json",
            "T14": rd / "track_a" / "readonly.json",
            "T15": rd / "track_a" / "readwrite.json",
            "T16": rd / "track_a" / "sweep" / "pareto_frontier.json",
            "T19": rd / "track_b" / "ril.json",
            "T20": rd / "track_b" / "deltanet.json",
            "T21": rd / "track_b" / "multi.json",
        }

    # Load all result files
    raw: dict[str, dict[str, Any] | None] = {}
    available_tasks: list[str] = []
    missing_tasks:   list[str] = []

    for tid, path in paths.items():
        data = load_json(path)
        raw[tid] = data
        if data is not None:
            status = data.get("status", "")
            if status == "pending_gpu_training":
                missing_tasks.append(f"{tid} (placeholder)")
                raw[tid] = None
            else:
                available_tasks.append(tid)
        else:
            missing_tasks.append(tid)

    # Extract metrics
    all_metrics: dict[str, dict[str, float | None]] = {}
    for tid in ("T7", "T8", "T14", "T15", "T16", "T19", "T20", "T21"):
        data = raw.get(tid)
        if data is not None:
            all_metrics[tid] = extract_metrics(data, tid)
        else:
            all_metrics[tid] = {k: None for k in _metric_keys()}

    t7_metrics = all_metrics["T7"]
    t8_metrics = all_metrics["T8"]

    # Gate B criteria evaluation
    gate_criteria = evaluate_gate_criteria(all_metrics, reference="T7", yarn_ref="T8")

    # Strategy comparison
    strategy = strategy_comparison(all_metrics, t7_metrics)

    # Efficiency analysis
    efficiency = efficiency_analysis(all_metrics, t7_metrics, t8_metrics)

    # Track A vs B comparison
    ab_comparison = track_a_vs_b(all_metrics, t7_metrics)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  GATE B EVALUATION — Latent Reservoir Scratchpads for LLMs")
    print("=" * 72)

    print("\n--- Data availability ---")
    print(f"  Available : {', '.join(available_tasks) if available_tasks else 'None'}")
    print(f"  Missing   : {', '.join(missing_tasks) if missing_tasks else 'None'}")

    print("\n--- Gate B Criteria ---")
    print(f"{'#':<3} {'Criterion':<50} {'Δ vs T7/T8':>12}  {'Result':>12}")
    print("-" * 84)
    for c in gate_criteria:
        go_str = _fmt_pct(c.get("gain_or_overhead"))
        pf_str = _pf_symbol(c["pass_fail"])
        name_short = c["name"][:48]
        print(f"  {c['id']}  {name_short:<50} {go_str:>12}  {pf_str:>14}")

    pass_count   = sum(1 for c in gate_criteria if c["pass_fail"] == "PASS")
    fail_count   = sum(1 for c in gate_criteria if c["pass_fail"] == "FAIL")
    pending_count = sum(1 for c in gate_criteria if c["pass_fail"] == "PENDING")

    print(f"\n  Summary: {pass_count} PASS / {fail_count} FAIL / {pending_count} PENDING out of {len(gate_criteria)}")

    if pending_count == len(gate_criteria):
        print("\n  ⧖ ALL PENDING — run GPU evaluations first.")
    elif fail_count > 0:
        print("\n  ✗ GATE B FAIL — review failing criteria.")
    elif pass_count == len(gate_criteria):
        print("\n  ✓ GATE B PASS — proceed to Track C.")
    else:
        print("\n  ⧖ PARTIAL — complete pending evaluations.")

    print("\n--- Track B Strategy Comparison ---")
    for name in ("ril", "deltanet", "multi"):
        sr = strategy["per_strategy"].get(name, {})
        label = {"ril": "T19 RIL    ", "deltanet": "T20 DeltaNet", "multi": "T21 Multi  "}[name]
        q_gain = _fmt_pct(sr.get("quality_gain_vs_t7"))
        ppl = _fmt_pct(sr.get("ppl_delta_vs_t7"))
        ep = _fmt_params(sr.get("extra_params"))
        print(f"  {label}: quality_gain={q_gain}, ppl_delta={ppl}, extra_params={ep}")
    print(f"  Multi-reservoir verdict: {strategy.get('multi_reservoir_verdict', 'PENDING')}")

    print("\n--- Track A vs Track B ---")
    print(f"  Track A best ({ab_comparison.get('track_a_best_id', '—')}): "
          f"quality={_fmt(ab_comparison.get('track_a_quality'))}, "
          f"gain={_fmt_pct(ab_comparison.get('track_a_gain_vs_t7'))}")
    print(f"  Track B best ({ab_comparison.get('track_b_best_id', '—')}): "
          f"quality={_fmt(ab_comparison.get('track_b_quality'))}, "
          f"gain={_fmt_pct(ab_comparison.get('track_b_gain_vs_t7'))}")
    print(f"  Verdict: {ab_comparison.get('verdict', 'PENDING')}")

    rec = efficiency.get("recommended_for_track_c", {})
    print(f"\n--- Track C Recommendation ---")
    print(f"  Recommended config: {rec.get('label', 'PENDING')}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    if not args.no_report:
        generated_at = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        report = build_report(
            all_metrics=all_metrics,
            gate_criteria=gate_criteria,
            strategy=strategy,
            efficiency=efficiency,
            ab_comparison=ab_comparison,
            available_tasks=available_tasks,
            missing_tasks=missing_tasks,
            generated_at=generated_at,
        )

        out_path: Path = args.report_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\n  Report written to: {out_path}")

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.json_out:
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "available_tasks": available_tasks,
            "missing_tasks": missing_tasks,
            "metrics": all_metrics,
            "gate_criteria": gate_criteria,
            "strategy_comparison": strategy,
            "efficiency_analysis": efficiency,
            "track_a_vs_b": ab_comparison,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  JSON output written to: {args.json_out}")

    print()


if __name__ == "__main__":
    main()
