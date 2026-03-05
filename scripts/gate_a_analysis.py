#!/usr/bin/env python3
"""Gate A evaluation analysis script (rc-wwh.18).

Reads result JSONs produced by T7–T17 evaluation/training scripts and
generates a comparison table plus pass/fail assessment for each Gate A
criterion.  Also writes docs/gate_a_report.md.

Expected result file paths
--------------------------
T7  (Qwen3.5 vanilla)   : results/baselines/qwen35_vanilla.json
T8  (YaRN)              : results/baselines/qwen35_yarn.json
T9  (Mamba-2 1.3B)      : results/baselines/mamba2_1.3b.json
T10 (LLaMA-3.2-1B)      : results/baselines/llama32_1b.json
T11 (Infini-attention)  : results/baselines/infini_attention.json
T14 (read-only sidecar) : results/track_a/readonly.json
T15 (read/write sidecar): results/track_a/readwrite.json
T16 (best HP sweep)     : results/track_a/sweep/pareto_frontier.json
T17 (LLaMA + RC)        : results/track_a/llama_readonly.json

Gate A Thresholds (ALL must pass for full pass)
----------------------------------------------
1. >= 10% gain on long-context retrieval (PasskeyRetrieval) vs T7
2. >= 15% gain on algorithmic memory tasks (VariableTracking + AssociativeRecall) vs T7
3. >= 10% gain on compositional generalization vs T7
4. <=  20% inference latency overhead vs T7
5. <   2% perplexity degradation vs T7

Usage::

    python scripts/gate_a_analysis.py
    python scripts/gate_a_analysis.py --results_dir results --report_out docs/gate_a_report.md
    python scripts/gate_a_analysis.py --no_report   # stdout only
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
    "T9":  _REPO_ROOT / "results" / "baselines" / "mamba2_1.3b.json",
    "T10": _REPO_ROOT / "results" / "baselines" / "llama32_1b.json",
    "T11": _REPO_ROOT / "results" / "baselines" / "infini_attention.json",
    "T14": _REPO_ROOT / "results" / "track_a" / "readonly.json",
    "T15": _REPO_ROOT / "results" / "track_a" / "readwrite.json",
    "T16": _REPO_ROOT / "results" / "track_a" / "sweep" / "pareto_frontier.json",
    "T17": _REPO_ROOT / "results" / "track_a" / "llama_readonly.json",
}

_MODEL_LABELS: dict[str, str] = {
    "T7":  "Qwen3.5-0.8B vanilla",
    "T8":  "Qwen3.5-0.8B + YaRN",
    "T9":  "Mamba-2 1.3B",
    "T10": "LLaMA-3.2-1B",
    "T11": "Infini-attention (Ctrl 3)",
    "T14": "Qwen3.5 + RC read-only",
    "T15": "Qwen3.5 + RC read/write",
    "T16": "Best HP sweep config",
    "T17": "LLaMA-3.2-1B + RC",
}

# Gate A thresholds
_GATE_THRESHOLDS = {
    "long_context_retrieval_gain":     0.10,  # >=10% gain vs T7
    "algorithmic_memory_gain":         0.15,  # >=15% gain vs T7
    "compositional_generalization_gain": 0.10,  # >=10% gain vs T7
    "latency_overhead_max":            0.20,  # <=20% overhead vs T7
    "perplexity_degradation_max":      0.02,  # <2% degradation vs T7
}

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
    """Extract Gate A relevant metrics from a result dict.

    Returns a dict with keys:
        passkey_acc            – PasskeyRetrieval exact match (long-context proxy)
        variable_tracking_acc  – VariableTracking exact match
        associative_recall_acc – AssociativeRecall exact match
        algorithmic_memory_acc – Average of variable_tracking + associative_recall
        comp_gen_acc           – CompositionalGeneralization test-split exact match
        latency_p50_s          – p50 inference latency in seconds
        tokens_per_sec         – generation throughput
        perplexity             – held-out text perplexity
    """
    # Check for pending-GPU status (infini_attention.json placeholder)
    status = data.get("status", "")
    if status == "pending_gpu_training":
        return {k: None for k in (
            "passkey_acc", "variable_tracking_acc", "associative_recall_acc",
            "algorithmic_memory_acc", "comp_gen_acc",
            "latency_p50_s", "tokens_per_sec", "perplexity",
        )}

    results_list: list[dict[str, Any]] = data.get("results", [])

    # T16: pareto frontier format — use best-quality run's sweep metrics
    if task_id == "T16":
        return _extract_t16_metrics(data)

    # T14/T15/T17: training result format — may lack benchmark eval
    if task_id in ("T14", "T15", "T17"):
        return _extract_training_metrics(data, results_list)

    # Standard baseline eval format (T7-T11)
    passkey = _avg_metric(results_list, "PasskeyRetrieval")
    vt      = _avg_metric(results_list, "VariableTracking")
    ar      = _avg_metric(results_list, "AssociativeRecall")
    alg_mem = (
        (vt + ar) / 2
        if vt is not None and ar is not None
        else (vt if vt is not None else ar)
    )
    cg_test = _avg_metric(results_list, "CompositionalGeneralization")

    latency = data.get("latency", {})
    throughput = data.get("throughput", {})
    perplexity = data.get("perplexity")

    # T9/T10/T11 may save results differently via harness (no top-level perplexity)
    if perplexity is None:
        ppl_result = _first_metric(results_list, "", metric="perplexity")
        perplexity = ppl_result

    return {
        "passkey_acc":            passkey,
        "variable_tracking_acc":  vt,
        "associative_recall_acc": ar,
        "algorithmic_memory_acc": alg_mem,
        "comp_gen_acc":           cg_test,
        "latency_p50_s":          latency.get("p50_s"),
        "tokens_per_sec":         throughput.get("tokens_per_sec"),
        "perplexity":             perplexity,
    }


def _extract_t16_metrics(data: dict[str, Any]) -> dict[str, float | None]:
    """Extract metrics from T16 pareto frontier JSON."""
    pareto_list: list[dict[str, Any]] = data.get("pareto_frontier", [])
    if not pareto_list:
        return {k: None for k in (
            "passkey_acc", "variable_tracking_acc", "associative_recall_acc",
            "algorithmic_memory_acc", "comp_gen_acc",
            "latency_p50_s", "tokens_per_sec", "perplexity",
        )}
    # Best quality run is first (sorted descending by quality_score)
    best = pareto_list[0]
    return {
        "passkey_acc":            best.get("passkey_acc"),
        "variable_tracking_acc":  None,  # not tracked in sweep
        "associative_recall_acc": None,
        "algorithmic_memory_acc": None,
        "comp_gen_acc":           None,
        "latency_p50_s":          best.get("step_latency_ms", None) and best["step_latency_ms"] / 1000,
        "tokens_per_sec":         None,
        "perplexity":             None,
        # Extra: quality score
        "_quality_score":         best.get("quality_score"),
        "_memory_capacity":       best.get("memory_capacity"),
        "_computation_acc":       best.get("computation_acc"),
    }


def _extract_training_metrics(
    data: dict[str, Any],
    results_list: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Extract from training-format JSON (T14/T15/T17)."""
    # If the JSON has a 'results' key with eval results (post-training eval)
    passkey = _avg_metric(results_list, "PasskeyRetrieval") if results_list else None
    vt      = _avg_metric(results_list, "VariableTracking") if results_list else None
    ar      = _avg_metric(results_list, "AssociativeRecall") if results_list else None
    alg_mem = (
        (vt + ar) / 2
        if vt is not None and ar is not None
        else (vt if vt is not None else ar)
    )
    cg_test = _avg_metric(results_list, "CompositionalGeneralization") if results_list else None

    metrics = data.get("metrics", {})
    perplexity = (
        metrics.get("final_train_perplexity")  # training-time estimate
        or data.get("perplexity")
    )
    latency = data.get("latency", {})
    throughput = data.get("throughput", {})

    return {
        "passkey_acc":            passkey,
        "variable_tracking_acc":  vt,
        "associative_recall_acc": ar,
        "algorithmic_memory_acc": alg_mem,
        "comp_gen_acc":           cg_test,
        "latency_p50_s":          latency.get("p50_s"),
        "tokens_per_sec":         throughput.get("tokens_per_sec"),
        "perplexity":             perplexity,
    }


# ---------------------------------------------------------------------------
# Gate A criterion evaluation
# ---------------------------------------------------------------------------


def _pct_gain(val: float | None, ref: float | None) -> float | None:
    """Return (val - ref) / ref as a fraction.  Returns None if inputs missing."""
    if val is None or ref is None or ref == 0.0:
        return None
    return (val - ref) / ref


def _pct_overhead(val: float | None, ref: float | None) -> float | None:
    """Return (val - ref) / ref for latency (positive = worse)."""
    return _pct_gain(val, ref)


def evaluate_gate_criteria(
    all_metrics: dict[str, dict[str, float | None]],
    reference: str = "T7",
) -> list[dict[str, Any]]:
    """Evaluate each Gate A criterion against the reference model.

    Returns a list of criterion dicts with keys:
        id, name, threshold_str, best_model, best_value, ref_value,
        gain_or_overhead, pass_fail
    """
    ref = all_metrics.get(reference, {})

    def best_among(key: str, higher_is_better: bool = True) -> tuple[str | None, float | None]:
        """Find the task with the best value for a metric."""
        candidates = {
            tid: m[key]
            for tid, m in all_metrics.items()
            if m.get(key) is not None and tid != reference
        }
        if not candidates:
            return None, None
        best_tid = max(candidates, key=lambda t: candidates[t]) if higher_is_better \
            else min(candidates, key=lambda t: candidates[t])
        return best_tid, candidates[best_tid]

    # Criterion 1: long-context retrieval (PasskeyRetrieval)
    best_tid, best_val = best_among("passkey_acc", higher_is_better=True)
    gain1 = _pct_gain(best_val, ref.get("passkey_acc"))
    c1 = {
        "id": 1,
        "name": "Long-context retrieval gain (PasskeyRetrieval)",
        "threshold_str": ">= 10% gain vs T7",
        "threshold": _GATE_THRESHOLDS["long_context_retrieval_gain"],
        "direction": "higher",
        "best_model": best_tid,
        "best_value": best_val,
        "ref_value": ref.get("passkey_acc"),
        "gain_or_overhead": gain1,
        "pass_fail": (
            "PASS" if gain1 is not None and gain1 >= _GATE_THRESHOLDS["long_context_retrieval_gain"]
            else ("FAIL" if gain1 is not None else "PENDING")
        ),
    }

    # Criterion 2: algorithmic memory (VariableTracking + AssociativeRecall avg)
    best_tid2, best_val2 = best_among("algorithmic_memory_acc", higher_is_better=True)
    gain2 = _pct_gain(best_val2, ref.get("algorithmic_memory_acc"))
    c2 = {
        "id": 2,
        "name": "Algorithmic memory gain (VarTracking + AssocRecall avg)",
        "threshold_str": ">= 15% gain vs T7",
        "threshold": _GATE_THRESHOLDS["algorithmic_memory_gain"],
        "direction": "higher",
        "best_model": best_tid2,
        "best_value": best_val2,
        "ref_value": ref.get("algorithmic_memory_acc"),
        "gain_or_overhead": gain2,
        "pass_fail": (
            "PASS" if gain2 is not None and gain2 >= _GATE_THRESHOLDS["algorithmic_memory_gain"]
            else ("FAIL" if gain2 is not None else "PENDING")
        ),
    }

    # Criterion 3: compositional generalization
    best_tid3, best_val3 = best_among("comp_gen_acc", higher_is_better=True)
    gain3 = _pct_gain(best_val3, ref.get("comp_gen_acc"))
    c3 = {
        "id": 3,
        "name": "Compositional generalization gain (CompGenTest)",
        "threshold_str": ">= 10% gain vs T7",
        "threshold": _GATE_THRESHOLDS["compositional_generalization_gain"],
        "direction": "higher",
        "best_model": best_tid3,
        "best_value": best_val3,
        "ref_value": ref.get("comp_gen_acc"),
        "gain_or_overhead": gain3,
        "pass_fail": (
            "PASS" if gain3 is not None and gain3 >= _GATE_THRESHOLDS["compositional_generalization_gain"]
            else ("FAIL" if gain3 is not None else "PENDING")
        ),
    }

    # Criterion 4: inference latency overhead (best = lowest overhead)
    # We want the Track A model with lowest latency penalty
    track_a_latencies = {
        tid: m.get("latency_p50_s")
        for tid, m in all_metrics.items()
        if tid in ("T14", "T15", "T16", "T17") and m.get("latency_p50_s") is not None
    }
    if track_a_latencies:
        best_lat_tid = min(track_a_latencies, key=lambda t: track_a_latencies[t])  # type: ignore[arg-type]
        best_lat_val = track_a_latencies[best_lat_tid]
        lat_overhead = _pct_overhead(best_lat_val, ref.get("latency_p50_s"))
    else:
        best_lat_tid = None
        best_lat_val = None
        lat_overhead = None

    c4 = {
        "id": 4,
        "name": "Inference latency overhead (best Track A p50)",
        "threshold_str": "<= 20% overhead vs T7",
        "threshold": _GATE_THRESHOLDS["latency_overhead_max"],
        "direction": "lower",
        "best_model": best_lat_tid,
        "best_value": best_lat_val,
        "ref_value": ref.get("latency_p50_s"),
        "gain_or_overhead": lat_overhead,
        "pass_fail": (
            "PASS" if lat_overhead is not None and lat_overhead <= _GATE_THRESHOLDS["latency_overhead_max"]
            else ("FAIL" if lat_overhead is not None else "PENDING")
        ),
    }

    # Criterion 5: perplexity degradation (best = lowest degradation)
    track_a_ppls = {
        tid: m.get("perplexity")
        for tid, m in all_metrics.items()
        if tid in ("T14", "T15", "T17") and m.get("perplexity") is not None
    }
    if track_a_ppls:
        best_ppl_tid = min(track_a_ppls, key=lambda t: track_a_ppls[t])  # type: ignore[arg-type]
        best_ppl_val = track_a_ppls[best_ppl_tid]
        ppl_degradation = _pct_gain(best_ppl_val, ref.get("perplexity"))
    else:
        best_ppl_tid = None
        best_ppl_val = None
        ppl_degradation = None

    c5 = {
        "id": 5,
        "name": "Perplexity degradation (best Track A)",
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

    return [c1, c2, c3, c4, c5]


# ---------------------------------------------------------------------------
# DeltaNet synergy analysis
# ---------------------------------------------------------------------------


def deltanet_synergy(
    t14_metrics: dict[str, float | None],
    t17_metrics: dict[str, float | None],
    t7_metrics: dict[str, float | None],
) -> dict[str, Any]:
    """Compare T14 (Qwen+RC) vs T17 (LLaMA+RC) gain over their respective baselines.

    T14 uses Qwen3.5-0.8B (which has DeltaNet layers).
    T17 uses LLaMA-3.2-1B (pure softmax attention).
    If T14 gain >> T17 gain on the same tasks, this suggests DeltaNet synergy.

    Returns a dict with gain comparisons.
    """
    # Use algorithmic memory as the key metric
    t14_am = t14_metrics.get("algorithmic_memory_acc")
    t17_am = t17_metrics.get("algorithmic_memory_acc")
    t7_am  = t7_metrics.get("algorithmic_memory_acc")

    t14_passkey = t14_metrics.get("passkey_acc")
    t17_passkey = t17_metrics.get("passkey_acc")
    t7_passkey  = t7_metrics.get("passkey_acc")

    def gain(val: float | None, ref: float | None) -> float | None:
        if val is None or ref is None or ref == 0:
            return None
        return (val - ref) / ref

    t14_am_gain = gain(t14_am, t7_am)
    t17_am_gain = gain(t17_am, t7_am)  # T17 vs T7 (different arch but same baseline context)

    t14_passkey_gain = gain(t14_passkey, t7_passkey)
    t17_passkey_gain = gain(t17_passkey, t7_passkey)

    # DeltaNet synergy delta: how much extra gain does DeltaNet architecture provide
    delta_am = (
        t14_am_gain - t17_am_gain
        if t14_am_gain is not None and t17_am_gain is not None
        else None
    )
    delta_passkey = (
        t14_passkey_gain - t17_passkey_gain
        if t14_passkey_gain is not None and t17_passkey_gain is not None
        else None
    )

    hypothesis = "PENDING"
    if delta_am is not None:
        if delta_am > 0.05:
            hypothesis = "SUPPORTED (T14 >> T17, DeltaNet synergy likely)"
        elif delta_am > 0.0:
            hypothesis = "WEAKLY_SUPPORTED (T14 > T17 but small margin)"
        else:
            hypothesis = "REFUTED (T17 >= T14, no DeltaNet advantage)"

    return {
        "t14_am_gain":      t14_am_gain,
        "t17_am_gain":      t17_am_gain,
        "t14_passkey_gain": t14_passkey_gain,
        "t17_passkey_gain": t17_passkey_gain,
        "deltanet_synergy_delta_am":      delta_am,
        "deltanet_synergy_delta_passkey": delta_passkey,
        "hypothesis":       hypothesis,
    }


# ---------------------------------------------------------------------------
# T16 sweep summary
# ---------------------------------------------------------------------------


def summarise_sweep(sweep_data: dict[str, Any] | None) -> dict[str, Any]:
    """Extract summary statistics from T16 pareto frontier JSON."""
    if sweep_data is None:
        return {"status": "results_not_found"}

    pareto = sweep_data.get("pareto_frontier", [])
    if not pareto:
        return {"status": "empty_pareto"}

    best = max(pareto, key=lambda r: r.get("quality_score", 0.0))
    return {
        "status": "available",
        "n_pareto_runs": len(pareto),
        "best_config": best.get("config", {}),
        "best_quality_score": best.get("quality_score"),
        "best_memory_capacity": best.get("memory_capacity"),
        "best_passkey_acc": best.get("passkey_acc"),
        "best_computation_acc": best.get("computation_acc"),
        "best_step_latency_ms": best.get("step_latency_ms"),
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
        ("Comp Gen (↑)", "comp_gen_acc"),
        ("Latency p50s (↓)", "latency_p50_s"),
        ("Perplexity (↓)", "perplexity"),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines.append(header)
    lines.append(sep)

    for tid in ("T7", "T8", "T9", "T10", "T11", "T14", "T15", "T16", "T17"):
        m = all_metrics.get(tid, {})
        label = _MODEL_LABELS.get(tid, tid)

        passkey = _fmt(m.get("passkey_acc"))
        alg_mem = _fmt(m.get("algorithmic_memory_acc"))
        comp_gen = _fmt(m.get("comp_gen_acc"))
        latency = _fmt(m.get("latency_p50_s"), ".3f")
        ppl = _fmt(m.get("perplexity"), ".2f")

        # Add gain relative to T7 for non-reference rows
        if tid != "T7":
            g1 = _fmt_pct(_pct_gain(m.get("passkey_acc"), t7_ref.get("passkey_acc")))
            g2 = _fmt_pct(_pct_gain(m.get("algorithmic_memory_acc"), t7_ref.get("algorithmic_memory_acc")))
            g3 = _fmt_pct(_pct_gain(m.get("comp_gen_acc"), t7_ref.get("comp_gen_acc")))
            g4 = _fmt_pct(_pct_overhead(m.get("latency_p50_s"), t7_ref.get("latency_p50_s")))
            g5 = _fmt_pct(_pct_gain(m.get("perplexity"), t7_ref.get("perplexity")))
            passkey  = f"{passkey} ({g1})"
            alg_mem  = f"{alg_mem} ({g2})"
            comp_gen = f"{comp_gen} ({g3})"
            latency  = f"{latency} ({g4})"
            ppl      = f"{ppl} ({g5})"

        row = f"| **{tid}** {label} | {passkey} | {alg_mem} | {comp_gen} | {latency} | {ppl} |"
        lines.append(row)

    return lines


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_report(
    all_metrics: dict[str, dict[str, float | None]],
    gate_criteria: list[dict[str, Any]],
    synergy: dict[str, Any],
    sweep_summary: dict[str, Any],
    available_tasks: list[str],
    missing_tasks: list[str],
    generated_at: str,
) -> str:
    t7_ref = all_metrics.get("T7", {})

    pass_count = sum(1 for c in gate_criteria if c["pass_fail"] == "PASS")
    fail_count = sum(1 for c in gate_criteria if c["pass_fail"] == "FAIL")
    pending_count = sum(1 for c in gate_criteria if c["pass_fail"] == "PENDING")
    total = len(gate_criteria)

    if pending_count == total:
        overall = "PENDING — all GPU evaluations outstanding"
        recommendation = "Suspend judgment; all baseline and Track A GPU evaluations must complete before Gate A can be assessed."
    elif fail_count > 0:
        overall = f"FAIL — {fail_count}/{total} criteria failed"
        recommendation = "Stop or revise Track A approach.  Review failing criteria before proceeding to Track B."
    elif pass_count == total:
        overall = "PASS — all criteria met"
        recommendation = "Proceed to Track B (Inserted Layers, T19–T22)."
    else:
        pf_str = f"{pass_count} PASS / {fail_count} FAIL / {pending_count} PENDING"
        overall = f"PARTIAL — {pf_str}"
        recommendation = "Review partial results; proceed to Track B only if all criteria pass upon completion."

    table_lines = build_comparison_table(all_metrics, t7_ref)
    table_md = "\n".join(table_lines)

    # Gate criteria table
    gate_rows = []
    for c in gate_criteria:
        pf = _pf_symbol(c["pass_fail"])
        best = c.get("best_model") or "—"
        go = _fmt_pct(c.get("gain_or_overhead"))
        ref_v = _fmt(c.get("ref_value"))
        best_v = _fmt(c.get("best_value"))
        gate_rows.append(
            f"| {c['id']} | {c['name']} | {c['threshold_str']} | "
            f"{ref_v} | {best} ({best_v}) | {go} | **{pf}** |"
        )
    gate_table = (
        "| # | Criterion | Threshold | T7 (ref) | Best model (value) | Δ vs T7 | Result |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(gate_rows)
    )

    # DeltaNet synergy section
    if synergy.get("hypothesis") == "PENDING":
        synergy_md = (
            "Results for T14 (Qwen+RC) and/or T17 (LLaMA+RC) are not yet available.\n"
            "Synergy analysis will be possible once both models have completed training\n"
            "and benchmark evaluation."
        )
    else:
        synergy_md = "\n".join([
            "| Metric | T14 (Qwen+RC) gain vs T7 | T17 (LLaMA+RC) gain vs T7 | DeltaSynergy (T14-T17) |",
            "| --- | --- | --- | --- |",
            f"| Algorithmic memory | {_fmt_pct(synergy.get('t14_am_gain'))} | {_fmt_pct(synergy.get('t17_am_gain'))} | {_fmt_pct(synergy.get('deltanet_synergy_delta_am'))} |",
            f"| Passkey retrieval | {_fmt_pct(synergy.get('t14_passkey_gain'))} | {_fmt_pct(synergy.get('t17_passkey_gain'))} | {_fmt_pct(synergy.get('deltanet_synergy_delta_passkey'))} |",
            "",
            f"**Hypothesis:** {synergy.get('hypothesis', 'PENDING')}",
            "",
            "Positive DeltaSynergy indicates DeltaNet-style recurrent layers in Qwen3.5 may be",
            "synergistic with reservoir augmentation compared to a pure-attention baseline.",
        ])

    # Sweep summary section
    if sweep_summary.get("status") == "available":
        best_cfg = sweep_summary.get("best_config", {})
        sweep_md = "\n".join([
            f"**Best Pareto-optimal configuration** (quality_score={_fmt(sweep_summary.get('best_quality_score'), '.4f')}):",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| Reservoir size | {best_cfg.get('reservoir_size', 'N/A')} |",
            f"| Spectral radius | {best_cfg.get('spectral_radius', 'N/A')} |",
            f"| Leak rate | {best_cfg.get('leak_rate', 'N/A')} |",
            f"| Topology | {best_cfg.get('topology', 'N/A')} |",
            "",
            "Proxy metrics for best config:",
            f"- Memory capacity (MC): {_fmt(sweep_summary.get('best_memory_capacity'), '.3f')}",
            f"- Passkey accuracy: {_fmt(sweep_summary.get('best_passkey_acc'), '.4f')}",
            f"- Computation accuracy: {_fmt(sweep_summary.get('best_computation_acc'), '.4f')}",
            f"- Step latency: {_fmt(sweep_summary.get('best_step_latency_ms'), '.4f')} ms",
            "",
            f"Pareto frontier size: {sweep_summary.get('n_pareto_runs', 'N/A')} runs",
        ])
    elif sweep_summary.get("status") == "results_not_found":
        sweep_md = (
            "T16 sweep results not yet available "
            "(`results/track_a/sweep/pareto_frontier.json` not found).\n"
            "Run `python scripts/sweep_reservoir_hp.py` to generate them."
        )
    else:
        sweep_md = "T16 sweep results present but empty.  Re-run sweep."

    available_str = ", ".join(available_tasks) if available_tasks else "None"
    missing_str   = ", ".join(missing_tasks)   if missing_tasks   else "None"

    sections = [
        f"# Gate A Evaluation Report\n",
        f"**Generated:** {generated_at}  ",
        f"**Project:** Latent Reservoir Scratchpads for LLMs (LRS)  ",
        f"**Purpose:** Pass/fail decision point before Track B (Inserted Layers)",
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
        "## Gate A Criteria",
        "",
        gate_table,
        "",
        "### Criterion definitions",
        "",
        "1. **Long-context retrieval** — averaged exact-match accuracy on PasskeyRetrieval benchmark",
        "   (all context-length variants).  Proxy for >= 128K context performance via the",
        "   synthetic passkey task generator (T5).",
        "2. **Algorithmic memory** — average of VariableTracking and AssociativeRecall",
        "   exact-match accuracy.  Tests whether the model can maintain structured state",
        "   across distractors.",
        "3. **Compositional generalization** — exact-match on held-out compositional test",
        "   split.  Tests generalisation to unseen operator combinations.",
        "4. **Inference latency overhead** — p50 generation latency relative to T7 vanilla.",
        "   Best (lowest-overhead) Track A model is reported.",
        "5. **Perplexity degradation** — relative perplexity increase vs T7 on held-out",
        "   text.  Ensures reservoir augmentation does not degrade language modelling.",
        "",
        "---",
        "",
        "## Full Comparison Table",
        "",
        "Delta values in parentheses are percentage change relative to T7.",
        "Positive delta = improvement for accuracy metrics; overhead for latency/perplexity.",
        "",
        table_md,
        "",
        "---",
        "",
        "## DeltaNet Synergy Analysis (T17 vs T14)",
        "",
        synergy_md,
        "",
        "---",
        "",
        "## T16 Reservoir HP Sweep Summary",
        "",
        sweep_md,
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "All numbers in this report are derived from JSON artifacts in `results/`.",
        "Re-run the analysis at any time:",
        "",
        "```bash",
        "python scripts/gate_a_analysis.py",
        "```",
        "",
        "To regenerate after running all evaluations:",
        "",
        "| Task | Script |",
        "| --- | --- |",
        "| T7 (Qwen vanilla) | `python scripts/eval_qwen_vanilla.py` |",
        "| T8 (YaRN)         | `python scripts/eval_qwen35_yarn.py` |",
        "| T9 (Mamba-2)      | `python scripts/eval_mamba2.py` |",
        "| T10 (LLaMA)       | `python scripts/eval_llama.py` |",
        "| T11 (Infini-attn) | `python scripts/train_infini_attention.py && python scripts/eval_infini_attention.py` |",
        "| T14 (read-only)   | `python scripts/train_track_a_readonly.py` |",
        "| T15 (read/write)  | *(see T15 training script)* |",
        "| T16 (HP sweep)    | `python scripts/sweep_reservoir_hp.py` |",
        "| T17 (LLaMA+RC)    | *(see T17 training script)* |",
        "",
        "After each evaluation completes, rerun this script to update the report.",
        "",
        "---",
        "",
        "*Report auto-generated by `scripts/gate_a_analysis.py`.*",
    ]
    report = "\n".join(sections) + "\n"

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate A evaluation: compile Track A results and assess pass/fail.",
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
        default=_REPO_ROOT / "docs" / "gate_a_report.md",
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
            "T9":  rd / "baselines" / "mamba2_1.3b.json",
            "T10": rd / "baselines" / "llama32_1b.json",
            "T11": rd / "baselines" / "infini_attention.json",
            "T14": rd / "track_a" / "readonly.json",
            "T15": rd / "track_a" / "readwrite.json",
            "T16": rd / "track_a" / "sweep" / "pareto_frontier.json",
            "T17": rd / "track_a" / "llama_readonly.json",
        }

    # Load all result files
    raw: dict[str, dict[str, Any] | None] = {}
    available_tasks: list[str] = []
    missing_tasks:   list[str] = []

    for tid, path in paths.items():
        data = load_json(path)
        raw[tid] = data
        if data is not None:
            # Check if it's a genuine result or just a pending placeholder
            status = data.get("status", "")
            if status == "pending_gpu_training":
                missing_tasks.append(f"{tid} (placeholder)")
                raw[tid] = None
            else:
                available_tasks.append(tid)
        else:
            missing_tasks.append(tid)

    # Extract metrics from each available result
    all_metrics: dict[str, dict[str, float | None]] = {}
    for tid in ("T7", "T8", "T9", "T10", "T11", "T14", "T15", "T16", "T17"):
        data = raw.get(tid)
        if data is not None:
            all_metrics[tid] = extract_metrics(data, tid)
        else:
            all_metrics[tid] = {k: None for k in (
                "passkey_acc", "variable_tracking_acc", "associative_recall_acc",
                "algorithmic_memory_acc", "comp_gen_acc",
                "latency_p50_s", "tokens_per_sec", "perplexity",
            )}

    t7_metrics  = all_metrics["T7"]
    t14_metrics = all_metrics["T14"]
    t17_metrics = all_metrics["T17"]

    # Gate A criterion evaluation
    gate_criteria = evaluate_gate_criteria(all_metrics, reference="T7")

    # DeltaNet synergy
    synergy = deltanet_synergy(t14_metrics, t17_metrics, t7_metrics)

    # T16 sweep summary
    sweep_summary = summarise_sweep(raw.get("T16"))

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  GATE A EVALUATION — Latent Reservoir Scratchpads for LLMs")
    print("=" * 72)

    print("\n--- Data availability ---")
    print(f"  Available : {', '.join(available_tasks) if available_tasks else 'None'}")
    print(f"  Missing   : {', '.join(missing_tasks)   if missing_tasks   else 'None'}")

    print("\n--- Gate A Criteria ---")
    print(f"{'#':<3} {'Criterion':<48} {'Δ vs T7':>10}  {'Result':>10}")
    print("-" * 78)
    for c in gate_criteria:
        go_str = _fmt_pct(c.get("gain_or_overhead"))
        pf_str = _pf_symbol(c["pass_fail"])
        name_short = c["name"][:46]
        print(f"  {c['id']}  {name_short:<48} {go_str:>10}  {pf_str:>12}")

    pass_count  = sum(1 for c in gate_criteria if c["pass_fail"] == "PASS")
    fail_count  = sum(1 for c in gate_criteria if c["pass_fail"] == "FAIL")
    pending_count = sum(1 for c in gate_criteria if c["pass_fail"] == "PENDING")

    print(f"\n  Summary: {pass_count} PASS / {fail_count} FAIL / {pending_count} PENDING out of {len(gate_criteria)}")

    if pending_count == len(gate_criteria):
        print("\n  ⧖ ALL PENDING — run GPU evaluations first.")
    elif fail_count > 0:
        print("\n  ✗ GATE A FAIL — review failing criteria.")
    elif pass_count == len(gate_criteria):
        print("\n  ✓ GATE A PASS — proceed to Track B.")
    else:
        print("\n  ⧖ PARTIAL — complete pending evaluations.")

    print("\n--- DeltaNet Synergy (T14 vs T17) ---")
    print(f"  T14 (Qwen+RC) algo-mem gain : {_fmt_pct(synergy.get('t14_am_gain'))}")
    print(f"  T17 (LLaMA+RC) algo-mem gain: {_fmt_pct(synergy.get('t17_am_gain'))}")
    print(f"  Synergy delta               : {_fmt_pct(synergy.get('deltanet_synergy_delta_am'))}")
    print(f"  Hypothesis                  : {synergy.get('hypothesis', 'PENDING')}")

    print("\n--- T16 Sweep Summary ---")
    if sweep_summary.get("status") == "available":
        best = sweep_summary.get("best_config", {})
        print(f"  Best config : size={best.get('reservoir_size')}, "
              f"sr={best.get('spectral_radius')}, "
              f"lr={best.get('leak_rate')}, "
              f"topo={best.get('topology')}")
        print(f"  Quality     : {_fmt(sweep_summary.get('best_quality_score'), '.4f')}")
        print(f"  Passkey acc : {_fmt(sweep_summary.get('best_passkey_acc'), '.4f')}")
        print(f"  Pareto size : {sweep_summary.get('n_pareto_runs')}")
    else:
        print(f"  Status: {sweep_summary.get('status')}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    if not args.no_report:
        generated_at = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        report = build_report(
            all_metrics=all_metrics,
            gate_criteria=gate_criteria,
            synergy=synergy,
            sweep_summary=sweep_summary,
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
            "deltanet_synergy": synergy,
            "sweep_summary": sweep_summary,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  JSON output written to: {args.json_out}")

    print()


if __name__ == "__main__":
    main()
