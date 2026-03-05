#!/usr/bin/env python3
"""Evaluate Mamba-2 1.3B on the full benchmark suite.

T9 (Control 2) — tests whether SSM architectures natively solve the memory gap
without reservoir augmentation.

Usage:
    python scripts/eval_mamba2.py
    python scripts/eval_mamba2.py --device cuda --dtype bfloat16 --n 200
    python scripts/eval_mamba2.py --compare results/baselines/qwen35_0.8b.json

Output: results/baselines/mamba2_1.3b.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.eval.benchmarks.suite import build_benchmark_suite
from src.eval.harness import EvalConfig, evaluate
from src.models.eval_adapter import TextEvalAdapter
from src.models.loader import load_model

# TextGenerationWrapper moved to src.models.eval_adapter.TextEvalAdapter


# ---------------------------------------------------------------------------
# VRAM / throughput helpers
# ---------------------------------------------------------------------------


def _vram_mb() -> float:
    """Return current GPU memory allocated in MB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**2


def _max_vram_mb() -> float:
    """Return peak GPU memory allocated in MB since last reset."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _measure_throughput(
    wrapper: TextEvalAdapter,
    prompt: str = "Hello, world!",
    n_runs: int = 10,
) -> dict[str, float]:
    """Measure tokens-per-second throughput over n_runs warmup + timed runs."""
    # Warmup
    wrapper.generate(prompt, do_sample=False)

    start = time.perf_counter()
    total_tokens = 0
    for _ in range(n_runs):
        _ = wrapper.generate(prompt, do_sample=False)
        total_tokens += wrapper.max_new_tokens
    elapsed = time.perf_counter() - start

    return {
        "tokens_per_second": total_tokens / elapsed,
        "ms_per_token": elapsed * 1000 / total_tokens,
        "n_runs": n_runs,
    }


# build_benchmarks moved to src.eval.benchmarks.suite.build_benchmark_suite


def _benchmark_name(b: Any) -> str:
    """Derive a unique benchmark name from its class + key parameters."""
    cls = type(b).__name__
    extras: list[str] = []
    if hasattr(b, "context_length"):
        extras.append(f"ctx{b.context_length}")
    if hasattr(b, "num_variables"):
        extras.append(f"vars{b.num_variables}")
    if hasattr(b, "num_pairs"):
        extras.append(f"pairs{b.num_pairs}")
    if hasattr(b, "operation"):
        extras.append(b.operation)
    if hasattr(b, "bracket_types"):
        extras.append(f"bt{b.bracket_types}")
    if hasattr(b, "split"):
        extras.append(b.split)
    if hasattr(b, "test_multiplier"):
        extras.append(f"x{b.test_multiplier:.1f}")
    if hasattr(b, "family"):
        extras.append(b.family)
    suffix = "_".join(extras) if extras else ""
    return f"{cls}_{suffix}" if suffix else cls


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_comparison_table(
    mamba_results: list[dict[str, Any]],
    t7_path: str | None,
) -> None:
    """Print a comparison table of Mamba-2 vs T7 (Qwen3.5 vanilla)."""
    t7_map: dict[str, float] = {}
    if t7_path and Path(t7_path).exists():
        with open(t7_path) as f:
            t7_data = json.load(f)
        for r in t7_data.get("results", []):
            key = f"{r['task']}::{r['metric']}"
            t7_map[key] = r["value"]

    print("\n" + "=" * 80)
    print(f"{'Task':<45} {'Metric':<14} {'Mamba-2':>8} {'Qwen3.5':>8} {'Delta':>8}")
    print("-" * 80)
    for r in sorted(mamba_results, key=lambda x: (x["task"], x["metric"])):
        task, metric, val = r["task"], r["metric"], r["value"]
        key = f"{task}::{metric}"
        t7_val = t7_map.get(key)
        delta_str = f"{val - t7_val:+.3f}" if t7_val is not None else "  n/a"
        t7_str = f"{t7_val:.3f}" if t7_val is not None else "  n/a"
        print(f"{task:<45} {metric:<14} {val:8.3f} {t7_str:>8} {delta_str:>8}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Mamba-2 1.3B — T9 baseline")
    p.add_argument("--device", default="cuda", help="Torch device (cuda/cpu)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Model weight dtype",
    )
    p.add_argument("--n", type=int, default=500, help="Examples per benchmark task")
    p.add_argument(
        "--batch-size", type=int, default=1, help="Eval batch size (Mamba-2 is sequential)"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=32, help="Max new tokens per generation"
    )
    p.add_argument(
        "--output",
        default="results/baselines/mamba2_1.3b.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--compare",
        default=None,
        help="Path to T7 (Qwen3.5) results JSON for comparison table",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Override HuggingFace model ID (default: state-spaces/mamba2-1.3b-hf)",
    )
    p.add_argument("--throughput-runs", type=int, default=10, help="Runs for throughput benchmark")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading Mamba-2 1.3B on {args.device} ({args.dtype})…")
    _reset_vram_stats()
    model_wrapper = load_model(
        "mamba2-1.3b",
        dtype=dtype,
        device=args.device,
        trust_remote_code=True,
        model_id=args.model_id,
    )
    vram_after_load_mb = _vram_mb()
    print(f"  VRAM after load: {vram_after_load_mb:.0f} MB")

    wrapper = TextEvalAdapter(model_wrapper, max_new_tokens=args.max_new_tokens)

    # -----------------------------------------------------------------------
    # Throughput benchmark
    # -----------------------------------------------------------------------
    print("Measuring throughput…")
    throughput = _measure_throughput(wrapper, n_runs=args.throughput_runs)
    print(
        f"  Throughput: {throughput['tokens_per_second']:.1f} tok/s  "
        f"({throughput['ms_per_token']:.1f} ms/tok)"
    )

    # -----------------------------------------------------------------------
    # Build benchmark suite and assign names
    # -----------------------------------------------------------------------
    benchmarks = build_benchmark_suite(n=args.n)
    # Inject `name` attribute so harness uses our descriptive names
    for b in benchmarks:
        b.name = _benchmark_name(b)  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    _reset_vram_stats()
    config = EvalConfig(
        batch_size=args.batch_size,
        num_few_shot=0,
        decode_mode="greedy",
        metrics=["exact_match", "accuracy"],
        output_file=args.output,
        model_name="mamba2-1.3b",
        seed=42,
    )

    print(f"Running {len(benchmarks)} benchmarks ({args.n} examples each)…")
    eval_start = time.perf_counter()
    results = evaluate(wrapper, benchmarks, config)
    eval_elapsed = time.perf_counter() - eval_start
    vram_peak_mb = _max_vram_mb()

    print(f"  Evaluation complete in {eval_elapsed / 60:.1f} min")
    print(f"  Peak VRAM during eval: {vram_peak_mb:.0f} MB")

    # -----------------------------------------------------------------------
    # Augment the output JSON with hardware/throughput metadata
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    if output_path.exists():
        with output_path.open() as f:
            saved = json.load(f)
    else:
        saved = {"results": [r.__dict__ for r in results]}  # fallback

    saved["hardware"] = {
        "device": args.device,
        "dtype": args.dtype,
        "vram_after_load_mb": vram_after_load_mb,
        "vram_peak_eval_mb": vram_peak_mb,
    }
    saved["throughput"] = throughput
    saved["eval_elapsed_seconds"] = eval_elapsed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(saved, f, indent=2)
    print(f"Results written to {output_path}")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    result_dicts = [
        {"task": r.task, "metric": r.metric, "value": r.value} for r in results
    ]
    _print_comparison_table(result_dicts, args.compare)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if results:
        em_results = [r for r in results if r.metric == "exact_match"]
        if em_results:
            avg_em = sum(r.value for r in em_results) / len(em_results)
            print(f"\nAverage exact-match across all tasks: {avg_em:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
