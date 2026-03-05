#!/usr/bin/env python3
"""Evaluate Qwen3.5-0.8B with YaRN RoPE scaling on long-context benchmarks.

T8 (Control 1) — tests whether YaRN-style positional extension solves the
memory gap by extending the native context window to 128K+.

YaRN (Yet another RoPE extensioN) scales the RoPE frequencies to allow the
model to generalise to context lengths beyond its training maximum, using a
non-uniform interpolation scheme that preserves high-frequency components.

Usage:
    python scripts/eval_qwen35_yarn.py
    python scripts/eval_qwen35_yarn.py --device cuda --dtype bfloat16
    python scripts/eval_qwen35_yarn.py --compare results/baselines/qwen35_0.8b.json

Output: results/baselines/qwen35_yarn.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.benchmarks.memory import PasskeyRetrieval, VariableTracking  # noqa: E402
from src.eval.benchmarks.suite import build_benchmark_suite  # noqa: E402
from src.eval.harness import EvalConfig, evaluate  # noqa: E402
from src.models.eval_adapter import TextEvalAdapter  # noqa: E402
from src.models.loader import MODEL_REGISTRY  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Native Qwen3.5-0.8B max position embeddings (from model config)
QWEN35_NATIVE_MAX_POS = 32768  # 32K tokens native context

# YaRN scale factor: extend to 128K (4x) by default
# Users can override with --yarn-scale
DEFAULT_YARN_SCALE = 4.0

DEFAULT_OUTPUT = str(_REPO_ROOT / "results" / "baselines" / "qwen35_yarn.json")

# Approximate words-per-token ratio for context sizing
# Each distractor word is ~1.3 tokens on average
_WORDS_PER_TOKEN = 1.3

# Long-context passkey retrieval target depths (in tokens)
LONG_CTX_TOKEN_DEPTHS = [32_000, 64_000, 128_000]


# ---------------------------------------------------------------------------
# YaRN application
# ---------------------------------------------------------------------------


def apply_yarn_to_config(config: Any, scale_factor: float) -> None:
    """Patch a HuggingFace model config with YaRN rope_scaling in place.

    Reads the native max_position_embeddings from config and sets the
    rope_scaling dict to use the 'yarn' type with the given scale factor.
    Extended max = native_max * scale_factor.

    Args:
        config: HuggingFace PretrainedConfig (modified in place).
        scale_factor: Multiplicative factor (e.g. 4.0 = extend 32K → 128K).
    """
    native_max = getattr(config, "max_position_embeddings", QWEN35_NATIVE_MAX_POS)
    config.rope_scaling = {
        "type": "yarn",
        "factor": float(scale_factor),
        "original_max_position_embeddings": native_max,
    }
    config.max_position_embeddings = int(native_max * scale_factor)


def load_qwen35_with_yarn(
    dtype: torch.dtype,
    device: str | torch.device,
    yarn_scale: float = DEFAULT_YARN_SCALE,
    model_id: str | None = None,
) -> Any:
    """Load Qwen3.5-0.8B-Base with YaRN RoPE scaling applied.

    Applies the rope_scaling config before loading weights so the positional
    embedding tables are initialised with the extended range.

    Args:
        dtype: Weight dtype (fp16 or bf16).
        device: Target device.
        yarn_scale: YaRN scale factor (default 4.0 → 128K context).
        model_id: Override HuggingFace model ID (default: Qwen/Qwen3.5-0.8B-Base).

    Returns:
        Tuple of (raw HuggingFace model, tokenizer).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    hf_id = model_id or MODEL_REGISTRY["qwen3.5-0.8b"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    apply_yarn_to_config(config, yarn_scale)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        config=config,
        torch_dtype=dtype,
        device_map=str(device),
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


# QwenYaRNEvalAdapter moved to src.models.eval_adapter.TextEvalAdapter


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


def _words_for_tokens(target_tokens: int) -> int:
    """Convert target token count to approximate word count for PasskeyRetrieval."""
    return max(100, int(target_tokens / _WORDS_PER_TOKEN))


# build_standard_benchmarks moved to src.eval.benchmarks.suite.build_benchmark_suite


def build_long_context_benchmarks(n: int) -> list[Any]:
    """Return long-context benchmarks targeting 32K, 64K, 128K+ token positions.

    Uses PasskeyRetrieval with large context_length values and VariableTracking
    with many operations/long distractors to stress long-range memory.
    """
    benchmarks = []

    # Passkey retrieval at 32K, 64K, 128K token-equivalent depths
    for target_tokens in LONG_CTX_TOKEN_DEPTHS:
        ctx_words = _words_for_tokens(target_tokens)
        label = f"{target_tokens // 1000}k"
        b = PasskeyRetrieval(n=n, context_length=ctx_words, seed=42)
        b.name = f"PasskeyRetrieval_long_{label}"  # type: ignore[attr-defined]
        benchmarks.append(b)

        # Also test passkey at end of context (hardest position)
        b_end = PasskeyRetrieval(
            n=n,
            context_length=ctx_words,
            passkey_position=0.9,  # passkey near end
            seed=43,
        )
        b_end.name = f"PasskeyRetrieval_long_{label}_end"  # type: ignore[attr-defined]
        benchmarks.append(b_end)

    # Variable tracking at long range (many operations across long distractors)
    for distractor_len, label in [(500, "dist500"), (2000, "dist2k"), (5000, "dist5k")]:
        b = VariableTracking(
            n=n,
            num_variables=5,
            num_operations=20,
            distractor_length=distractor_len,
            seed=42,
        )
        b.name = f"VariableTracking_long_{label}"  # type: ignore[attr-defined]
        benchmarks.append(b)

    return benchmarks


def _benchmark_name(b: Any) -> str:
    """Derive a unique benchmark name from class + key parameters."""
    if hasattr(b, "name"):
        return b.name
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
# VRAM helpers
# ---------------------------------------------------------------------------


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**2


def _max_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_comparison_table(
    yarn_results: list[dict[str, Any]],
    t7_path: str | None,
) -> None:
    """Print a comparison table: YaRN vs T7 (Qwen3.5 vanilla)."""
    t7_map: dict[str, float] = {}
    if t7_path and Path(t7_path).exists():
        with open(t7_path) as f:
            t7_data = json.load(f)
        for r in t7_data.get("results", []):
            key = f"{r['task']}::{r['metric']}"
            t7_map[key] = r["value"]

    print("\n" + "=" * 88)
    print(f"{'Task':<50} {'Metric':<12} {'YaRN':>7} {'T7 Vanilla':>10} {'Delta':>8}")
    print("-" * 88)
    for r in sorted(yarn_results, key=lambda x: (x["task"], x["metric"])):
        task, metric, val = r["task"], r["metric"], r["value"]
        key = f"{task}::{metric}"
        t7_val = t7_map.get(key)
        delta_str = f"{val - t7_val:+.3f}" if t7_val is not None else "   n/a"
        t7_str = f"{t7_val:.3f}" if t7_val is not None else "     n/a"
        print(f"{task:<50} {metric:<12} {val:7.3f} {t7_str:>10} {delta_str:>8}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-0.8B + YaRN RoPE scaling — T8 baseline"
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda/cpu)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Model weight dtype",
    )
    p.add_argument("--n", type=int, default=200, help="Examples per benchmark task")
    p.add_argument(
        "--n-long",
        type=int,
        default=50,
        help="Examples per long-context benchmark task (default: 50, reduced for runtime)",
    )
    p.add_argument(
        "--batch-size", type=int, default=1, help="Eval batch size (default: 1)"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=32, help="Max new tokens per generation"
    )
    p.add_argument(
        "--yarn-scale",
        type=float,
        default=DEFAULT_YARN_SCALE,
        help=f"YaRN scale factor (default: {DEFAULT_YARN_SCALE} → extends 32K to 128K)",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--compare",
        default=None,
        help="Path to T7 (Qwen3.5 vanilla) results JSON for comparison table",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Override HuggingFace model ID (default: Qwen/Qwen3.5-0.8B-Base)",
    )
    p.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip the standard (short-context) benchmark suite",
    )
    p.add_argument(
        "--skip-long",
        action="store_true",
        help="Skip the long-context benchmark suite",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip benchmark tasks already present in the output file",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    extended_max = int(QWEN35_NATIVE_MAX_POS * args.yarn_scale)
    print(
        f"Loading Qwen3.5-0.8B with YaRN scale={args.yarn_scale:.1f} "
        f"(native {QWEN35_NATIVE_MAX_POS // 1024}K → extended {extended_max // 1024}K) "
        f"on {args.device} ({args.dtype})…"
    )

    _reset_vram_stats()
    model, tokenizer = load_qwen35_with_yarn(
        dtype=dtype,
        device=device,
        yarn_scale=args.yarn_scale,
        model_id=args.model_id,
    )
    vram_after_load_mb = _vram_mb()
    print(f"  VRAM after load: {vram_after_load_mb:.0f} MB")
    print(f"  Model max_position_embeddings: {model.config.max_position_embeddings}")
    print(f"  RoPE scaling config: {model.config.rope_scaling}")

    adapter = TextEvalAdapter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_input_length=extended_max,
    )

    # -----------------------------------------------------------------------
    # Build benchmark suites
    # -----------------------------------------------------------------------
    benchmarks: list[Any] = []

    if not args.skip_standard:
        standard = build_benchmark_suite(n=args.n)
        for b in standard:
            b.name = _benchmark_name(b)  # type: ignore[attr-defined]
        benchmarks.extend(standard)
        print(f"Standard benchmarks: {len(standard)} tasks ({args.n} examples each)")

    if not args.skip_long:
        long_ctx = build_long_context_benchmarks(args.n_long)
        benchmarks.extend(long_ctx)
        token_depths = ", ".join(f"{t // 1000}K" for t in LONG_CTX_TOKEN_DEPTHS)
        print(
            f"Long-context benchmarks: {len(long_ctx)} tasks "
            f"({args.n_long} examples each, depths: {token_depths})"
        )

    if not benchmarks:
        print("No benchmarks selected. Use --skip-standard or --skip-long to select subsets.")
        return

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    _reset_vram_stats()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(
        batch_size=args.batch_size,
        num_few_shot=0,
        decode_mode="greedy",
        metrics=["exact_match", "accuracy"],
        output_file=args.output,
        model_name="qwen3.5-0.8b-yarn",
        resume=args.resume,
        seed=42,
    )

    print(f"\nRunning {len(benchmarks)} benchmark tasks…")
    eval_start = time.perf_counter()
    results = evaluate(adapter, benchmarks, config)
    eval_elapsed = time.perf_counter() - eval_start
    vram_peak_mb = _max_vram_mb()

    print(f"  Evaluation complete in {eval_elapsed / 60:.1f} min")
    print(f"  Peak VRAM during eval: {vram_peak_mb:.0f} MB")

    # -----------------------------------------------------------------------
    # Augment output JSON with YaRN / hardware metadata
    # -----------------------------------------------------------------------
    if output_path.exists():
        with output_path.open() as f:
            saved = json.load(f)
    else:
        saved = {"results": []}

    saved["yarn_config"] = {
        "scale_factor": args.yarn_scale,
        "native_max_position_embeddings": QWEN35_NATIVE_MAX_POS,
        "extended_max_position_embeddings": extended_max,
        "rope_type": "yarn",
    }
    saved["hardware"] = {
        "device": args.device,
        "dtype": args.dtype,
        "vram_after_load_mb": vram_after_load_mb,
        "vram_peak_eval_mb": vram_peak_mb,
    }
    saved["eval_elapsed_seconds"] = eval_elapsed
    saved["long_context_token_depths"] = LONG_CTX_TOKEN_DEPTHS

    with output_path.open("w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nResults written to {output_path}")

    # -----------------------------------------------------------------------
    # Comparison table vs T7
    # -----------------------------------------------------------------------
    result_dicts = [
        {"task": r.task, "metric": r.metric, "value": r.value} for r in results
    ]
    _print_comparison_table(result_dicts, args.compare)

    # -----------------------------------------------------------------------
    # Long-context summary
    # -----------------------------------------------------------------------
    long_ctx_results = [
        r for r in results
        if any(depth_label in r.task for depth_label in ["_long_", "32k", "64k", "128k"])
    ]
    if long_ctx_results:
        print("\n=== Long-Context Results ===")
        print(f"{'Task':<55} {'Metric':<12} {'Score':>7}")
        print("-" * 76)
        for r in sorted(long_ctx_results, key=lambda x: (x.task, x.metric)):
            print(f"{r.task:<55} {r.metric:<12} {r.value:7.4f}")

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    if results:
        em_results = [r for r in results if r.metric == "exact_match"]
        if em_results:
            avg_em = sum(r.value for r in em_results) / len(em_results)
            print(f"\nAverage exact-match across all tasks: {avg_em:.3f}")

        long_em = [r for r in long_ctx_results if r.metric == "exact_match"]
        if long_em:
            avg_long_em = sum(r.value for r in long_em) / len(long_em)
            print(f"Average exact-match on long-context tasks: {avg_long_em:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
