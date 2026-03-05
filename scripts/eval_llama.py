"""Evaluate LLaMA-3.2-1B (pure softmax attention) on the full benchmark suite.

Architecture control for the DeltaNet synergy hypothesis — needed to compare
against T17 (LLaMA + reservoir) and T7 (Qwen3.5 vanilla baseline).

Output: results/baselines/llama32_1b.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.benchmarks.suite import build_benchmark_suite  # noqa: E402
from src.eval.harness import EvalConfig, evaluate  # noqa: E402
from src.models.eval_adapter import TextEvalAdapter  # noqa: E402
from src.models.loader import load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = str(_REPO_ROOT / "results" / "baselines" / "llama32_1b.json")

# LlamaEvalAdapter and build_benchmarks moved to shared modules:
# - src.models.eval_adapter.TextEvalAdapter
# - src.eval.benchmarks.suite.build_benchmark_suite


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLaMA-3.2-1B on the full synthetic benchmark suite."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Model weight dtype (default: fp16)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path for results JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=200,
        help="Examples per benchmark task (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Eval batch size — currently sequential (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per example (default: 64)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip benchmark tasks already present in the output file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading LLaMA-3.2-1B on {args.device} ({args.dtype})…")
    wrapper = load_model("llama-3.2-1b", dtype=dtype, device=args.device)
    model = TextEvalAdapter(wrapper, max_new_tokens=args.max_new_tokens)

    benchmarks = build_benchmark_suite(n=args.n_examples)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(
        batch_size=args.batch_size,
        num_few_shot=0,
        decode_mode="greedy",
        metrics=["exact_match", "accuracy", "f1"],
        output_file=args.output,
        model_name="llama-3.2-1b",
        resume=args.resume,
    )

    print(f"Running {len(benchmarks)} benchmark tasks…")
    results = evaluate(model, benchmarks, config)

    print(f"\nDone. {len(results)} metric results saved to {args.output}")

    # Print a summary table
    print("\n=== Results Summary ===")
    print(f"{'Task':<45} {'Metric':<15} {'Score':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r.task:<45} {r.metric:<15} {r.value:>7.4f}")


if __name__ == "__main__":
    main()
