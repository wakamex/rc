"""Evaluate LLaMA-3.2-1B (pure softmax attention) on the full benchmark suite.

Architecture control for the DeltaNet synergy hypothesis — needed to compare
against T17 (LLaMA + reservoir) and T7 (Qwen3.5 vanilla baseline).

Output: results/baselines/llama32_1b.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.benchmarks.computation import (  # noqa: E402
    DyckLanguage,
    ModularArithmetic,
    MultiDigitArithmetic,
    ProgramTrace,
)
from src.eval.benchmarks.emergent import (  # noqa: E402
    AlgorithmicTransfer,
    CompositionalGeneralization,
    LengthExtrapolation,
)
from src.eval.benchmarks.memory import (  # noqa: E402
    AssociativeRecall,
    PasskeyRetrieval,
    VariableTracking,
)
from src.eval.harness import EvalConfig, evaluate  # noqa: E402
from src.models.loader import load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = str(_REPO_ROOT / "results" / "baselines" / "llama32_1b.json")

# ---------------------------------------------------------------------------
# String-level model adapter
# ---------------------------------------------------------------------------
# The eval harness calls model.generate(prompt: str) and expects a str back.
# ModelWrapperImpl.generate expects a tensor.  This adapter bridges the gap.


class LlamaEvalAdapter:
    """Wraps ModelWrapperImpl to accept / return plain strings for the harness."""

    def __init__(self, wrapper: Any, max_new_tokens: int = 64) -> None:
        self._wrapper = wrapper
        self._tok = wrapper.tokenizer
        self.max_new_tokens = max_new_tokens

    # -- ModelWrapper protocol ------------------------------------------------

    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        return self._wrapper.forward(input_ids, **kwargs)

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        """Accept a string prompt, return a string continuation."""
        # Remove harness-internal kwargs that HuggingFace generate() doesn't support
        kwargs.pop("seed", None)

        input_ids = self._tok.encode(str(prompt), padding=False, truncation=True, max_length=1024)
        input_ids = input_ids.to(self._wrapper.device)

        with torch.no_grad():
            output_ids = self._wrapper.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tok.eos_token_id,
                **kwargs,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0, input_ids.shape[-1]:]
        return self._tok.decode(new_ids)

    def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
        return self._wrapper.get_hidden(input_ids, layer=layer, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark suite (mirrors T5 generators, same as T7 baseline)
# ---------------------------------------------------------------------------


def build_benchmarks(n: int = 200) -> list:
    """Return the full benchmark suite used across all baseline evaluations.

    Uses n=200 examples per benchmark for a balance of coverage and runtime
    on the ~3-hour GPU budget specified for T10.

    Args:
        n: Examples per benchmark task.

    Returns:
        List of Generator objects compatible with the eval harness.
    """
    return [
        # Memory benchmarks
        PasskeyRetrieval(n=n, context_length=200, seed=42),
        PasskeyRetrieval(n=n, context_length=500, seed=43),
        VariableTracking(n=n, num_variables=3, num_operations=5, seed=42),
        VariableTracking(n=n, num_variables=5, num_operations=10, seed=43),
        AssociativeRecall(n=n, num_pairs=5, delay_length=30, seed=42),
        AssociativeRecall(n=n, num_pairs=10, delay_length=50, seed=43),
        # Computation benchmarks
        MultiDigitArithmetic(n=n, digit_count=3, operation="addition", seed=42),
        MultiDigitArithmetic(n=n, digit_count=4, operation="addition", seed=43),
        MultiDigitArithmetic(n=n, digit_count=3, operation="multiplication", seed=44),
        ModularArithmetic(n=n, modulus=97, seed=42),
        DyckLanguage(n=n, max_depth=3, bracket_types=1, seed=42),
        DyckLanguage(n=n, max_depth=4, bracket_types=2, seed=43),
        ProgramTrace(n=n, num_steps=4, num_vars=3, seed=42),
        ProgramTrace(n=n, num_steps=6, num_vars=3, seed=43),
        # Emergent benchmarks
        CompositionalGeneralization(n=n, split="train", seed=42),
        CompositionalGeneralization(n=n, split="test", seed=43),
        LengthExtrapolation(n=n, train_length=5, test_multiplier=1.0, seed=42),
        LengthExtrapolation(n=n, train_length=5, test_multiplier=2.0, seed=43),
        LengthExtrapolation(n=n, train_length=5, test_multiplier=4.0, seed=44),
        AlgorithmicTransfer(n=n, family="sorting", split="train", seed=42),
        AlgorithmicTransfer(n=n, family="sorting", split="test", seed=43),
        AlgorithmicTransfer(n=n, family="search", split="train", seed=42),
        AlgorithmicTransfer(n=n, family="search", split="test", seed=43),
    ]


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
    model = LlamaEvalAdapter(wrapper, max_new_tokens=args.max_new_tokens)

    benchmarks = build_benchmarks(n=args.n_examples)

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
