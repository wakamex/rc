"""Canonical benchmark suite used across all evaluation scripts.

The standard suite contains 23 benchmarks with consistent parameters and seeds.
All eval scripts should import from here instead of defining their own.
"""

from __future__ import annotations

from typing import Any

from src.eval.benchmarks.computation import (
    DyckLanguage,
    ModularArithmetic,
    MultiDigitArithmetic,
    ProgramTrace,
)
from src.eval.benchmarks.emergent import (
    AlgorithmicTransfer,
    CompositionalGeneralization,
    LengthExtrapolation,
)
from src.eval.benchmarks.memory import (
    AssociativeRecall,
    PasskeyRetrieval,
    VariableTracking,
)


def build_benchmark_suite(n: int = 200) -> list[Any]:
    """Return the standard 23-benchmark suite with consistent parameters.

    Args:
        n: Number of examples per benchmark task.

    Returns:
        List of Generator objects compatible with src.eval.harness.evaluate.
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
