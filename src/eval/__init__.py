"""Evaluation sub-package.

Public API
----------
evaluate        : Run benchmark suite against any ModelWrapper; write JSON results.
EvalConfig      : Configuration dataclass for evaluation runs.
exact_match     : Exact-match accuracy metric (normalised string comparison).
token_f1        : Token-level F1 metric.
compute_perplexity : Perplexity from per-token log-probabilities.

Benchmark generators live in src.eval.benchmarks:
  PasskeyRetrievalGenerator, VariableTrackingGenerator,
  AssociativeRecallGenerator, CompositionalGenGenerator.
"""

from src.eval.harness import EvalConfig, compute_perplexity, evaluate, exact_match, token_f1

__all__ = [
    "evaluate",
    "EvalConfig",
    "exact_match",
    "token_f1",
    "compute_perplexity",
]
