"""Evaluation harness: run any benchmark suite against any model, output standardized JSON."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.types import BenchmarkExample, EvalResult, Generator, ModelWrapper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    batch_size: int = 8
    num_few_shot: int = 0  # 0 = zero-shot, k > 0 = k-shot
    decode_mode: str = "greedy"  # "greedy" | "sampling"
    temperature: float = 1.0
    metrics: list[str] = field(default_factory=lambda: ["exact_match"])
    output_file: str | None = None
    resume: bool = False  # if True, skip already-completed tasks
    model_name: str = ""
    seed: int | None = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return text.strip().lower()


def exact_match(prediction: str, target: str) -> float:
    """Return 1.0 if normalized prediction equals normalized target, else 0.0."""
    return float(_normalize(prediction) == _normalize(target))


def token_f1(prediction: str, target: str) -> float:
    """Token-level F1 score between prediction and target."""
    pred_tokens = _normalize(prediction).split()
    target_tokens = _normalize(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    target_set = set(target_tokens)
    common = pred_set & target_set
    if not common:
        return 0.0
    precision = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    recall = sum(1 for t in target_tokens if t in common) / len(target_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_perplexity(log_probs: list[float]) -> float:
    """Compute perplexity from per-token log-probabilities (base e)."""
    if not log_probs:
        return float("inf")
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)


def _aggregate_metric(metric: str, predictions: list[str], targets: list[str]) -> float:
    """Compute an aggregate metric over all predictions and targets."""
    if not predictions:
        return 0.0
    if metric in ("exact_match", "accuracy"):
        return sum(exact_match(p, t) for p, t in zip(predictions, targets)) / len(predictions)
    if metric == "f1":
        return sum(token_f1(p, t) for p, t in zip(predictions, targets)) / len(predictions)
    raise ValueError(f"Unknown aggregate metric: {metric!r}")


# ---------------------------------------------------------------------------
# Reproducibility utilities
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Few-shot prompt construction
# ---------------------------------------------------------------------------


def _make_prompt(example: BenchmarkExample, shots: list[BenchmarkExample]) -> str:
    """Build a (optionally few-shot) text prompt for an example."""
    parts: list[str] = []
    for shot in shots:
        parts.append(f"Input: {shot.input}\nOutput: {shot.target}")
    parts.append(f"Input: {example.input}\nOutput:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Resume / persistence
# ---------------------------------------------------------------------------


def _load_existing_results(output_file: str) -> dict[str, list[EvalResult]]:
    """Load task → results mapping from a previous output file."""
    path = Path(output_file)
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        existing: dict[str, list[EvalResult]] = {}
        for entry in data.get("results", []):
            task = entry["task"]
            existing.setdefault(task, []).append(EvalResult(**entry))
        return existing
    except Exception:
        return {}


def _save_results(
    output_file: str,
    results: list[EvalResult],
    model_name: str,
    config: EvalConfig,
    git_hash: str,
    timestamp: float,
) -> None:
    """Write results to a JSON file with full reproducibility metadata."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "git_hash": git_hash,
        "timestamp": timestamp,
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------


def evaluate(
    model: ModelWrapper,
    benchmarks: list[Generator],
    config: EvalConfig,
) -> list[EvalResult]:
    """Evaluate a model on a list of benchmark generators.

    Args:
        model: Any object implementing the ModelWrapper protocol.
        benchmarks: List of Generator objects yielding BenchmarkExample items.
        config: EvalConfig controlling batch size, metrics, decode mode, etc.

    Returns:
        List of EvalResult with per-task, per-metric aggregate values.
    """
    git_hash = _git_hash()
    timestamp = time.time()

    # Load existing results when resuming
    existing: dict[str, list[EvalResult]] = {}
    if config.resume and config.output_file:
        existing = _load_existing_results(config.output_file)

    all_results: list[EvalResult] = [r for rs in existing.values() for r in rs]

    # Try to import tqdm; fall back to a no-op wrapper
    try:
        from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

        def _progress(iterable: Any, **kwargs: Any) -> Any:
            return _tqdm(iterable, **kwargs)

    except ImportError:

        def _progress(iterable: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            return iterable

    for benchmark in _progress(benchmarks, desc="Benchmarks", unit="benchmark"):
        task_name: str = getattr(benchmark, "name", type(benchmark).__name__)

        # Skip tasks already finished when resuming
        if config.resume and task_name in existing:
            continue

        examples: list[BenchmarkExample] = list(benchmark)
        if not examples:
            continue

        # Reserve the first num_few_shot examples as in-context demonstrations
        shots: list[BenchmarkExample] = []
        if config.num_few_shot > 0:
            shots = examples[: config.num_few_shot]
            examples = examples[config.num_few_shot :]
        if not examples:
            continue

        predictions: list[str] = []
        targets: list[str] = []

        generate_kwargs: dict[str, Any] = {}
        if config.decode_mode == "greedy":
            generate_kwargs["do_sample"] = False
        elif config.decode_mode == "sampling":
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = config.temperature
        if config.seed is not None:
            generate_kwargs["seed"] = config.seed

        # Batch evaluation
        batch_iter = range(0, len(examples), config.batch_size)
        for batch_start in _progress(batch_iter, desc=f"  {task_name}", unit="batch", leave=False):
            batch = examples[batch_start : batch_start + config.batch_size]
            for example in batch:
                prompt = _make_prompt(example, shots)
                raw = model.generate(prompt, **generate_kwargs)
                # Accept str or list output from model.generate
                if isinstance(raw, list):
                    prediction = str(raw[0]) if raw else ""
                else:
                    prediction = str(raw)
                predictions.append(prediction)
                targets.append(example.target)

        # Build per-task config dict for reproducibility
        task_config: dict[str, Any] = {
            "model_name": config.model_name,
            "git_hash": git_hash,
            "timestamp": timestamp,
            "num_few_shot": config.num_few_shot,
            "decode_mode": config.decode_mode,
            "batch_size": config.batch_size,
        }

        for metric in config.metrics:
            if metric == "perplexity":
                # Perplexity requires per-token log-probs from model.forward;
                # the harness exposes compute_perplexity() for callers that have them.
                logging.getLogger(__name__).warning(
                    "Skipping perplexity metric for task %r — "
                    "compute it via compute_perplexity() with model.forward log-probs.",
                    task_name,
                )
                continue
            value = _aggregate_metric(metric, predictions, targets)
            all_results.append(
                EvalResult(task=task_name, metric=metric, value=value, config=task_config)
            )

    if config.output_file:
        _save_results(
            config.output_file, all_results, config.model_name, config, git_hash, timestamp
        )

    return all_results
