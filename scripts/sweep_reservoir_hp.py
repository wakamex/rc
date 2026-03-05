#!/usr/bin/env python3
"""Sweep reservoir hyperparameters for Track A (rc-wwh.16).

Sequential sweep strategy — one dimension at a time, fix best value, move on:
  1. Reservoir size: 500, 2K, 10K, 50K          (4 runs)
  2. Spectral radius: 0.5, 0.9, 0.99, 1.1       (4 runs)
  3. Leak rate: 0.1, 0.3, 0.7, 1.0              (4 runs)
  4. Topology: erdos_renyi vs small_world        (2 runs)
  5. Best combo fine-tune                        (1 run)

Total: 15 runs.

Evaluation: lightweight ridge regression readout on ESN states applied
to memory + computation benchmark subsets (50 examples each) for fast
proxy scoring — no full LLM training required for the sweep itself.

Metrics tracked per run:
  - memory_capacity: sum of R² for k-step delay recall (k=1..20)
  - passkey_acc: exact-match accuracy on PasskeyRetrieval (50 examples)
  - computation_acc: exact-match on MultiDigitArithmetic (50 examples)
  - step_latency_ms: mean time per ESN step (1 000 trials, µs precision)
  - quality_score: 0.4*MC + 0.4*passkey_acc + 0.2*computation_acc

Pareto frontier (reservoir_size vs quality_score vs step_latency_ms) is
saved to results/track_a/sweep/pareto_frontier.json and the best overall
config to results/track_a/sweep/best_config.yaml.

Usage::

    # Full sequential sweep (15 runs)
    python scripts/sweep_reservoir_hp.py

    # Run a single configuration by index (0-based)
    python scripts/sweep_reservoir_hp.py --run_id 3

    # Resume from existing partial results
    python scripts/sweep_reservoir_hp.py --resume

    # Dry-run: print all configs without executing
    python scripts/sweep_reservoir_hp.py --dry_run

    # Disable wandb
    python scripts/sweep_reservoir_hp.py --no_wandb
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/track_a/sweep")
CONFIGS_DIR = Path("configs/sweep")

# ---------------------------------------------------------------------------
# Base configuration (best from T15 / track_a_readonly defaults)
# ---------------------------------------------------------------------------

BASE_CONFIG: dict[str, Any] = {
    "reservoir_size": 10_000,
    "spectral_radius": 0.9,
    "leak_rate": 0.5,
    "topology": "erdos_renyi",
    "input_scaling": 1.0,
    "sparsity": 0.01,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Sweep dimensions  (dim_label, param_key, values)
# ---------------------------------------------------------------------------

SWEEP_DIMENSIONS: list[tuple[str, str, list[Any]]] = [
    ("size",            "reservoir_size",  [500, 2_000, 10_000, 50_000]),
    ("spectral_radius", "spectral_radius", [0.5, 0.9, 0.99, 1.1]),
    ("leak_rate",       "leak_rate",       [0.1, 0.3, 0.7, 1.0]),
    ("topology",        "topology",        ["erdos_renyi", "small_world"]),
]

# ---------------------------------------------------------------------------
# Character encoding (for text → ESN input)
# ---------------------------------------------------------------------------

_PRINTABLE = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
_CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(_PRINTABLE)}
_VOCAB_SIZE: int = len(_PRINTABLE)  # 95 printable ASCII chars


def _encode_text(text: str, max_len: int = 512) -> np.ndarray:
    """Return one-hot character matrix, shape (T, vocab_size)."""
    text = text[:max_len]
    X = np.zeros((len(text), _VOCAB_SIZE), dtype=np.float32)
    for t, ch in enumerate(text):
        idx = _CHAR_TO_IDX.get(ch, 0)
        X[t, idx] = 1.0
    return X


# ---------------------------------------------------------------------------
# Ridge regression readout (pure NumPy, no sklearn dependency)
# ---------------------------------------------------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Fit ridge regression: w = (X^T X + λI)^{-1} X^T y.

    Args:
        X: Feature matrix, shape (N, d).
        y: Target matrix, shape (N, C).
        lam: L2 regularisation coefficient.

    Returns:
        Weight matrix W, shape (d, C).
    """
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return predictions, shape (N, C)."""
    return X @ W


# ---------------------------------------------------------------------------
# Reservoir state extraction helpers
# ---------------------------------------------------------------------------


def _run_reservoir(
    esn: Any,
    sequences: list[np.ndarray],
    use_final: bool = True,
) -> np.ndarray:
    """Run a list of input sequences through the ESN.

    Args:
        esn: ESN instance.
        sequences: List of arrays, each shape (T_i, input_dim).
        use_final: If True, return only the final state per sequence
                   (shape (N, n)).  If False, return mean state
                   (shape (N, n)).

    Returns:
        Feature matrix, shape (N, n).
    """
    features = []
    for seq in sequences:
        esn.reset()
        r = None
        if use_final:
            for t in range(seq.shape[0]):
                r = esn.step(seq[t])
            features.append(r if r is not None else np.zeros(esn.n, dtype=np.float32))
        else:
            states = []
            for t in range(seq.shape[0]):
                r = esn.step(seq[t])
                states.append(r.copy())
            features.append(np.stack(states).mean(axis=0))
    return np.stack(features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Memory capacity (standard RC metric)
# ---------------------------------------------------------------------------


def _memory_capacity(
    esn: Any,
    n_steps: int = 500,
    max_delay: int = 20,
    seed: int = 0,
) -> float:
    """Compute memory capacity: sum of R² for k-step delay recall.

    Generates a random ±1 input sequence, runs through ESN, then for
    each delay k trains a linear readout to recall the k-step-ago input
    and measures R².  MC = Σ_{k=1}^{max_delay} R²(k).

    Args:
        esn: ESN instance (will be reset).
        n_steps: Length of the input sequence.
        max_delay: Maximum delay to test.
        seed: RNG seed.

    Returns:
        Total memory capacity (float in [0, max_delay]).
    """
    rng = np.random.default_rng(seed)
    u = rng.choice([-1.0, 1.0], size=(n_steps,)).astype(np.float32)
    u_2d = u.reshape(-1, 1)

    esn.reset()
    states = np.zeros((n_steps, esn.n), dtype=np.float32)
    for t in range(n_steps):
        states[t] = esn.step(u_2d[t])

    mc_total = 0.0
    # Use only the second half of states (after washout)
    washout = n_steps // 4
    S = states[washout:]
    T_eval = S.shape[0]

    for k in range(1, max_delay + 1):
        if k >= T_eval:
            break
        # Target: u[t - k] for t in [washout, n_steps)
        target = u[washout - k : n_steps - k].reshape(-1, 1).astype(np.float32)
        W = _ridge_fit(S[: T_eval - k], target[: T_eval - k], lam=1e-4)
        pred = _ridge_predict(W, S[: T_eval - k])
        # R² = 1 - SS_res / SS_tot
        ss_res = float(np.sum((pred - target[: T_eval - k]) ** 2))
        ss_tot = float(np.sum((target[: T_eval - k] - target[: T_eval - k].mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        mc_total += max(0.0, r2)

    return mc_total


# ---------------------------------------------------------------------------
# Benchmark evaluation (passkey + computation)
# ---------------------------------------------------------------------------


def _eval_passkey(esn: Any, n_examples: int = 50, seed: int = 42) -> float:
    """Evaluate passkey retrieval accuracy using ESN + linear readout.

    Returns exact-match accuracy in [0, 1].
    """
    from src.eval.benchmarks.memory import PasskeyRetrieval

    gen = PasskeyRetrieval(n=n_examples, context_length=50, seed=seed)
    examples = list(gen)

    sequences = [_encode_text(ex.input, max_len=256) for ex in examples]
    targets_str = [ex.target for ex in examples]

    # Encode targets as class indices
    unique_targets = sorted(set(targets_str))
    target_map = {t: i for i, t in enumerate(unique_targets)}
    n_classes = len(unique_targets)

    X = _run_reservoir(esn, sequences, use_final=True)

    if n_classes < 2:
        return 1.0  # trivially correct if only one class

    # One-hot encode targets
    y_oh = np.zeros((len(examples), n_classes), dtype=np.float32)
    for i, t in enumerate(targets_str):
        y_oh[i, target_map[t]] = 1.0

    # Train / test split (80 / 20)
    n_train = max(1, int(0.8 * len(examples)))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_oh[:n_train], y_oh[n_train:]
    targets_test = targets_str[n_train:]

    if X_test.shape[0] == 0:
        return 0.0

    W = _ridge_fit(X_train, y_train, lam=1.0)
    pred_scores = _ridge_predict(W, X_test)
    pred_indices = np.argmax(pred_scores, axis=1)
    pred_strings = [unique_targets[i] for i in pred_indices]

    correct = sum(p == t for p, t in zip(pred_strings, targets_test))
    return correct / len(targets_test)


def _eval_computation(esn: Any, n_examples: int = 50, seed: int = 42) -> float:
    """Evaluate multi-digit arithmetic using ESN + linear readout.

    Returns exact-match accuracy in [0, 1].
    """
    from src.eval.benchmarks.computation import MultiDigitArithmetic

    gen = MultiDigitArithmetic(n=n_examples, digit_count=3, operation="addition", seed=seed)
    examples = list(gen)

    sequences = [_encode_text(ex.input, max_len=128) for ex in examples]
    targets_str = [ex.target for ex in examples]

    unique_targets = sorted(set(targets_str))
    target_map = {t: i for i, t in enumerate(unique_targets)}
    n_classes = len(unique_targets)

    X = _run_reservoir(esn, sequences, use_final=True)

    if n_classes < 2:
        return 1.0

    y_oh = np.zeros((len(examples), n_classes), dtype=np.float32)
    for i, t in enumerate(targets_str):
        y_oh[i, target_map[t]] = 1.0

    n_train = max(1, int(0.8 * len(examples)))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_oh[:n_train], y_oh[n_train:]
    targets_test = targets_str[n_train:]

    if X_test.shape[0] == 0:
        return 0.0

    W = _ridge_fit(X_train, y_train, lam=1.0)
    pred_scores = _ridge_predict(W, X_test)
    pred_indices = np.argmax(pred_scores, axis=1)
    pred_strings = [unique_targets[i] for i in pred_indices]

    correct = sum(p == t for p, t in zip(pred_strings, targets_test))
    return correct / len(targets_test)


# ---------------------------------------------------------------------------
# Step latency measurement
# ---------------------------------------------------------------------------


def _measure_latency(esn: Any, n_trials: int = 1_000) -> float:
    """Return mean step latency in milliseconds."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(esn.input_dim).astype(np.float32)

    # Warm-up
    for _ in range(10):
        esn.step(x)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        esn.step(x)
    elapsed = time.perf_counter() - t0
    return elapsed / n_trials * 1_000  # ms


# ---------------------------------------------------------------------------
# Run config dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """A single sweep run configuration."""

    run_id: int
    run_name: str
    dimension: str  # which sweep dimension this belongs to
    param_key: str
    param_value: Any
    reservoir_size: int
    spectral_radius: float
    leak_rate: float
    topology: str
    input_scaling: float
    sparsity: float
    seed: int


@dataclass
class RunResult:
    """Metrics from a single sweep run."""

    run_id: int
    run_name: str
    config: dict[str, Any]
    memory_capacity: float
    passkey_acc: float
    computation_acc: float
    step_latency_ms: float
    quality_score: float
    elapsed_seconds: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _make_run_name(dim_label: str, param_key: str, value: Any) -> str:
    """Produce a filesystem-safe run name."""
    val_str = str(value).replace(".", "p").replace("-", "m")
    return f"{dim_label}_{val_str}"


def build_all_runs(best_so_far: dict[str, Any] | None = None) -> list[RunConfig]:
    """Build the full list of 15 run configurations.

    Sequential sweep: after each dimension we fix the best value and
    move to the next.  Since we don't know the best values in advance,
    *best_so_far* is only used when resuming a partial sweep.

    Args:
        best_so_far: Mapping of param_key -> best_value discovered so
                     far (used when resuming).  Pass None to start fresh.

    Returns:
        Ordered list of RunConfig objects (15 total).
    """
    current_cfg = dict(BASE_CONFIG)
    if best_so_far:
        current_cfg.update(best_so_far)

    runs: list[RunConfig] = []
    run_id = 0

    for dim_label, param_key, values in SWEEP_DIMENSIONS:
        for value in values:
            cfg = dict(current_cfg)
            cfg[param_key] = value
            runs.append(
                RunConfig(
                    run_id=run_id,
                    run_name=_make_run_name(dim_label, param_key, value),
                    dimension=dim_label,
                    param_key=param_key,
                    param_value=value,
                    reservoir_size=cfg["reservoir_size"],
                    spectral_radius=cfg["spectral_radius"],
                    leak_rate=cfg["leak_rate"],
                    topology=cfg["topology"],
                    input_scaling=cfg["input_scaling"],
                    sparsity=cfg["sparsity"],
                    seed=cfg["seed"],
                )
            )
            run_id += 1

    # Run 14 (index 14): best combo fine-tune
    runs.append(
        RunConfig(
            run_id=run_id,
            run_name="best_combo",
            dimension="best_combo",
            param_key="",
            param_value=None,
            reservoir_size=current_cfg["reservoir_size"],
            spectral_radius=current_cfg["spectral_radius"],
            leak_rate=current_cfg["leak_rate"],
            topology=current_cfg["topology"],
            input_scaling=current_cfg["input_scaling"],
            sparsity=current_cfg["sparsity"],
            seed=current_cfg["seed"],
        )
    )

    return runs


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------


def _build_esn(run_cfg: RunConfig) -> Any:
    """Build an ESN from a RunConfig."""
    from src.reservoir.esn import ESN
    from src.types import ReservoirConfig

    res_cfg = ReservoirConfig(
        size=run_cfg.reservoir_size,
        spectral_radius=run_cfg.spectral_radius,
        leak_rate=run_cfg.leak_rate,
        input_scaling=run_cfg.input_scaling,
        topology=run_cfg.topology,
        sparsity=run_cfg.sparsity,
        seed=run_cfg.seed,
    )
    return ESN(res_cfg, input_dim=_VOCAB_SIZE)


def execute_run(
    run_cfg: RunConfig,
    *,
    n_eval_examples: int = 50,
    mc_steps: int = 500,
    no_wandb: bool = False,
    results_dir: Path = RESULTS_DIR,
) -> RunResult:
    """Execute a single sweep run and return its metrics.

    Args:
        run_cfg: Configuration for this run.
        n_eval_examples: Number of examples per benchmark subset.
        mc_steps: Steps used for memory capacity measurement.
        no_wandb: If True, skip wandb logging.
        results_dir: Directory to write per-run results.

    Returns:
        RunResult with all metrics.
    """
    t0 = time.perf_counter()
    logger.info(
        "run_id=%d  name=%s  size=%d  sr=%.3f  lr=%.2f  topo=%s",
        run_cfg.run_id,
        run_cfg.run_name,
        run_cfg.reservoir_size,
        run_cfg.spectral_radius,
        run_cfg.leak_rate,
        run_cfg.topology,
    )

    # Build ESN
    esn = _build_esn(run_cfg)

    # Evaluate
    logger.info("  measuring memory capacity ...")
    mc = _memory_capacity(esn, n_steps=mc_steps, max_delay=20)

    logger.info("  evaluating passkey retrieval (%d examples) ...", n_eval_examples)
    passkey_acc = _eval_passkey(esn, n_examples=n_eval_examples)

    logger.info("  evaluating computation (%d examples) ...", n_eval_examples)
    comp_acc = _eval_computation(esn, n_examples=n_eval_examples)

    logger.info("  measuring step latency ...")
    latency_ms = _measure_latency(esn, n_trials=1_000)

    # Normalise MC to [0, 1] range (max possible is 20 for max_delay=20)
    mc_norm = min(1.0, mc / 20.0)
    quality = 0.4 * mc_norm + 0.4 * passkey_acc + 0.2 * comp_acc

    elapsed = time.perf_counter() - t0

    result = RunResult(
        run_id=run_cfg.run_id,
        run_name=run_cfg.run_name,
        config={
            "reservoir_size": run_cfg.reservoir_size,
            "spectral_radius": run_cfg.spectral_radius,
            "leak_rate": run_cfg.leak_rate,
            "topology": run_cfg.topology,
            "input_scaling": run_cfg.input_scaling,
            "sparsity": run_cfg.sparsity,
            "seed": run_cfg.seed,
        },
        memory_capacity=mc,
        passkey_acc=passkey_acc,
        computation_acc=comp_acc,
        step_latency_ms=latency_ms,
        quality_score=quality,
        elapsed_seconds=elapsed,
    )

    logger.info(
        "  MC=%.3f  passkey=%.3f  comp=%.3f  latency=%.3fms  quality=%.3f  t=%.1fs",
        mc,
        passkey_acc,
        comp_acc,
        latency_ms,
        quality,
        elapsed,
    )

    # Save per-run result
    run_out = results_dir / run_cfg.run_name
    run_out.mkdir(parents=True, exist_ok=True)
    with (run_out / "metrics.json").open("w") as f:
        json.dump(asdict(result), f, indent=2)

    # Wandb
    if not no_wandb:
        _wandb_log(run_cfg, result)

    return result


def _wandb_log(run_cfg: RunConfig, result: RunResult) -> None:
    """Log run result to wandb (silently skip if wandb not installed)."""
    try:
        import wandb  # type: ignore[import]

        run = wandb.init(
            project="lrs-track-a",
            name=f"sweep/{run_cfg.run_name}",
            group="reservoir_hp_sweep",
            config={
                **result.config,
                "dimension": run_cfg.dimension,
                "param_key": run_cfg.param_key,
                "param_value": run_cfg.param_value,
            },
            reinit=True,
        )
        wandb.log(
            {
                "memory_capacity": result.memory_capacity,
                "passkey_acc": result.passkey_acc,
                "computation_acc": result.computation_acc,
                "step_latency_ms": result.step_latency_ms,
                "quality_score": result.quality_score,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        if run is not None:
            run.finish()
    except Exception as exc:
        logger.debug("wandb logging skipped: %s", exc)


# ---------------------------------------------------------------------------
# Sequential sweep orchestration
# ---------------------------------------------------------------------------


def _load_existing_result(run_name: str, results_dir: Path) -> RunResult | None:
    """Return a RunResult if already computed, else None."""
    p = results_dir / run_name / "metrics.json"
    if not p.exists():
        return None
    try:
        with p.open() as f:
            d = json.load(f)
        return RunResult(**d)
    except Exception:
        return None


def _best_in_dimension(
    results: list[RunResult],
    param_key: str,
) -> Any:
    """Return the param_value of the run with the highest quality_score."""
    if not results:
        return BASE_CONFIG.get(param_key)
    best = max(results, key=lambda r: r.quality_score)
    return best.config.get(param_key)


def run_sweep(
    args: argparse.Namespace,
    results_dir: Path,
) -> list[RunResult]:
    """Execute the full 15-run sequential sweep.

    Returns the list of all RunResult objects.
    """
    all_results: list[RunResult] = []
    best_per_dim: dict[str, Any] = {}
    current_cfg = dict(BASE_CONFIG)

    dim_runs: dict[str, list[RunConfig]] = {}
    run_id = 0

    for dim_label, param_key, values in SWEEP_DIMENSIONS:
        dim_runs[dim_label] = []
        for value in values:
            cfg = dict(current_cfg)
            cfg[param_key] = value
            dim_runs[dim_label].append(
                RunConfig(
                    run_id=run_id,
                    run_name=_make_run_name(dim_label, param_key, value),
                    dimension=dim_label,
                    param_key=param_key,
                    param_value=value,
                    reservoir_size=cfg["reservoir_size"],
                    spectral_radius=cfg["spectral_radius"],
                    leak_rate=cfg["leak_rate"],
                    topology=cfg["topology"],
                    input_scaling=cfg["input_scaling"],
                    sparsity=cfg["sparsity"],
                    seed=cfg["seed"],
                )
            )
            run_id += 1

        # Execute all runs in this dimension
        dim_results: list[RunResult] = []
        for run_cfg in dim_runs[dim_label]:
            # Check if args.run_id is set (single-run mode)
            if args.run_id is not None and run_cfg.run_id != args.run_id:
                continue

            # Resume: skip already-completed runs
            if args.resume:
                existing = _load_existing_result(run_cfg.run_name, results_dir)
                if existing is not None:
                    logger.info("Resuming: skipping run %s (already done)", run_cfg.run_name)
                    dim_results.append(existing)
                    all_results.append(existing)
                    continue

            if args.dry_run:
                logger.info("DRY RUN: %s  (id=%d)", run_cfg.run_name, run_cfg.run_id)
                continue

            result = execute_run(
                run_cfg,
                n_eval_examples=args.n_eval_examples,
                mc_steps=args.mc_steps,
                no_wandb=args.no_wandb,
                results_dir=results_dir,
            )
            dim_results.append(result)
            all_results.append(result)

        # Fix best value for this dimension before moving on
        if dim_results:
            best_val = _best_in_dimension(dim_results, param_key)
            best_per_dim[param_key] = best_val
            current_cfg[param_key] = best_val
            logger.info(
                "Dimension '%s' best: %s = %s", dim_label, param_key, best_val
            )

    # Run 14: best combo fine-tune
    best_combo_cfg = RunConfig(
        run_id=run_id,
        run_name="best_combo",
        dimension="best_combo",
        param_key="",
        param_value=None,
        reservoir_size=current_cfg["reservoir_size"],
        spectral_radius=current_cfg["spectral_radius"],
        leak_rate=current_cfg["leak_rate"],
        topology=current_cfg["topology"],
        input_scaling=current_cfg["input_scaling"],
        sparsity=current_cfg["sparsity"],
        seed=current_cfg["seed"],
    )

    if args.run_id is None or args.run_id == run_id:
        if args.dry_run:
            logger.info("DRY RUN: best_combo  (id=%d)", run_id)
        elif args.resume:
            existing = _load_existing_result("best_combo", results_dir)
            if existing is not None:
                logger.info("Resuming: skipping best_combo (already done)")
                all_results.append(existing)
            else:
                result = execute_run(
                    best_combo_cfg,
                    n_eval_examples=args.n_eval_examples,
                    mc_steps=args.mc_steps,
                    no_wandb=args.no_wandb,
                    results_dir=results_dir,
                )
                all_results.append(result)
        else:
            result = execute_run(
                best_combo_cfg,
                n_eval_examples=args.n_eval_examples,
                mc_steps=args.mc_steps,
                no_wandb=args.no_wandb,
                results_dir=results_dir,
            )
            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def compute_pareto_frontier(results: list[RunResult]) -> list[RunResult]:
    """Identify Pareto-optimal runs wrt (quality_score ↑, step_latency_ms ↓).

    A run is Pareto-optimal if no other run dominates it on both axes.
    """
    pareto: list[RunResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if (
                other.quality_score >= r.quality_score
                and other.step_latency_ms <= r.step_latency_ms
                and (
                    other.quality_score > r.quality_score
                    or other.step_latency_ms < r.step_latency_ms
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return sorted(pareto, key=lambda r: r.quality_score, reverse=True)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------


def write_summary(results: list[RunResult], results_dir: Path) -> None:
    """Write pareto frontier and best config to results_dir."""
    if not results:
        return

    pareto = compute_pareto_frontier(results)
    best = max(results, key=lambda r: r.quality_score)

    pareto_path = results_dir / "pareto_frontier.json"
    with pareto_path.open("w") as f:
        json.dump(
            {
                "pareto_frontier": [asdict(r) for r in pareto],
                "axes": {
                    "x": "reservoir_size",
                    "y": "quality_score (higher is better)",
                    "z": "step_latency_ms (lower is better)",
                },
            },
            f,
            indent=2,
        )
    logger.info("Pareto frontier (%d runs) written to %s", len(pareto), pareto_path)

    best_cfg_path = results_dir / "best_config.yaml"
    try:
        import yaml  # type: ignore[import-untyped]

        with best_cfg_path.open("w") as f:
            yaml.dump(
                {
                    **best.config,
                    "_run_name": best.run_name,
                    "_quality_score": best.quality_score,
                    "_step_latency_ms": best.step_latency_ms,
                },
                f,
                default_flow_style=False,
            )
    except ImportError:
        # Fallback to JSON if PyYAML not available
        best_cfg_path = results_dir / "best_config.json"
        with best_cfg_path.open("w") as f:
            json.dump(
                {
                    **best.config,
                    "_run_name": best.run_name,
                    "_quality_score": best.quality_score,
                    "_step_latency_ms": best.step_latency_ms,
                },
                f,
                indent=2,
            )
    logger.info("Best config written to %s", best_cfg_path)

    # Print leaderboard
    print("\n=== Reservoir HP Sweep Results ===")
    print(f"{'Run':<24}  {'Quality':>8}  {'MC':>6}  {'Pass':>6}  {'Comp':>6}  {'ms/step':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.quality_score, reverse=True):
        print(
            f"{r.run_name:<24}  {r.quality_score:8.4f}  "
            f"{r.memory_capacity:6.2f}  {r.passkey_acc:6.3f}  "
            f"{r.computation_acc:6.3f}  {r.step_latency_ms:8.4f}"
        )
    print("\nPareto-optimal runs:")
    for r in pareto:
        print(
            f"  {r.run_name}  quality={r.quality_score:.4f}  "
            f"latency={r.step_latency_ms:.4f}ms  size={r.config['reservoir_size']}"
        )
    print(f"\nBest overall: {best.run_name}  (quality={best.quality_score:.4f})")


# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------


def write_sweep_configs(results_dir: Path, configs_dir: Path) -> None:
    """Write per-run YAML configs to configs/sweep/ for reproducibility."""
    configs_dir.mkdir(parents=True, exist_ok=True)

    runs = build_all_runs()
    for run_cfg in runs:
        cfg_dict = {
            # Inherit non-swept settings from track_a_readonly.yaml
            "model_name": "qwen3.5-0.8b",
            "dtype": "bfloat16",
            "device": "cuda",
            # Reservoir settings
            "reservoir_size": run_cfg.reservoir_size,
            "spectral_radius": run_cfg.spectral_radius,
            "leak_rate": run_cfg.leak_rate,
            "topology": run_cfg.topology,
            "input_scaling": run_cfg.input_scaling,
            "reservoir_sparsity": run_cfg.sparsity,
            "reservoir_seed": run_cfg.seed,
            # Training (short runs for sweep)
            "max_steps": 2000,
            "batch_size": 4,
            "grad_accum": 4,
            "warmup_steps": 100,
            "lr": 2.0e-4,
            "interface_lr": 1.0e-3,
            "weight_decay": 0.01,
            "max_seq_length": 2048,
            "gradient_checkpointing": True,
            "seed": 42,
            # Data
            "dataset_name": "HuggingFaceFW/fineweb",
            "dataset_config": "sample-10BT",
            # Output
            "output_dir": f"checkpoints/track_a/sweep/{run_cfg.run_name}",
            "results_file": f"results/track_a/sweep/{run_cfg.run_name}/train_metrics.json",
            "log_interval": 50,
            "save_interval": 1000,
            # Sidecar
            "sidecar_layers": None,
            "num_heads": 8,
            "sidecar_dropout": 0.0,
            # LoRA
            "lora_rank": 16,
            "lora_alpha": 32.0,
            "lora_dropout": 0.05,
            "lora_targets": ["q_proj", "v_proj"],
            # Wandb
            "no_wandb": False,
            "wandb_project": "lrs-track-a",
            "wandb_run_name": f"sweep/{run_cfg.run_name}",
            # Sweep metadata
            "_sweep_dimension": run_cfg.dimension,
            "_sweep_param": run_cfg.param_key,
            "_sweep_value": run_cfg.param_value,
            "_run_id": run_cfg.run_id,
        }

        cfg_path = configs_dir / f"{run_cfg.run_name}.yaml"
        try:
            import yaml  # type: ignore[import-untyped]

            with cfg_path.open("w") as f:
                yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback: write JSON
            cfg_path = configs_dir / f"{run_cfg.run_name}.json"
            with cfg_path.open("w") as f:
                json.dump(cfg_dict, f, indent=2)

    logger.info("Written %d sweep configs to %s", len(runs), configs_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep reservoir hyperparameters for Track A.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Run a single configuration by its 0-based index (0-14).  "
             "Omit to run the full sweep.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose results already exist in results_dir.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all configurations without executing any runs.",
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    p.add_argument(
        "--n_eval_examples",
        type=int,
        default=50,
        help="Number of examples per benchmark subset for proxy evaluation.",
    )
    p.add_argument(
        "--mc_steps",
        type=int,
        default=500,
        help="Sequence length for memory capacity measurement.",
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write results.",
    )
    p.add_argument(
        "--configs_dir",
        type=Path,
        default=CONFIGS_DIR,
        help="Directory to write per-run YAML configs.",
    )
    p.add_argument(
        "--gen_configs_only",
        action="store_true",
        help="Only generate YAML config files; do not run any sweep.",
    )
    p.add_argument(
        "--list_runs",
        action="store_true",
        help="List all planned runs and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Generate YAML configs always (they are lightweight)
    write_sweep_configs(args.results_dir, args.configs_dir)

    if args.gen_configs_only:
        logger.info("Config generation complete.  Exiting (--gen_configs_only).")
        return

    if args.list_runs:
        runs = build_all_runs()
        print(f"{'ID':>4}  {'Name':<28}  {'Dim':<18}  {'Param':<20}  Value")
        print("-" * 90)
        for r in runs:
            print(
                f"{r.run_id:>4}  {r.run_name:<28}  {r.dimension:<18}  "
                f"{r.param_key:<20}  {r.param_value}"
            )
        return

    all_results = run_sweep(args, args.results_dir)

    if all_results:
        write_summary(all_results, args.results_dir)
    else:
        logger.info("No results produced (dry_run or all skipped).")


if __name__ == "__main__":
    main()
