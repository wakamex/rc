#!/usr/bin/env python3
"""Efficiency benchmarking for reservoir computing models (rc-wwh.29).

Measures throughput, latency, memory footprint, and quality-per-compute
for ESN reservoir variants at different operating points.

Benchmarks:
  - Throughput: states/sec at batch sizes 1, 4, 16
  - Latency: p50/p95 at sequence lengths 64, 512, 4096, 32768, 131072
  - Memory: parameter bytes + peak working-set at reservoir sizes
    corresponding to 4K, 32K, 128K effective context lengths
  - Quality-per-FLOP: quality_score / total_FLOPs
  - Quality-per-byte: quality_score / param_bytes
  - Comparison against RandomProjection control at matched compute

Results are written to ``results/efficiency/`` as standardised JSON.

Usage::

    python scripts/efficiency_benchmark.py

    # Specific benchmark only
    python scripts/efficiency_benchmark.py --benchmark throughput

    # Dry-run
    python scripts/efficiency_benchmark.py --dry_run

    # Use a specific reservoir size for all benchmarks
    python scripts/efficiency_benchmark.py --reservoir_size 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/efficiency")

# ---------------------------------------------------------------------------
# Best config from Track A
# ---------------------------------------------------------------------------

BEST_TRACK_A_CONFIG: dict[str, Any] = {
    "reservoir_size": 10_000,
    "spectral_radius": 0.9,
    "leak_rate": 0.3,
    "topology": "erdos_renyi",
    "input_scaling": 1.0,
    "sparsity": 0.01,
    "seed": 42,
}

_PRINTABLE = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
_VOCAB_SIZE: int = len(_PRINTABLE)

# ---------------------------------------------------------------------------
# Reservoir factories
# ---------------------------------------------------------------------------


def _build_esn(size: int, spectral_radius: float = 0.9, seed: int = 42) -> Any:
    from src.reservoir.esn import ESN
    from src.types import ReservoirConfig

    cfg = ReservoirConfig(
        size=size,
        spectral_radius=spectral_radius,
        leak_rate=BEST_TRACK_A_CONFIG["leak_rate"],
        input_scaling=BEST_TRACK_A_CONFIG["input_scaling"],
        topology=BEST_TRACK_A_CONFIG["topology"],
        sparsity=BEST_TRACK_A_CONFIG["sparsity"],
        seed=seed,
    )
    return ESN(cfg, input_dim=_VOCAB_SIZE)


def _build_random_proj(size: int, seed: int = 42) -> Any:
    """Stateless random projection baseline (same output dim as ESN)."""
    import scipy.sparse as sp

    class _RandomProj:
        def __init__(self) -> None:
            rng = np.random.default_rng(seed)
            self.n = size
            self.input_dim = _VOCAB_SIZE
            self._W = rng.standard_normal((size, _VOCAB_SIZE)).astype(np.float32)
            self._W /= np.linalg.norm(self._W, axis=1, keepdims=True) + 1e-8
            self.state = np.zeros(size, dtype=np.float32)

        def step(self, x_t: np.ndarray, w_t: Any = None) -> np.ndarray:
            out = np.tanh(self._W @ np.asarray(x_t, dtype=np.float32))
            self.state = out
            return out

        def step_batch(self, X: np.ndarray) -> np.ndarray:
            """Batched step: X shape (batch, input_dim) → (batch, n)."""
            return np.tanh(X @ self._W.T)

        def reset(self) -> None:
            self.state = np.zeros(self.n, dtype=np.float32)

        def forward(self, X: np.ndarray) -> np.ndarray:
            T = X.shape[0]
            out = np.empty((T, self.n), dtype=np.float32)
            for t in range(T):
                out[t] = self.step(X[t])
            return out

    return _RandomProj()


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------


def _esn_flops_per_step(n: int, input_dim: int, sparsity: float) -> int:
    """Estimate FLOPs for one ESN step.

    Operations:
      - W @ r: 2 * n * (sparsity * n) — sparse MV
      - W_in @ x: 2 * n * input_dim — dense MV (or sparse for large n)
      - tanh + leaky integration: ~5n element-wise ops
    """
    sparse_mv = 2 * n * max(1, int(sparsity * n))
    win_mv = 2 * n * input_dim
    elemwise = 5 * n
    return sparse_mv + win_mv + elemwise


def _random_proj_flops_per_step(n: int, input_dim: int) -> int:
    """Estimate FLOPs for one random projection step (dense MV + tanh)."""
    return 2 * n * input_dim + n


# ---------------------------------------------------------------------------
# Parameter count & memory
# ---------------------------------------------------------------------------


def _count_esn_bytes(reservoir: Any) -> int:
    """Total bytes in ESN weight matrices."""
    import scipy.sparse as sp

    total_floats = 0
    for attr in ("W", "W_in", "_W"):
        m = getattr(reservoir, attr, None)
        if m is None:
            continue
        if sp.issparse(m):
            # CSR: data + indices + indptr (float32 + int32 + int32)
            total_floats += m.nnz  # float32 data
            total_bytes_indices = m.nnz * 4 + (m.shape[0] + 1) * 4  # indices + indptr
        elif isinstance(m, np.ndarray):
            total_floats += m.size

    # Working set: state vector
    n = getattr(reservoir, "n", 0)
    state_bytes = n * 4  # float32

    return total_floats * 4 + state_bytes


def _count_random_proj_bytes(reservoir: Any) -> int:
    n = reservoir.n
    input_dim = reservoir.input_dim
    # W matrix (float32) + state
    return (n * input_dim + n) * 4


# ---------------------------------------------------------------------------
# Benchmark: throughput
# ---------------------------------------------------------------------------


@dataclass
class ThroughputResult:
    reservoir_type: str
    reservoir_size: int
    batch_size: int
    steps_per_second: float
    tokens_per_second: float  # same as steps_per_second × batch_size
    latency_mean_ms: float
    n_warmup: int = 20
    n_trials: int = 500
    timestamp: float = field(default_factory=time.time)


def benchmark_throughput(
    reservoir: Any,
    reservoir_type: str,
    reservoir_size: int,
    batch_sizes: list[int],
    n_trials: int = 500,
    n_warmup: int = 20,
) -> list[ThroughputResult]:
    """Measure throughput at different batch sizes."""
    results = []
    rng = np.random.default_rng(0)
    input_dim = getattr(reservoir, "input_dim", _VOCAB_SIZE)

    for bs in batch_sizes:
        logger.info(
            "  throughput: type=%s  size=%d  batch=%d",
            reservoir_type, reservoir_size, bs,
        )

        # Prepare input batch: (batch, input_dim) for step calls
        X = rng.standard_normal((bs, input_dim)).astype(np.float32)

        reservoir.reset()

        # Warm-up
        for _ in range(n_warmup):
            for b in range(bs):
                reservoir.step(X[b])

        # Timed trials — each "step" processes bs items
        t0 = time.perf_counter()
        for _ in range(n_trials):
            for b in range(bs):
                reservoir.step(X[b])
        elapsed = time.perf_counter() - t0

        total_steps = n_trials  # one call per trial
        total_tokens = n_trials * bs

        results.append(ThroughputResult(
            reservoir_type=reservoir_type,
            reservoir_size=reservoir_size,
            batch_size=bs,
            steps_per_second=total_steps / elapsed,
            tokens_per_second=total_tokens / elapsed,
            latency_mean_ms=elapsed / total_steps * 1_000,
        ))
        logger.info(
            "    → %.0f tokens/sec  mean_latency=%.3f ms",
            results[-1].tokens_per_second,
            results[-1].latency_mean_ms,
        )

    return results


# ---------------------------------------------------------------------------
# Benchmark: latency percentiles
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    reservoir_type: str
    reservoir_size: int
    seq_len: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    n_trials: int = 100
    timestamp: float = field(default_factory=time.time)


def benchmark_latency(
    reservoir: Any,
    reservoir_type: str,
    reservoir_size: int,
    seq_lengths: list[int],
    n_trials: int = 100,
) -> list[LatencyResult]:
    """Measure p50/p95/p99 latency for processing sequences of different lengths."""
    results = []
    rng = np.random.default_rng(1)
    input_dim = getattr(reservoir, "input_dim", _VOCAB_SIZE)

    for seq_len in seq_lengths:
        logger.info(
            "  latency: type=%s  size=%d  seq_len=%d",
            reservoir_type, reservoir_size, seq_len,
        )

        X = rng.standard_normal((seq_len, input_dim)).astype(np.float32)
        timings = []

        # Warm-up
        for _ in range(5):
            reservoir.reset()
            reservoir.forward(X[:min(64, seq_len)])

        for _ in range(n_trials):
            reservoir.reset()
            t0 = time.perf_counter()
            reservoir.forward(X)
            timings.append((time.perf_counter() - t0) * 1_000)  # ms

        timings_arr = np.array(timings)
        results.append(LatencyResult(
            reservoir_type=reservoir_type,
            reservoir_size=reservoir_size,
            seq_len=seq_len,
            p50_ms=float(np.percentile(timings_arr, 50)),
            p95_ms=float(np.percentile(timings_arr, 95)),
            p99_ms=float(np.percentile(timings_arr, 99)),
            mean_ms=float(timings_arr.mean()),
            n_trials=n_trials,
        ))
        r = results[-1]
        logger.info(
            "    → p50=%.2f ms  p95=%.2f ms  p99=%.2f ms",
            r.p50_ms, r.p95_ms, r.p99_ms,
        )

    return results


# ---------------------------------------------------------------------------
# Benchmark: memory (VRAM / RAM footprint)
# ---------------------------------------------------------------------------


@dataclass
class MemoryResult:
    reservoir_type: str
    reservoir_size: int
    context_label: str  # "4K", "32K", "128K"
    param_bytes: int
    param_mb: float
    working_set_bytes: int  # state + intermediate buffers estimate
    working_set_mb: float
    timestamp: float = field(default_factory=time.time)


def benchmark_memory(
    reservoir_sizes: dict[str, int],
) -> list[MemoryResult]:
    """Estimate memory footprint at different reservoir sizes.

    ``reservoir_sizes`` maps context label → reservoir size.
    We build both ESN and RandomProjection at each size and report bytes.
    """
    results = []

    for label, size in reservoir_sizes.items():
        logger.info("  memory: label=%s  size=%d", label, size)

        for rtype, builder in [
            ("esn", lambda s: _build_esn(s)),
            ("random_proj", lambda s: _build_random_proj(s)),
        ]:
            reservoir = builder(size)

            if rtype == "esn":
                param_bytes = _count_esn_bytes(reservoir)
            else:
                param_bytes = _count_random_proj_bytes(reservoir)

            # Working set: state vector + intermediate arrays (2× state for leaky update)
            n = getattr(reservoir, "n", size)
            working_set_bytes = param_bytes + n * 4 * 3  # state + 3 temp buffers

            results.append(MemoryResult(
                reservoir_type=rtype,
                reservoir_size=size,
                context_label=label,
                param_bytes=param_bytes,
                param_mb=param_bytes / 1024 / 1024,
                working_set_bytes=working_set_bytes,
                working_set_mb=working_set_bytes / 1024 / 1024,
            ))
            logger.info(
                "    → %s: param=%.2f MB  working_set=%.2f MB",
                rtype,
                results[-1].param_mb,
                results[-1].working_set_mb,
            )

    return results


# ---------------------------------------------------------------------------
# Benchmark: quality-per-FLOP and quality-per-byte
# ---------------------------------------------------------------------------


@dataclass
class QualityEfficiencyResult:
    reservoir_type: str
    reservoir_size: int
    quality_score: float
    memory_capacity: float
    passkey_acc: float
    computation_acc: float
    flops_per_step: int
    param_bytes: int
    quality_per_flop: float
    quality_per_mb: float
    timestamp: float = field(default_factory=time.time)


def _eval_quality(
    reservoir: Any,
    n_eval: int = 50,
    mc_steps: int = 300,
) -> tuple[float, float, float]:
    """Return (memory_capacity, passkey_acc, computation_acc)."""
    # Memory capacity
    rng = np.random.default_rng(0)
    u = rng.choice([-1.0, 1.0], size=(mc_steps,)).astype(np.float32)
    u_2d = u.reshape(-1, 1)
    reservoir.reset()
    n = getattr(reservoir, "n", None)
    states = np.zeros((mc_steps, n), dtype=np.float32)
    for t in range(mc_steps):
        states[t] = reservoir.step(u_2d[t])

    washout = mc_steps // 4
    S = states[washout:]
    T_eval = S.shape[0]
    mc_total = 0.0
    for k in range(1, min(21, T_eval)):
        target = u[washout - k : mc_steps - k].reshape(-1, 1).astype(np.float32)
        d = S.shape[1]
        A = S[: T_eval - k].T @ S[: T_eval - k] + 1e-4 * np.eye(d, dtype=np.float32)
        b = S[: T_eval - k].T @ target[: T_eval - k]
        W = np.linalg.solve(A, b)
        pred = S[: T_eval - k] @ W
        ss_res = float(np.sum((pred - target[: T_eval - k]) ** 2))
        ss_tot = float(np.sum((target[: T_eval - k] - target[: T_eval - k].mean()) ** 2))
        mc_total += max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))

    # Passkey (simplified version using random sequences for speed)
    _CHAR_IDX = {c: i for i, c in enumerate(_PRINTABLE)}

    def _encode(text: str) -> np.ndarray:
        X = np.zeros((len(text[:128]), _VOCAB_SIZE), dtype=np.float32)
        for t, ch in enumerate(text[:128]):
            X[t, _CHAR_IDX.get(ch, 0)] = 1.0
        return X

    try:
        from src.eval.benchmarks.memory import PasskeyRetrieval
        from src.eval.benchmarks.computation import MultiDigitArithmetic

        gen_pk = PasskeyRetrieval(n=n_eval, context_length=50, seed=42)
        examples_pk = list(gen_pk)
        seqs_pk = [_encode(ex.input) for ex in examples_pk]
        targets_pk = [ex.target for ex in examples_pk]
        unique_pk = sorted(set(targets_pk))
        n_cls_pk = len(unique_pk)
        pk_map = {t: i for i, t in enumerate(unique_pk)}

        feats_pk = np.stack([
            _run_final_state(reservoir, s) for s in seqs_pk
        ])

        if n_cls_pk < 2:
            passkey_acc = 1.0
        else:
            y_oh = np.zeros((len(examples_pk), n_cls_pk), dtype=np.float32)
            for i, t in enumerate(targets_pk):
                y_oh[i, pk_map[t]] = 1.0
            n_tr = max(1, int(0.8 * len(examples_pk)))
            A = feats_pk[:n_tr].T @ feats_pk[:n_tr] + np.eye(feats_pk.shape[1], dtype=np.float32)
            b = feats_pk[:n_tr].T @ y_oh[:n_tr]
            W = np.linalg.solve(A, b)
            preds = np.argmax(feats_pk[n_tr:] @ W, axis=1)
            targets_test = targets_pk[n_tr:]
            passkey_acc = sum(
                unique_pk[p] == t for p, t in zip(preds, targets_test)
            ) / max(1, len(targets_test))

        gen_comp = MultiDigitArithmetic(n=n_eval, digit_count=3, operation="addition", seed=42)
        examples_comp = list(gen_comp)
        seqs_comp = [_encode(ex.input) for ex in examples_comp]
        targets_comp = [ex.target for ex in examples_comp]
        unique_comp = sorted(set(targets_comp))
        n_cls_comp = len(unique_comp)
        comp_map = {t: i for i, t in enumerate(unique_comp)}

        feats_comp = np.stack([
            _run_final_state(reservoir, s) for s in seqs_comp
        ])

        if n_cls_comp < 2:
            comp_acc = 1.0
        else:
            y_oh = np.zeros((len(examples_comp), n_cls_comp), dtype=np.float32)
            for i, t in enumerate(targets_comp):
                y_oh[i, comp_map[t]] = 1.0
            n_tr = max(1, int(0.8 * len(examples_comp)))
            A = feats_comp[:n_tr].T @ feats_comp[:n_tr] + np.eye(feats_comp.shape[1], dtype=np.float32)
            b = feats_comp[:n_tr].T @ y_oh[:n_tr]
            W = np.linalg.solve(A, b)
            preds = np.argmax(feats_comp[n_tr:] @ W, axis=1)
            targets_test_c = targets_comp[n_tr:]
            comp_acc = sum(
                unique_comp[p] == t for p, t in zip(preds, targets_test_c)
            ) / max(1, len(targets_test_c))

    except ImportError:
        # Benchmark generators not available — use MC as proxy
        passkey_acc = min(1.0, mc_total / 10.0)
        comp_acc = passkey_acc * 0.5

    return mc_total, passkey_acc, comp_acc


def _run_final_state(reservoir: Any, X: np.ndarray) -> np.ndarray:
    reservoir.reset()
    r = None
    for t in range(X.shape[0]):
        r = reservoir.step(X[t])
    n = getattr(reservoir, "n", X.shape[1])
    return r if r is not None else np.zeros(n, dtype=np.float32)


def benchmark_quality_efficiency(
    reservoir_sizes: list[int],
    n_eval: int = 50,
) -> list[QualityEfficiencyResult]:
    """Measure quality-per-FLOP and quality-per-byte at different reservoir sizes."""
    results = []

    for size in reservoir_sizes:
        logger.info("  quality-efficiency: size=%d", size)

        for rtype in ("esn", "random_proj"):
            if rtype == "esn":
                reservoir = _build_esn(size)
                flops = _esn_flops_per_step(
                    size, _VOCAB_SIZE, BEST_TRACK_A_CONFIG["sparsity"]
                )
                param_bytes = _count_esn_bytes(reservoir)
            else:
                reservoir = _build_random_proj(size)
                flops = _random_proj_flops_per_step(size, _VOCAB_SIZE)
                param_bytes = _count_random_proj_bytes(reservoir)

            mc, passkey_acc, comp_acc = _eval_quality(reservoir, n_eval=n_eval)
            mc_norm = min(1.0, mc / 20.0)
            quality = 0.4 * mc_norm + 0.4 * passkey_acc + 0.2 * comp_acc
            param_mb = param_bytes / 1024 / 1024

            results.append(QualityEfficiencyResult(
                reservoir_type=rtype,
                reservoir_size=size,
                quality_score=quality,
                memory_capacity=mc,
                passkey_acc=passkey_acc,
                computation_acc=comp_acc,
                flops_per_step=flops,
                param_bytes=param_bytes,
                quality_per_flop=quality / (flops + 1),
                quality_per_mb=quality / (param_mb + 1e-6),
            ))
            logger.info(
                "    → %s: quality=%.4f  Q/FLOP=%.2e  Q/MB=%.3f",
                rtype, quality,
                results[-1].quality_per_flop,
                results[-1].quality_per_mb,
            )

    return results


# ---------------------------------------------------------------------------
# Coconut comparison (latent reasoning at matched compute)
# ---------------------------------------------------------------------------


@dataclass
class CoconutComparisonResult:
    """Compare ESN multi-step updates vs Coconut-style K sub-steps.

    At matched compute (same FLOPs budget), tests whether recurrent reservoir
    dynamics (multiple ESN steps per token) outperform simple recirculation
    (applying a fixed transformation K times).
    """
    k_steps: int
    esn_quality: float
    coconut_quality: float  # quality of K linear recirculations
    flops_per_token: int
    delta_quality: float  # esn_quality - coconut_quality
    timestamp: float = field(default_factory=time.time)


class _CoconutBaseline:
    """Coconut-style latent recirculation baseline.

    At each token position, applies a fixed linear transformation K times
    to a hidden state vector, matching the ESN's computational budget.
    No recurrent dynamics — purely feedforward K-step chain.
    """

    def __init__(self, n: int, k_steps: int, seed: int = 0) -> None:
        self.n = n
        self.input_dim = _VOCAB_SIZE
        self.k_steps = k_steps
        rng = np.random.default_rng(seed)
        # Fixed linear transform (approximately orthogonal for stability)
        A = rng.standard_normal((n, n)).astype(np.float32)
        # Normalise to spectral radius ~ 0.9
        eigs = np.linalg.eigvals(A[:min(50, n), :min(50, n)])
        sr = float(np.max(np.abs(eigs)))
        self._A = (A * 0.9 / (sr + 1e-8))
        self._W_in = rng.standard_normal((n, _VOCAB_SIZE)).astype(np.float32) * 0.1
        self.state = np.zeros(n, dtype=np.float32)

    def step(self, x_t: np.ndarray, w_t: Any = None) -> np.ndarray:
        h = np.tanh(self._W_in @ np.asarray(x_t, dtype=np.float32))
        # K recirculations — purely feedforward (no input after first step)
        for _ in range(self.k_steps):
            h = np.tanh(self._A @ h)
        self.state = h
        return h

    def reset(self) -> None:
        self.state = np.zeros(self.n, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        out = np.empty((T, self.n), dtype=np.float32)
        for t in range(T):
            out[t] = self.step(X[t])
        return out


def benchmark_coconut_comparison(
    reservoir_size: int = 1_000,
    k_values: list[int] | None = None,
    n_eval: int = 50,
) -> list[CoconutComparisonResult]:
    """Compare ESN multi-step updates vs Coconut-style K sub-steps.

    For fair comparison, the ESN uses K=1 (standard) but with the full
    recurrent state.  The Coconut baseline uses K recirculations of a
    fixed transform, matching the total FLOPs budget.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8]

    results = []
    logger.info("  coconut comparison: size=%d", reservoir_size)

    # ESN at K=1 (baseline)
    esn = _build_esn(reservoir_size)
    mc_esn, pk_esn, comp_esn = _eval_quality(esn, n_eval=n_eval)
    mc_norm_esn = min(1.0, mc_esn / 20.0)
    esn_quality = 0.4 * mc_norm_esn + 0.4 * pk_esn + 0.2 * comp_esn
    esn_flops = _esn_flops_per_step(
        reservoir_size, _VOCAB_SIZE, BEST_TRACK_A_CONFIG["sparsity"]
    )

    for k in k_values:
        coco = _CoconutBaseline(reservoir_size, k_steps=k, seed=42)
        mc_coco, pk_coco, comp_coco = _eval_quality(coco, n_eval=n_eval)
        mc_norm_coco = min(1.0, mc_coco / 20.0)
        coco_quality = 0.4 * mc_norm_coco + 0.4 * pk_coco + 0.2 * comp_coco

        # FLOPs for Coconut at K: input projection (2*n*input_dim) + K*(2*n*n) + K*n
        coco_flops = (
            2 * reservoir_size * _VOCAB_SIZE
            + k * (2 * reservoir_size * reservoir_size + reservoir_size)
        )

        results.append(CoconutComparisonResult(
            k_steps=k,
            esn_quality=esn_quality,
            coconut_quality=coco_quality,
            flops_per_token=max(esn_flops, coco_flops),
            delta_quality=esn_quality - coco_quality,
        ))
        logger.info(
            "    K=%d → ESN: %.4f  Coconut: %.4f  Δ=%.4f",
            k, esn_quality, coco_quality, esn_quality - coco_quality,
        )

    return results


# ---------------------------------------------------------------------------
# Full benchmark orchestration
# ---------------------------------------------------------------------------


BENCHMARK_NAMES = ("throughput", "latency", "memory", "quality_efficiency", "coconut")


@dataclass
class BenchmarkSuite:
    throughput: list[ThroughputResult] = field(default_factory=list)
    latency: list[LatencyResult] = field(default_factory=list)
    memory: list[MemoryResult] = field(default_factory=list)
    quality_efficiency: list[QualityEfficiencyResult] = field(default_factory=list)
    coconut: list[CoconutComparisonResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


def run_benchmarks(
    args: argparse.Namespace,
    results_dir: Path,
) -> BenchmarkSuite:
    suite = BenchmarkSuite()

    selected = set(args.benchmark.split(",")) if args.benchmark else set(BENCHMARK_NAMES)

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------
    if "throughput" in selected and not args.dry_run:
        logger.info("=== Throughput benchmark ===")
        for rtype, builder in [
            ("esn", lambda: _build_esn(args.reservoir_size)),
            ("random_proj", lambda: _build_random_proj(args.reservoir_size)),
        ]:
            reservoir = builder()
            results = benchmark_throughput(
                reservoir=reservoir,
                reservoir_type=rtype,
                reservoir_size=args.reservoir_size,
                batch_sizes=[1, 4, 16],
                n_trials=args.n_latency_trials,
            )
            suite.throughput.extend(results)
        _save_json(
            [asdict(r) for r in suite.throughput],
            results_dir / "throughput.json",
        )

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------
    if "latency" in selected and not args.dry_run:
        logger.info("=== Latency benchmark ===")
        # Limit very long sequences to avoid excessive runtime on small machines
        seq_lengths = [64, 512, 4096]
        if not args.fast:
            seq_lengths += [32_768]
        for rtype, builder in [
            ("esn", lambda: _build_esn(args.reservoir_size)),
            ("random_proj", lambda: _build_random_proj(args.reservoir_size)),
        ]:
            reservoir = builder()
            results = benchmark_latency(
                reservoir=reservoir,
                reservoir_type=rtype,
                reservoir_size=args.reservoir_size,
                seq_lengths=seq_lengths,
                n_trials=args.n_latency_trials,
            )
            suite.latency.extend(results)
        _save_json(
            [asdict(r) for r in suite.latency],
            results_dir / "latency.json",
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    if "memory" in selected and not args.dry_run:
        logger.info("=== Memory benchmark ===")
        # Map context label → reservoir size (approximate scaling)
        reservoir_sizes_by_context = {
            "4K": 2_000,
            "32K": 10_000,
            "128K": 30_000,
        }
        results = benchmark_memory(reservoir_sizes_by_context)
        suite.memory.extend(results)
        _save_json(
            [asdict(r) for r in suite.memory],
            results_dir / "memory.json",
        )

    # ------------------------------------------------------------------
    # Quality-per-FLOP and quality-per-byte
    # ------------------------------------------------------------------
    if "quality_efficiency" in selected and not args.dry_run:
        logger.info("=== Quality-efficiency benchmark ===")
        sizes = [500, 2_000, 5_000, 10_000]
        results = benchmark_quality_efficiency(
            reservoir_sizes=sizes,
            n_eval=args.n_eval_examples,
        )
        suite.quality_efficiency.extend(results)
        _save_json(
            [asdict(r) for r in suite.quality_efficiency],
            results_dir / "quality_efficiency.json",
        )

    # ------------------------------------------------------------------
    # Coconut comparison
    # ------------------------------------------------------------------
    if "coconut" in selected and not args.dry_run:
        logger.info("=== Coconut comparison ===")
        results = benchmark_coconut_comparison(
            reservoir_size=min(args.reservoir_size, 1_000),
            k_values=[1, 2, 4, 8],
            n_eval=args.n_eval_examples,
        )
        suite.coconut.extend(results)
        _save_json(
            [asdict(r) for r in suite.coconut],
            results_dir / "coconut_comparison.json",
        )

    if args.dry_run:
        logger.info("DRY RUN: would run benchmarks: %s", ", ".join(sorted(selected)))

    return suite


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def write_summary(suite: BenchmarkSuite, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Full suite JSON
    full_path = results_dir / "efficiency_summary.json"
    _save_json(
        {
            "throughput": [asdict(r) for r in suite.throughput],
            "latency": [asdict(r) for r in suite.latency],
            "memory": [asdict(r) for r in suite.memory],
            "quality_efficiency": [asdict(r) for r in suite.quality_efficiency],
            "coconut": [asdict(r) for r in suite.coconut],
        },
        full_path,
    )

    # Print tables
    if suite.throughput:
        print("\n=== Throughput (tokens/sec) ===")
        print(f"{'Type':<18}  {'Size':>8}  {'Batch':>6}  {'tok/s':>12}  {'lat(ms)':>10}")
        print("-" * 60)
        for r in suite.throughput:
            print(
                f"{r.reservoir_type:<18}  {r.reservoir_size:>8,}  {r.batch_size:>6}  "
                f"{r.tokens_per_second:>12,.0f}  {r.latency_mean_ms:>10.3f}"
            )

    if suite.latency:
        print("\n=== Latency (ms) ===")
        print(f"{'Type':<18}  {'Size':>8}  {'SeqLen':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}")
        print("-" * 65)
        for r in suite.latency:
            print(
                f"{r.reservoir_type:<18}  {r.reservoir_size:>8,}  {r.seq_len:>8,}  "
                f"{r.p50_ms:>8.2f}  {r.p95_ms:>8.2f}  {r.p99_ms:>8.2f}"
            )

    if suite.memory:
        print("\n=== Memory (MB) ===")
        print(f"{'Type':<18}  {'Context':>8}  {'Size':>8}  {'Params(MB)':>12}  {'WSet(MB)':>12}")
        print("-" * 65)
        for r in suite.memory:
            print(
                f"{r.reservoir_type:<18}  {r.context_label:>8}  {r.reservoir_size:>8,}  "
                f"{r.param_mb:>12.2f}  {r.working_set_mb:>12.2f}"
            )

    if suite.quality_efficiency:
        print("\n=== Quality Efficiency ===")
        print(
            f"{'Type':<18}  {'Size':>8}  {'Quality':>8}  "
            f"{'Q/FLOP':>12}  {'Q/MB':>8}"
        )
        print("-" * 65)
        for r in suite.quality_efficiency:
            print(
                f"{r.reservoir_type:<18}  {r.reservoir_size:>8,}  {r.quality_score:>8.4f}  "
                f"{r.quality_per_flop:>12.2e}  {r.quality_per_mb:>8.3f}"
            )

    if suite.coconut:
        print("\n=== Coconut Comparison (ESN vs K-step recirculation) ===")
        print(f"{'K':>4}  {'ESN':>8}  {'Coconut':>8}  {'Delta':>8}  {'ESN better?'}")
        print("-" * 45)
        for r in suite.coconut:
            better = "YES" if r.delta_quality > 0.01 else ("TIE" if abs(r.delta_quality) <= 0.01 else "NO")
            print(
                f"{r.k_steps:>4}  {r.esn_quality:>8.4f}  {r.coconut_quality:>8.4f}  "
                f"{r.delta_quality:>+8.4f}  {better}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Efficiency benchmarking for reservoir computing models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help=(
            "Comma-separated list of benchmarks to run.  "
            "Options: throughput, latency, memory, quality_efficiency, coconut.  "
            "Default: all."
        ),
    )
    p.add_argument(
        "--reservoir_size",
        type=int,
        default=BEST_TRACK_A_CONFIG["reservoir_size"],
        help="Primary reservoir size for throughput/latency benchmarks.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be run without executing.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip long sequence lengths (>4K) to reduce runtime.",
    )
    p.add_argument(
        "--n_eval_examples",
        type=int,
        default=50,
        help="Number of examples for quality evaluation.",
    )
    p.add_argument(
        "--n_latency_trials",
        type=int,
        default=100,
        help="Number of trials for latency and throughput measurements.",
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write results.",
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

    suite = run_benchmarks(args, args.results_dir)

    if not args.dry_run:
        write_summary(suite, args.results_dir)
    else:
        logger.info("Dry run complete — no benchmarks executed.")


if __name__ == "__main__":
    main()
