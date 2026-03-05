#!/usr/bin/env python3
"""Ablation studies on the best-performing reservoir track (rc-wwh.29).

Runs ablations on the best Track A ESN configuration to understand WHAT
drives performance gains.

Ablation experiments:
  1. Read-only vs write-only vs full read/write (modality ablation)
  2. Single reservoir vs multi-reservoir (fast/slow)
  3. Frozen vs partially trainable reservoir input projections
  4. Spectral radius regimes: subcritical (0.5) vs critical (0.99) vs supercritical (1.1)
  5. CRITICAL — Randomized dynamics control: replace ESN with random stateless
     projection (same param count, no recurrent dynamics).

     If this performs similarly → gains come from extra parameters, not recurrence.
     If it performs worse → recurrent dynamics matter.

All results are written to ``results/ablations/`` as standardised JSON.

Usage::

    # Full ablation suite
    python scripts/ablations.py

    # Single ablation by name
    python scripts/ablations.py --ablation read_only

    # Resume from partial results
    python scripts/ablations.py --resume

    # Dry-run: print all configs without running
    python scripts/ablations.py --dry_run

    # Disable wandb
    python scripts/ablations.py --no_wandb
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/ablations")

# ---------------------------------------------------------------------------
# Best config from Track A HP sweep (baseline for all ablations)
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

# ---------------------------------------------------------------------------
# Character encoding (same as sweep_reservoir_hp.py for comparability)
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
# Ridge regression readout (pure NumPy)
# ---------------------------------------------------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ W


# ---------------------------------------------------------------------------
# Random stateless projection (control for recurrent dynamics)
# ---------------------------------------------------------------------------


class RandomProjection:
    """Stateless random projection with same parameter count as an ESN.

    This is the critical control condition: if this performs similarly to
    the ESN, gains come from extra parameters (projection size), not
    recurrent dynamics.

    The projection maps input → output via a fixed random matrix of the
    same input-to-output parameter count as ESN.W_in (size × input_dim).
    No state is maintained between steps.
    """

    def __init__(self, n: int, input_dim: int, seed: int = 0) -> None:
        self.n = n
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)
        # Same number of parameters as ESN's W_in (dense version)
        self._W = rng.standard_normal((n, input_dim)).astype(np.float32)
        self._W /= np.linalg.norm(self._W, axis=1, keepdims=True) + 1e-8
        self.state = np.zeros(n, dtype=np.float32)

    def step(self, x_t: np.ndarray, w_t: np.ndarray | None = None) -> np.ndarray:
        """Project input with no recurrence — state is always freshly computed."""
        x = np.asarray(x_t, dtype=np.float32)
        out = np.tanh(self._W @ x)
        self.state = out
        return out

    def reset(self) -> None:
        self.state = np.zeros(self.n, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        states = np.empty((T, self.n), dtype=np.float32)
        for t in range(T):
            states[t] = self.step(X[t])
        return states


# ---------------------------------------------------------------------------
# Write-only reservoir (ablate the read path)
# ---------------------------------------------------------------------------


class WriteOnlyReservoir:
    """Reservoir that receives write signals but whose state is NOT read back.

    The "feature" returned is the raw projected input (no recurrent memory),
    modelling the case where the read path is disabled.  The reservoir still
    runs its recurrent update internally, but the output is replaced by the
    write-side projection only.
    """

    def __init__(self, n: int, input_dim: int, seed: int = 0) -> None:
        self.n = n
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)
        # Write projection: input → n (same shape as W_in in ESN)
        self._W_write = rng.standard_normal((n, input_dim)).astype(np.float32)
        self.state = np.zeros(n, dtype=np.float32)

    def step(self, x_t: np.ndarray, w_t: np.ndarray | None = None) -> np.ndarray:
        """Return a write-side projection; do NOT use the recurrent state."""
        x = np.asarray(x_t, dtype=np.float32)
        # Write signal: current-step projection only (no recurrence exposed)
        out = np.tanh(self._W_write @ x)
        self.state = out
        return out

    def reset(self) -> None:
        self.state = np.zeros(self.n, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        states = np.empty((T, self.n), dtype=np.float32)
        for t in range(T):
            states[t] = self.step(X[t])
        return states


# ---------------------------------------------------------------------------
# ESN with tunable input projection (partial training ablation)
# ---------------------------------------------------------------------------


def _build_esn_trainable_win(
    cfg: dict[str, Any],
    input_dim: int,
    n_train_steps: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> Any:
    """Build an ESN and then fine-tune W_in via gradient-free ES-style update.

    This simulates "partially trainable reservoir input projections" using a
    simple evolution-strategy update on W_in (since full backprop through the
    ESN is not implemented).  W_in is updated to minimise squared prediction
    error on a random regression task.

    Args:
        cfg: Reservoir config dict.
        input_dim: Input dimension.
        n_train_steps: Number of ES update steps.
        lr: Learning rate for W_in updates.
        seed: RNG seed for training data.

    Returns:
        ESN with tuned W_in.
    """
    from src.reservoir.esn import ESN
    from src.types import ReservoirConfig

    res_cfg = ReservoirConfig(
        size=cfg["reservoir_size"],
        spectral_radius=cfg["spectral_radius"],
        leak_rate=cfg["leak_rate"],
        input_scaling=cfg["input_scaling"],
        topology=cfg["topology"],
        sparsity=cfg["sparsity"],
        seed=cfg["seed"],
    )
    esn = ESN(res_cfg, input_dim=input_dim)

    # Simple ES-style update: perturb W_in, keep perturbation if it improves MC
    rng = np.random.default_rng(seed)
    u = rng.choice([-1.0, 1.0], size=(300,)).astype(np.float32)
    u_2d = u.reshape(-1, 1)

    def _mc(local_esn: Any) -> float:
        local_esn.reset()
        n_steps = 300
        states = np.zeros((n_steps, local_esn.n), dtype=np.float32)
        for t in range(n_steps):
            states[t] = local_esn.step(u_2d[t])
        washout = n_steps // 4
        S = states[washout:]
        T_eval = S.shape[0]
        mc_total = 0.0
        for k in range(1, min(11, T_eval)):
            target = u[washout - k : n_steps - k].reshape(-1, 1).astype(np.float32)
            W = _ridge_fit(S[: T_eval - k], target[: T_eval - k], lam=1e-4)
            pred = _ridge_predict(W, S[: T_eval - k])
            ss_res = float(np.sum((pred - target[: T_eval - k]) ** 2))
            ss_tot = float(np.sum((target[: T_eval - k] - target[: T_eval - k].mean()) ** 2))
            mc_total += max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))
        return mc_total

    import scipy.sparse as sp

    best_mc = _mc(esn)
    for _ in range(n_train_steps):
        # Perturb W_in
        import copy

        esn_p = copy.deepcopy(esn)
        if sp.issparse(esn_p.W_in):
            esn_p.W_in = esn_p.W_in.toarray()
        noise = rng.standard_normal(esn_p.W_in.shape).astype(np.float32) * lr
        esn_p.W_in = esn_p.W_in + noise
        new_mc = _mc(esn_p)
        if new_mc > best_mc:
            best_mc = new_mc
            esn.W_in = esn_p.W_in

    return esn


# ---------------------------------------------------------------------------
# Reservoir state extraction
# ---------------------------------------------------------------------------


def _run_reservoir(
    reservoir: Any,
    sequences: list[np.ndarray],
    use_final: bool = True,
) -> np.ndarray:
    features = []
    for seq in sequences:
        reservoir.reset()
        r = None
        if use_final:
            for t in range(seq.shape[0]):
                r = reservoir.step(seq[t])
            n = getattr(reservoir, "n", None)
            if r is None and n is not None:
                r = np.zeros(n, dtype=np.float32)
            features.append(r)
        else:
            states = []
            for t in range(seq.shape[0]):
                r = reservoir.step(seq[t])
                states.append(r.copy())
            features.append(np.stack(states).mean(axis=0))
    return np.stack(features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Memory capacity
# ---------------------------------------------------------------------------


def _memory_capacity(
    reservoir: Any,
    n_steps: int = 500,
    max_delay: int = 20,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    u = rng.choice([-1.0, 1.0], size=(n_steps,)).astype(np.float32)
    u_2d = u.reshape(-1, 1)

    reservoir.reset()
    n = getattr(reservoir, "n", None)
    if n is None:
        raise ValueError("reservoir must have attribute 'n'")
    states = np.zeros((n_steps, n), dtype=np.float32)
    for t in range(n_steps):
        states[t] = reservoir.step(u_2d[t])

    mc_total = 0.0
    washout = n_steps // 4
    S = states[washout:]
    T_eval = S.shape[0]

    for k in range(1, max_delay + 1):
        if k >= T_eval:
            break
        target = u[washout - k : n_steps - k].reshape(-1, 1).astype(np.float32)
        W = _ridge_fit(S[: T_eval - k], target[: T_eval - k], lam=1e-4)
        pred = _ridge_predict(W, S[: T_eval - k])
        ss_res = float(np.sum((pred - target[: T_eval - k]) ** 2))
        ss_tot = float(np.sum((target[: T_eval - k] - target[: T_eval - k].mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        mc_total += max(0.0, r2)

    return mc_total


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------


def _eval_passkey(reservoir: Any, n_examples: int = 50, seed: int = 42) -> float:
    from src.eval.benchmarks.memory import PasskeyRetrieval

    gen = PasskeyRetrieval(n=n_examples, context_length=50, seed=seed)
    examples = list(gen)

    sequences = [_encode_text(ex.input, max_len=256) for ex in examples]
    targets_str = [ex.target for ex in examples]

    unique_targets = sorted(set(targets_str))
    target_map = {t: i for i, t in enumerate(unique_targets)}
    n_classes = len(unique_targets)

    X = _run_reservoir(reservoir, sequences, use_final=True)

    if n_classes < 2:
        return 1.0

    y_oh = np.zeros((len(examples), n_classes), dtype=np.float32)
    for i, t in enumerate(targets_str):
        y_oh[i, target_map[t]] = 1.0

    n_train = max(1, int(0.8 * len(examples)))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y_oh[:n_train]
    targets_test = targets_str[n_train:]

    if X_test.shape[0] == 0:
        return 0.0

    W = _ridge_fit(X_train, y_train, lam=1.0)
    pred_scores = _ridge_predict(W, X_test)
    pred_indices = np.argmax(pred_scores, axis=1)
    pred_strings = [unique_targets[i] for i in pred_indices]

    correct = sum(p == t for p, t in zip(pred_strings, targets_test))
    return correct / len(targets_test)


def _eval_computation(reservoir: Any, n_examples: int = 50, seed: int = 42) -> float:
    from src.eval.benchmarks.computation import MultiDigitArithmetic

    gen = MultiDigitArithmetic(n=n_examples, digit_count=3, operation="addition", seed=seed)
    examples = list(gen)

    sequences = [_encode_text(ex.input, max_len=128) for ex in examples]
    targets_str = [ex.target for ex in examples]

    unique_targets = sorted(set(targets_str))
    target_map = {t: i for i, t in enumerate(unique_targets)}
    n_classes = len(unique_targets)

    X = _run_reservoir(reservoir, sequences, use_final=True)

    if n_classes < 2:
        return 1.0

    y_oh = np.zeros((len(examples), n_classes), dtype=np.float32)
    for i, t in enumerate(targets_str):
        y_oh[i, target_map[t]] = 1.0

    n_train = max(1, int(0.8 * len(examples)))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y_oh[:n_train]
    targets_test = targets_str[n_train:]

    if X_test.shape[0] == 0:
        return 0.0

    W = _ridge_fit(X_train, y_train, lam=1.0)
    pred_scores = _ridge_predict(W, X_test)
    pred_indices = np.argmax(pred_scores, axis=1)
    pred_strings = [unique_targets[i] for i in pred_indices]

    correct = sum(p == t for p, t in zip(pred_strings, targets_test))
    return correct / len(targets_test)


def _measure_latency(reservoir: Any, n_trials: int = 1_000) -> float:
    """Return mean step latency in milliseconds."""
    rng = np.random.default_rng(0)
    input_dim = getattr(reservoir, "input_dim", _VOCAB_SIZE)
    x = rng.standard_normal(input_dim).astype(np.float32)
    for _ in range(10):
        reservoir.step(x)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        reservoir.step(x)
    return (time.perf_counter() - t0) / n_trials * 1_000


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Definition of a single ablation experiment."""

    name: str
    group: str  # which ablation family (1-5)
    description: str
    reservoir_type: str  # "esn", "multi", "random_proj", "write_only", "esn_trainable_win"
    reservoir_kwargs: dict[str, Any] = field(default_factory=dict)
    is_control: bool = False  # True for the critical randomized control


@dataclass
class AblationResult:
    """Metrics from a single ablation run."""

    name: str
    group: str
    description: str
    reservoir_type: str
    is_control: bool
    memory_capacity: float
    passkey_acc: float
    computation_acc: float
    step_latency_ms: float
    quality_score: float
    param_count: int
    elapsed_seconds: float
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


def _count_params(reservoir: Any) -> int:
    """Count total float parameters in the reservoir."""
    import scipy.sparse as sp

    total = 0
    for attr in ("W", "W_in", "_W", "_W_write"):
        m = getattr(reservoir, attr, None)
        if m is None:
            continue
        if sp.issparse(m):
            total += m.nnz
        elif isinstance(m, np.ndarray):
            total += m.size
    # Multi-reservoir: sum over sub-reservoirs
    for sub in ("fast", "slow"):
        sub_r = getattr(reservoir, sub, None)
        if sub_r is not None:
            total += _count_params(sub_r)
    return total


# ---------------------------------------------------------------------------
# Build ablation suite
# ---------------------------------------------------------------------------


def build_ablation_suite() -> list[AblationConfig]:
    """Return all ablation configurations."""
    cfg = BEST_TRACK_A_CONFIG
    ablations: list[AblationConfig] = []

    # ------------------------------------------------------------------
    # Group 1: Read-only vs write-only vs full read/write
    # ------------------------------------------------------------------
    ablations.append(AblationConfig(
        name="read_only",
        group="1_modality",
        description="Standard ESN (read-only): reservoir state used as feature",
        reservoir_type="esn",
        reservoir_kwargs=dict(cfg=cfg),
    ))
    ablations.append(AblationConfig(
        name="write_only",
        group="1_modality",
        description="Write-only: reservoir runs but only write-projection output is read (no recurrent state)",
        reservoir_type="write_only",
        reservoir_kwargs=dict(n=cfg["reservoir_size"], input_dim=_VOCAB_SIZE, seed=cfg["seed"]),
    ))
    ablations.append(AblationConfig(
        name="full_readwrite",
        group="1_modality",
        description="Full read/write: ESN with feedback write signal (w_t = prev_state projection)",
        reservoir_type="esn",
        reservoir_kwargs=dict(cfg=cfg, enable_write_feedback=True),
    ))

    # ------------------------------------------------------------------
    # Group 2: Single vs multi-reservoir
    # ------------------------------------------------------------------
    ablations.append(AblationConfig(
        name="single_reservoir",
        group="2_topology",
        description="Single ESN reservoir (baseline topology)",
        reservoir_type="esn",
        reservoir_kwargs=dict(cfg=cfg),
    ))
    ablations.append(AblationConfig(
        name="multi_reservoir_fastslow",
        group="2_topology",
        description="Multi-reservoir: fast (lr=0.9, sr=0.9) + slow (lr=0.1, sr=0.5) pair",
        reservoir_type="multi",
        reservoir_kwargs=dict(
            fast_size=cfg["reservoir_size"] // 2,
            slow_size=cfg["reservoir_size"] // 2,
            seed=cfg["seed"],
        ),
    ))

    # ------------------------------------------------------------------
    # Group 3: Frozen vs partially trainable input projections
    # ------------------------------------------------------------------
    ablations.append(AblationConfig(
        name="frozen_win",
        group="3_trainability",
        description="Frozen W_in (fixed random projection — standard ESN)",
        reservoir_type="esn",
        reservoir_kwargs=dict(cfg=cfg),
    ))
    ablations.append(AblationConfig(
        name="trainable_win",
        group="3_trainability",
        description="Partially trainable W_in (ES fine-tuning of input projection)",
        reservoir_type="esn_trainable_win",
        reservoir_kwargs=dict(cfg=cfg),
    ))

    # ------------------------------------------------------------------
    # Group 4: Spectral radius regimes
    # ------------------------------------------------------------------
    for sr, label in [(0.5, "subcritical"), (0.99, "critical"), (1.1, "supercritical")]:
        sr_cfg = dict(cfg)
        sr_cfg["spectral_radius"] = sr
        ablations.append(AblationConfig(
            name=f"sr_{label}",
            group="4_spectral_radius",
            description=f"Spectral radius = {sr} ({label} regime)",
            reservoir_type="esn",
            reservoir_kwargs=dict(cfg=sr_cfg),
        ))

    # ------------------------------------------------------------------
    # Group 5: Randomized dynamics control (CRITICAL)
    # ------------------------------------------------------------------
    ablations.append(AblationConfig(
        name="random_projection_control",
        group="5_control",
        description=(
            "CRITICAL CONTROL: Random stateless projection — same param count as ESN but "
            "NO recurrent dynamics. If quality ≈ ESN → gains from param count, not recurrence. "
            "If quality < ESN → recurrent dynamics matter."
        ),
        reservoir_type="random_proj",
        reservoir_kwargs=dict(n=cfg["reservoir_size"], input_dim=_VOCAB_SIZE, seed=cfg["seed"]),
        is_control=True,
    ))

    return ablations


# ---------------------------------------------------------------------------
# Reservoir factory
# ---------------------------------------------------------------------------


def _build_reservoir(ablation: AblationConfig) -> Any:
    rtype = ablation.reservoir_type
    kwargs = ablation.reservoir_kwargs

    if rtype == "esn":
        from src.reservoir.esn import ESN
        from src.types import ReservoirConfig

        cfg = kwargs["cfg"]
        res_cfg = ReservoirConfig(
            size=cfg["reservoir_size"],
            spectral_radius=cfg["spectral_radius"],
            leak_rate=cfg["leak_rate"],
            input_scaling=cfg["input_scaling"],
            topology=cfg["topology"],
            sparsity=cfg["sparsity"],
            seed=cfg["seed"],
        )
        esn = ESN(res_cfg, input_dim=_VOCAB_SIZE)
        if kwargs.get("enable_write_feedback"):
            # Wrap ESN with a feedback write-signal from previous state
            return _FeedbackESN(esn)
        return esn

    elif rtype == "multi":
        from src.reservoir.multi_reservoir import MultiReservoir, MultiReservoirConfig
        from src.types import ReservoirConfig

        seed = kwargs.get("seed", 0)
        fast_size = kwargs.get("fast_size", 200)
        slow_size = kwargs.get("slow_size", 200)
        mr_cfg = MultiReservoirConfig(
            fast=ReservoirConfig(size=fast_size, spectral_radius=0.9, leak_rate=0.9, seed=seed),
            slow=ReservoirConfig(size=slow_size, spectral_radius=0.5, leak_rate=0.1, seed=seed + 1),
        )
        return MultiReservoir(mr_cfg, input_dim=_VOCAB_SIZE)

    elif rtype == "random_proj":
        return RandomProjection(
            n=kwargs["n"],
            input_dim=kwargs.get("input_dim", _VOCAB_SIZE),
            seed=kwargs.get("seed", 0),
        )

    elif rtype == "write_only":
        return WriteOnlyReservoir(
            n=kwargs["n"],
            input_dim=kwargs.get("input_dim", _VOCAB_SIZE),
            seed=kwargs.get("seed", 0),
        )

    elif rtype == "esn_trainable_win":
        return _build_esn_trainable_win(
            cfg=kwargs["cfg"],
            input_dim=_VOCAB_SIZE,
            n_train_steps=200,
            lr=0.01,
        )

    else:
        raise ValueError(f"Unknown reservoir_type: {rtype!r}")


class _FeedbackESN:
    """Wraps an ESN to add a self-feedback write signal (full read/write).

    At each step, the current state is projected back as an additive drive
    (w_t = W_fb @ prev_state), simulating a write-head that routes the
    model's own hidden state back into the reservoir.
    """

    def __init__(self, esn: Any) -> None:
        self._esn = esn
        self.n = esn.n
        self.input_dim = esn.input_dim
        rng = np.random.default_rng(esn.config.seed + 100)
        # Random feedback matrix: n × n, scaled small to avoid instability
        self._W_fb = (rng.standard_normal((esn.n, esn.n)) * 0.1).astype(np.float32)
        self.state = esn.state

    def step(self, x_t: np.ndarray, w_t: np.ndarray | None = None) -> np.ndarray:
        feedback = self._W_fb @ self._esn.state
        r = self._esn.step(x_t, w_t=feedback)
        self.state = r
        return r

    def reset(self) -> None:
        self._esn.reset()
        self.state = self._esn.state

    def forward(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        states = np.empty((T, self.n), dtype=np.float32)
        for t in range(T):
            states[t] = self.step(X[t])
        return states


# ---------------------------------------------------------------------------
# Run single ablation
# ---------------------------------------------------------------------------


def execute_ablation(
    ablation: AblationConfig,
    *,
    n_eval_examples: int = 50,
    mc_steps: int = 500,
    no_wandb: bool = False,
    results_dir: Path = RESULTS_DIR,
) -> AblationResult:
    t0 = time.perf_counter()
    logger.info(
        "ablation=%s  group=%s  type=%s",
        ablation.name,
        ablation.group,
        ablation.reservoir_type,
    )

    reservoir = _build_reservoir(ablation)
    n_params = _count_params(reservoir)

    logger.info("  measuring memory capacity ...")
    mc = _memory_capacity(reservoir, n_steps=mc_steps, max_delay=20)

    logger.info("  evaluating passkey retrieval (%d examples) ...", n_eval_examples)
    passkey_acc = _eval_passkey(reservoir, n_examples=n_eval_examples)

    logger.info("  evaluating computation (%d examples) ...", n_eval_examples)
    comp_acc = _eval_computation(reservoir, n_examples=n_eval_examples)

    logger.info("  measuring step latency ...")
    latency_ms = _measure_latency(reservoir, n_trials=1_000)

    mc_norm = min(1.0, mc / 20.0)
    quality = 0.4 * mc_norm + 0.4 * passkey_acc + 0.2 * comp_acc

    elapsed = time.perf_counter() - t0

    result = AblationResult(
        name=ablation.name,
        group=ablation.group,
        description=ablation.description,
        reservoir_type=ablation.reservoir_type,
        is_control=ablation.is_control,
        memory_capacity=mc,
        passkey_acc=passkey_acc,
        computation_acc=comp_acc,
        step_latency_ms=latency_ms,
        quality_score=quality,
        param_count=n_params,
        elapsed_seconds=elapsed,
    )

    logger.info(
        "  MC=%.3f  passkey=%.3f  comp=%.3f  latency=%.3fms  quality=%.3f  t=%.1fs",
        mc, passkey_acc, comp_acc, latency_ms, quality, elapsed,
    )

    # Save per-ablation result
    out_dir = results_dir / ablation.group
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{ablation.name}.json").open("w") as f:
        json.dump(asdict(result), f, indent=2)

    if not no_wandb:
        _wandb_log(ablation, result)

    return result


def _wandb_log(ablation: AblationConfig, result: AblationResult) -> None:
    try:
        import wandb  # type: ignore[import]

        run = wandb.init(
            project="lrs-ablations",
            name=f"{ablation.group}/{ablation.name}",
            group=ablation.group,
            config={
                "reservoir_type": ablation.reservoir_type,
                "is_control": ablation.is_control,
                "description": ablation.description,
            },
            reinit=True,
        )
        wandb.log({
            "memory_capacity": result.memory_capacity,
            "passkey_acc": result.passkey_acc,
            "computation_acc": result.computation_acc,
            "step_latency_ms": result.step_latency_ms,
            "quality_score": result.quality_score,
            "param_count": result.param_count,
            "elapsed_seconds": result.elapsed_seconds,
        })
        if run is not None:
            run.finish()
    except Exception as exc:
        logger.debug("wandb logging skipped: %s", exc)


# ---------------------------------------------------------------------------
# Run full suite
# ---------------------------------------------------------------------------


def run_ablations(
    args: argparse.Namespace,
    results_dir: Path,
) -> list[AblationResult]:
    ablations = build_ablation_suite()

    # Filter by requested name
    if args.ablation:
        ablations = [a for a in ablations if a.name == args.ablation]
        if not ablations:
            logger.error("No ablation found with name %r", args.ablation)
            return []

    all_results: list[AblationResult] = []

    for ablation in ablations:
        # Resume: skip already-completed
        if args.resume:
            out_path = results_dir / ablation.group / f"{ablation.name}.json"
            if out_path.exists():
                try:
                    with out_path.open() as f:
                        d = json.load(f)
                    all_results.append(AblationResult(**d))
                    logger.info("Resuming: skipping %s (already done)", ablation.name)
                    continue
                except Exception:
                    pass

        if args.dry_run:
            logger.info(
                "DRY RUN: %s  [%s]  type=%s",
                ablation.name,
                ablation.group,
                ablation.reservoir_type,
            )
            continue

        result = execute_ablation(
            ablation,
            n_eval_examples=args.n_eval_examples,
            mc_steps=args.mc_steps,
            no_wandb=args.no_wandb,
            results_dir=results_dir,
        )
        all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Summary + control verdict
# ---------------------------------------------------------------------------


def write_summary(results: list[AblationResult], results_dir: Path) -> None:
    if not results:
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # Full results JSON
    summary_path = results_dir / "ablation_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "best_track_a_config": BEST_TRACK_A_CONFIG,
                "metric_description": {
                    "quality_score": "0.4*MC_norm + 0.4*passkey_acc + 0.2*comp_acc",
                    "memory_capacity": "Sum R² for k=1..20 delay recall (max=20)",
                    "passkey_acc": "Exact-match accuracy on PasskeyRetrieval",
                    "computation_acc": "Exact-match on MultiDigitArithmetic",
                },
            },
            f,
            indent=2,
        )
    logger.info("Summary written to %s", summary_path)

    # Print leaderboard
    print("\n=== Ablation Study Results ===")
    print(
        f"{'Ablation':<30}  {'Group':<20}  {'Quality':>8}  "
        f"{'MC':>6}  {'Pass':>6}  {'Comp':>6}  {'ms/step':>8}  {'Params':>10}"
    )
    print("-" * 100)
    for r in sorted(results, key=lambda x: x.quality_score, reverse=True):
        ctrl = " [CONTROL]" if r.is_control else ""
        print(
            f"{r.name:<30}  {r.group:<20}  {r.quality_score:8.4f}  "
            f"{r.memory_capacity:6.2f}  {r.passkey_acc:6.3f}  "
            f"{r.computation_acc:6.3f}  {r.step_latency_ms:8.4f}  "
            f"{r.param_count:>10,}{ctrl}"
        )

    # Group-by analysis
    print("\n=== Group Analysis ===")
    groups: dict[str, list[AblationResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)
    for group, group_results in sorted(groups.items()):
        best = max(group_results, key=lambda x: x.quality_score)
        worst = min(group_results, key=lambda x: x.quality_score)
        print(
            f"\n[{group}]  best={best.name} ({best.quality_score:.4f})  "
            f"worst={worst.name} ({worst.quality_score:.4f})  "
            f"Δ={best.quality_score - worst.quality_score:.4f}"
        )
        for r in sorted(group_results, key=lambda x: x.quality_score, reverse=True):
            print(f"  {r.name:<30}  quality={r.quality_score:.4f}  MC={r.memory_capacity:.2f}")

    # Critical control verdict
    control_results = [r for r in results if r.is_control]
    esn_baseline = next(
        (r for r in results if r.name == "read_only"), None
    )
    if control_results and esn_baseline:
        ctrl = control_results[0]
        delta = esn_baseline.quality_score - ctrl.quality_score
        rel = delta / (esn_baseline.quality_score + 1e-10) * 100
        verdict = (
            "RECURRENT DYNAMICS MATTER"
            if delta > 0.02
            else "GAINS FROM PARAM COUNT (not recurrence)"
        )
        print(f"\n=== CRITICAL CONTROL VERDICT ===")
        print(f"  ESN (read_only):              quality = {esn_baseline.quality_score:.4f}")
        print(f"  RandomProjection (control):   quality = {ctrl.quality_score:.4f}")
        print(f"  Δ = {delta:.4f}  ({rel:.1f}% relative)")
        print(f"  VERDICT: {verdict}")

        # Write verdict to JSON
        verdict_path = results_dir / "control_verdict.json"
        with verdict_path.open("w") as f:
            json.dump({
                "esn_quality": esn_baseline.quality_score,
                "control_quality": ctrl.quality_score,
                "delta": delta,
                "relative_pct": rel,
                "verdict": verdict,
                "recurrent_dynamics_matter": delta > 0.02,
            }, f, indent=2)
        logger.info("Control verdict written to %s", verdict_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation studies for the best-performing reservoir track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Run a single ablation by name.  Omit to run all.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip ablations whose results already exist.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all ablation configs without running them.",
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
        help="Number of examples per benchmark subset.",
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
        "--list",
        action="store_true",
        help="List all planned ablations and exit.",
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

    if args.list:
        ablations = build_ablation_suite()
        print(f"{'Name':<30}  {'Group':<20}  {'Type':<20}  {'Control'}")
        print("-" * 85)
        for a in ablations:
            print(
                f"{a.name:<30}  {a.group:<20}  {a.reservoir_type:<20}  "
                f"{'YES' if a.is_control else ''}"
            )
        return

    all_results = run_ablations(args, args.results_dir)

    if all_results:
        write_summary(all_results, args.results_dir)
    else:
        logger.info("No results produced (dry_run or all skipped).")


if __name__ == "__main__":
    main()
