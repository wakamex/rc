"""Tests for the multi-reservoir module (rc-wwh.21)."""
from __future__ import annotations

import numpy as np
import pytest

from src.reservoir.multi_reservoir import MultiReservoir, MultiReservoirConfig
from src.types import Reservoir, ReservoirConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def autocorr_decay(states: np.ndarray, max_lag: int = 50) -> float:
    """Return the mean lag at which the mean autocorrelation first drops below 0.5.

    A shorter lag indicates faster decay (fast reservoir).
    Computes per-neuron autocorrelation and averages across neurons.

    Args:
        states: State trajectory, shape ``(T, n)``.
        max_lag: Maximum lag to examine.

    Returns:
        The first lag at which the mean autocorrelation falls below 0.5,
        or ``max_lag`` if it never does.
    """
    T, n = states.shape
    mean = states.mean(axis=0)
    centered = states - mean
    var = (centered ** 2).mean(axis=0)
    var = np.where(var < 1e-12, 1.0, var)  # avoid division by zero

    for lag in range(1, max_lag + 1):
        cov = (centered[:T - lag] * centered[lag:]).mean(axis=0)
        ac = (cov / var).mean()
        if ac < 0.5:
            return float(lag)
    return float(max_lag)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mr_default() -> MultiReservoir:
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=100, spectral_radius=0.9, leak_rate=0.9, seed=10),
        slow=ReservoirConfig(size=100, spectral_radius=0.5, leak_rate=0.1, seed=20),
    )
    return MultiReservoir(cfg, input_dim=5)


# ---------------------------------------------------------------------------
# Timescale tests
# ---------------------------------------------------------------------------


def test_fast_autocorrelation_decays_faster():
    """Fast reservoir autocorrelation should decay faster than slow reservoir."""
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=100, spectral_radius=0.9, leak_rate=0.9, seed=1),
        slow=ReservoirConfig(size=100, spectral_radius=0.5, leak_rate=0.1, seed=2),
    )
    mr = MultiReservoir(cfg, input_dim=4)

    rng = np.random.default_rng(42)
    T = 500
    X = rng.standard_normal((T, 4)).astype(np.float32)

    fast_states = np.empty((T, mr.fast_dim), dtype=np.float32)
    slow_states = np.empty((T, mr.slow_dim), dtype=np.float32)

    for t in range(T):
        mr.step(X[t])
        fast_states[t] = mr.fast.state.copy()
        slow_states[t] = mr.slow.state.copy()

    fast_lag = autocorr_decay(fast_states)
    slow_lag = autocorr_decay(slow_states)

    assert fast_lag < slow_lag, (
        f"Expected fast reservoir to decay before slow reservoir: "
        f"fast_lag={fast_lag}, slow_lag={slow_lag}"
    )


# ---------------------------------------------------------------------------
# Read interface / shape tests
# ---------------------------------------------------------------------------


def test_read_state_shape(mr_default: MultiReservoir):
    rng = np.random.default_rng(0)
    mr_default.step(rng.standard_normal(5))
    state = mr_default.read()
    expected = mr_default.fast_dim + mr_default.slow_dim
    assert state.shape == (expected,), f"Expected ({expected},), got {state.shape}"


def test_state_dim_property(mr_default: MultiReservoir):
    assert mr_default.state_dim == mr_default.fast_dim + mr_default.slow_dim


def test_step_returns_concatenated_state(mr_default: MultiReservoir):
    rng = np.random.default_rng(1)
    result = mr_default.step(rng.standard_normal(5))
    assert result.shape == (mr_default.state_dim,)


def test_forward_sequence_shape():
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=30, spectral_radius=0.9, leak_rate=0.8, seed=5),
        slow=ReservoirConfig(size=50, spectral_radius=0.5, leak_rate=0.2, seed=6),
    )
    mr = MultiReservoir(cfg, input_dim=3)
    X = np.random.default_rng(7).standard_normal((20, 3)).astype(np.float32)
    states = mr.forward(X)
    assert states.shape == (20, 80), f"Expected (20, 80), got {states.shape}"


def test_asymmetric_reservoir_sizes():
    """fast_dim and slow_dim can differ; read() must concatenate correctly."""
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=64, spectral_radius=0.95, leak_rate=0.85, seed=3),
        slow=ReservoirConfig(size=128, spectral_radius=0.4, leak_rate=0.15, seed=4),
    )
    mr = MultiReservoir(cfg, input_dim=8)
    assert mr.fast_dim == 64
    assert mr.slow_dim == 128
    assert mr.state_dim == 192

    mr.step(np.zeros(8, dtype=np.float32))
    state = mr.read()
    assert state.shape == (192,)


# ---------------------------------------------------------------------------
# Step and reset correctness
# ---------------------------------------------------------------------------


def test_reset_zeroes_both_reservoirs(mr_default: MultiReservoir):
    rng = np.random.default_rng(9)
    for _ in range(20):
        mr_default.step(rng.standard_normal(5))

    mr_default.reset()
    assert np.allclose(mr_default.fast.state, 0.0), "fast reservoir state not zeroed"
    assert np.allclose(mr_default.slow.state, 0.0), "slow reservoir state not zeroed"


def test_step_modifies_state(mr_default: MultiReservoir):
    mr_default.reset()
    x = np.ones(5, dtype=np.float32)
    mr_default.step(x)
    assert not np.allclose(mr_default.fast.state, 0.0), "fast state unchanged after step"
    assert not np.allclose(mr_default.slow.state, 0.0), "slow state unchanged after step"


# ---------------------------------------------------------------------------
# Independent state tests
# ---------------------------------------------------------------------------


def test_reservoirs_maintain_independent_state():
    """Fast and slow states must be different after the same input sequence."""
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=50, spectral_radius=0.9, leak_rate=0.9, seed=11),
        slow=ReservoirConfig(size=50, spectral_radius=0.5, leak_rate=0.1, seed=12),
    )
    mr = MultiReservoir(cfg, input_dim=3)

    rng = np.random.default_rng(42)
    for _ in range(100):
        mr.step(rng.standard_normal(3))

    # States should differ because the two ESNs have different weights and dynamics.
    assert not np.allclose(mr.fast.state, mr.slow.state), (
        "Fast and slow reservoir states should differ after processing the same input"
    )


def test_independent_write_heads():
    """Writing to fast reservoir only must not change slow reservoir state."""
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=50, spectral_radius=0.9, leak_rate=0.9, seed=13),
        slow=ReservoirConfig(size=50, spectral_radius=0.5, leak_rate=0.1, seed=14),
        shared_write_heads=False,
    )
    mr_a = MultiReservoir(cfg, input_dim=3)
    mr_b = MultiReservoir(cfg, input_dim=3)

    x = np.zeros(3, dtype=np.float32)
    w = np.ones(50, dtype=np.float32) * 2.0

    # mr_a: write to fast only; mr_b: no write
    mr_a.step(x, w_fast=w, w_slow=None)
    mr_b.step(x, w_fast=None, w_slow=None)

    assert not np.allclose(mr_a.fast.state, mr_b.fast.state), (
        "w_fast had no effect on fast reservoir"
    )
    assert np.allclose(mr_a.slow.state, mr_b.slow.state), (
        "w_fast leaked into slow reservoir (should not)"
    )


def test_shared_write_head_broadcasts():
    """With shared_write_heads=True, w_fast is applied to both reservoirs."""
    cfg = MultiReservoirConfig(
        fast=ReservoirConfig(size=50, spectral_radius=0.9, leak_rate=0.9, seed=15),
        slow=ReservoirConfig(size=50, spectral_radius=0.5, leak_rate=0.1, seed=16),
        shared_write_heads=True,
    )
    mr_with = MultiReservoir(cfg, input_dim=3)

    cfg_no_share = MultiReservoirConfig(
        fast=ReservoirConfig(size=50, spectral_radius=0.9, leak_rate=0.9, seed=15),
        slow=ReservoirConfig(size=50, spectral_radius=0.5, leak_rate=0.1, seed=16),
        shared_write_heads=False,
    )
    mr_without = MultiReservoir(cfg_no_share, input_dim=3)

    x = np.zeros(3, dtype=np.float32)
    w = np.ones(50, dtype=np.float32) * 3.0

    mr_with.step(x, w_fast=w)
    mr_without.step(x, w_fast=None, w_slow=None)

    # The shared write head should cause the slow reservoir state to differ from
    # the no-write case.
    assert not np.allclose(mr_with.slow.state, mr_without.slow.state), (
        "shared_write_heads=True did not broadcast w_fast to slow reservoir"
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_reservoir_protocol(mr_default: MultiReservoir):
    assert isinstance(mr_default, Reservoir)
