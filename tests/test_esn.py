"""Tests for the ESN reservoir (rc-wwh.2)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.reservoir.esn import ESN
from src.types import Reservoir, ReservoirConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def esn_50k() -> ESN:
    """ESN with 50K nodes for performance testing (built once per module)."""
    cfg = ReservoirConfig(
        size=50_000,
        spectral_radius=0.9,
        leak_rate=0.3,
        sparsity=0.0001,  # 0.01% → ~250K non-zeros
        seed=0,
    )
    return ESN(cfg, input_dim=10)


# ---------------------------------------------------------------------------
# Echo state / fading memory
# ---------------------------------------------------------------------------


def test_echo_state_property():
    """States from different initial conditions converge under the same input."""
    cfg = ReservoirConfig(size=200, spectral_radius=0.9, leak_rate=0.5, seed=42)
    esn1 = ESN(cfg, input_dim=5)
    esn2 = ESN(cfg, input_dim=5)

    # Different random initial states
    rng0 = np.random.default_rng(0)
    esn1.state = rng0.standard_normal(200)
    esn2.state = rng0.standard_normal(200)

    rng = np.random.default_rng(999)
    inputs = rng.standard_normal((500, 5))

    r1, r2 = None, None
    for x in inputs:
        r1 = esn1.step(x)
        r2 = esn2.step(x)

    assert r1 is not None and r2 is not None
    diff = float(np.linalg.norm(r1 - r2))
    assert diff < 1.0, f"Echo state property violated: ||r1 - r2|| = {diff:.4f}"


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_step_performance(esn_50k: ESN):
    """Each step must complete in <1 ms at 50 K nodes on CPU."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(10)

    # Warm up JIT / OS cache
    for _ in range(5):
        esn_50k.step(x)

    n_trials = 50
    t0 = time.perf_counter()
    for _ in range(n_trials):
        esn_50k.step(x)
    elapsed = (time.perf_counter() - t0) / n_trials

    assert elapsed < 1e-3, f"step() took {elapsed * 1e3:.2f} ms (limit 1 ms)"


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        ReservoirConfig(size=100, spectral_radius=0.9, leak_rate=0.3, seed=42),
        ReservoirConfig(size=100, spectral_radius=0.5, leak_rate=0.1, seed=43),
        ReservoirConfig(size=100, spectral_radius=0.99, leak_rate=0.9, seed=44),
        ReservoirConfig(size=100, spectral_radius=0.7, leak_rate=0.5, topology="small_world", seed=45),
    ],
)
def test_state_norm_bounded(cfg: ReservoirConfig):
    """State norm must not explode over 10 K steps for stable configs (sr < 1)."""
    esn = ESN(cfg, input_dim=3)
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((10_000, 3))

    for x in inputs:
        r = esn.step(x)

    norm = float(np.linalg.norm(r))
    assert norm < 1e4, f"State norm exploded: {norm:.2e} for {cfg}"


# ---------------------------------------------------------------------------
# API correctness
# ---------------------------------------------------------------------------


def test_reset_zeroes_state():
    cfg = ReservoirConfig(size=100, seed=42)
    esn = ESN(cfg, input_dim=3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        esn.step(rng.standard_normal(3))
    esn.reset()
    assert np.allclose(esn.state, 0.0), "reset() did not zero the state"


def test_step_returns_correct_shape():
    cfg = ReservoirConfig(size=50, seed=0)
    esn = ESN(cfg, input_dim=4)
    r = esn.step(np.zeros(4))
    assert r.shape == (50,)


def test_batched_step_shape():
    cfg = ReservoirConfig(size=50, seed=0)
    esn = ESN(cfg, input_dim=4)
    X = np.random.default_rng(1).standard_normal((8, 4))
    R = esn.step(X)
    assert R.shape == (8, 50)


def test_w_t_additive_drive():
    """Passing w_t changes the state compared to not passing it."""
    cfg = ReservoirConfig(size=50, seed=7)
    esn_a = ESN(cfg, input_dim=2)
    esn_b = ESN(cfg, input_dim=2)
    x = np.ones(2)
    w = np.ones(50) * 5.0
    r_a = esn_a.step(x, w_t=None)
    r_b = esn_b.step(x, w_t=w)
    assert not np.allclose(r_a, r_b), "w_t additive drive had no effect"


def test_forward_sequence_shape():
    cfg = ReservoirConfig(size=30, seed=0)
    esn = ESN(cfg, input_dim=3)
    X = np.random.default_rng(2).standard_normal((20, 3))
    states = esn.forward(X)
    assert states.shape == (20, 30)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_erdos_renyi_sparsity():
    cfg = ReservoirConfig(size=300, topology="erdos_renyi", sparsity=0.05, seed=1)
    esn = ESN(cfg, input_dim=2)
    actual = esn.W.nnz / (300 * 300)
    # Binomial variance: expect within 3 std of target
    assert 0.02 < actual < 0.10, f"Density {actual:.4f} far from target 0.05"


def test_small_world_topology():
    cfg = ReservoirConfig(size=200, topology="small_world", sparsity=0.05, seed=2)
    esn = ESN(cfg, input_dim=2)
    assert esn.W.nnz > 0
    r = esn.step(np.ones(2))
    assert r.shape == (200,)


def test_unknown_topology_raises():
    cfg = ReservoirConfig(size=10, topology="invalid", seed=0)
    with pytest.raises(ValueError, match="Unknown topology"):
        ESN(cfg, input_dim=1)


# ---------------------------------------------------------------------------
# Spectral radius
# ---------------------------------------------------------------------------


def test_spectral_radius_after_rescale():
    """Actual spectral radius should be close to the configured value."""
    cfg = ReservoirConfig(size=200, spectral_radius=0.7, sparsity=0.05, seed=5)
    esn = ESN(cfg, input_dim=2)
    # Dense eigendecomposition for accuracy (small matrix)
    eigs = np.linalg.eigvals(esn.W.toarray())
    actual_sr = float(np.max(np.abs(eigs)))
    assert abs(actual_sr - 0.7) < 0.05, f"Spectral radius {actual_sr:.4f}, expected ~0.7"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_reservoir_protocol():
    cfg = ReservoirConfig(size=50, seed=0)
    esn = ESN(cfg, input_dim=3)
    assert isinstance(esn, Reservoir)
