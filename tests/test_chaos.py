"""Tests for src/data/chaos.py — chaotic systems data pipeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.data.chaos import (
    generate_trajectory,
    lyapunov_time,
    max_lyapunov_exponent,
    normalize,
    split_trajectory,
)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def test_lorenz63_shape():
    traj = generate_trajectory("lorenz63", T=10.0, dt=0.02, transient=10.0)
    assert traj.shape == (500, 3)


def test_mackey_glass_shape():
    traj = generate_trajectory("mackey_glass", T=10.0, dt=0.02, transient=10.0)
    assert traj.shape == (500, 1)


def test_ks_shape():
    traj = generate_trajectory("ks", params={"N": 32, "L": 22.0}, T=5.0, dt=0.05, transient=50.0)
    assert traj.shape == (100, 32)


def test_unknown_system_raises():
    with pytest.raises(ValueError, match="Unknown system"):
        generate_trajectory("bogus_system")


# ---------------------------------------------------------------------------
# Splits: non-overlapping and correct sizes
# ---------------------------------------------------------------------------


def test_splits_non_overlapping():
    traj = generate_trajectory("lorenz63", T=20.0, dt=0.02, transient=10.0)
    splits = split_trajectory(traj, train=0.7, val=0.15, test=0.15)

    n = len(traj)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    assert len(splits["train"]) == n_train
    assert len(splits["val"]) == n_val
    # test gets the remainder
    assert len(splits["test"]) == n - n_train - n_val

    # Verify temporal ordering (no overlap): last train index < first val index
    assert np.allclose(splits["train"][-1], traj[n_train - 1])
    assert np.allclose(splits["val"][0], traj[n_train])


def test_splits_cover_all_data():
    traj = generate_trajectory("lorenz63", T=10.0, dt=0.02, transient=10.0)
    splits = split_trajectory(traj)
    total = sum(len(v) for v in splits.values())
    assert total == len(traj)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalisation_train_zero_mean_unit_std():
    traj = generate_trajectory("lorenz63", T=50.0, dt=0.02, transient=10.0)
    splits = split_trajectory(traj)
    norm_splits, mu, sigma = normalize(splits)
    train_norm = norm_splits["train"]
    np.testing.assert_allclose(train_norm.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(train_norm.std(axis=0), 1.0, atol=1e-4)


def test_normalisation_returns_stats():
    traj = generate_trajectory("lorenz63", T=10.0, dt=0.02, transient=10.0)
    splits = split_trajectory(traj)
    _, mu, sigma = normalize(splits)
    assert mu.shape == (3,)
    assert sigma.shape == (3,)
    assert np.all(sigma > 0)


# ---------------------------------------------------------------------------
# Lyapunov exponents
# ---------------------------------------------------------------------------


def test_lorenz63_lyapunov_exponent():
    """Lorenz-63 max LE should be ≈ 0.906."""
    le = max_lyapunov_exponent(
        "lorenz63",
        params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        dt=0.01,
        T_transient=100.0,
        T_compute=500.0,
        renorm_interval=10,
    )
    # Published value ≈ 0.906; allow ±0.15 tolerance for finite-time estimate
    assert 0.70 < le < 1.10, f"Lorenz LE = {le:.4f}, expected ~0.906"


def test_lyapunov_time_positive():
    lt = lyapunov_time("lorenz63")
    assert lt > 0


def test_mackey_glass_chaotic_tau17():
    """At τ=17, Mackey-Glass should be chaotic (positive LE)."""
    le = max_lyapunov_exponent(
        "mackey_glass",
        params={"tau": 17.0},
        dt=0.1,
        T_transient=200.0,
        T_compute=300.0,
    )
    assert le > 0, f"MG τ=17 should be chaotic, got LE={le:.4f}"


def test_mackey_glass_non_chaotic_tau10():
    """At τ=10, Mackey-Glass is periodic (LE ≤ 0)."""
    le = max_lyapunov_exponent(
        "mackey_glass",
        params={"tau": 10.0},
        dt=0.1,
        T_transient=200.0,
        T_compute=300.0,
    )
    assert le < 0.05, f"MG τ=10 should be near-zero or negative LE, got {le:.4f}"


def test_ks_chaotic():
    """KS system should have positive LE."""
    le = max_lyapunov_exponent(
        "ks",
        params={"L": 22.0, "N": 64},
        dt=0.05,
        T_transient=100.0,
        T_compute=200.0,
    )
    assert le > 0, f"KS should be chaotic, got LE={le:.4f}"


# ---------------------------------------------------------------------------
# Performance: data loads in <1s
# ---------------------------------------------------------------------------


def test_lorenz63_loads_fast():
    t0 = time.perf_counter()
    traj = generate_trajectory("lorenz63", T=50.0, dt=0.02, transient=10.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"Lorenz generation took {elapsed:.2f}s, expected <1s"
    assert len(traj) > 0


def test_mackey_glass_loads_fast():
    t0 = time.perf_counter()
    traj = generate_trajectory("mackey_glass", T=50.0, dt=0.1, transient=100.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"MG generation took {elapsed:.2f}s, expected <1s"
    assert len(traj) > 0
