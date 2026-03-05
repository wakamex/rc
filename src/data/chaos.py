"""Chaotic dynamical systems data pipeline.

Implements trajectory generation for:
- Lorenz-63 (ODE)
- Mackey-Glass (delay-differential equation)
- Kuramoto-Sivashinsky (PDE, spatial discretisation via Lawson-RK4)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Lorenz-63
# ---------------------------------------------------------------------------


def _lorenz63_step(
    state: NDArray[np.float64], dt: float, sigma: float, rho: float, beta: float
) -> NDArray[np.float64]:
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    k1 = np.array([dx, dy, dz])

    x2, y2, z2 = state + 0.5 * dt * k1
    k2 = np.array([sigma * (y2 - x2), x2 * (rho - z2) - y2, x2 * y2 - beta * z2])

    x3, y3, z3 = state + 0.5 * dt * k2
    k3 = np.array([sigma * (y3 - x3), x3 * (rho - z3) - y3, x3 * y3 - beta * z3])

    x4, y4, z4 = state + dt * k3
    k4 = np.array([sigma * (y4 - x4), x4 * (rho - z4) - y4, x4 * y4 - beta * z4])

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _generate_lorenz63(
    params: dict[str, Any], T: float, dt: float, transient: float
) -> NDArray[np.float64]:
    sigma = float(params.get("sigma", 10.0))
    rho = float(params.get("rho", 28.0))
    beta = float(params.get("beta", 8.0 / 3.0))
    seed = params.get("seed", 42)

    rng = np.random.default_rng(seed)
    state = rng.standard_normal(3)

    n_trans = int(transient / dt)
    for _ in range(n_trans):
        state = _lorenz63_step(state, dt, sigma, rho, beta)

    n_steps = int(T / dt)
    traj = np.empty((n_steps, 3), dtype=np.float64)
    for i in range(n_steps):
        traj[i] = state
        state = _lorenz63_step(state, dt, sigma, rho, beta)

    return traj


# ---------------------------------------------------------------------------
# Mackey-Glass DDE
# ---------------------------------------------------------------------------


def _mg_euler_step(x: float, x_tau: float, beta: float, gamma: float, n: float, dt: float) -> float:
    dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x
    return x + dt * dx


def _generate_mackey_glass(
    params: dict[str, Any], T: float, dt: float, transient: float
) -> NDArray[np.float64]:
    tau = float(params.get("tau", 17.0))
    beta_mg = float(params.get("beta", 0.2))
    gamma = float(params.get("gamma", 0.1))
    n_mg = float(params.get("n", 10.0))
    seed = params.get("seed", 42)

    rng = np.random.default_rng(seed)
    n_delay = max(1, int(tau / dt))
    total_steps = int((transient + T) / dt) + n_delay + 2

    hist = np.full(total_steps, float(rng.uniform(0.5, 1.5)))

    for i in range(n_delay, total_steps - 1):
        hist[i + 1] = _mg_euler_step(hist[i], hist[i - n_delay], beta_mg, gamma, n_mg, dt)

    n_steps = int(T / dt)
    start = n_delay + int(transient / dt)
    traj_1d = hist[start : start + n_steps]
    return traj_1d.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Kuramoto-Sivashinsky PDE (Lawson-RK4 in Fourier space)
# ---------------------------------------------------------------------------

def _ks_nonlin_hat(u_hat: NDArray, k: NDArray, N: int) -> NDArray:
    """Nonlinear part: F(-0.5 * d(u^2)/dx) = -0.5j * k * F(u^2), with 2/3 dealiasing."""
    u = np.fft.irfft(u_hat, n=N)
    u2_hat = np.fft.rfft(u**2)
    # 2/3 dealiasing: zero out top 1/3 of modes
    cutoff = len(k) * 2 // 3
    u2_hat[cutoff:] = 0.0
    return -0.5j * k * u2_hat


def _ks_lawson_step(
    u_hat: NDArray,
    dt: float,
    E: NDArray,
    E2: NDArray,
    k: NDArray,
    N: int,
) -> NDArray:
    """One Lawson-RK4 step for KS in Fourier space.

    Lawson-RK4: u_{n+1} = E*u_n + (h/6)*(E*k1 + 2*E2*k2 + 2*E2*k3 + k4)
    where ki are the nonlinear contributions only.
    """
    k1 = _ks_nonlin_hat(u_hat, k, N)
    a_hat = E2 * (u_hat + (dt / 2) * k1)
    k2 = _ks_nonlin_hat(a_hat, k, N)
    b_hat = E2 * (u_hat + (dt / 2) * k2)
    k3 = _ks_nonlin_hat(b_hat, k, N)
    c_hat = E * (u_hat + dt * k3)
    k4 = _ks_nonlin_hat(c_hat, k, N)
    return E * u_hat + (dt / 6.0) * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)


def _ks_precompute(N: int, L: float, dt: float) -> tuple[NDArray, NDArray, NDArray]:
    """Precompute wave numbers and matrix exponentials for KS Lawson-RK4."""
    k = (2 * np.pi / L) * np.fft.rfftfreq(N, d=1.0 / N)
    lin = k**2 - k**4  # linear operator
    E = np.exp(lin * dt)
    E2 = np.exp(lin * dt / 2)
    return k, E, E2


def _generate_ks(
    params: dict[str, Any], T: float, dt: float, transient: float
) -> NDArray[np.float64]:
    L_domain = float(params.get("L", 22.0))
    N = int(params.get("N", 64))
    seed = params.get("seed", 42)

    rng = np.random.default_rng(seed)
    u = rng.standard_normal(N) * 0.1

    k, E, E2 = _ks_precompute(N, L_domain, dt)
    u_hat = np.fft.rfft(u)

    n_trans = int(transient / dt)
    for _ in range(n_trans):
        u_hat = _ks_lawson_step(u_hat, dt, E, E2, k, N)

    n_steps = int(T / dt)
    traj = np.empty((n_steps, N), dtype=np.float64)
    for i in range(n_steps):
        traj[i] = np.fft.irfft(u_hat, n=N)
        u_hat = _ks_lawson_step(u_hat, dt, E, E2, k, N)

    return traj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "lorenz63": _generate_lorenz63,
    "mackey_glass": _generate_mackey_glass,
    "ks": _generate_ks,
}


def generate_trajectory(
    system: str,
    params: dict[str, Any] | None = None,
    T: float = 100.0,
    dt: float = 0.02,
    transient: float = 100.0,
) -> NDArray[np.float64]:
    """Generate a trajectory for a chaotic dynamical system.

    Parameters
    ----------
    system:
        One of ``"lorenz63"``, ``"mackey_glass"``, ``"ks"``.
    params:
        System-specific parameters.  Missing keys use defaults.
    T:
        Total integration time *after* the transient.
    dt:
        Time step.
    transient:
        Duration of initial transient to discard.

    Returns
    -------
    NDArray of shape ``(n_steps, d)`` where ``n_steps = int(T / dt)``.
    """
    if system not in _SYSTEMS:
        raise ValueError(f"Unknown system '{system}'. Choose from {list(_SYSTEMS)}")
    params = params or {}
    return _SYSTEMS[system](params, T, dt, transient)


def split_trajectory(
    traj: NDArray[np.float64],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
) -> dict[str, NDArray[np.float64]]:
    """Temporal (non-overlapping) train/val/test split.

    Parameters
    ----------
    traj:
        Array of shape ``(T, d)``.
    train, val, test:
        Fractional sizes; must sum to 1.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    assert abs(train + val + test - 1.0) < 1e-9, "Fractions must sum to 1"
    n = len(traj)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": traj[:n_train],
        "val": traj[n_train : n_train + n_val],
        "test": traj[n_train + n_val :],
    }


def normalize(
    splits: dict[str, NDArray[np.float64]],
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """Per-variable z-score normalisation fitted on the train split.

    Returns
    -------
    normalised_splits, mean, std
    """
    train = splits["train"]
    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    normalised = {k: (v - mu) / sigma for k, v in splits.items()}
    return normalised, mu.squeeze(0), sigma.squeeze(0)


# ---------------------------------------------------------------------------
# Lyapunov exponent estimation
# ---------------------------------------------------------------------------


def lyapunov_time(
    system: str,
    params: dict[str, Any] | None = None,
    dt: float = 0.02,
) -> float:
    """Return 1/λ_max (Lyapunov time) for the given system."""
    params = params or {}
    le = max_lyapunov_exponent(system, params, dt=dt)
    if le <= 0:
        return float("inf")
    return 1.0 / le


def max_lyapunov_exponent(
    system: str,
    params: dict[str, Any] | None = None,
    dt: float = 0.02,
    T_transient: float = 100.0,
    T_compute: float = 200.0,
    renorm_interval: int = 10,
) -> float:
    """Estimate the maximal Lyapunov exponent via the standard perturbation method.

    Returns λ_max in units of 1/time.
    """
    params = params or {}

    if system == "mackey_glass":
        return _le_mackey_glass(params, dt, T_transient, T_compute, renorm_interval)

    # --- Generic perturbation method for ODE/PDE systems ---
    # Warm-up: get a point on the attractor
    traj = generate_trajectory(system, params, T=dt * 100, dt=dt, transient=T_transient)
    state = traj[-1]

    eps = 1e-8
    rng = np.random.default_rng(0)
    perturb = rng.standard_normal(state.shape)
    perturb = perturb / np.linalg.norm(perturb) * eps

    n_steps = int(T_compute / dt)
    log_growth = 0.0
    count = 0

    # Pre-compute KS operators once
    ks_ops: tuple | None = None
    if system == "ks":
        L_domain = float(params.get("L", 22.0))
        N = int(params.get("N", 64))
        k, E, E2 = _ks_precompute(N, L_domain, dt)
        ks_ops = (k, E, E2, N)

    for i in range(n_steps):
        state_p = state + perturb

        state_new = _advance_state(system, state, params, dt, ks_ops)
        state_p_new = _advance_state(system, state_p, params, dt, ks_ops)

        delta = state_p_new - state_new
        norm_delta = np.linalg.norm(delta)

        if (i + 1) % renorm_interval == 0:
            if norm_delta > 0:
                log_growth += np.log(norm_delta / eps)
                count += 1
            perturb = delta / norm_delta * eps if norm_delta > 0 else perturb
        else:
            perturb = delta

        state = state_new

    if count == 0:
        return 0.0
    return log_growth / (count * renorm_interval * dt)


def _le_mackey_glass(
    params: dict[str, Any],
    dt: float,
    T_transient: float,
    T_compute: float,
    renorm_interval: int,
) -> float:
    """LE estimation for Mackey-Glass DDE with proper delay history tracking.

    Uses the discrete history segment as the state, renormalising the full
    perturbation vector (current state + delay history) at each interval.
    """
    tau = float(params.get("tau", 17.0))
    beta_mg = float(params.get("beta", 0.2))
    gamma = float(params.get("gamma", 0.1))
    n_mg = float(params.get("n", 10.0))

    eps = 1e-7
    n_delay = max(1, int(tau / dt))
    n_trans = int(T_transient / dt)
    n_steps = int(T_compute / dt)
    total = n_delay + n_trans + n_steps + 2

    # Build reference trajectory
    hist_ref = np.full(total, 0.9)
    for i in range(n_delay, total - 1):
        hist_ref[i + 1] = _mg_euler_step(hist_ref[i], hist_ref[i - n_delay], beta_mg, gamma, n_mg, dt)

    start = n_delay + n_trans  # index where measurement begins

    # Initialize perturbed trajectory (window of size n_delay+1)
    window_size = n_delay + 1
    ref_window = hist_ref[start - n_delay : start + 1].copy()
    pert_window = ref_window.copy()
    pert_window[-1] += eps  # perturb current state

    log_growth = 0.0
    count = 0

    for i in range(n_steps):
        t = start + i

        # Advance perturbed: current = pert_window[-1], delayed = pert_window[0]
        x_pert_new = _mg_euler_step(
            pert_window[-1], pert_window[0], beta_mg, gamma, n_mg, dt
        )

        # Slide window
        pert_window[:-1] = pert_window[1:]
        pert_window[-1] = x_pert_new

        # Reference window at t+1
        ref_window_next = hist_ref[t + 1 - n_delay : t + 2]

        delta_vec = pert_window - ref_window_next
        norm_delta = np.linalg.norm(delta_vec)

        if (i + 1) % renorm_interval == 0:
            if norm_delta > 0:
                log_growth += np.log(norm_delta / eps)
                count += 1
                pert_window = ref_window_next + eps * delta_vec / norm_delta
            else:
                pert_window = ref_window_next.copy()
                pert_window[-1] += eps

    if count == 0:
        return 0.0
    return log_growth / (count * renorm_interval * dt)


def _advance_state(
    system: str,
    state: NDArray[np.float64],
    params: dict[str, Any],
    dt: float,
    ks_ops: tuple | None = None,
) -> NDArray[np.float64]:
    """Advance state by one dt for ODE/PDE systems."""
    if system == "lorenz63":
        sigma = float(params.get("sigma", 10.0))
        rho = float(params.get("rho", 28.0))
        beta = float(params.get("beta", 8.0 / 3.0))
        return _lorenz63_step(state, dt, sigma, rho, beta)
    elif system == "ks":
        if ks_ops is not None:
            k, E, E2, N = ks_ops
        else:
            L_domain = float(params.get("L", 22.0))
            N = int(params.get("N", 64))
            k, E, E2 = _ks_precompute(N, L_domain, dt)
        u_hat = np.fft.rfft(state)
        u_hat_new = _ks_lawson_step(u_hat, dt, E, E2, k, N)
        return np.fft.irfft(u_hat_new, n=N)
    else:
        raise ValueError(f"Unknown system for _advance_state: {system}")


# ---------------------------------------------------------------------------
# DataLoader-compatible dataset
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset

    class TrajectoryDataset(Dataset):  # type: ignore[misc]
        """PyTorch Dataset wrapping a trajectory array for sequence modelling.

        Each sample is a (input_seq, target_seq) pair of length ``seq_len``.
        """

        def __init__(self, traj: NDArray[np.float64], seq_len: int = 50) -> None:
            self.data = torch.from_numpy(traj.astype(np.float32))
            self.seq_len = seq_len
            self.n = len(traj) - seq_len

        def __len__(self) -> int:
            return max(0, self.n)

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            x = self.data[idx : idx + self.seq_len]
            y = self.data[idx + 1 : idx + self.seq_len + 1]
            return x, y

except ImportError:
    pass  # torch not available; TrajectoryDataset won't be importable
