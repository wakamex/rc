"""Echo State Network (ESN) reservoir implementation.

State update:
    r_t = (1 - α) * r_{t-1} + α * tanh(W @ r_{t-1} + W_in @ x_t + w_t)
where α is the leak rate.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.types import ReservoirConfig

# Internal float type.  float32 is 5x faster for tanh/SpMV at large n.
_DTYPE = np.float32


class ESN:
    """Echo State Network with sparse reservoir topology.

    Supports Erdős–Rényi and Watts-Strogatz small-world topologies.
    Uses scipy.sparse.csr_matrix for memory-efficient weight storage.

    For large reservoirs (n ≥ 10 000), W_in is stored sparse to keep
    per-step memory bandwidth manageable; for smaller n it stays dense.
    """

    def __init__(self, config: ReservoirConfig, input_dim: int) -> None:
        self.config = config
        self.n = config.size
        self.alpha = _DTYPE(config.leak_rate)
        self.input_dim = input_dim

        rng = np.random.default_rng(config.seed)
        self.W: sp.csr_matrix = self._build_reservoir(rng)
        self.W_in: np.ndarray | sp.csr_matrix = self._build_W_in(rng, input_dim)
        self.state: np.ndarray = np.zeros(self.n, dtype=_DTYPE)

    # ------------------------------------------------------------------
    # Reservoir construction
    # ------------------------------------------------------------------

    def _build_W_in(
        self, rng: np.random.Generator, input_dim: int
    ) -> np.ndarray | sp.csr_matrix:
        """Build input weight matrix.

        For large n, use a sparse W_in (≈10 connections per neuron) to
        reduce memory bandwidth per step.  For small n, use a dense matrix.
        """
        n = self.n
        scale = _DTYPE(self.config.input_scaling)
        if n >= 10_000:
            # ~10 connections per neuron regardless of n
            density = min(1.0, 10.0 * input_dim / n)
            W_in = sp.random(n, input_dim, density=density, format="csr", random_state=rng)
            W_in.data[:] = (rng.uniform(-1.0, 1.0, W_in.nnz) * scale).astype(_DTYPE)
            return W_in.astype(_DTYPE)
        else:
            return (rng.uniform(-1.0, 1.0, (n, input_dim)) * scale).astype(_DTYPE)

    def _build_reservoir(self, rng: np.random.Generator) -> sp.csr_matrix:
        topology = self.config.topology
        if topology == "erdos_renyi":
            W = self._erdos_renyi(rng)
        elif topology == "small_world":
            W = self._small_world(rng)
        else:
            raise ValueError(f"Unknown topology: {topology!r}")
        return self._rescale(W, self.config.spectral_radius).astype(_DTYPE)

    def _erdos_renyi(self, rng: np.random.Generator) -> sp.csr_matrix:
        """Erdős–Rényi random graph with Gaussian edge weights."""
        W = sp.random(
            self.n,
            self.n,
            density=self.config.sparsity,
            format="csr",
            random_state=rng,
        )
        W.data[:] = rng.standard_normal(W.nnz)
        return W

    def _small_world(self, rng: np.random.Generator) -> sp.csr_matrix:
        """Watts-Strogatz small-world graph with Gaussian edge weights.

        Builds a ring lattice with mean degree k ≈ sparsity * n,
        then randomly rewires each edge with probability beta = 0.1.
        """
        n = self.n
        k = max(2, int(self.config.sparsity * n))
        k = k + (k % 2)  # round up to even
        k = min(k, n - 1)
        half_k = k // 2

        # Vectorised ring-lattice construction
        i_idx = np.repeat(np.arange(n), half_k * 2)
        offsets = np.tile(
            np.concatenate([np.arange(1, half_k + 1), np.arange(-half_k, 0)]),
            n,
        )
        j_idx = (i_idx + offsets) % n

        # Rewire edges with probability beta
        beta = 0.1
        rewire_mask = rng.random(len(i_idx)) < beta
        j_idx[rewire_mask] = rng.integers(0, n, size=int(rewire_mask.sum()))

        data = rng.standard_normal(len(i_idx))
        W = sp.csr_matrix((data, (i_idx, j_idx)), shape=(n, n))
        return W

    def _rescale(self, W: sp.csr_matrix, target_sr: float) -> sp.csr_matrix:
        """Rescale W so its spectral radius equals target_sr."""
        if W.nnz == 0:
            return W

        try:
            k = min(6, W.shape[0] - 2)
            if k < 1:
                raise ValueError("Matrix too small for eigs")
            eigs = spla.eigs(W, k=k, which="LM", return_eigenvectors=False, tol=1e-4)
            current_sr = float(np.max(np.abs(eigs)))
        except Exception:
            # Fallback: 20 steps of power iteration
            v = np.ones(W.shape[0])
            current_sr = 1.0
            for _ in range(20):
                v = W.dot(v)
                nrm = float(np.linalg.norm(v))
                if nrm == 0:
                    break
                v /= nrm
                current_sr = nrm

        if current_sr > 1e-10:
            W = W * (target_sr / current_sr)
        return W

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def step(self, x_t: np.ndarray, w_t: np.ndarray | None = None) -> np.ndarray:
        """Advance reservoir state one step.

        Args:
            x_t: Input vector, shape ``(input_dim,)`` or ``(batch, input_dim)``.
            w_t: Optional additive drive with the same leading shape as the
                 reservoir state (``(n,)`` or ``(batch, n)``).

        Returns:
            New reservoir state r_t with shape ``(n,)`` or ``(batch, n)``.
        """
        x_t = np.asarray(x_t, dtype=_DTYPE)
        batched = x_t.ndim == 2

        if batched:
            return self._step_batched(x_t, w_t)

        # ------ single-step (performance-critical path) ------
        r = self.state

        # pre = W @ r  (sparse MV; returns new ndarray in _DTYPE)
        pre = self.W.dot(r)

        # pre += W_in @ x_t
        if sp.issparse(self.W_in):
            pre += self.W_in.dot(x_t)
        else:
            pre += np.einsum("ij,j->i", self.W_in, x_t)

        if w_t is not None:
            pre += np.asarray(w_t, dtype=_DTYPE)

        # in-place leaky integration: state = (1-α)*state + α*tanh(pre)
        np.tanh(pre, out=pre)
        pre *= self.alpha
        r *= _DTYPE(1.0) - self.alpha
        r += pre
        return r

    def _step_batched(
        self, x_t: np.ndarray, w_t: np.ndarray | None
    ) -> np.ndarray:
        batch = x_t.shape[0]
        r = self.state if self.state.ndim == 2 else np.tile(self.state, (batch, 1))

        # W @ r^T → (n, batch), then transpose to (batch, n)
        Wr = self.W.dot(r.T).T

        if sp.issparse(self.W_in):
            Win_x = (self.W_in.dot(x_t.T)).T  # (batch, n)
        else:
            Win_x = x_t @ self.W_in.T  # (batch, n)

        pre = Wr + Win_x
        if w_t is not None:
            pre += np.asarray(w_t, dtype=_DTYPE)

        r_new = (_DTYPE(1.0) - self.alpha) * r + self.alpha * np.tanh(pre)
        self.state = r_new
        return r_new

    def reset(self) -> None:
        """Reset reservoir state to zero."""
        self.state = np.zeros(self.n, dtype=_DTYPE)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a full input sequence through the reservoir.

        Args:
            X: Input sequence, shape ``(T, input_dim)``.

        Returns:
            State sequence, shape ``(T, n)``.
        """
        T = X.shape[0]
        states = np.empty((T, self.n), dtype=_DTYPE)
        for t in range(T):
            states[t] = self.step(X[t])
        return states
