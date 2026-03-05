"""Multi-reservoir module with fast and slow timescales.

Wraps two ESN instances:
- Fast reservoir: high leak rate (α ≈ 0.7–1.0), spectral radius near 1.0.
  Responds quickly to recent input; short memory horizon.
- Slow reservoir: low leak rate (α ≈ 0.1–0.3), spectral radius near 0.5.
  Smooths over long sequences; retains information longer.

Both reservoirs receive the same input (or separate signals via independent
write heads).  The shared read interface returns a concatenated state vector
``[r_fast; r_slow]`` for downstream cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.reservoir.esn import ESN
from src.types import ReservoirConfig

_DTYPE = np.float32

# Default configs expose the intended timescale characteristics.
_DEFAULT_FAST_CFG = ReservoirConfig(
    size=200,
    spectral_radius=0.9,
    leak_rate=0.9,
    seed=0,
)
_DEFAULT_SLOW_CFG = ReservoirConfig(
    size=200,
    spectral_radius=0.5,
    leak_rate=0.1,
    seed=1,
)


@dataclass
class MultiReservoirConfig:
    """Configuration for a dual-timescale reservoir pair."""

    fast: ReservoirConfig = field(default_factory=lambda: ReservoirConfig(
        size=200, spectral_radius=0.9, leak_rate=0.9, seed=0,
    ))
    slow: ReservoirConfig = field(default_factory=lambda: ReservoirConfig(
        size=200, spectral_radius=0.5, leak_rate=0.1, seed=1,
    ))
    # When True, a single write signal is broadcast to both reservoirs.
    # When False (default), fast and slow accept independent write signals.
    shared_write_heads: bool = False
    # "concat" (default): read state = [r_fast; r_slow]
    # "attention": placeholder for future attention-based merging
    state_merge: str = "concat"


class MultiReservoir:
    """Parallel fast/slow reservoir pair.

    Both reservoirs share the same input projection but evolve at
    different timescales controlled by their individual leak rates and
    spectral radii.

    The concatenated read state ``[r_fast; r_slow]`` has dimension
    ``fast.n + slow.n``, which is the dimensionality seen by any
    downstream reader (e.g. an LLM cross-attention module).

    Args:
        config: Full multi-reservoir configuration.
        input_dim: Dimensionality of the input signal ``x_t``.
    """

    def __init__(self, config: MultiReservoirConfig, input_dim: int) -> None:
        self.config = config
        self.fast = ESN(config.fast, input_dim)
        self.slow = ESN(config.slow, input_dim)
        self.input_dim = input_dim

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fast_dim(self) -> int:
        """Dimensionality of the fast reservoir state."""
        return self.fast.n

    @property
    def slow_dim(self) -> int:
        """Dimensionality of the slow reservoir state."""
        return self.slow.n

    @property
    def state_dim(self) -> int:
        """Total concatenated state dimensionality."""
        return self.fast_dim + self.slow_dim

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Return the concatenated reservoir state ``[r_fast; r_slow]``.

        Returns:
            Array of shape ``(fast_dim + slow_dim,)``.
        """
        if self.config.state_merge != "concat":
            raise NotImplementedError(
                f"state_merge={self.config.state_merge!r} is not yet implemented; "
                "only 'concat' is supported."
            )
        return np.concatenate(
            [
                np.asarray(self.fast.state, dtype=_DTYPE),
                np.asarray(self.slow.state, dtype=_DTYPE),
            ]
        )

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(
        self,
        x_t: np.ndarray,
        w_fast: np.ndarray | None = None,
        w_slow: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance both reservoirs by one step.

        Both reservoirs receive the same input ``x_t``.  Write signals
        ``w_fast`` and ``w_slow`` are applied independently (unless
        ``shared_write_heads=True``, in which case a single write signal
        may be passed via ``w_fast`` and it is broadcast to both).

        Args:
            x_t: Input vector, shape ``(input_dim,)``.
            w_fast: Optional additive drive for the fast reservoir,
                shape ``(fast_dim,)``.
            w_slow: Optional additive drive for the slow reservoir,
                shape ``(slow_dim,)``.  Ignored when
                ``shared_write_heads=True`` (``w_fast`` is used for
                both).

        Returns:
            Concatenated state ``[r_fast; r_slow]``, shape
            ``(fast_dim + slow_dim,)``.
        """
        if self.config.shared_write_heads and w_fast is not None:
            # Broadcast the single write signal to both reservoirs.
            w_slow = w_fast

        self.fast.step(x_t, w_t=w_fast)
        self.slow.step(x_t, w_t=w_slow)
        return self.read()

    def reset(self) -> None:
        """Reset both reservoir states to zero."""
        self.fast.reset()
        self.slow.reset()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a full sequence through both reservoirs.

        Args:
            X: Input sequence, shape ``(T, input_dim)``.

        Returns:
            Concatenated state sequence, shape ``(T, fast_dim + slow_dim)``.
        """
        T = X.shape[0]
        states = np.empty((T, self.state_dim), dtype=_DTYPE)
        for t in range(T):
            states[t] = self.step(X[t])
        return states
