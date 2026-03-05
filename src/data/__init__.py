"""Data sub-package.

Public API
----------
generate_trajectory  : Generate trajectory sequences for Lorenz-63, Mackey-Glass,
                       or Kuramoto-Sivashinsky systems.
split_trajectory     : Split a trajectory into train/val/test segments.
normalize            : Normalize a trajectory to zero mean, unit std.
lyapunov_time        : Estimate the Lyapunov time (inverse of max Lyapunov exponent).

These generators are used as representation diagnostics (verifying stable
bounded reservoir dynamics) and as training data for the chaotic-systems
curriculum phase.
"""

from src.data.chaos import (
    generate_trajectory,
    lyapunov_time,
    normalize,
    split_trajectory,
)

__all__ = [
    "generate_trajectory",
    "split_trajectory",
    "normalize",
    "lyapunov_time",
]
