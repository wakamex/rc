"""Shared protocols and types for the LRS project."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReservoirConfig:
    """Configuration for an Echo State Network reservoir."""

    size: int = 1000
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    topology: str = "erdos_renyi"  # "erdos_renyi" | "small_world"
    sparsity: float = 0.01
    seed: int | None = None


@dataclass
class BenchmarkExample:
    """A single example from a benchmark generator."""

    input: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from evaluating a model on a benchmark task."""

    task: str
    metric: str
    value: float
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Reservoir(Protocol):
    """Protocol for reservoir computing modules."""

    def step(self, x_t: Any, w_t: Any | None = None) -> Any:
        """Advance reservoir state by one step given input x_t and optional write signal w_t."""
        ...

    def reset(self) -> None:
        """Reset reservoir state to zero."""
        ...


@runtime_checkable
class ModelWrapper(Protocol):
    """Protocol for wrapped language models."""

    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        """Run a forward pass and return model outputs."""
        ...

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        """Generate token sequences."""
        ...

    def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
        """Return hidden states at the specified layer."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Protocol for benchmark example generators."""

    def __iter__(self) -> Iterator[BenchmarkExample]:
        """Iterate over generated benchmark examples."""
        ...

    def __len__(self) -> int:
        """Return the number of examples (may be approximate for infinite generators)."""
        ...


@runtime_checkable
class DataPipeline(Protocol):
    """Protocol for data pipelines."""

    def iterate(self, split: str = "train") -> Iterator[Any]:
        """Iterate over batches for the given split."""
        ...

    def split(self, ratios: dict[str, float]) -> dict[str, DataPipeline]:
        """Return a mapping of split name to sub-pipeline."""
        ...
