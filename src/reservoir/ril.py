"""Reservoir Interaction Layer (RIL) — inserts between transformer blocks.

Architecture:
    RIL sits between transformer block N and block N+1.
    Given hidden state h from block N:
    1. Read reservoir state r via ReadProjection → memory_signal  (grad stops here)
    2. Compute write signal w from h via WriteHead → update reservoir
    3. Modulate hidden state: h' = h + gate * f(memory_signal)
    4. Return h' to block N+1.

Insertion:
    RILWrapper wraps a transformer model and inserts RIL modules at
    every ``insert_every`` layers (e.g. every 6 blocks in a 36-layer model).
    Reservoir can be shared across all RIL instances or kept separate per layer.

Gradient boundary:
    Reservoir weights (W, W_in) are frozen numpy arrays → no gradient.
    Interface modules (ReadProjection, WriteHead, gate) are trainable.
    Gradient stops at the numpy ↔ torch boundary automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.reservoir.esn import ESN
from src.reservoir.interface import ReadProjection, WriteHead
from src.types import ReservoirConfig

_ACTIVATIONS: dict[str, Any] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": nn.functional.gelu,
    "identity": lambda x: x,
}


@dataclass
class RILConfig:
    """Configuration for the Reservoir Interaction Layer."""

    # ESN reservoir configuration
    reservoir: ReservoirConfig = field(default_factory=lambda: ReservoirConfig(
        size=256, spectral_radius=0.9, leak_rate=0.3, seed=0
    ))
    # Share one reservoir across all RIL instances (vs. separate per layer)
    shared_reservoir: bool = True
    # Insert RIL after every k transformer blocks (0 = disabled)
    insert_every: int = 6
    # Gate initial value (0 → near-identity at start of training)
    gate_init: float = 0.0
    # Activation applied to the read projection output
    read_activation: str = "tanh"  # "tanh" | "relu" | "gelu" | "identity"


class ReservoirInteractionLayer(nn.Module):
    """Single RIL instance inserted between two transformer blocks.

    Forward:
        1. ReadProjection: reservoir.state → memory_signal (hidden_dim)
        2. WriteHead: h_mean → write signal (numpy, reservoir_dim) → reservoir.step()
        3. h' = h + gate * activation(memory_signal)

    Gradient flows through ReadProjection, WriteHead, and gate.
    Gradient stops at the reservoir boundary (numpy ↔ torch).

    Args:
        hidden_dim: Hidden dimension of the transformer.
        reservoir: The ESN reservoir instance to read from and write to.
            Its ``input_dim`` must equal ``hidden_dim``.
        gate_init: Initial scalar gate value (default 0 for stable training).
        read_activation: Nonlinearity applied to memory_signal before gating.
    """

    def __init__(
        self,
        hidden_dim: int,
        reservoir: ESN,
        gate_init: float = 0.0,
        read_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if read_activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown read_activation {read_activation!r}. "
                f"Choose from {sorted(_ACTIVATIONS)!r}."
            )
        self.hidden_dim = hidden_dim
        self.reservoir = reservoir
        reservoir_dim = reservoir.n

        # ReadProjection: reservoir_dim → hidden_dim  (grad flows through weights)
        self.read_proj = ReadProjection(reservoir_dim, hidden_dim)
        # WriteHead: hidden_dim → reservoir_dim  (used as additive drive w_t)
        self.write_head = WriteHead(hidden_dim, reservoir_dim)
        # Learnable scalar gate; init near 0 → near-identity at training start
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self._activation = _ACTIVATIONS[read_activation]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Modulate hidden state h using reservoir memory.

        Args:
            h: Hidden state from block N.
               Shape ``(batch, seq_len, hidden_dim)`` or ``(seq_len, hidden_dim)``.

        Returns:
            h' of the same shape as h.
        """
        # ------ Step 1: Read current reservoir state ------
        r_state = self.reservoir.state  # numpy (reservoir_dim,)
        memory_signal = self.read_proj(r_state, device=h.device)  # (hidden_dim,)

        # ------ Step 2: Compute write signal, update reservoir ------
        # Pool over the sequence dimension; detach so no grad enters the reservoir.
        if h.ndim == 3:
            h_pool = h.detach().mean(dim=1)     # (batch, hidden_dim)
        else:
            h_pool = h.detach().mean(dim=0, keepdim=True)  # (1, hidden_dim)

        # x_t: first-batch mean-pooled hidden, fed into reservoir via W_in
        x_np = h_pool[0].cpu().float().numpy()  # (hidden_dim,) = input_dim

        # w_t: WriteHead output used as additive drive (hidden_dim → reservoir_dim)
        w_np = self.write_head.to_numpy(h_pool)  # (batch, reservoir_dim) or (1, r_dim)
        w_np = w_np[0] if w_np.ndim > 1 else w_np  # (reservoir_dim,)

        # Advance reservoir state (side-effect; no torch gradient)
        self.reservoir.step(x_t=x_np, w_t=w_np)

        # ------ Step 3: Modulate hidden state ------
        activated = self._activation(memory_signal)  # (hidden_dim,) — broadcasts
        return h + self.gate * activated


class _BlockWithRIL(nn.Module):
    """Wraps a single transformer block and appends a RIL pass after it."""

    def __init__(self, block: nn.Module, ril: ReservoirInteractionLayer) -> None:
        super().__init__()
        self.block = block
        self.ril = ril

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> Any:
        out = self.block(hidden_states, **kwargs)
        if isinstance(out, torch.Tensor):
            return self.ril(out)
        elif isinstance(out, tuple):
            return (self.ril(out[0]),) + out[1:]
        # Unsupported output type — pass through unchanged
        return out


class RILWrapper(nn.Module):
    """Inserts RIL modules into a transformer model between every k blocks.

    Modifies the model's transformer layer list in-place so that the
    existing forward pass automatically applies RIL at the configured
    insertion points.

    Supports HuggingFace CausalLM models with transformer blocks at:
    - ``model.model.layers`` (Qwen, LLaMA, etc.)
    - ``model.transformer.h`` (GPT-style)
    - ``model.layers`` (direct access)

    Args:
        model: The transformer model to augment.
        hidden_dim: Transformer hidden dimension.
        config: RIL configuration (uses defaults if None).
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int,
        config: RILConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = RILConfig()
        self.model = model
        self.config = config

        # Build shared reservoir once (if requested)
        shared_res = (
            ESN(config.reservoir, input_dim=hidden_dim)
            if config.shared_reservoir
            else None
        )

        layers = self._get_layers(model)
        inserted: list[ReservoirInteractionLayer] = []
        for i in range(len(layers)):
            # Insert after every k-th block (1-indexed so layer 6, 12, 18, …)
            if config.insert_every > 0 and (i + 1) % config.insert_every == 0:
                reservoir = shared_res if config.shared_reservoir else ESN(
                    config.reservoir, input_dim=hidden_dim
                )
                ril = ReservoirInteractionLayer(
                    hidden_dim=hidden_dim,
                    reservoir=reservoir,
                    gate_init=config.gate_init,
                    read_activation=config.read_activation,
                )
                layers[i] = _BlockWithRIL(layers[i], ril)
                inserted.append(ril)

        # Plain Python list — parameters are already registered through model.layers
        self._ril_instances: list[ReservoirInteractionLayer] = inserted

    @staticmethod
    def _get_layers(model: nn.Module) -> nn.ModuleList:
        """Return the transformer block ModuleList from a HuggingFace-style model."""
        for attr_path in ("model.layers", "transformer.h", "layers"):
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList):
                    return obj
            except AttributeError:
                continue
        raise AttributeError(
            f"Cannot find a nn.ModuleList of transformer blocks in "
            f"{type(model).__name__}. Tried: model.layers, transformer.h, layers."
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped model; RIL is applied inside each wrapped block."""
        return self.model(*args, **kwargs)

    @property
    def ril_layers(self) -> list[ReservoirInteractionLayer]:
        """Access the inserted RIL module instances."""
        return self._ril_instances
