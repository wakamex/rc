"""DeltaNet block replacement module.

Replaces selected Gated DeltaNet blocks in Qwen3.5 with ESN reservoir modules
that serve as richer recurrent modules with higher-dimensional state.

Classes:
- ReservoirBlock: drop-in attention replacement backed by an ESN reservoir.
- DeltaNetReplacer: identifies DeltaNet layers (T3) and swaps selected ones.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.loader import get_deltanet_layers
from src.reservoir.esn import ESN
from src.reservoir.interface import ReadProjection, WriteHead
from src.types import ReservoirConfig


class ReservoirBlock(nn.Module):
    """ESN reservoir replacement for a DeltaNet attention block.

    Drop-in for any attention module: accepts ``(batch, seq_len, hidden_dim)``
    or ``(seq_len, hidden_dim)`` and returns a tensor of the same shape.

    Forward path (ESN mode):
        1. WriteHead:   hidden_states  → ESN input  (numpy, gradient stops here)
        2. ESN.step:    run reservoir  → state sequence (numpy)
        3. ReadProjection: state       → hidden_dim   (trainable, gradient flows)

    A/B swap: call :meth:`swap_to_original` to fall back to the stored
    DeltaNet module and :meth:`swap_to_esn` to return to ESN mode.

    Args:
        hidden_dim: LLM hidden dimension (input = output dim of this block).
        reservoir_config: ESN configuration.
        esn_input_dim: Reservoir input dimension. Defaults to ``hidden_dim``.
        original_module: Original DeltaNet module kept for A/B swap.
    """

    def __init__(
        self,
        hidden_dim: int,
        reservoir_config: ReservoirConfig,
        esn_input_dim: int | None = None,
        original_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        esn_input_dim = esn_input_dim or hidden_dim

        # Trainable interface modules (T12)
        self.write_head = WriteHead(hidden_dim, esn_input_dim)
        self.read_proj = ReadProjection(reservoir_config.size, hidden_dim)

        # ESN reservoir (non-trainable numpy computation)
        self._esn = ESN(reservoir_config, esn_input_dim)
        self._reservoir_config = reservoir_config

        # Original module stored for A/B swap
        self._original = original_module
        self._use_esn: bool = True

    # ------------------------------------------------------------------
    # A/B swap API
    # ------------------------------------------------------------------

    def swap_to_esn(self) -> None:
        """Switch to ESN reservoir mode."""
        self._use_esn = True

    def swap_to_original(self) -> None:
        """Switch back to original DeltaNet mode.

        Raises:
            RuntimeError: if no original module was stored at construction.
        """
        if self._original is None:
            raise RuntimeError(
                "No original module stored; cannot swap to DeltaNet. "
                "Pass original_module= when constructing ReservoirBlock."
            )
        self._use_esn = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run ESN reservoir (or original DeltaNet) on the hidden states.

        Args:
            hidden_states: Shape ``(batch, seq_len, hidden_dim)`` or
                           ``(seq_len, hidden_dim)``.
            **kwargs: Forwarded to the original module when in DeltaNet mode.

        Returns:
            Tensor of same shape as ``hidden_states``.
        """
        if not self._use_esn and self._original is not None:
            return self._original(hidden_states, **kwargs)

        # Handle 2-D input (no batch dim)
        squeeze_batch = hidden_states.ndim == 2
        if squeeze_batch:
            hidden_states = hidden_states.unsqueeze(0)

        batch, seq_len, _ = hidden_states.shape

        # Write: project LLM hidden → ESN input (gradient boundary)
        esn_input_np = self.write_head.to_numpy(hidden_states)  # (B, T, esn_input_dim)

        # Step ESN through the sequence
        self._esn.reset()
        states: list[np.ndarray] = []
        for t in range(seq_len):
            state = self._esn.step(esn_input_np[:, t, :])  # (B, n)
            states.append(state.copy())

        # (B, T, reservoir_size)
        reservoir_arr = np.stack(states, axis=1)

        # Read: project ESN state → hidden_dim (trainable)
        out = self.read_proj(reservoir_arr)  # (B, T, hidden_dim)

        if squeeze_batch:
            out = out.squeeze(0)

        return out


class DeltaNetReplacer:
    """Replace selected DeltaNet blocks in a model with ESN reservoir modules.

    Uses T3's :func:`~src.models.loader.get_deltanet_layers` to identify
    DeltaNet layer paths, then replaces the layers at the given indices
    with :class:`ReservoirBlock` instances (in-place model modification).

    Args:
        model: The model to modify (Qwen3.5 or compatible ``nn.Module``).
        layer_indices: Indices into the discovered DeltaNet layer list to
            replace. E.g. ``[0, 1, 2]`` replaces the first 3 DeltaNet layers.
        reservoir_config: ESN configuration. Defaults to a 512-unit ESN.
        hidden_dim: LLM hidden dimension. Inferred from the replaced module's
            parameters if not provided.
        esn_input_dim: Reservoir input dimension. Defaults to ``hidden_dim``.

    Raises:
        IndexError: if any element of ``layer_indices`` is out of range.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        reservoir_config: ReservoirConfig | None = None,
        hidden_dim: int | None = None,
        esn_input_dim: int | None = None,
    ) -> None:
        self.model = model
        self.reservoir_config = reservoir_config or ReservoirConfig(size=512, seed=42)

        # T3: identify DeltaNet layers by name
        all_paths = get_deltanet_layers(model)
        self._deltanet_paths: list[str] = self._unique_top_level(all_paths)

        n = len(self._deltanet_paths)
        for idx in layer_indices:
            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Layer index {idx} out of range; "
                    f"model has {n} DeltaNet layers (indices 0–{n - 1})."
                )

        self._layer_indices = list(layer_indices)
        self._params_removed: int = 0
        self._params_added: int = 0

        self._do_replace(hidden_dim, esn_input_dim)

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_top_level(paths: list[str]) -> list[str]:
        """Return only paths with no proper-prefix ancestor also in the list."""
        path_set = set(paths)
        seen: set[str] = set()
        result: list[str] = []
        for path in paths:
            parts = path.split(".")
            dominated = any(
                ".".join(parts[:k]) in path_set for k in range(1, len(parts))
            )
            if not dominated and path not in seen:
                seen.add(path)
                result.append(path)
        return result

    def _get_module_at(self, path: str) -> nn.Module:
        m: nn.Module = self.model
        for part in path.split("."):
            m = getattr(m, part)
        return m

    def _set_module_at(self, path: str, new_module: nn.Module) -> None:
        parts = path.split(".")
        parent: nn.Module = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    # ------------------------------------------------------------------
    # Replacement logic
    # ------------------------------------------------------------------

    def _infer_hidden_dim(self, module: nn.Module) -> int:
        """Infer hidden_dim from the first 2-D parameter (output dimension)."""
        for p in module.parameters():
            if p.ndim >= 2:
                return p.shape[0]
        return 512

    def _do_replace(self, hidden_dim: int | None, esn_input_dim: int | None) -> None:
        for idx in self._layer_indices:
            path = self._deltanet_paths[idx]
            original = self._get_module_at(path)

            self._params_removed += sum(p.numel() for p in original.parameters())

            hdim = hidden_dim or self._infer_hidden_dim(original)
            block = ReservoirBlock(
                hidden_dim=hdim,
                reservoir_config=self.reservoir_config,
                esn_input_dim=esn_input_dim,
                original_module=original,
            )
            self._params_added += sum(p.numel() for p in block.parameters())
            self._set_module_at(path, block)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def params_removed(self) -> int:
        """Trainable parameters removed (from replaced DeltaNet blocks)."""
        return self._params_removed

    @property
    def params_added(self) -> int:
        """Trainable parameters added (ReservoirBlock interface modules)."""
        return self._params_added

    @property
    def param_delta(self) -> int:
        """Net parameter change: params_added - params_removed."""
        return self._params_added - self._params_removed

    @property
    def total_deltanet_layers(self) -> int:
        """Total DeltaNet layers found in the model."""
        return len(self._deltanet_paths)

    @property
    def num_replaced(self) -> int:
        """Number of DeltaNet layers replaced."""
        return len(self._layer_indices)

    def param_report(self) -> dict:
        """Return a dict with parameter accounting."""
        return {
            "params_removed": self._params_removed,
            "params_added": self._params_added,
            "param_delta": self.param_delta,
            "num_replaced": self.num_replaced,
            "total_deltanet": self.total_deltanet_layers,
        }

    def swap_all_to_esn(self) -> None:
        """Switch all replaced blocks to ESN mode."""
        for idx in self._layer_indices:
            block = self._get_module_at(self._deltanet_paths[idx])
            if isinstance(block, ReservoirBlock):
                block.swap_to_esn()

    def swap_all_to_original(self) -> None:
        """Switch all replaced blocks back to original DeltaNet mode."""
        for idx in self._layer_indices:
            block = self._get_module_at(self._deltanet_paths[idx])
            if isinstance(block, ReservoirBlock):
                block.swap_to_original()
