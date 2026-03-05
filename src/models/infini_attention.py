"""Infini-attention: compressive memory baseline for Qwen3.5-0.8B.

Implements the Infini-attention mechanism from:
  "Leave No Context Behind: Efficient Infinite Context Transformers
  with Infini-attention" (Munkhdalai et al., 2024, arXiv:2404.07143)

The module adds a compressive associative memory to selected attention layers.
For each augmented layer the model can:
  1. Retrieve compressed past context using the current queries.
  2. Update the memory with the current keys / values.
  3. Mix the memory-retrieved values with the standard attention output via
     a learned per-head gate (beta).

Usage example::

    from src.models.loader import load_model
    from src.models.infini_attention import apply_infini_attention, reset_infini_memory

    wrapper = load_model("qwen3.5-0.8b", device="cuda")
    infini_layers = apply_infini_attention(
        wrapper.model,
        layer_indices=[4, 8, 12, 16],   # augment every 4th layer
    )
    # train wrapper.model with LoRA + infini_layers params ...
    # reset memory between sequences during training:
    reset_infini_memory(wrapper.model)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Compressive memory (associative matrix memory)
# ---------------------------------------------------------------------------


class CompressiveMemory(nn.Module):
    """Fixed-size associative memory for Infini-attention.

    Maintains a per-head memory matrix M ∈ R^{head_dim × head_dim} and a
    normalisation vector z ∈ R^{head_dim}.  The memory is a running
    accumulation — it is NOT a learnable parameter.
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # State buffers — updated in-place during forward; not trained.
        self.register_buffer("M", torch.zeros(num_heads, head_dim, head_dim))
        self.register_buffer("z", torch.zeros(num_heads, head_dim))

    def reset(self) -> None:
        """Zero out memory state (call at the start of each new sequence)."""
        self.M.zero_()
        self.z.zero_()

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """Feature map φ(x) = ELU(x) + 1.  Keeps values strictly positive."""
        return F.elu(x) + 1.0

    def retrieve(self, Q: torch.Tensor) -> torch.Tensor:
        """Retrieve memory values for the given queries.

        Args:
            Q: (batch, heads, seq, head_dim)

        Returns:
            Retrieved values of the same shape as Q.
        """
        phi_Q = self._phi(Q)  # (B, H, T, d)
        # M: (H, d, d) — broadcast across batch
        retrieved = torch.einsum("bhtd,hde->bhte", phi_Q, self.M)  # (B, H, T, d)
        norm = torch.einsum("bhtd,hd->bht", phi_Q, self.z).unsqueeze(-1)  # (B, H, T, 1)
        return retrieved / norm.clamp(min=1e-6)

    def update(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """Update memory with new key-value pairs.

        The update is applied with detached K / V so that gradients flow only
        through the *retrieval* path, not back through historical updates.

        Args:
            K: (batch, heads, seq, head_dim)
            V: (batch, heads, seq, head_dim)
        """
        phi_K = self._phi(K.detach())  # (B, H, T, d)
        V_det = V.detach()  # (B, H, T, d)
        # Average over batch for a single shared memory state.
        phi_K_b = phi_K.mean(0)  # (H, T, d)
        V_b = V_det.mean(0)  # (H, T, d)
        # M += Σ_t φ(K_t)^T V_t
        self.M = self.M + torch.einsum("htd,hte->hde", phi_K_b, V_b)
        # z += Σ_t φ(K_t)
        self.z = self.z + phi_K_b.sum(1)  # (H, d)


# ---------------------------------------------------------------------------
# Per-layer wrapper that adds compressive memory to an attention module
# ---------------------------------------------------------------------------


class InfiniAttentionLayer(nn.Module):
    """Wraps an existing attention module with a compressive memory side-path.

    The base attention layer is frozen (wrapped as-is); only the new memory
    projections and the mixing gate (beta) are trainable.

    Architecture (per forward call):
      1. Run base attention → attn_out  (standard local attention)
      2. Project input hidden states to Q_m, K_m, V_m  (memory projections)
      3. Retrieve A_mem = memory.retrieve(Q_m)
      4. Update memory with K_m, V_m
      5. Mix: output = attn_out + sigmoid(β).mean() * out_proj(A_mem)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        hidden_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)

        # Compressive memory (state only, not learned)
        self.memory = CompressiveMemory(num_heads=self.num_heads, head_dim=self.head_dim)

        mem_inner = self.num_heads * self.head_dim
        # Adapter projections — these are the only NEW trainable parameters.
        self.mem_q_proj = nn.Linear(hidden_dim, mem_inner, bias=False)
        self.mem_k_proj = nn.Linear(hidden_dim, mem_inner, bias=False)
        self.mem_v_proj = nn.Linear(hidden_dim, mem_inner, bias=False)
        self.mem_out_proj = nn.Linear(mem_inner, hidden_dim, bias=False)

        # Learned mixing gate: one scalar per head.
        self.beta = nn.Parameter(torch.zeros(self.num_heads))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialise to near-zero so training starts close to unmodified model.
        nn.init.normal_(self.mem_q_proj.weight, std=0.02)
        nn.init.normal_(self.mem_k_proj.weight, std=0.02)
        nn.init.normal_(self.mem_v_proj.weight, std=0.02)
        nn.init.zeros_(self.mem_out_proj.weight)

    def reset_memory(self) -> None:
        """Reset memory state for a new sequence."""
        self.memory.reset()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run base attention + memory retrieval; mix and return combined output.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            *args / **kwargs: Forwarded verbatim to the base attention layer.

        Returns:
            Same type as the base layer return (tensor or tuple with the
            attention output as the first element).
        """
        # --- 1. Standard (frozen) local attention ---
        base_out = self.base_layer(hidden_states, *args, **kwargs)

        base_is_tuple = isinstance(base_out, tuple)
        if base_is_tuple:
            attn_out: torch.Tensor = base_out[0]
            rest = base_out[1:]
        else:
            attn_out = base_out
            rest = ()

        B, T, _ = hidden_states.shape

        # --- 2. Memory projections ---
        Q_m = (
            self.mem_q_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, T, d)
        K_m = (
            self.mem_k_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V_m = (
            self.mem_v_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # --- 3. Retrieve from memory ---
        A_mem = self.memory.retrieve(Q_m)  # (B, H, T, d)

        # --- 4. Update memory ---
        self.memory.update(K_m, V_m)

        # --- 5. Project memory output and mix with attention output ---
        mem_flat = A_mem.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, H*d)
        mem_out = self.dropout(self.mem_out_proj(mem_flat))  # (B, T, hidden_dim)

        beta_scalar = torch.sigmoid(self.beta).mean()  # scalar ∈ (0, 1)
        mixed = attn_out + beta_scalar * mem_out

        if base_is_tuple:
            return (mixed,) + rest
        return mixed


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------


def apply_infini_attention(
    model: nn.Module,
    layer_indices: list[int] | None = None,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    dropout: float = 0.0,
) -> dict[int, InfiniAttentionLayer]:
    """Apply Infini-attention wrappers to selected attention layers.

    Replaces the ``self_attn`` (or equivalent) sub-module of each specified
    transformer layer with an :class:`InfiniAttentionLayer`.  The base
    attention weights are preserved and should be frozen before training
    (or left for LoRA to adapt).

    Args:
        model: The language model (e.g. Qwen3.5-0.8B loaded via
            :func:`src.models.loader.load_model`).  The *inner* HF model
            should be passed, not the :class:`ModelWrapperImpl`.
        layer_indices: Indices of transformer layers to augment.  ``None``
            augments every other layer (0, 2, 4, …) as a sensible default.
        hidden_dim: Hidden state dimension.  Inferred from ``model.config``
            if not provided.
        num_heads: Number of attention heads.  Inferred from ``model.config``
            if not provided.
        head_dim: Per-head dimension.  Defaults to
            ``hidden_dim // num_heads``.
        dropout: Dropout rate for the memory output projection.

    Returns:
        Dict mapping each augmented layer index to the
        :class:`InfiniAttentionLayer` that was installed.

    Raises:
        ValueError: If the transformer layer list cannot be located.
    """
    cfg = getattr(model, "config", None)

    if hidden_dim is None:
        hidden_dim = int(getattr(cfg, "hidden_size", 1024))
    if num_heads is None:
        num_heads = int(getattr(cfg, "num_attention_heads", hidden_dim // 64))
    if head_dim is None:
        head_dim = hidden_dim // num_heads

    # Locate the list of transformer layers.
    transformer_layers: nn.ModuleList | None = None
    for attr_path in ("model.layers", "transformer.h", "layers", "model.decoder.layers"):
        obj: Any = model
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            transformer_layers = obj  # type: ignore[assignment]
            break

    if transformer_layers is None:
        raise ValueError(
            "Could not locate transformer layers in model.  "
            "Tried: model.layers, transformer.h, layers, model.decoder.layers"
        )

    n_layers = len(transformer_layers)
    if layer_indices is None:
        # Default: augment every other full-attention layer (skip DeltaNet layers
        # which are at even indices in Qwen3.5's hybrid architecture).
        layer_indices = list(range(1, n_layers, 2))  # odd layers = full-attn

    wrapped: dict[int, InfiniAttentionLayer] = {}
    attn_names = ("self_attn", "attn", "attention", "self_attention")

    for idx in layer_indices:
        if idx >= n_layers:
            continue
        layer = transformer_layers[idx]

        # Find the attention sub-module.
        attn_mod: nn.Module | None = None
        attn_attr: str | None = None
        for name in attn_names:
            candidate = getattr(layer, name, None)
            if candidate is not None:
                attn_mod = candidate
                attn_attr = name
                break

        if attn_mod is None:
            continue  # layer has no standard attention sub-module — skip

        infini_layer = InfiniAttentionLayer(
            base_layer=attn_mod,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        setattr(layer, attn_attr, infini_layer)
        wrapped[idx] = infini_layer

    return wrapped


def reset_infini_memory(model: nn.Module) -> None:
    """Reset all :class:`InfiniAttentionLayer` memory states in *model*.

    Call this at the start of each new sequence during training and before
    each evaluation episode.
    """
    for module in model.modules():
        if isinstance(module, InfiniAttentionLayer):
            module.reset_memory()


def get_infini_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the infini-attention adapter parameters.

    Useful for constructing an optimizer that trains infini-attention params
    separately from LoRA params.
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, InfiniAttentionLayer):
            params.extend(
                [
                    *module.mem_q_proj.parameters(),
                    *module.mem_k_proj.parameters(),
                    *module.mem_v_proj.parameters(),
                    *module.mem_out_proj.parameters(),
                    module.beta,
                ]
            )
    return params


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


from dataclasses import dataclass, field  # noqa: E402 (imports after code body)


@dataclass
class InfiniAttentionConfig:
    """Configuration for Infini-attention baseline training."""

    # Model
    model_name: str = "qwen3.5-0.8b"
    dtype: str = "bfloat16"

    # Infini-attention architecture
    layer_indices: list[int] = field(default_factory=list)  # empty → auto
    dropout: float = 0.0

    # LoRA
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    seed: int = 42

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "sample-10BT"

    # Logging / output
    output_dir: str = "checkpoints/infini_attention"
    eval_output: str = "results/baselines/infini_attention.json"
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    wandb_project: str = "lrs-baselines"
    wandb_run_name: str = "infini-attention-qwen3.5-0.8b"
