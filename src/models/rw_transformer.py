"""Reservoir-Workspace Transformer (RW-Transformer).

A decoder-only hybrid architecture that fuses standard transformer components
(causal self-attention + FFN) with a bidirectional multi-reservoir ESN workspace.

Architecture overview::

                        input_ids
                            │
                      [Embedding]
                            │
                            ▼
    h ──────────────────────────────────────────────────────────┐
        │              │                      │                 │
    [RMSNorm]      [RMSNorm]             [RMSNorm]*             │
        │              │                      │                 │
    [CausalAttn]   [SwiGLU MLP]     [ReservoirWorkspace]*      │
        │              │                      │                 │
        └──────┬────────┘               g3 · res_out *         │
               │                            │                  │
          h + g1·attn + g2·mlp ─────────────┘*                 │
               │                                               │
              h₁ ──────────── repeat N layers ─────────────────┘
               │
           [RMSNorm]
               │
           [LM Head]
               │
            logits

    * Only in reservoir-augmented layers (every ``reservoir_every_n`` layers).

Key design decisions:

- Parallel branch formulation (PaLM-style): attention and MLP operate on the
  same normalized input and are added together, not applied sequentially.
- All three branches share a gated residual: h' = h + g1·attn + g2·mlp + g3·res.
  g1 = g2 = 1.0 at init; g3 = 0.0 so reservoir starts dormant.
- Bidirectional reservoir: run forward and backward over the sequence, concatenate
  states.  This gives each position access to both past and future context via
  the ESN workspace, complementing the causal attention branch.
- Reservoir weights are frozen numpy arrays (not PyTorch parameters).
  Gradients flow through ``g3`` and ``reservoir_ws.out_proj`` but NOT into the
  ESN itself.  The input to the ESN is detached from the computation graph.

Parameter count (default config, ~0.8B trainable):

    vocab_size=151936, hidden_dim=1024, num_layers=32, num_heads=16,
    head_dim=64, ffn_dim=5120, tie_embeddings=True, reservoir_every_n=4,
    fast_reservoir_size=200, slow_reservoir_size=200.

    embed_tokens                       : 151,936 × 1024 = 155,582,464
    24 non-reservoir blocks × 19,924,994              = 477,899,856
     8    reservoir blocks × 20,745,219              = 165,961,752
    final norm                                         =          1,024
    ─────────────────────────────────────────────────────────────────
    Total trainable                                    ≈ 799,445,096 ≈ 0.799 B

    LM head is weight-tied with embed_tokens (0 extra params).
    Reservoir ESN weights (numpy) are NOT counted.

Compatibility notes:

- CPU-only: reservoir uses numpy; all PyTorch ops work on CPU.
- BF16 / FP16: all PyTorch ops are precision-agnostic; reservoir is float32
  internally and the result is cast back to ``x.dtype`` before ``out_proj``.
- Gradient checkpointing: attention and MLP blocks can be wrapped with
  ``torch.utils.checkpoint.checkpoint`` for memory savings.  Reservoir layers
  should NOT be checkpointed (they are stateful numpy objects; recomputation
  would advance their internal state a second time, producing incorrect states).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.reservoir.multi_reservoir import MultiReservoir, MultiReservoirConfig
from src.types import ReservoirConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RWTransformerConfig:
    """Configuration for the Reservoir-Workspace Transformer.

    Default values target ~0.8B trainable parameters with the Qwen2/3
    tokenizer (vocab_size=151936).  Override vocab_size if using the
    extended Qwen3.5 tokenizer (~248K tokens).

    Args:
        vocab_size: Vocabulary size.  Use 151936 for the standard Qwen2/3
            tokenizer or the appropriate size for Qwen3.5.
        hidden_dim: Transformer hidden (model) dimension.
        num_layers: Number of decoder blocks.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension.  Must satisfy
            ``num_heads * head_dim == hidden_dim``.
        ffn_dim: Intermediate dimension for the SwiGLU MLP.
        max_seq_len: Maximum supported sequence length (for RoPE precomputation).
        dropout: Dropout probability applied inside attention.
        tie_embeddings: If True, share weights between the embedding table
            and the LM head (saves vocab_size × hidden_dim parameters).
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        rope_theta: Base frequency for RoPE positional embeddings.
        reservoir_every_n: Reservoir workspace is inserted at layer indices
            where ``layer_index % reservoir_every_n == 0``.
        fast_reservoir_size: Number of neurons in the fast ESN.
        slow_reservoir_size: Number of neurons in the slow ESN.
        fast_spectral_radius: Spectral radius for the fast reservoir.
        slow_spectral_radius: Spectral radius for the slow reservoir.
        fast_leak_rate: Leak rate α for the fast reservoir (high → fast dynamics).
        slow_leak_rate: Leak rate α for the slow reservoir (low → slow dynamics).
        reservoir_seed: Base random seed for reservoir construction.  Layer i
            uses seeds ``reservoir_seed + 2*i`` and ``reservoir_seed + 2*i + 1``.
    """

    # Vocabulary
    vocab_size: int = 151936

    # Architecture
    hidden_dim: int = 1024
    num_layers: int = 32
    num_heads: int = 16
    head_dim: int = 64
    ffn_dim: int = 5120
    max_seq_len: int = 4096
    dropout: float = 0.0
    tie_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # Reservoir
    reservoir_every_n: int = 4
    fast_reservoir_size: int = 200
    slow_reservoir_size: int = 200
    fast_spectral_radius: float = 0.9
    slow_spectral_radius: float = 0.5
    fast_leak_rate: float = 0.9
    slow_leak_rate: float = 0.1
    reservoir_seed: int = 42

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.hidden_dim:
            raise ValueError(
                f"num_heads ({self.num_heads}) * head_dim ({self.head_dim}) "
                f"must equal hidden_dim ({self.hidden_dim}), "
                f"got {self.num_heads * self.head_dim}."
            )


# ---------------------------------------------------------------------------
# Utility: RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (weight only, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm) * self.weight


# ---------------------------------------------------------------------------
# Utility: RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------


def _precompute_rope(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cosine and sine tensors.

    Returns:
        cos, sin: Shape (max_seq_len, head_dim).
    """
    # half-dim frequencies
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (T, head_dim // 2)
    # Tile to full head_dim using the "rotate-half" convention
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (T, head_dim)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by 90°: [-x2, x1] where x = [x1, x2] split along last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: Shape (B, H, T, head_dim).
        cos, sin: Shape (T, head_dim) — already sliced to sequence length.

    Returns:
        Rotated (q, k) with the same shape.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Attention branch
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE positional embeddings."""

    def __init__(self, config: RWTransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        inner_dim = config.num_heads * config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.hidden_dim, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim)
            cos, sin: (T, head_dim)

        Returns:
            (B, T, hidden_dim)
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True
        )  # (B, H, T, D)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# MLP branch (SwiGLU)
# ---------------------------------------------------------------------------


class RWTMLP(nn.Module):
    """SwiGLU feed-forward network: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: RWTransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Reservoir workspace branch
# ---------------------------------------------------------------------------


class ReservoirWorkspaceLayer(nn.Module):
    """Bidirectional multi-reservoir (fast + slow ESN) workspace.

    Processes the sequence through a fast+slow ESN pair in both the forward
    and backward directions, then concatenates the two state trajectories and
    projects the result back to ``hidden_dim`` via a learnable linear layer.

    **Frozen ESN weights**: The ``MultiReservoir`` is a plain Python/numpy
    object — its weights are *not* PyTorch parameters.  Only ``out_proj`` is
    trainable and receives gradients during backpropagation.  The ESN input is
    ``detach()``ed, so no gradient flows into the ESN.

    **Initialisation**: ``out_proj.weight`` is zero-initialized so the
    reservoir starts with no influence on the output (dormant start).  The
    gate scalar ``g3`` in the parent block is also zero-initialized.

    Args:
        config: Model configuration.
        layer_idx: Index of the enclosing decoder layer (used to vary the
            random seed across layers so each reservoir is unique).
    """

    def __init__(self, config: RWTransformerConfig, layer_idx: int) -> None:
        super().__init__()

        mr_config = MultiReservoirConfig(
            fast=ReservoirConfig(
                size=config.fast_reservoir_size,
                spectral_radius=config.fast_spectral_radius,
                leak_rate=config.fast_leak_rate,
                seed=config.reservoir_seed + layer_idx * 2,
            ),
            slow=ReservoirConfig(
                size=config.slow_reservoir_size,
                spectral_radius=config.slow_spectral_radius,
                leak_rate=config.slow_leak_rate,
                seed=config.reservoir_seed + layer_idx * 2 + 1,
            ),
        )
        # Stored as a plain Python object — not a PyTorch Module.
        # Reservoir weights are numpy arrays, not tracked by autograd.
        self.reservoir = MultiReservoir(mr_config, input_dim=config.hidden_dim)
        self.state_dim = self.reservoir.state_dim  # fast_size + slow_size

        # Bidirectional: fwd + bwd → 2 × state_dim input features
        self.out_proj = nn.Linear(2 * self.state_dim, config.hidden_dim, bias=False)
        # Zero init → dormant at training start; g3 gate also starts at 0
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bidirectional reservoir and project back to hidden_dim.

        Args:
            x: (B, T, hidden_dim) — detached from the computation graph
               internally before passing to the numpy ESN.

        Returns:
            (B, T, hidden_dim) — differentiable through ``out_proj`` only.
        """
        B, T, _ = x.shape

        # Detach: reservoir is a non-differentiable numpy computation.
        x_np = x.detach().cpu().float().numpy()  # (B, T, hidden_dim)

        fwd_states = np.empty((B, T, self.state_dim), dtype=np.float32)
        bwd_states = np.empty((B, T, self.state_dim), dtype=np.float32)

        for b in range(B):
            # Forward pass (left → right)
            self.reservoir.reset()
            fwd_states[b] = self.reservoir.forward(x_np[b])

            # Backward pass (right → left, then re-reverse to align positions)
            self.reservoir.reset()
            bwd_raw = self.reservoir.forward(np.ascontiguousarray(x_np[b, ::-1]))
            bwd_states[b] = bwd_raw[::-1].copy()

        combined = np.concatenate([fwd_states, bwd_states], axis=-1)  # (B, T, 2·state_dim)
        combined_t = torch.from_numpy(combined).to(dtype=x.dtype, device=x.device)
        return self.out_proj(combined_t)  # (B, T, hidden_dim)


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------


class RWTransformerBlock(nn.Module):
    """One RW-Transformer decoder block.

    Contains up to three parallel branches, all operating on the same
    RMSNorm-normalised hidden state:

        1. Causal self-attention (with RoPE)
        2. SwiGLU MLP
        3. Bidirectional reservoir workspace  *(only in reservoir layers)*

    Output mixing::

        h' = h + g1 · attn_out + g2 · mlp_out [+ g3 · reservoir_out]

    Gate initialisation: g1 = g2 = 1.0 (standard residual);
    g3 = 0.0 (reservoir starts dormant).

    Args:
        config: Model configuration.
        has_reservoir: Whether this block includes the reservoir branch.
        layer_idx: Global layer index (passed to reservoir for seed variety).
    """

    def __init__(
        self,
        config: RWTransformerConfig,
        has_reservoir: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.has_reservoir = has_reservoir

        # Branch 1: attention
        self.norm1 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)

        # Branch 2: MLP
        self.norm2 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = RWTMLP(config)

        # Gate scalars (attention and MLP start at 1 → standard residual)
        self.g1 = nn.Parameter(torch.ones(1))
        self.g2 = nn.Parameter(torch.ones(1))

        # Branch 3: reservoir workspace (optional)
        if has_reservoir:
            self.norm3 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
            self.reservoir_ws = ReservoirWorkspaceLayer(config, layer_idx)
            self.g3 = nn.Parameter(torch.zeros(1))  # dormant at init

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through this decoder block.

        Args:
            x: Hidden states, shape (B, T, hidden_dim).
            cos, sin: RoPE buffers, shape (T, head_dim).

        Returns:
            Updated hidden states, shape (B, T, hidden_dim).
        """
        attn_out = self.attn(self.norm1(x), cos, sin)
        mlp_out = self.mlp(self.norm2(x))

        h = x + self.g1 * attn_out + self.g2 * mlp_out

        if self.has_reservoir:
            res_out = self.reservoir_ws(self.norm3(x))
            h = h + self.g3 * res_out

        return h


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class RWTransformer(nn.Module):
    """Reservoir-Workspace Transformer (decoder-only, ~0.8B trainable params).

    A from-scratch hybrid language model that augments every N-th transformer
    layer with a bidirectional fast+slow ESN reservoir workspace.  The reservoir
    weights are frozen; all other weights (~0.8B) are trainable.

    See module docstring for the architecture diagram and parameter count.

    Example usage::

        from src.models.rw_transformer import RWTransformer, RWTransformerConfig

        cfg = RWTransformerConfig(vocab_size=1000, num_layers=2, hidden_dim=64,
                                  num_heads=4, head_dim=16, ffn_dim=256,
                                  max_seq_len=32)
        model = RWTransformer(cfg)
        input_ids = torch.randint(0, 1000, (2, 16))
        logits = model(input_ids)  # shape: (2, 16, 1000)

        # Backward pass
        loss = logits.mean()
        loss.backward()

        # Reset reservoir state between sequences
        model.reset_reservoirs()

    BF16 mixed precision::

        model = model.to(dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids)

    Gradient checkpointing (attention/MLP layers only)::

        from torch.utils.checkpoint import checkpoint

        # Wrap non-reservoir layer forwards manually; do NOT checkpoint
        # reservoir layers (they are stateful and would advance ESN state
        # twice during gradient recomputation).
    """

    def __init__(self, config: RWTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Set of layer indices that include a reservoir workspace branch
        self._reservoir_indices: frozenset[int] = frozenset(
            i for i in range(config.num_layers) if i % config.reservoir_every_n == 0
        )

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Decoder blocks
        self.layers = nn.ModuleList(
            [
                RWTransformerBlock(
                    config=config,
                    has_reservoir=(i in self._reservoir_indices),
                    layer_idx=i,
                )
                for i in range(config.num_layers)
            ]
        )

        # Final norm + language model head
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Pre-compute and cache RoPE buffers
        cos, sin = _precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialise weights (reservoir out_proj stays zero from its own __init__)
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Normal initialisation for all Linear and Embedding modules.

        Reservoir workspace ``out_proj`` weights are zero-initialized inside
        :class:`ReservoirWorkspaceLayer.__init__`; we restore them to zero
        after the blanket normal-init pass to ensure the dormant start.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        # Restore reservoir out_proj to zero (overwritten by the loop above)
        for layer in self.layers:
            if layer.has_reservoir:
                nn.init.zeros_(layer.reservoir_ws.out_proj.weight)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset_reservoirs(self) -> None:
        """Reset all reservoir states to zero.

        Call this at the start of each new sequence during inference or
        between sequences during training.
        """
        for layer in self.layers:
            if layer.has_reservoir:
                layer.reservoir_ws.reservoir.reset()

    def count_trainable_params(self) -> int:
        """Number of trainable parameters (deduplicates tied weights)."""
        seen: set[int] = set()
        count = 0
        for p in self.parameters():
            if p.requires_grad and p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                count += p.numel()
        return count

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass through the RW-Transformer.

        Args:
            input_ids: Token indices, shape (B, T).
            attention_mask: Reserved for future use (causal masking is handled
                automatically via ``is_causal=True`` in SDPA).

        Returns:
            Logits over the vocabulary, shape (B, T, vocab_size).

        Raises:
            AssertionError: If ``T > max_seq_len``.
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embed_tokens(input_ids)  # (B, T, hidden_dim)

        cos = self.rope_cos[:T]  # (T, head_dim)
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)

        x = self.norm(x)
        return self.lm_head(x)  # (B, T, vocab_size)
