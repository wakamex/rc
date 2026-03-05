"""Tests for the RW-Transformer architecture (rc-wwh.24).

Covers:
- ~0.8B parameter count for the default config
- Correct logit shape from a forward pass
- Gradient flow through attention, MLP, and reservoir gate/projection
- Gate initialisation (g1=g2=1.0, g3=0.0)
- Reservoir insertion at the correct layer indices
- Reservoir reset resets internal ESN state
- BF16 / mixed-precision compatibility
"""
from __future__ import annotations

import pytest
import torch

from src.models.rw_transformer import (
    RWTransformer,
    RWTransformerConfig,
    RWTransformerBlock,
    ReservoirWorkspaceLayer,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny model for fast tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cfg() -> RWTransformerConfig:
    """A small config that keeps tests fast (CPU, seconds)."""
    return RWTransformerConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        head_dim=16,
        ffn_dim=256,
        max_seq_len=32,
        reservoir_every_n=2,
        fast_reservoir_size=20,
        slow_reservoir_size=20,
        tie_embeddings=True,
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg: RWTransformerConfig) -> RWTransformer:
    return RWTransformer(tiny_cfg)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def test_tiny_model_param_count(tiny_cfg: RWTransformerConfig, tiny_model: RWTransformer):
    """Verify exact parameter count for the tiny model against the formula.

    Formula (tie_embeddings=True):
        embed_tokens       : vocab_size × hidden_dim
        n_res blocks       : (4×h² + 3×h×f + 2×h + 2) + (h + 2·state_dim·h + 1)
        n_plain blocks     : 4×h² + 3×h×f + 2×h + 2
        final norm         : h
    where h = hidden_dim, f = ffn_dim, state_dim = fast + slow.
    """
    h = tiny_cfg.hidden_dim       # 64
    f = tiny_cfg.ffn_dim          # 256
    n = tiny_cfg.num_layers       # 4
    v = tiny_cfg.vocab_size       # 1000
    state_dim = tiny_cfg.fast_reservoir_size + tiny_cfg.slow_reservoir_size  # 40
    bidirectional_dim = 2 * state_dim  # 80

    # Layer indices with reservoirs: 0, 2 (every 2 out of 4)
    n_res = sum(1 for i in range(n) if i % tiny_cfg.reservoir_every_n == 0)
    n_plain = n - n_res

    # Params per plain block
    block_plain = (
        4 * h * h          # q, k, v, out projections
        + 3 * h * f        # gate, up, down MLP
        + 2 * h            # norm1 + norm2 weights
        + 2                # g1 + g2 scalars
    )

    # Extra params per reservoir block
    block_res_extra = (
        h                  # norm3 weight
        + bidirectional_dim * h  # out_proj weight
        + 1                # g3 scalar
    )
    block_res = block_plain + block_res_extra

    expected = (
        v * h              # embed_tokens (= lm_head, tied)
        + n_plain * block_plain
        + n_res * block_res
        + h                # final norm
    )

    actual = tiny_model.count_trainable_params()
    assert actual == expected, (
        f"Expected {expected} trainable params, got {actual}.\n"
        f"  n_plain={n_plain}, n_res={n_res}, block_plain={block_plain}, "
        f"block_res_extra={block_res_extra}"
    )


@pytest.mark.slow
def test_default_model_approx_08b():
    """Default config model should have ~0.8B trainable parameters (±10%).

    This test creates a full-size model (~800M params, ~3.2 GB float32).
    It is marked 'slow' and may be skipped with: pytest -m 'not slow'.
    """
    cfg = RWTransformerConfig()
    model = RWTransformer(cfg)
    n = model.count_trainable_params()
    target = 800_000_000
    tolerance = 0.10
    assert abs(n - target) / target <= tolerance, (
        f"Expected ~0.8B params (±10%), got {n:,} ({n / 1e9:.3f}B)"
    )


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------


def test_forward_shape(tiny_model: RWTransformer, tiny_cfg: RWTransformerConfig):
    """Forward pass must return logits of shape (B, T, vocab_size)."""
    B, T = 2, 16
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits = tiny_model(input_ids)
    assert logits.shape == (B, T, tiny_cfg.vocab_size), (
        f"Expected logits shape ({B}, {T}, {tiny_cfg.vocab_size}), got {logits.shape}"
    )


def test_forward_batch_size_1(tiny_model: RWTransformer, tiny_cfg: RWTransformerConfig):
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    logits = tiny_model(input_ids)
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)


def test_forward_single_token(tiny_model: RWTransformer, tiny_cfg: RWTransformerConfig):
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 1))
    logits = tiny_model(input_ids)
    assert logits.shape == (1, 1, tiny_cfg.vocab_size)


def test_forward_max_seq_len(tiny_model: RWTransformer, tiny_cfg: RWTransformerConfig):
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.max_seq_len))
    logits = tiny_model(input_ids)
    assert logits.shape == (1, tiny_cfg.max_seq_len, tiny_cfg.vocab_size)


def test_forward_exceeds_seq_len_raises(
    tiny_model: RWTransformer, tiny_cfg: RWTransformerConfig
):
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.max_seq_len + 1))
    with pytest.raises(AssertionError):
        tiny_model(input_ids)


# ---------------------------------------------------------------------------
# Backward pass / gradient flow
# ---------------------------------------------------------------------------


def test_backward_gradients_flow(tiny_cfg: RWTransformerConfig):
    """Backward pass must not raise and key parameters must receive gradients."""
    model = RWTransformer(tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

    # Attention projections must have gradients
    for i, layer in enumerate(model.layers):
        assert layer.attn.q_proj.weight.grad is not None, (
            f"Layer {i} q_proj has no gradient"
        )
        assert layer.mlp.gate_proj.weight.grad is not None, (
            f"Layer {i} gate_proj has no gradient"
        )
        # Gate scalars
        assert layer.g1.grad is not None, f"Layer {i} g1 has no gradient"
        assert layer.g2.grad is not None, f"Layer {i} g2 has no gradient"

        if layer.has_reservoir:
            # g3 gate must receive gradient (it multiplies the reservoir output)
            assert layer.g3.grad is not None, (
                f"Layer {i} g3 (reservoir gate) has no gradient"
            )
            # out_proj must receive gradient (projects reservoir→hidden)
            assert layer.reservoir_ws.out_proj.weight.grad is not None, (
                f"Layer {i} reservoir out_proj has no gradient"
            )


def test_backward_embedding_gradient(tiny_cfg: RWTransformerConfig):
    """Embedding table receives gradients from the language model loss."""
    model = RWTransformer(tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
    logits = model(input_ids)
    logits.mean().backward()
    assert model.embed_tokens.weight.grad is not None


# ---------------------------------------------------------------------------
# Gate initialisation
# ---------------------------------------------------------------------------


def test_gate_initialisation(tiny_model: RWTransformer):
    """g1=g2=1.0 (standard residual), g3=0.0 (reservoir starts dormant)."""
    for layer in tiny_model.layers:
        assert torch.allclose(layer.g1.data, torch.ones(1)), (
            f"g1 should be 1.0 at init, got {layer.g1.data}"
        )
        assert torch.allclose(layer.g2.data, torch.ones(1)), (
            f"g2 should be 1.0 at init, got {layer.g2.data}"
        )
        if layer.has_reservoir:
            assert torch.allclose(layer.g3.data, torch.zeros(1)), (
                f"g3 should be 0.0 at init, got {layer.g3.data}"
            )


def test_reservoir_out_proj_zero_init(tiny_model: RWTransformer):
    """Reservoir out_proj must be zero-initialized (dormant start)."""
    for layer in tiny_model.layers:
        if layer.has_reservoir:
            w = layer.reservoir_ws.out_proj.weight
            assert torch.all(w == 0), (
                f"reservoir_ws.out_proj.weight not zero-initialized: max={w.abs().max()}"
            )


# ---------------------------------------------------------------------------
# Architecture / reservoir insertion
# ---------------------------------------------------------------------------


def test_reservoir_layer_indices(tiny_cfg: RWTransformerConfig, tiny_model: RWTransformer):
    """Reservoir workspace present exactly at indices where i % reservoir_every_n == 0."""
    for i, layer in enumerate(tiny_model.layers):
        expected = (i % tiny_cfg.reservoir_every_n == 0)
        assert layer.has_reservoir == expected, (
            f"Layer {i}: expected has_reservoir={expected}, got {layer.has_reservoir}"
        )


def test_reservoir_indices_set(tiny_cfg: RWTransformerConfig, tiny_model: RWTransformer):
    expected = frozenset(
        i for i in range(tiny_cfg.num_layers) if i % tiny_cfg.reservoir_every_n == 0
    )
    assert tiny_model._reservoir_indices == expected


def test_no_reservoir_layers_have_no_norm3(
    tiny_cfg: RWTransformerConfig, tiny_model: RWTransformer
):
    for i, layer in enumerate(tiny_model.layers):
        if not layer.has_reservoir:
            assert not hasattr(layer, "norm3"), (
                f"Layer {i} (no reservoir) should not have norm3"
            )
            assert not hasattr(layer, "reservoir_ws"), (
                f"Layer {i} (no reservoir) should not have reservoir_ws"
            )
            assert not hasattr(layer, "g3"), (
                f"Layer {i} (no reservoir) should not have g3"
            )


# ---------------------------------------------------------------------------
# Reservoir reset
# ---------------------------------------------------------------------------


def test_reset_reservoirs_zeroes_esn_state(
    tiny_cfg: RWTransformerConfig, tiny_model: RWTransformer
):
    """reset_reservoirs() must zero out all ESN states."""
    import numpy as np

    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    tiny_model(input_ids)  # advances ESN state

    tiny_model.reset_reservoirs()

    for layer in tiny_model.layers:
        if layer.has_reservoir:
            mr = layer.reservoir_ws.reservoir
            assert np.allclose(mr.fast.state, 0.0), "fast ESN not zeroed"
            assert np.allclose(mr.slow.state, 0.0), "slow ESN not zeroed"


# ---------------------------------------------------------------------------
# BF16 / mixed precision
# ---------------------------------------------------------------------------


def test_forward_bf16(tiny_cfg: RWTransformerConfig):
    """Model should produce finite logits in BF16."""
    model = RWTransformer(tiny_cfg).to(dtype=torch.bfloat16)
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    logits = model(input_ids)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all(), "BF16 logits contain inf/nan"


# ---------------------------------------------------------------------------
# Tied embeddings
# ---------------------------------------------------------------------------


def test_tied_embeddings_share_weight(
    tiny_cfg: RWTransformerConfig, tiny_model: RWTransformer
):
    """embed_tokens and lm_head must share the same weight tensor."""
    assert tiny_model.embed_tokens.weight is tiny_model.lm_head.weight, (
        "embed_tokens.weight and lm_head.weight should be the same tensor"
    )


def test_untied_embeddings_independent():
    """When tie_embeddings=False the weights must be separate objects."""
    cfg = RWTransformerConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        head_dim=16,
        ffn_dim=64,
        max_seq_len=16,
        reservoir_every_n=2,
        fast_reservoir_size=10,
        slow_reservoir_size=10,
        tie_embeddings=False,
    )
    model = RWTransformer(cfg)
    assert model.embed_tokens.weight is not model.lm_head.weight


# ---------------------------------------------------------------------------
# Reservoir workspace layer (unit)
# ---------------------------------------------------------------------------


def test_reservoir_workspace_output_shape(tiny_cfg: RWTransformerConfig):
    """ReservoirWorkspaceLayer output must match (B, T, hidden_dim)."""
    layer = ReservoirWorkspaceLayer(tiny_cfg, layer_idx=0)
    x = torch.randn(2, 8, tiny_cfg.hidden_dim)
    out = layer(x)
    assert out.shape == (2, 8, tiny_cfg.hidden_dim), (
        f"Expected (2, 8, {tiny_cfg.hidden_dim}), got {out.shape}"
    )


def test_reservoir_workspace_state_dim(tiny_cfg: RWTransformerConfig):
    """state_dim should equal fast_size + slow_size."""
    layer = ReservoirWorkspaceLayer(tiny_cfg, layer_idx=0)
    expected = tiny_cfg.fast_reservoir_size + tiny_cfg.slow_reservoir_size
    assert layer.state_dim == expected


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


def test_forward_deterministic(tiny_model: RWTransformer, tiny_cfg: RWTransformerConfig):
    """Two identical forward calls (after reservoir reset) must produce the
    same logits."""
    tiny_model.eval()
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (1, 6))

    tiny_model.reset_reservoirs()
    logits_a = tiny_model(input_ids)

    tiny_model.reset_reservoirs()
    logits_b = tiny_model(input_ids)

    assert torch.allclose(logits_a, logits_b, atol=1e-5), (
        "Forward pass not deterministic after reservoir reset"
    )
