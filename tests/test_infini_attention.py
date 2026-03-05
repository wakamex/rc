"""Tests for src/models/infini_attention.py.

All tests run on CPU without any HuggingFace model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.models.infini_attention import (
    CompressiveMemory,
    InfiniAttentionConfig,
    InfiniAttentionLayer,
    apply_infini_attention,
    get_infini_trainable_params,
    reset_infini_memory,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 16
BATCH = 2
SEQ = 8


def _make_identity_attn() -> nn.Module:
    """Minimal attention stub that returns hidden_states unchanged (as a tuple)."""

    class _FakeAttn(nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            # Mimic HF: return (attn_output, ...) tuple.
            return (hidden_states,)

    return _FakeAttn()


def _make_infini_layer(dropout: float = 0.0) -> InfiniAttentionLayer:
    return InfiniAttentionLayer(
        base_layer=_make_identity_attn(),
        hidden_dim=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        dropout=dropout,
    )


def _make_mock_model(num_layers: int = 6) -> nn.Module:
    """Build a minimal model stub with model.layers containing fake transformer layers."""

    class _FakeTransformerLayer(nn.Module):
        def __init__(self, is_attn: bool) -> None:
            super().__init__()
            if is_attn:
                self.self_attn = _make_identity_attn()

    class _FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = MagicMock()
            self.config.hidden_size = HIDDEN
            self.config.num_attention_heads = HEADS
            layers = []
            for i in range(num_layers):
                # Alternate: even layers have no attention (DeltaNet-like), odd have self_attn.
                layers.append(_FakeTransformerLayer(is_attn=(i % 2 == 1)))
            self.layers = nn.ModuleList(layers)

    # Wrap in outer model with model attribute
    class _Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _FakeModel()
            self.config = self.model.config

    return _Outer()


# ---------------------------------------------------------------------------
# CompressiveMemory
# ---------------------------------------------------------------------------


class TestCompressiveMemory:
    def test_init_shapes(self):
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        assert mem.M.shape == (HEADS, HEAD_DIM, HEAD_DIM)
        assert mem.z.shape == (HEADS, HEAD_DIM)

    def test_reset_zeros(self):
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        # Dirty the state.
        mem.M.fill_(1.0)
        mem.z.fill_(1.0)
        mem.reset()
        assert mem.M.sum().item() == 0.0
        assert mem.z.sum().item() == 0.0

    def test_phi_positive(self):
        x = torch.randn(2, HEADS, SEQ, HEAD_DIM)
        phi = CompressiveMemory._phi(x)
        assert (phi > 0).all(), "φ(x) must be strictly positive"

    def test_retrieve_output_shape(self):
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        Q = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)
        out = mem.retrieve(Q)
        assert out.shape == (BATCH, HEADS, SEQ, HEAD_DIM)

    def test_retrieve_with_zero_memory_near_zero(self):
        """With zero memory (M=0, z=0), retrieval output should be near zero."""
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        Q = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)
        out = mem.retrieve(Q)
        assert out.abs().max().item() < 1e-3

    def test_update_changes_memory(self):
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        K = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)
        V = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)
        M_before = mem.M.clone()
        z_before = mem.z.clone()
        mem.update(K, V)
        assert not torch.allclose(mem.M, M_before), "M should change after update"
        assert not torch.allclose(mem.z, z_before), "z should change after update"

    def test_update_then_retrieve_nonzero(self):
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        K = torch.ones(BATCH, HEADS, SEQ, HEAD_DIM) * 0.5
        V = torch.ones(BATCH, HEADS, SEQ, HEAD_DIM) * 0.5
        mem.update(K, V)
        Q = torch.ones(BATCH, HEADS, SEQ, HEAD_DIM) * 0.5
        out = mem.retrieve(Q)
        assert out.abs().max().item() > 0, "After update, retrieval should be non-zero"

    def test_update_detaches_from_graph(self):
        """Memory update should not build gradient graph through K/V."""
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        K = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, requires_grad=True)
        V = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, requires_grad=True)
        mem.update(K, V)
        # M should not have a grad_fn (it was updated with detached inputs).
        assert mem.M.grad_fn is None or not mem.M.requires_grad

    def test_sequential_updates_accumulate(self):
        mem = CompressiveMemory(num_heads=HEADS, head_dim=HEAD_DIM)
        K = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)
        V = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)
        mem.update(K, V)
        M_after_one = mem.M.clone()
        mem.update(K, V)
        M_after_two = mem.M.clone()
        assert not torch.allclose(M_after_one, M_after_two)


# ---------------------------------------------------------------------------
# InfiniAttentionLayer
# ---------------------------------------------------------------------------


class TestInfiniAttentionLayer:
    def test_output_shape_tensor(self):
        """When base layer returns a tensor, layer returns a tensor."""
        base = MagicMock(return_value=torch.zeros(BATCH, SEQ, HIDDEN))
        layer = InfiniAttentionLayer(base, HIDDEN, HEADS, HEAD_DIM)
        x = torch.randn(BATCH, SEQ, HIDDEN)
        out = layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (BATCH, SEQ, HIDDEN)

    def test_output_shape_tuple(self):
        """When base layer returns a tuple, layer returns a tuple."""
        layer = _make_infini_layer()
        x = torch.randn(BATCH, SEQ, HIDDEN)
        out = layer(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (BATCH, SEQ, HIDDEN)

    def test_memory_updated_after_forward(self):
        layer = _make_infini_layer()
        x = torch.randn(BATCH, SEQ, HIDDEN)
        M_before = layer.memory.M.clone()
        layer(x)
        assert not torch.allclose(layer.memory.M, M_before), "Memory should be updated after forward"

    def test_reset_memory(self):
        layer = _make_infini_layer()
        x = torch.randn(BATCH, SEQ, HIDDEN)
        layer(x)  # dirty the memory
        layer.reset_memory()
        assert layer.memory.M.sum().item() == 0.0
        assert layer.memory.z.sum().item() == 0.0

    def test_beta_is_parameter(self):
        layer = _make_infini_layer()
        assert isinstance(layer.beta, nn.Parameter)
        assert layer.beta.shape == (HEADS,)

    def test_gradient_flows_through_beta(self):
        layer = _make_infini_layer()
        x = torch.randn(BATCH, SEQ, HIDDEN)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        assert layer.beta.grad is not None, "Gradient should flow to beta"

    def test_gradient_flows_through_projections(self):
        """Q projection gets gradient through retrieval output.
        K/V projections are used only for the memory update (detached), so they
        do NOT get gradients within a single forward call — gradients would flow
        to K/V projections via the retrieval path in future forward calls that
        use the memory built from those projections.
        """
        layer = _make_infini_layer()
        x = torch.randn(BATCH, SEQ, HIDDEN)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        assert layer.mem_q_proj.weight.grad is not None, "Q projection should get gradient"
        # out_proj gets gradient because A_mem flows through it
        assert layer.mem_out_proj.weight.grad is not None, "out_proj should get gradient"

    def test_initialisation_near_zero_contribution(self):
        """With zero-init out_proj, memory contribution should start near zero."""
        layer = _make_infini_layer()
        x = torch.randn(BATCH, SEQ, HIDDEN)
        with torch.no_grad():
            out = layer(x)
        attn_reference = torch.zeros(BATCH, SEQ, HIDDEN)  # identity stub returns input
        if isinstance(out, tuple):
            out = out[0]
        # out should be close to x (the identity attention) since out_proj is zero-init.
        # (Not exactly x because beta is also applied to the mem_out_proj which starts at 0.)
        assert torch.allclose(out, x, atol=1e-5), \
            "With zero-init out_proj, layer should not modify attention output at init"

    def test_sequential_calls_accumulate_memory(self):
        layer = _make_infini_layer()
        x1 = torch.randn(BATCH, SEQ, HIDDEN)
        x2 = torch.randn(BATCH, SEQ, HIDDEN)
        layer(x1)
        M_after_x1 = layer.memory.M.clone()
        layer(x2)
        M_after_x2 = layer.memory.M.clone()
        assert not torch.allclose(M_after_x1, M_after_x2)


# ---------------------------------------------------------------------------
# apply_infini_attention
# ---------------------------------------------------------------------------


class TestApplyInfiniAttention:
    def test_wraps_correct_layers(self):
        model = _make_mock_model(num_layers=6)
        # Odd layers (1, 3, 5) have self_attn; even layers do not.
        wrapped = apply_infini_attention(model, layer_indices=[1, 3, 5],
                                         hidden_dim=HIDDEN, num_heads=HEADS)
        assert sorted(wrapped.keys()) == [1, 3, 5]

    def test_returns_infini_attention_layer_instances(self):
        model = _make_mock_model(num_layers=6)
        wrapped = apply_infini_attention(model, layer_indices=[1],
                                         hidden_dim=HIDDEN, num_heads=HEADS)
        assert isinstance(wrapped[1], InfiniAttentionLayer)

    def test_layer_installed_in_model(self):
        model = _make_mock_model(num_layers=6)
        apply_infini_attention(model, layer_indices=[1], hidden_dim=HIDDEN, num_heads=HEADS)
        layer_1 = model.model.layers[1]
        assert isinstance(layer_1.self_attn, InfiniAttentionLayer)

    def test_skips_layers_without_attn(self):
        """Even-indexed layers in the mock have no self_attn — should be skipped."""
        model = _make_mock_model(num_layers=6)
        wrapped = apply_infini_attention(model, layer_indices=[0, 1, 2],
                                         hidden_dim=HIDDEN, num_heads=HEADS)
        # Only layer 1 has self_attn; 0 and 2 do not.
        assert 0 not in wrapped
        assert 1 in wrapped
        assert 2 not in wrapped

    def test_out_of_range_index_skipped(self):
        model = _make_mock_model(num_layers=4)
        wrapped = apply_infini_attention(model, layer_indices=[1, 99],
                                         hidden_dim=HIDDEN, num_heads=HEADS)
        assert 99 not in wrapped

    def test_no_layer_indices_uses_default(self):
        """None layer_indices → automatically selects odd layers."""
        model = _make_mock_model(num_layers=6)
        wrapped = apply_infini_attention(model, hidden_dim=HIDDEN, num_heads=HEADS)
        # Default selects odd layers: 1, 3, 5 — all have self_attn.
        assert len(wrapped) > 0
        for idx in wrapped:
            assert idx % 2 == 1

    def test_raises_if_no_layers_found(self):
        """Model with no known layer attribute should raise ValueError."""

        class _NoLayerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = MagicMock()
                self.config.hidden_size = HIDDEN
                self.config.num_attention_heads = HEADS

        with pytest.raises(ValueError, match="Could not locate transformer layers"):
            apply_infini_attention(_NoLayerModel())


# ---------------------------------------------------------------------------
# reset_infini_memory
# ---------------------------------------------------------------------------


class TestResetInfiniMemory:
    def test_resets_all_infini_layers(self):
        model = _make_mock_model(num_layers=6)
        apply_infini_attention(model, layer_indices=[1, 3], hidden_dim=HIDDEN, num_heads=HEADS)

        # Dirty memory in both layers.
        for m in model.modules():
            if isinstance(m, InfiniAttentionLayer):
                m.memory.M.fill_(1.0)
                m.memory.z.fill_(1.0)

        reset_infini_memory(model)

        for m in model.modules():
            if isinstance(m, InfiniAttentionLayer):
                assert m.memory.M.sum().item() == 0.0
                assert m.memory.z.sum().item() == 0.0

    def test_no_op_on_model_without_infini(self):
        """reset_infini_memory should not raise on a plain model."""
        plain = nn.Linear(10, 10)
        reset_infini_memory(plain)  # should not raise


# ---------------------------------------------------------------------------
# get_infini_trainable_params
# ---------------------------------------------------------------------------


class TestGetInfiniTrainableParams:
    def test_returns_list_of_parameters(self):
        model = _make_mock_model(num_layers=4)
        apply_infini_attention(model, layer_indices=[1, 3], hidden_dim=HIDDEN, num_heads=HEADS)
        params = get_infini_trainable_params(model)
        assert isinstance(params, list)
        assert len(params) > 0
        for p in params:
            assert isinstance(p, nn.Parameter)

    def test_count_per_layer(self):
        """Each InfiniAttentionLayer contributes 5 parameter tensors:
        q_proj, k_proj, v_proj, out_proj weights + beta."""
        model = _make_mock_model(num_layers=4)
        apply_infini_attention(model, layer_indices=[1, 3], hidden_dim=HIDDEN, num_heads=HEADS)
        params = get_infini_trainable_params(model)
        # 2 layers × 5 parameter groups = 10 tensors
        assert len(params) == 10

    def test_no_duplicates(self):
        model = _make_mock_model(num_layers=4)
        apply_infini_attention(model, layer_indices=[1, 3], hidden_dim=HIDDEN, num_heads=HEADS)
        params = get_infini_trainable_params(model)
        # All parameter object ids should be unique.
        ids = [id(p) for p in params]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# InfiniAttentionConfig
# ---------------------------------------------------------------------------


class TestInfiniAttentionConfig:
    def test_default_instantiation(self):
        cfg = InfiniAttentionConfig()
        assert cfg.model_name == "qwen3.5-0.8b"
        assert cfg.max_steps == 5000
        assert cfg.lora_rank == 16

    def test_custom_values(self):
        cfg = InfiniAttentionConfig(max_steps=1000, lora_rank=8)
        assert cfg.max_steps == 1000
        assert cfg.lora_rank == 8
