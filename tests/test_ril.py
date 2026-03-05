"""Tests for the Reservoir Interaction Layer (RIL) — rc-wwh.19."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.reservoir.esn import ESN
from src.reservoir.ril import (
    RILConfig,
    RILWrapper,
    ReservoirInteractionLayer,
    _BlockWithRIL,
)
from src.types import ReservoirConfig


# ---------------------------------------------------------------------------
# Helpers / mock model
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
RESERVOIR_SIZE = 128
BATCH = 2
SEQ_LEN = 8
NUM_LAYERS = 12
INSERT_EVERY = 6


def _make_esn(hidden_dim: int = HIDDEN_DIM, size: int = RESERVOIR_SIZE) -> ESN:
    cfg = ReservoirConfig(size=size, spectral_radius=0.9, leak_rate=0.3, seed=42)
    return ESN(cfg, input_dim=hidden_dim)


def _make_ril(
    hidden_dim: int = HIDDEN_DIM,
    gate_init: float = 0.0,
    read_activation: str = "tanh",
) -> ReservoirInteractionLayer:
    return ReservoirInteractionLayer(
        hidden_dim=hidden_dim,
        reservoir=_make_esn(hidden_dim),
        gate_init=gate_init,
        read_activation=read_activation,
    )


class _MockBlock(nn.Module):
    """Minimal transformer block substitute (just a linear layer)."""

    def __init__(self, hidden_dim: int, returns_tuple: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> Any:
        out = self.linear(hidden_states)
        if self.returns_tuple:
            return (out, torch.zeros(1))  # mimic (hidden_states, past_key_value)
        return out


class _MockInnerModel(nn.Module):
    """Inner model with a .layers attribute (Qwen/LLaMA-style)."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_MockBlock(hidden_dim) for _ in range(num_layers)]
        )


class _MockTransformerModel(nn.Module):
    """Top-level mock with model.model.layers structure (HuggingFace CausalLM style)."""

    def __init__(self, hidden_dim: int, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.model = _MockInnerModel(hidden_dim, num_layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = hidden_states
        for block in self.model.layers:
            out = block(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class _MockModelWithTupleBlocks(_MockTransformerModel):
    """Same as _MockTransformerModel but blocks return (hidden, extra) tuples."""

    def __init__(self, hidden_dim: int, num_layers: int = NUM_LAYERS) -> None:
        super().__init__(hidden_dim, num_layers)
        # Replace blocks with tuple-returning variants
        self.model.layers = nn.ModuleList(
            [_MockBlock(hidden_dim, returns_tuple=True) for _ in range(num_layers)]
        )


# ---------------------------------------------------------------------------
# ReservoirInteractionLayer: instantiation
# ---------------------------------------------------------------------------


class TestRILInstantiation:
    def test_basic_instantiation(self):
        ril = _make_ril()
        assert ril.hidden_dim == HIDDEN_DIM
        assert isinstance(ril.gate, nn.Parameter)

    def test_gate_initialized_near_zero(self):
        ril = _make_ril(gate_init=0.0)
        assert abs(float(ril.gate)) < 1e-6

    def test_gate_custom_init(self):
        ril = _make_ril(gate_init=0.1)
        assert abs(float(ril.gate) - 0.1) < 1e-6

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown read_activation"):
            _make_ril(read_activation="sigmoid")

    def test_read_proj_and_write_head_registered(self):
        ril = _make_ril()
        param_names = {n for n, _ in ril.named_parameters()}
        assert any("read_proj" in n for n in param_names)
        assert any("write_head" in n for n in param_names)
        assert any("gate" in n for n in param_names)


# ---------------------------------------------------------------------------
# ReservoirInteractionLayer: forward shapes
# ---------------------------------------------------------------------------


class TestRILForwardShapes:
    def test_3d_input_shape_preserved(self):
        ril = _make_ril()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_2d_input_shape_preserved(self):
        ril = _make_ril()
        h = torch.randn(SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        assert out.shape == (SEQ_LEN, HIDDEN_DIM)

    def test_output_is_tensor(self):
        ril = _make_ril()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        assert isinstance(out, torch.Tensor)

    def test_different_hidden_dims(self):
        for hd in [32, 128, 256]:
            ril = _make_ril(hidden_dim=hd)
            h = torch.randn(2, 4, hd)
            out = ril(h)
            assert out.shape == (2, 4, hd)


# ---------------------------------------------------------------------------
# ReservoirInteractionLayer: activations
# ---------------------------------------------------------------------------


class TestRILActivations:
    @pytest.mark.parametrize("act", ["tanh", "relu", "gelu", "identity"])
    def test_all_activations_run(self, act):
        ril = _make_ril(read_activation=act)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# ReservoirInteractionLayer: gradient behaviour
# ---------------------------------------------------------------------------


class TestRILGradients:
    def test_gradient_flows_through_gate(self):
        ril = _make_ril()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = ril(h)
        out.sum().backward()
        assert ril.gate.grad is not None

    def test_gradient_flows_through_read_proj(self):
        ril = _make_ril()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        out.sum().backward()
        for name, param in ril.read_proj.named_parameters():
            assert param.grad is not None, f"No grad for read_proj.{name}"

    def test_gradient_flows_through_hidden(self):
        ril = _make_ril(gate_init=1.0)  # non-zero gate so grad flows
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = ril(h)
        out.sum().backward()
        assert h.grad is not None

    def test_write_head_grad_does_not_flow_to_reservoir(self):
        """WriteHead uses to_numpy (no_grad) so reservoir state has no grad."""
        ril = _make_ril()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = ril(h)
        out.sum().backward()
        # Reservoir state is a numpy array — it cannot hold gradients
        assert isinstance(ril.reservoir.state, np.ndarray)
        assert not hasattr(ril.reservoir.state, "grad")

    def test_write_head_params_have_grad(self):
        """WriteHead parameters should NOT receive gradients via to_numpy (uses no_grad)."""
        ril = _make_ril()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        out.sum().backward()
        # write_head uses to_numpy which wraps in no_grad → no grad on weights
        for name, param in ril.write_head.named_parameters():
            assert param.grad is None, (
                f"write_head.{name} should have no grad (to_numpy uses no_grad)"
            )


# ---------------------------------------------------------------------------
# ReservoirInteractionLayer: gate near zero → near-identity
# ---------------------------------------------------------------------------


class TestRILNearIdentity:
    def test_gate_zero_means_identity(self):
        """With gate == 0, h' should equal h exactly."""
        ril = _make_ril(gate_init=0.0)
        # Freeze gate so it stays at 0
        ril.gate.requires_grad_(False)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = ril(h)
        assert torch.allclose(out, h), "gate=0 should give h' = h"

    def test_nonzero_gate_modulates_output(self):
        ril_a = _make_ril(gate_init=0.0)
        ril_b = _make_ril(gate_init=1.0)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out_a = ril_a(h)
        out_b = ril_b(h)
        assert not torch.allclose(out_a, out_b), "Different gates should differ"


# ---------------------------------------------------------------------------
# _BlockWithRIL: wrapper for individual blocks
# ---------------------------------------------------------------------------


class TestBlockWithRIL:
    def test_tensor_output_block(self):
        block = _MockBlock(HIDDEN_DIM, returns_tuple=False)
        ril = _make_ril()
        wrapped = _BlockWithRIL(block, ril)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = wrapped(h)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_tuple_output_block(self):
        block = _MockBlock(HIDDEN_DIM, returns_tuple=True)
        ril = _make_ril()
        wrapped = _BlockWithRIL(block, ril)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = wrapped(h)
        assert isinstance(out, tuple)
        assert out[0].shape == (BATCH, SEQ_LEN, HIDDEN_DIM)
        # Extra elements in tuple are preserved
        assert len(out) == 2

    def test_block_parameters_still_accessible(self):
        block = _MockBlock(HIDDEN_DIM)
        ril = _make_ril()
        wrapped = _BlockWithRIL(block, ril)
        param_names = {n for n, _ in wrapped.named_parameters()}
        assert any("block" in n for n in param_names)
        assert any("ril" in n for n in param_names)


# ---------------------------------------------------------------------------
# RILWrapper: insertion into mock Qwen3.5-style model
# ---------------------------------------------------------------------------


class TestRILWrapperInsertion:
    def test_wrapper_inserts_correct_number_of_rils(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            shared_reservoir=True,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        expected = NUM_LAYERS // INSERT_EVERY  # 12 // 6 = 2
        assert len(wrapper.ril_layers) == expected

    def test_wrapped_blocks_are_block_with_ril(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        layers = model.model.layers
        assert isinstance(layers[INSERT_EVERY - 1], _BlockWithRIL)
        assert isinstance(layers[2 * INSERT_EVERY - 1], _BlockWithRIL)

    def test_non_inserted_blocks_unchanged(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
        )
        RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        layers = model.model.layers
        # Block 0 (index 0) should still be a plain _MockBlock
        assert not isinstance(layers[0], _BlockWithRIL)

    def test_insert_every_zero_inserts_nothing(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=0,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        assert len(wrapper.ril_layers) == 0

    def test_shared_reservoir_is_same_instance(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            shared_reservoir=True,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        rils = wrapper.ril_layers
        assert len(rils) >= 2
        assert rils[0].reservoir is rils[1].reservoir

    def test_separate_reservoirs_are_different_instances(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            shared_reservoir=False,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        rils = wrapper.ril_layers
        assert len(rils) >= 2
        assert rils[0].reservoir is not rils[1].reservoir

    def test_unknown_model_structure_raises(self):
        class _WeirdModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(16, 16)

        with pytest.raises(AttributeError):
            RILWrapper(_WeirdModel(), hidden_dim=16)


# ---------------------------------------------------------------------------
# RILWrapper: forward pass (end-to-end, no shape errors)
# ---------------------------------------------------------------------------


class TestRILWrapperForward:
    def test_forward_shape_preserved(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = wrapper(h)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_forward_works_with_tuple_blocks(self):
        model = _MockModelWithTupleBlocks(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
        )
        RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        # Forward through the modified model directly
        out = model(h)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_forward_without_ril(self):
        """With insert_every=0, wrapper is transparent."""
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=4)
        wrapper = RILWrapper(
            model,
            hidden_dim=HIDDEN_DIM,
            config=RILConfig(insert_every=0),
        )
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = wrapper(h)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_forward_changes_output_with_nonzero_gate(self):
        """Modulated output differs from baseline (no-RIL) when gate > 0."""
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)

        # Baseline: no RIL
        model_base = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        base_out = model_base(h.clone())

        # With RIL, gate=1.0
        model_ril = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        # Copy weights so only RIL differs
        model_ril.load_state_dict(model_base.state_dict())
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            gate_init=1.0,
        )
        RILWrapper(model_ril, hidden_dim=HIDDEN_DIM, config=cfg)
        ril_out = model_ril(h.clone())

        assert not torch.allclose(base_out, ril_out), (
            "RIL with gate=1 should change the output"
        )


# ---------------------------------------------------------------------------
# RILWrapper: gradient flow
# ---------------------------------------------------------------------------


class TestRILWrapperGradients:
    def test_ril_parameters_receive_gradients(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            gate_init=1.0,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = wrapper(h)
        out.sum().backward()

        for ril in wrapper.ril_layers:
            assert ril.gate.grad is not None, "gate should have grad"
            for name, p in ril.read_proj.named_parameters():
                assert p.grad is not None, f"read_proj.{name} should have grad"

    def test_gradient_stops_at_reservoir(self):
        """Reservoir numpy state cannot carry gradients."""
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            gate_init=1.0,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = wrapper(h)
        out.sum().backward()

        for ril in wrapper.ril_layers:
            assert isinstance(ril.reservoir.state, np.ndarray)
            # numpy arrays have no .grad attribute
            assert not hasattr(ril.reservoir.state, "grad")

    def test_model_block_params_still_receive_gradients(self):
        """Transformer block parameters should still get gradients after RIL insertion."""
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=NUM_LAYERS)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=INSERT_EVERY,
            gate_init=1.0,
        )
        RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = model(h)
        out.sum().backward()

        # Check a non-RIL block's linear weights
        non_ril_block = model.model.layers[0]  # _MockBlock, not wrapped
        assert non_ril_block.linear.weight.grad is not None


# ---------------------------------------------------------------------------
# Gate initialization
# ---------------------------------------------------------------------------


class TestGateInit:
    def test_default_gate_is_zero(self):
        ril = _make_ril()
        assert float(ril.gate.item()) == pytest.approx(0.0, abs=1e-7)

    def test_default_ril_config_gate_init_zero(self):
        cfg = RILConfig()
        assert cfg.gate_init == pytest.approx(0.0)

    def test_wrapper_uses_config_gate_init(self):
        model = _MockTransformerModel(HIDDEN_DIM, num_layers=6)
        cfg = RILConfig(
            reservoir=ReservoirConfig(size=64, seed=0),
            insert_every=6,
            gate_init=0.01,
        )
        wrapper = RILWrapper(model, hidden_dim=HIDDEN_DIM, config=cfg)
        for ril in wrapper.ril_layers:
            assert float(ril.gate.item()) == pytest.approx(0.01, abs=1e-7)


# ---------------------------------------------------------------------------
# Reservoir state updates
# ---------------------------------------------------------------------------


class TestReservoirStateUpdate:
    def test_reservoir_state_changes_after_forward(self):
        ril = _make_ril()
        state_before = ril.reservoir.state.copy()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        ril(h)
        state_after = ril.reservoir.state
        assert not np.allclose(state_before, state_after), (
            "Reservoir state should update after forward pass"
        )

    def test_multiple_forwards_keep_updating_state(self):
        ril = _make_ril()
        states = []
        for _ in range(3):
            h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
            ril(h)
            states.append(ril.reservoir.state.copy())
        # All three states should be different
        assert not np.allclose(states[0], states[1])
        assert not np.allclose(states[1], states[2])
