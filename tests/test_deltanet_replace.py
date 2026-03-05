"""Tests for DeltaNet block replacement module (rc-wwh.20)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.reservoir.deltanet_replace import DeltaNetReplacer, ReservoirBlock
from src.types import ReservoirConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
BATCH = 2
SEQ_LEN = 8
N_DELTANET = 18  # mirrors Qwen3.5-0.8B


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class _MockDeltaNetAttn(nn.Module):
    """Minimal mock that stands in for a GatedDeltaNet attention module."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(x)


class _MockLayer(nn.Module):
    """Transformer layer with either full attention or a deltanet sub-module."""

    def __init__(self, hidden_dim: int, use_deltanet: bool = False) -> None:
        super().__init__()
        if use_deltanet:
            # Named 'deltanet' so get_deltanet_layers() picks it up
            self.deltanet = _MockDeltaNetAttn(hidden_dim)
        else:
            self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self, "deltanet"):
            h = self.deltanet(x, **kwargs)
        else:
            h = self.attn(x)
        return h + self.mlp(x)


class MockQwen35(nn.Module):
    """Mock Qwen3.5 with N_DELTANET interleaved DeltaNet layers."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_deltanet: int = N_DELTANET) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(n_deltanet * 2):
            layers.append(_MockLayer(hidden_dim, use_deltanet=(i % 2 == 1)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


def _make_model() -> MockQwen35:
    return MockQwen35()


def _make_cfg(size: int = 128) -> ReservoirConfig:
    return ReservoirConfig(size=size, seed=42, sparsity=0.1)


# ---------------------------------------------------------------------------
# ReservoirBlock tests
# ---------------------------------------------------------------------------


class TestReservoirBlock:
    def test_output_shape_3d(self):
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg())
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_output_shape_2d(self):
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg())
        x = torch.randn(SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        assert out.shape == (SEQ_LEN, HIDDEN_DIM)

    def test_output_no_nan(self):
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg())
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_swap_to_esn_sets_flag(self):
        original = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg(), original_module=original)
        block.swap_to_original()
        assert not block._use_esn
        block.swap_to_esn()
        assert block._use_esn

    def test_swap_to_original_no_module_raises(self):
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg())
        with pytest.raises(RuntimeError, match="No original module"):
            block.swap_to_original()

    def test_original_mode_delegates(self):
        original = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg(), original_module=original)
        block.swap_to_original()
        x = torch.randn(BATCH, HIDDEN_DIM)
        out = block(x)
        expected = original(x)
        assert torch.allclose(out, expected)

    def test_gradient_flows_through_read_proj(self):
        """Gradients flow through read_proj; write_head uses to_numpy() so no grad."""
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg())
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        out.sum().backward()
        for name, p in block.read_proj.named_parameters():
            assert p.grad is not None, f"No grad for read_proj.{name}"
        # write_head.to_numpy() is wrapped in no_grad → no gradient
        for name, p in block.write_head.named_parameters():
            assert p.grad is None, f"Unexpected grad for write_head.{name}"

    def test_custom_esn_input_dim(self):
        esn_input_dim = 32
        block = ReservoirBlock(HIDDEN_DIM, _make_cfg(), esn_input_dim=esn_input_dim)
        assert block.write_head.input_dim == esn_input_dim
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# DeltaNetReplacer tests
# ---------------------------------------------------------------------------


class TestDeltaNetReplacer:
    def test_identifies_correct_number_of_deltanet_layers(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [], _make_cfg(), hidden_dim=HIDDEN_DIM)
        assert replacer.total_deltanet_layers == N_DELTANET

    def test_replace_one_layer(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0], _make_cfg(), hidden_dim=HIDDEN_DIM)
        assert replacer.num_replaced == 1

    def test_replace_six_layers(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, list(range(6)), _make_cfg(), hidden_dim=HIDDEN_DIM)
        assert replacer.num_replaced == 6

    def test_replace_all_18_layers(self):
        model = _make_model()
        replacer = DeltaNetReplacer(
            model, list(range(N_DELTANET)), _make_cfg(), hidden_dim=HIDDEN_DIM
        )
        assert replacer.num_replaced == N_DELTANET

    def test_output_valid_shape_after_replacement(self):
        model = _make_model()
        DeltaNetReplacer(model, [0, 1, 2], _make_cfg(), hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_output_no_nan_after_replacement(self):
        model = _make_model()
        DeltaNetReplacer(
            model, list(range(N_DELTANET)), _make_cfg(), hidden_dim=HIDDEN_DIM
        )
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_param_count_accounting_nonempty(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0], _make_cfg(), hidden_dim=HIDDEN_DIM)
        report = replacer.param_report()
        assert report["params_removed"] > 0
        assert report["params_added"] > 0
        assert report["num_replaced"] == 1
        assert report["total_deltanet"] == N_DELTANET
        # param_delta is reported (can be positive or negative)
        assert "param_delta" in report

    def test_param_delta_equals_added_minus_removed(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, list(range(6)), _make_cfg(), hidden_dim=HIDDEN_DIM)
        assert replacer.param_delta == replacer.params_added - replacer.params_removed

    def test_ab_swap_esn_mode_produces_valid_output(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0, 1], _make_cfg(), hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        replacer.swap_all_to_esn()
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_ab_swap_original_mode_produces_valid_output(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0, 1], _make_cfg(), hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        replacer.swap_all_to_original()
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_ab_swap_produces_different_outputs(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0], _make_cfg(), hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)

        replacer.swap_all_to_esn()
        out_esn = model(x)

        replacer.swap_all_to_original()
        out_orig = model(x)

        assert not torch.allclose(out_esn, out_orig), (
            "ESN and original DeltaNet should produce different outputs"
        )

    def test_ab_swap_round_trip(self):
        """ESN → original → ESN gives same ESN output each time (deterministic)."""
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0], _make_cfg(), hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)

        replacer.swap_all_to_esn()
        out1 = model(x)

        replacer.swap_all_to_original()
        model(x)  # run in original mode

        replacer.swap_all_to_esn()
        out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_invalid_index_raises(self):
        model = _make_model()
        with pytest.raises(IndexError):
            DeltaNetReplacer(model, [N_DELTANET], _make_cfg(), hidden_dim=HIDDEN_DIM)

    def test_negative_index_raises(self):
        model = _make_model()
        with pytest.raises(IndexError):
            DeltaNetReplacer(model, [-1], _make_cfg(), hidden_dim=HIDDEN_DIM)

    def test_no_replacement_leaves_model_intact(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [], _make_cfg(), hidden_dim=HIDDEN_DIM)
        assert replacer.num_replaced == 0
        assert replacer.params_removed == 0
        assert replacer.params_added == 0
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_replaced_modules_are_reservoir_blocks(self):
        model = _make_model()
        replacer = DeltaNetReplacer(model, [0, 2, 4], _make_cfg(), hidden_dim=HIDDEN_DIM)
        for idx in [0, 2, 4]:
            path = replacer._deltanet_paths[idx]
            block = replacer._get_module_at(path)
            assert isinstance(block, ReservoirBlock), (
                f"Layer {idx} at '{path}' should be a ReservoirBlock"
            )
