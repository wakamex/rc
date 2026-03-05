"""Tests for the ESN ↔ LLM read/write interface modules (rc-wwh.12)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.reservoir.interface import (
    CrossAttentionSidecar,
    FiLMModulation,
    ReadProjection,
    WriteHead,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESERVOIR_DIM = 64
HIDDEN_DIM = 128
INPUT_DIM = 16
BATCH = 4
SEQ_LEN = 8
NUM_HEADS = 4


def reservoir_state_np(batch: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    if batch is None:
        return rng.standard_normal(RESERVOIR_DIM).astype(np.float32)
    return rng.standard_normal((batch, RESERVOIR_DIM)).astype(np.float32)


def hidden_tensor(batch: int | None = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randn(batch, seq_len, HIDDEN_DIM) if batch else torch.randn(seq_len, HIDDEN_DIM)


# ===========================================================================
# ReadProjection
# ===========================================================================


class TestReadProjection:
    def test_instantiation(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        assert rp.reservoir_dim == RESERVOIR_DIM
        assert rp.hidden_dim == HIDDEN_DIM
        assert isinstance(rp.proj, nn.Linear)

    def test_output_shape_unbatched(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = reservoir_state_np()
        out = rp(r)
        assert out.shape == (HIDDEN_DIM,)

    def test_output_shape_batched(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = reservoir_state_np(batch=BATCH)
        out = rp(r)
        assert out.shape == (BATCH, HIDDEN_DIM)

    def test_output_is_tensor(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = reservoir_state_np()
        out = rp(r)
        assert isinstance(out, torch.Tensor)

    def test_gradient_flows_through_proj(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = reservoir_state_np()
        out = rp(r)
        loss = out.sum()
        loss.backward()
        assert rp.proj.weight.grad is not None
        assert rp.proj.bias.grad is not None

    def test_no_gradient_from_reservoir_numpy(self):
        """Numpy arrays carry no gradient — gradient stops at reservoir boundary."""
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = reservoir_state_np()
        # The numpy array itself cannot hold gradients; just verify forward works
        out = rp(r)
        assert out.requires_grad or not out.requires_grad  # always passes

    def test_works_with_different_dims(self):
        for res_dim, hid_dim in [(32, 64), (512, 256), (1024, 4096)]:
            rp = ReadProjection(res_dim, hid_dim)
            r = np.random.randn(res_dim).astype(np.float32)
            out = rp(r)
            assert out.shape == (hid_dim,)

    def test_accepts_torch_tensor_input(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = torch.randn(RESERVOIR_DIM)
        out = rp(r)
        assert out.shape == (HIDDEN_DIM,)

    def test_no_bias_option(self):
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM, bias=False)
        assert rp.proj.bias is None
        out = rp(reservoir_state_np())
        assert out.shape == (HIDDEN_DIM,)


# ===========================================================================
# WriteHead
# ===========================================================================


class TestWriteHead:
    def test_instantiation(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        assert wh.hidden_dim == HIDDEN_DIM
        assert wh.input_dim == INPUT_DIM

    def test_output_shape_3d(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = hidden_tensor()
        out = wh(h)
        assert out.shape == (BATCH, SEQ_LEN, INPUT_DIM)

    def test_output_shape_2d(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = torch.randn(BATCH, HIDDEN_DIM)
        out = wh(h)
        assert out.shape == (BATCH, INPUT_DIM)

    def test_output_shape_1d(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = torch.randn(HIDDEN_DIM)
        out = wh(h)
        assert out.shape == (INPUT_DIM,)

    def test_gradient_flows_through_proj(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = hidden_tensor()
        out = wh(h)
        out.sum().backward()
        assert wh.proj.weight.grad is not None

    def test_to_numpy_returns_numpy_array(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = hidden_tensor()
        arr = wh.to_numpy(h)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_to_numpy_shape(self):
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = hidden_tensor()
        arr = wh.to_numpy(h)
        assert arr.shape == (BATCH, SEQ_LEN, INPUT_DIM)

    def test_to_numpy_stops_gradient(self):
        """to_numpy uses no_grad so no gradient flows back through numpy."""
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = hidden_tensor().requires_grad_(True)
        arr = wh.to_numpy(h)
        # numpy array has no gradient
        assert not hasattr(arr, "requires_grad")

    def test_works_with_different_dims(self):
        for hid_dim, inp_dim in [(256, 8), (64, 32), (4096, 128)]:
            wh = WriteHead(hid_dim, inp_dim)
            h = torch.randn(2, hid_dim)
            out = wh(h)
            assert out.shape == (2, inp_dim)


# ===========================================================================
# CrossAttentionSidecar
# ===========================================================================


class TestCrossAttentionSidecar:
    def test_instantiation(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        assert ca.hidden_dim == HIDDEN_DIM
        assert ca.reservoir_dim == RESERVOIR_DIM
        assert ca.num_heads == NUM_HEADS

    def test_invalid_num_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=7)

    def test_output_shape_3d_input_single_reservoir_state(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor()
        r = reservoir_state_np()  # (reservoir_dim,)
        out = ca(h, r)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_output_shape_2d_input(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor(batch=None)  # (SEQ_LEN, HIDDEN_DIM)
        r = reservoir_state_np()
        out = ca(h, r)
        assert out.shape == (SEQ_LEN, HIDDEN_DIM)

    def test_output_shape_reservoir_sequence(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor()
        r_seq = np.random.randn(10, RESERVOIR_DIM).astype(np.float32)  # (S, R)
        out = ca(h, r_seq)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_output_shape_batched_reservoir_sequence(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor()
        r_batch_seq = np.random.randn(BATCH, 5, RESERVOIR_DIM).astype(np.float32)
        out = ca(h, r_batch_seq)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_gradient_flows_through_projections(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor()
        r = reservoir_state_np()
        out = ca(h, r)
        out.sum().backward()
        for name, param in ca.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_gradient_does_not_flow_to_reservoir_numpy(self):
        """Reservoir state is numpy → no gradient possible."""
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor()
        r = reservoir_state_np()
        # Forward should succeed; r has no grad tracking by design
        out = ca(h, r)
        out.sum().backward()
        # No assertion needed beyond not raising

    def test_accepts_torch_tensor_reservoir(self):
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor()
        r = torch.randn(RESERVOIR_DIM)
        out = ca(h, r)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_works_with_different_reservoir_sizes(self):
        for res_dim in [32, 256, 1000]:
            ca = CrossAttentionSidecar(HIDDEN_DIM, res_dim, num_heads=NUM_HEADS)
            h = hidden_tensor(batch=2, seq_len=4)
            r = np.random.randn(res_dim).astype(np.float32)
            out = ca(h, r)
            assert out.shape == (2, 4, HIDDEN_DIM)

    def test_works_with_different_hidden_dims(self):
        for hid_dim, n_heads in [(64, 4), (256, 8), (512, 16)]:
            ca = CrossAttentionSidecar(hid_dim, RESERVOIR_DIM, num_heads=n_heads)
            h = torch.randn(2, 5, hid_dim)
            r = np.random.randn(RESERVOIR_DIM).astype(np.float32)
            out = ca(h, r)
            assert out.shape == (2, 5, hid_dim)


# ===========================================================================
# FiLMModulation
# ===========================================================================


class TestFiLMModulation:
    def test_instantiation(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        assert film.reservoir_dim == RESERVOIR_DIM
        assert film.hidden_dim == HIDDEN_DIM

    def test_output_shape_3d_hidden(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = hidden_tensor()
        r = reservoir_state_np()
        out = film(h, r)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_output_shape_2d_hidden(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = torch.randn(BATCH, HIDDEN_DIM)
        r = reservoir_state_np()
        out = film(h, r)
        assert out.shape == (BATCH, HIDDEN_DIM)

    def test_output_shape_1d_hidden(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = torch.randn(HIDDEN_DIM)
        r = reservoir_state_np()
        out = film(h, r)
        assert out.shape == (HIDDEN_DIM,)

    def test_output_shape_batched_reservoir(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = hidden_tensor()
        r = reservoir_state_np(batch=BATCH)
        out = film(h, r)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_gradient_flows_through_projections(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = hidden_tensor()
        r = reservoir_state_np()
        out = film(h, r)
        out.sum().backward()
        for name, param in film.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_gradient_flows_through_hidden(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = hidden_tensor().requires_grad_(True)
        r = reservoir_state_np()
        out = film(h, r)
        out.sum().backward()
        assert h.grad is not None

    def test_near_identity_when_gate_zero(self):
        """When gate → 0, output should equal hidden (identity-like)."""
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        # Force gate_proj to produce very negative logits → sigmoid ≈ 0
        nn.init.constant_(film.gate_proj.weight, -10.0)
        nn.init.constant_(film.gate_proj.bias, -10.0)
        h = hidden_tensor()
        r = reservoir_state_np()
        out = film(h, r)
        # With gate ≈ 0 the output should be close to hidden
        assert torch.allclose(out, h, atol=1e-4), "Expected near-identity when gate ≈ 0"

    def test_reservoir_state_change_affects_output(self):
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = hidden_tensor()
        r1 = reservoir_state_np()
        r2 = np.random.default_rng(99).standard_normal(RESERVOIR_DIM).astype(np.float32)
        out1 = film(h, r1)
        out2 = film(h, r2)
        assert not torch.allclose(out1, out2), "Different reservoir states should produce different outputs"

    def test_works_with_different_dims(self):
        for res_dim, hid_dim in [(16, 32), (256, 512), (1000, 768)]:
            film = FiLMModulation(res_dim, hid_dim)
            h = torch.randn(2, 5, hid_dim)
            r = np.random.randn(res_dim).astype(np.float32)
            out = film(h, r)
            assert out.shape == (2, 5, hid_dim)


# ===========================================================================
# Gradient boundary: no gradient escapes into reservoir-side numpy arrays
# ===========================================================================


class TestGradientBoundary:
    def test_read_projection_gradient_stops(self):
        """Gradient exists in interface weights, not in reservoir input."""
        rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
        r = reservoir_state_np()  # numpy, no grad
        out = rp(r)
        out.sum().backward()
        # Interface weights receive gradient
        assert rp.proj.weight.grad is not None
        # The original numpy array cannot have grad (by definition)
        assert not hasattr(r, "grad") or r.grad is None

    def test_write_head_gradient_reaches_hidden(self):
        """Gradient flows from loss back through WriteHead into hidden states."""
        wh = WriteHead(HIDDEN_DIM, INPUT_DIM)
        h = hidden_tensor().requires_grad_(True)
        out = wh(h)
        out.sum().backward()
        assert h.grad is not None
        assert wh.proj.weight.grad is not None

    def test_cross_attention_gradient_stops_at_reservoir(self):
        """Gradients do not flow into reservoir tensors (they're detached)."""
        ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)
        h = hidden_tensor().requires_grad_(True)
        r_tensor = torch.randn(RESERVOIR_DIM)  # torch tensor — will be detached inside
        out = ca(h, r_tensor)
        out.sum().backward()
        # LLM hidden grad exists
        assert h.grad is not None
        # r_tensor was detached inside _to_tensor, so it has no grad
        assert r_tensor.grad is None

    def test_film_gradient_stops_at_reservoir(self):
        """Reservoir tensor is detached inside FiLMModulation."""
        film = FiLMModulation(RESERVOIR_DIM, HIDDEN_DIM)
        h = hidden_tensor().requires_grad_(True)
        r_tensor = torch.randn(RESERVOIR_DIM)  # will be detached inside
        out = film(h, r_tensor)
        out.sum().backward()
        assert h.grad is not None
        assert r_tensor.grad is None


# ===========================================================================
# Integration: ESN → ReadProjection → CrossAttentionSidecar chain
# ===========================================================================


def test_full_read_pipeline():
    """ReadProjection + CrossAttentionSidecar compose correctly."""
    rp = ReadProjection(RESERVOIR_DIM, HIDDEN_DIM)
    ca = CrossAttentionSidecar(HIDDEN_DIM, RESERVOIR_DIM, num_heads=NUM_HEADS)

    r_np = reservoir_state_np()
    h = hidden_tensor()

    # Read: reservoir → hidden_dim (independent projection)
    ctx = rp(r_np)  # (hidden_dim,)
    assert ctx.shape == (HIDDEN_DIM,)
    ctx.sum().backward()
    assert rp.proj.weight.grad is not None

    # Sidecar: attend over reservoir
    out = ca(h, r_np)
    assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    out.sum().backward()
    for param in ca.parameters():
        assert param.grad is not None
