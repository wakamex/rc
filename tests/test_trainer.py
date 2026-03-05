"""Tests for src/training/lora_trainer.py.

Test coverage:
- Smoke test: 10-step training loop on a tiny dummy model (CPU, no OOM)
- Loss decreases over training steps
- Checkpoint save + load round-trip
- Only expected parameters have gradients after freeze_base_model
- MixedDataLoader mixing works
- Optimizer has separate param groups for LoRA vs interface
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from src.training.lora_trainer import (
    LoRATrainer,
    LoRATrainingConfig,
    MixedDataLoader,
)

# ---------------------------------------------------------------------------
# Tiny models for testing
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
HIDDEN_SIZE = 16
SEQ_LEN = 6
BATCH_SIZE = 2
LORA_RANK = 4


class _LMOutput:
    """Minimal output object with .loss and .logits attributes."""

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class TinyLoRAModel(nn.Module):
    """Tiny model with LoRA params baked into the forward pass.

    Base weights (_q_weight, _v_weight, lm_head.weight, embed.weight) are
    meant to be frozen. The LoRA params (lora_A, lora_B) remain trainable
    and are used in the forward computation, so gradients flow through them.

    This simulates what PEFT does (W + BA) without requiring PEFT.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        rank: int = LORA_RANK,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # Base projection weights (will be frozen)
        self._q_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self._v_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        # LoRA decomposition: effective = _q_weight + lora_B @ lora_A
        self.lora_A = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, rank))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> _LMOutput:
        x = self.embed(input_ids)
        # q-projection with LoRA: (W + BA)x
        q = nn.functional.linear(x, self._q_weight) + nn.functional.linear(
            nn.functional.linear(x, self.lora_A), self.lora_B
        )
        v = nn.functional.linear(x, self._v_weight)
        x = q + v
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return _LMOutput(loss=loss, logits=logits)


class TinyLM(nn.Module):
    """Tiny causal LM with nn.Linear projections (PEFT-compatible)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> _LMOutput:
        x = self.embed(input_ids)
        x = self.q_proj(x)
        x = self.v_proj(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return _LMOutput(loss=loss, logits=logits)


class TinyInterface(nn.Module):
    """Minimal interface module (stand-in for ReadProjection/WriteHead)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _make_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _make_dataset(n: int = 20) -> list[dict[str, torch.Tensor]]:
    return [_make_batch() for _ in range(n)]


# ---------------------------------------------------------------------------
# PEFT helper
# ---------------------------------------------------------------------------


def _apply_lora_with_peft(model: nn.Module, config: LoRATrainingConfig) -> nn.Module:
    """Try applying PEFT LoRA; skip the test if it fails."""
    from src.training.lora_trainer import HAS_PEFT

    if not HAS_PEFT:
        pytest.skip("peft not installed")
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias=config.bias,
        )
        return get_peft_model(model, lora_config)
    except Exception as exc:
        pytest.skip(f"PEFT apply_lora failed: {exc}")


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config(tmp_path: Path) -> LoRATrainingConfig:
    return LoRATrainingConfig(
        lora_rank=LORA_RANK,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        lora_lr=1e-2,
        interface_lr=1e-2,
        num_warmup_steps=0,
        num_training_steps=10,
        gradient_checkpointing=False,
        mixed_precision=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_every_n_steps=100,
        val_every_n_steps=50,
        val_steps=2,
        use_wandb=False,
    )


# ---------------------------------------------------------------------------
# Tests: freeze_base_model
# ---------------------------------------------------------------------------


class TestFreezeBaseModel:
    def test_only_lora_params_have_grad(self) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, f"Expected {name!r} to require grad"
            else:
                assert not param.requires_grad, f"Expected {name!r} to NOT require grad"

    def test_non_lora_params_frozen(self) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        # embed, _q_weight, _v_weight, lm_head should all be frozen
        assert len(frozen) >= 3
        assert any("embed" in n or "weight" in n for n in frozen)

    def test_lora_params_remain_trainable(self) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        lora_params = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(lora_params) == 2  # lora_A and lora_B
        assert all("lora_" in n for n in lora_params)


# ---------------------------------------------------------------------------
# Tests: optimizer param groups
# ---------------------------------------------------------------------------


class TestOptimizerParamGroups:
    def test_lora_group_uses_lora_lr(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)
        opt = trainer._build_optimizer(model, None, base_config)

        lrs = [g["lr"] for g in opt.param_groups]
        assert base_config.lora_lr in lrs

    def test_interface_group_uses_interface_lr(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        interface = TinyInterface()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)
        opt = trainer._build_optimizer(model, interface, base_config)

        lrs = [g["lr"] for g in opt.param_groups]
        assert base_config.interface_lr in lrs

    def test_two_param_groups_with_interface(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        interface = TinyInterface()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)
        opt = trainer._build_optimizer(model, interface, base_config)
        assert len(opt.param_groups) >= 2

    def test_no_interface_single_lora_group(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)
        opt = trainer._build_optimizer(model, None, base_config)
        assert len(opt.param_groups) == 1  # only LoRA group

    def test_raises_when_no_trainable_params(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLM()
        # Freeze everything — no LoRA params, no interface
        for p in model.parameters():
            p.requires_grad_(False)
        trainer = LoRATrainer()
        with pytest.raises(ValueError, match="No trainable parameters"):
            trainer._build_optimizer(model, None, base_config)


# ---------------------------------------------------------------------------
# Tests: 10-step smoke training
# ---------------------------------------------------------------------------


class TestSmokeTraining:
    """10-step smoke test: pipeline runs without OOM on a small dummy model."""

    def test_smoke_10_steps_no_oom(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        result = trainer.train(model, None, None, _make_dataset(), base_config)

        assert result["steps_trained"] == 10
        assert result["final_loss"] is not None
        assert len(result["losses"]) == 10

    def test_smoke_with_interface(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        interface = TinyInterface()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        result = trainer.train(model, None, interface, _make_dataset(), base_config)

        assert result["steps_trained"] == 10
        assert result["final_loss"] is not None

    def test_smoke_finite_loss(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        result = trainer.train(model, None, None, _make_dataset(), base_config)

        assert all(torch.isfinite(torch.tensor(lo)) for lo in result["losses"])

    def test_smoke_returns_dict_keys(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        result = trainer.train(model, None, None, _make_dataset(), base_config)

        assert "steps_trained" in result
        assert "final_loss" in result
        assert "losses" in result
        assert "best_val_loss" in result


# ---------------------------------------------------------------------------
# Tests: loss decreases
# ---------------------------------------------------------------------------


class TestLossDecreases:
    """Verify that training on a fixed dataset reduces loss over steps."""

    def test_loss_decreases_over_steps(self, base_config: LoRATrainingConfig) -> None:
        torch.manual_seed(42)
        # Fixed tiny dataset — model should quickly memorise it
        dataset = _make_dataset(n=4)

        config = replace(
            base_config,
            num_training_steps=40,
            lora_lr=5e-2,
        )

        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        result = trainer.train(model, None, None, dataset, config)
        losses = result["losses"]

        # Second-half average should be lower than first-half average
        mid = len(losses) // 2
        first_avg = sum(losses[:mid]) / mid
        second_avg = sum(losses[mid:]) / len(losses[mid:])
        assert second_avg < first_avg, (
            f"Loss did not decrease: first_half={first_avg:.4f}, second_half={second_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_checkpoint_creates_dir(
        self, base_config: LoRATrainingConfig, tmp_path: Path
    ) -> None:
        config = replace(base_config, checkpoint_dir=str(tmp_path / "ckpts"))
        model = TinyLoRAModel()
        trainer = LoRATrainer()

        ckpt_path = trainer.save_checkpoint(model, None, step=5, config=config)
        assert ckpt_path.exists()
        assert (ckpt_path / "meta.pt").exists()

    def test_save_checkpoint_with_interface(
        self, base_config: LoRATrainingConfig, tmp_path: Path
    ) -> None:
        config = replace(base_config, checkpoint_dir=str(tmp_path / "ckpts"))
        model = TinyLoRAModel()
        interface = TinyInterface()
        trainer = LoRATrainer()

        ckpt_path = trainer.save_checkpoint(model, interface, step=10, config=config)
        assert (ckpt_path / "interface_weights.pt").exists()

    def test_load_checkpoint_restores_interface(
        self, base_config: LoRATrainingConfig, tmp_path: Path
    ) -> None:
        config = replace(base_config, checkpoint_dir=str(tmp_path / "ckpts"))

        interface_orig = TinyInterface()
        nn.init.constant_(interface_orig.proj.weight, 0.123)

        model = TinyLoRAModel()
        trainer = LoRATrainer()

        ckpt_path = trainer.save_checkpoint(model, interface_orig, step=1, config=config)

        interface_new = TinyInterface()
        nn.init.zeros_(interface_new.proj.weight)

        meta = trainer.load_checkpoint(model, interface_new, ckpt_path)

        assert torch.allclose(
            interface_new.proj.weight,
            torch.full_like(interface_new.proj.weight, 0.123),
        ), "Interface weights not restored correctly after load_checkpoint"
        assert meta.get("step") == 1

    def test_load_checkpoint_meta_contains_step(
        self, base_config: LoRATrainingConfig, tmp_path: Path
    ) -> None:
        config = replace(base_config, checkpoint_dir=str(tmp_path / "ckpts"))
        model = TinyLoRAModel()
        trainer = LoRATrainer()

        ckpt_path = trainer.save_checkpoint(model, None, step=42, config=config)
        meta = trainer.load_checkpoint(model, None, ckpt_path)
        assert meta["step"] == 42

    def test_train_saves_checkpoint_at_interval(
        self, base_config: LoRATrainingConfig, tmp_path: Path
    ) -> None:
        config = replace(
            base_config,
            checkpoint_dir=str(tmp_path / "ckpts"),
            num_training_steps=10,
            save_every_n_steps=5,
        )
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        trainer.train(model, None, None, _make_dataset(), config)

        assert (tmp_path / "ckpts" / "step_5").exists()
        assert (tmp_path / "ckpts" / "step_10").exists()

    def test_lora_weights_round_trip(
        self, base_config: LoRATrainingConfig, tmp_path: Path
    ) -> None:
        """Save lora_A/lora_B and verify they load back correctly."""
        config = replace(base_config, checkpoint_dir=str(tmp_path / "ckpts"))
        model = TinyLoRAModel()
        trainer = LoRATrainer()

        # Set known LoRA values
        nn.init.constant_(model.lora_A, 0.5)
        nn.init.constant_(model.lora_B, 0.1)

        ckpt_path = trainer.save_checkpoint(model, None, step=1, config=config)

        model2 = TinyLoRAModel()
        nn.init.zeros_(model2.lora_A)
        nn.init.zeros_(model2.lora_B)

        trainer.load_checkpoint(model2, None, ckpt_path)

        assert torch.allclose(model2.lora_A, torch.full_like(model2.lora_A, 0.5))
        assert torch.allclose(model2.lora_B, torch.full_like(model2.lora_B, 0.1))


# ---------------------------------------------------------------------------
# Tests: PEFT integration
# ---------------------------------------------------------------------------


class TestPeftLoRA:
    """Tests that use PEFT directly (skipped if peft is not installed/compatible)."""

    def test_apply_lora_adds_trainable_params(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLM()
        lora_model = _apply_lora_with_peft(model, base_config)

        trainable = [n for n, p in lora_model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0
        assert any("lora_" in n for n in trainable)

    def test_freeze_after_lora_leaves_lora_trainable(
        self, base_config: LoRATrainingConfig
    ) -> None:
        model = TinyLM()
        lora_model = _apply_lora_with_peft(model, base_config)

        trainer = LoRATrainer()
        trainer.freeze_base_model(lora_model)

        for name, param in lora_model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, f"LoRA param {name!r} should require grad"
            else:
                assert not param.requires_grad, f"Base param {name!r} should be frozen"

    def test_peft_smoke_10_steps(self, base_config: LoRATrainingConfig) -> None:
        model = TinyLM()
        lora_model = _apply_lora_with_peft(model, base_config)

        trainer = LoRATrainer()
        trainer.freeze_base_model(lora_model)

        result = trainer.train(lora_model, None, None, _make_dataset(), base_config)

        assert result["steps_trained"] == 10
        assert result["final_loss"] is not None


# ---------------------------------------------------------------------------
# Tests: MixedDataLoader
# ---------------------------------------------------------------------------


class TestMixedDataLoader:
    def test_produces_batches(self) -> None:
        loader = MixedDataLoader({"a": _make_dataset(5), "b": _make_dataset(5)}, {"a": 0.5, "b": 0.5})
        batch = next(iter(loader))
        assert "input_ids" in batch

    def test_normalises_ratios(self) -> None:
        loader = MixedDataLoader({"a": _make_dataset(5), "b": _make_dataset(5)}, {"a": 3.0, "b": 1.0})
        assert abs(loader.ratios["a"] - 0.75) < 1e-6
        assert abs(loader.ratios["b"] - 0.25) < 1e-6

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            MixedDataLoader({}, {})

    def test_train_with_mixed_loader(self, base_config: LoRATrainingConfig) -> None:
        loader = MixedDataLoader(
            {"a": _make_dataset(10), "b": _make_dataset(10)},
            {"a": 0.7, "b": 0.3},
        )
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        result = trainer.train(model, None, None, loader, base_config)
        assert result["steps_trained"] == 10


# ---------------------------------------------------------------------------
# Tests: early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_early_stopping_stops_training(self, base_config: LoRATrainingConfig) -> None:
        config = replace(
            base_config,
            num_training_steps=100,
            val_every_n_steps=5,
            val_steps=2,
            early_stopping_patience=2,
            early_stopping_min_delta=1e10,  # impossible to beat → stops immediately
        )
        model = TinyLoRAModel()
        trainer = LoRATrainer()
        trainer.freeze_base_model(model)

        dataset = _make_dataset(20)
        result = trainer.train(model, None, None, dataset, config, val_dataset=dataset)

        # Should stop well before 100 steps
        assert result["steps_trained"] < 100
