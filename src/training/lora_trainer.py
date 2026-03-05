"""LoRA + adapter training pipeline for frozen LLM + reservoir interface.

Implements the training loop for:
  frozen LLM + LoRA adapters + reservoir read/write interface modules.

Only LoRA params and interface params (ReadProjection, WriteHead,
CrossAttentionSidecar) are trained; the base model is frozen.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Names that identify interface module params (trained at interface_lr)
_INTERFACE_MODULE_NAMES = ("ReadProjection", "WriteHead", "CrossAttentionSidecar")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainingConfig:
    """Full configuration for LoRA + adapter training."""

    # LoRA adapter settings
    lora_rank: int = 16  # rank r (4–64)
    lora_alpha: int = 32  # LoRA scaling alpha
    lora_dropout: float = 0.1
    target_modules: list[str] | None = None  # None → use model defaults
    bias: str = "none"  # "none" | "all" | "lora_only"

    # Separate learning rates
    lora_lr: float = 2e-4
    interface_lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_eps: float = 1e-8

    # LR schedule: cosine with linear warmup
    num_warmup_steps: int = 100
    num_training_steps: int = 1000

    # Training loop
    gradient_checkpointing: bool = True
    mixed_precision: bool = True  # BF16 on CUDA; ignored on CPU

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 100

    # Early stopping (uses val loss)
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Validation
    val_every_n_steps: int = 50
    val_steps: int = 10  # number of val batches per evaluation

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "lrs-lora"
    wandb_run_name: str | None = None

    # Dataset mixing: {dataset_name: weight} (weights are normalised)
    dataset_mixing_ratios: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


class MixedDataLoader:
    """Infinite iterator that mixes multiple datasets by sampling ratios.

    Args:
        datasets: Mapping of dataset name → iterable of batches.
        ratios: Sampling weights per dataset (need not sum to 1).
    """

    def __init__(self, datasets: dict[str, Any], ratios: dict[str, float]) -> None:
        if not datasets:
            raise ValueError("datasets must be non-empty")
        total = sum(ratios.values())
        self.datasets = datasets
        self.ratios = {k: v / total for k, v in ratios.items()}

    def __iter__(self):  # type: ignore[override]
        iterators = {k: iter(v) for k, v in self.datasets.items()}
        names = list(self.ratios.keys())
        weights = torch.tensor([self.ratios[n] for n in names], dtype=torch.float)
        while True:
            idx = int(torch.multinomial(weights, 1).item())
            name = names[idx]
            try:
                yield next(iterators[name])
            except StopIteration:
                iterators[name] = iter(self.datasets[name])
                yield next(iterators[name])


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _cosine_with_warmup(num_warmup_steps: int, num_training_steps: int):
    """Return a LambdaLR lambda for cosine decay with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


class LoRATrainer:
    """Trains a frozen LLM with LoRA adapters + optional reservoir interface.

    Usage::

        config = LoRATrainingConfig(lora_rank=16, num_training_steps=1000)
        trainer = LoRATrainer()
        model = trainer.apply_lora(base_model, config)
        trainer.freeze_base_model(model)
        result = trainer.train(model, reservoir, interface, dataset, config)
    """

    # ------------------------------------------------------------------
    # LoRA application
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        model: nn.Module,
        config: LoRATrainingConfig,
        target_modules: list[str] | None = None,
    ) -> nn.Module:
        """Wrap *model* with PEFT LoRA adapters.

        Args:
            model: Base model (HuggingFace PreTrainedModel or nn.Module).
            config: Training config supplying lora_rank, lora_alpha, etc.
            target_modules: Override config.target_modules.

        Returns:
            PEFT-wrapped model with LoRA adapters added.

        Raises:
            ImportError: If ``peft`` is not installed.
        """
        if not HAS_PEFT:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft>=0.10"
            )
        targets = target_modules or config.target_modules
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=targets,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_config)

    # ------------------------------------------------------------------
    # Parameter freezing
    # ------------------------------------------------------------------

    def freeze_base_model(self, model: nn.Module) -> None:
        """Freeze all parameters that are not LoRA adapter weights.

        After calling this, only parameters whose names contain ``lora_``
        will have ``requires_grad=True``.
        """
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(
        self,
        model: nn.Module,
        interface: nn.Module | None,
        config: LoRATrainingConfig,
    ) -> AdamW:
        """Build AdamW with separate learning rates for LoRA vs interface.

        Parameter groups:
        - LoRA params (``lora_`` in name): trained at ``config.lora_lr``
        - Interface params: trained at ``config.interface_lr``
        - Any other trainable params: trained at ``config.lora_lr`` as fallback
        """
        lora_params: list[nn.Parameter] = []
        interface_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        if interface is not None:
            for param in interface.parameters():
                if param.requires_grad:
                    interface_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if lora_params:
            param_groups.append(
                {"params": lora_params, "lr": config.lora_lr, "weight_decay": config.weight_decay}
            )
        if interface_params:
            param_groups.append(
                {
                    "params": interface_params,
                    "lr": config.interface_lr,
                    "weight_decay": config.weight_decay,
                }
            )
        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": config.lora_lr,
                    "weight_decay": config.weight_decay,
                }
            )

        if not param_groups:
            raise ValueError(
                "No trainable parameters found. "
                "Did you call freeze_base_model before apply_lora?"
            )

        return AdamW(param_groups, eps=config.adam_eps)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(self, model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        """Compute language modelling loss from a batch dict.

        Tries, in order:
        1. Pass ``labels`` and use ``outputs.loss`` (HuggingFace convention).
        2. Compute cross-entropy from ``outputs.logits`` manually.

        Batch keys used:
        - ``input_ids``: required
        - ``labels``: optional; defaults to ``input_ids``
        - ``attention_mask``: optional
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask")

        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, labels=labels, **kwargs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if not isinstance(logits, torch.Tensor):
            raise ValueError("Cannot compute loss: model output has no .loss and no .logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        model: nn.Module,
        interface: nn.Module | None,
        step: int,
        config: LoRATrainingConfig,
        val_loss: float | None = None,
    ) -> Path:
        """Save LoRA adapter weights and interface state dict.

        Saves only the *trainable* parts (LoRA + interface), not the frozen
        base model, keeping checkpoint files small.

        Returns:
            Path to the checkpoint directory.
        """
        ckpt_dir = Path(config.checkpoint_dir) / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # LoRA weights — prefer PEFT's built-in save
        if HAS_PEFT and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(ckpt_dir / "lora_adapter"))
        else:
            lora_state = {
                k: v
                for k, v in model.state_dict().items()
                if any(tag in k for tag in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"])
            }
            torch.save(lora_state, ckpt_dir / "lora_weights.pt")

        # Interface weights
        if interface is not None:
            torch.save(interface.state_dict(), ckpt_dir / "interface_weights.pt")

        # Metadata
        meta: dict[str, Any] = {"step": step}
        if val_loss is not None:
            meta["val_loss"] = val_loss
        # Store config as plain dict (avoids pickle issues with dataclass)
        meta["config"] = {k: v for k, v in config.__dict__.items()}
        torch.save(meta, ckpt_dir / "meta.pt")

        return ckpt_dir

    def load_checkpoint(
        self,
        model: nn.Module,
        interface: nn.Module | None,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        """Load LoRA adapter + interface weights from a saved checkpoint.

        Args:
            model: Model with LoRA adapters already applied (same architecture).
            interface: Interface module (same architecture as when saved).
            checkpoint_path: Path to checkpoint directory returned by save_checkpoint.

        Returns:
            Metadata dict (step, val_loss, config, …).
        """
        ckpt_dir = Path(checkpoint_path)

        lora_adapter_dir = ckpt_dir / "lora_adapter"
        lora_weights_file = ckpt_dir / "lora_weights.pt"

        if lora_adapter_dir.exists() and HAS_PEFT and hasattr(model, "load_adapter"):
            model.load_adapter(str(lora_adapter_dir), adapter_name="default")
        elif lora_weights_file.exists():
            lora_state = torch.load(lora_weights_file, map_location="cpu", weights_only=True)
            model.load_state_dict(lora_state, strict=False)

        if interface is not None:
            iface_file = ckpt_dir / "interface_weights.pt"
            if iface_file.exists():
                iface_state = torch.load(iface_file, map_location="cpu", weights_only=True)
                interface.load_state_dict(iface_state)

        meta_file = ckpt_dir / "meta.pt"
        if meta_file.exists():
            return torch.load(meta_file, map_location="cpu", weights_only=False)
        return {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        model: nn.Module,
        val_dataset: Any,
        config: LoRATrainingConfig,
        device: torch.device,
        use_bf16: bool,
    ) -> float:
        """Evaluate model on val_dataset; returns mean loss."""
        model.eval()
        val_losses: list[float] = []
        val_iter = iter(val_dataset)

        with torch.no_grad():
            for _ in range(config.val_steps):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = self._compute_loss(model, batch)
                else:
                    loss = self._compute_loss(model, batch)
                val_losses.append(loss.item())

        model.train()
        return sum(val_losses) / len(val_losses) if val_losses else float("inf")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        model: nn.Module,
        reservoir: Any | None,
        interface: nn.Module | None,
        dataset: Any,
        config: LoRATrainingConfig,
        val_dataset: Any | None = None,
    ) -> dict[str, Any]:
        """Run the LoRA + adapter training loop.

        Args:
            model: Language model with LoRA already applied and base frozen.
                   The model should return outputs with a ``.loss`` attribute
                   (standard HuggingFace CausalLM), or the trainer falls back
                   to computing cross-entropy from ``.logits``.
            reservoir: Reservoir object (``Reservoir`` protocol) used during
                       inference; its state is reset each batch but its weights
                       are **not** optimised. Pass ``None`` if unused.
            interface: Interface module (ReadProjection / WriteHead /
                       CrossAttentionSidecar) — its params are trained at
                       ``config.interface_lr``. Pass ``None`` if unused.
            dataset: Iterable of batch dicts:
                     ``{"input_ids": Tensor, "labels": Tensor, ...}``.
                     Supports :class:`MixedDataLoader` or any Python iterable.
            config: :class:`LoRATrainingConfig` instance.
            val_dataset: Optional iterable for early stopping / validation.

        Returns:
            Dict with keys:
            - ``steps_trained``: int
            - ``final_loss``: float
            - ``losses``: list[float] — per-step training losses
            - ``best_val_loss``: float | None
        """
        device = next(model.parameters()).device
        use_bf16 = config.mixed_precision and device.type == "cuda"

        # Gradient checkpointing
        if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # Optimizer
        optimizer = self._build_optimizer(model, interface, config)

        # LR scheduler: cosine with warmup
        scheduler = LambdaLR(
            optimizer,
            _cosine_with_warmup(config.num_warmup_steps, config.num_training_steps),
        )

        # Mixed-precision scaler (CUDA only)
        scaler: torch.cuda.amp.GradScaler | None = None
        if use_bf16:
            scaler = torch.cuda.amp.GradScaler()

        # Wandb
        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={k: v for k, v in config.__dict__.items()},
                reinit=True,
            )

        model.train()
        if interface is not None:
            interface.train()

        best_val_loss = float("inf")
        patience_counter = 0
        losses: list[float] = []
        final_step = 0

        data_iter = iter(dataset)

        for step in range(config.num_training_steps):
            # Fetch next batch (restart iterator on exhaustion)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataset)
                batch = next(data_iter)

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Reset reservoir state each batch (stateless per-sequence)
            if reservoir is not None and hasattr(reservoir, "reset"):
                reservoir.reset()

            optimizer.zero_grad()

            if use_bf16 and scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = self._compute_loss(model, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                trainable = [p for p in model.parameters() if p.requires_grad]
                if interface is not None:
                    trainable += [p for p in interface.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self._compute_loss(model, batch)
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                if interface is not None:
                    trainable += [p for p in interface.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()

            scheduler.step()
            loss_val = loss.item()
            losses.append(loss_val)
            final_step = step + 1

            # Compute gradient norm for logging
            grad_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = math.sqrt(grad_norm)

            # Wandb logging
            if config.use_wandb and HAS_WANDB:
                log_dict: dict[str, Any] = {
                    "train/loss": loss_val,
                    "train/grad_norm": grad_norm,
                    "train/lr_lora": optimizer.param_groups[0]["lr"],
                    "step": step,
                }
                if len(optimizer.param_groups) > 1:
                    log_dict["train/lr_interface"] = optimizer.param_groups[1]["lr"]
                # Reservoir state stats
                if reservoir is not None and hasattr(reservoir, "state"):
                    r_state = reservoir.state
                    if isinstance(r_state, torch.Tensor):
                        log_dict["reservoir/state_norm"] = r_state.norm().item()
                        log_dict["reservoir/state_mean"] = r_state.mean().item()
                        log_dict["reservoir/state_std"] = r_state.std().item()
                wandb.log(log_dict)

            # Validation + early stopping
            if val_dataset is not None and (step + 1) % config.val_every_n_steps == 0:
                val_loss = self._validate(model, val_dataset, config, device, use_bf16)

                if config.use_wandb and HAS_WANDB:
                    wandb.log({"val/loss": val_loss, "step": step})

                if val_loss < best_val_loss - config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        break

            # Periodic checkpointing
            if (step + 1) % config.save_every_n_steps == 0:
                self.save_checkpoint(model, interface, step + 1, config)

        if config.use_wandb and HAS_WANDB:
            wandb.finish()

        return {
            "steps_trained": final_step,
            "final_loss": losses[-1] if losses else None,
            "losses": losses,
            "best_val_loss": best_val_loss if val_dataset is not None else None,
        }
