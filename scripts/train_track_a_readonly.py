#!/usr/bin/env python3
"""Train Track A Read-Only Sidecar on Qwen3.5-0.8B-Base.

Architecture:
  - Frozen Qwen3.5-0.8B-Base + LoRA adapters (Q/V projections, rank 16)
  - ESN reservoir (10K nodes) processes input embeddings → reservoir states
  - ReadProjection + CrossAttentionSidecar inject reservoir memory at selected layers
  - NO WriteHead — reservoir only reads input embeddings, LLM cannot write to reservoir

This is the first real LRS experiment (Track A).

Usage::

    python scripts/train_track_a_readonly.py
    python scripts/train_track_a_readonly.py --max_steps 5000 --output_dir checkpoints/track_a_readonly

Requires: peft, transformers, torch, wandb (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Track A Read-Only Sidecar")
    p.add_argument("--model_name", default="qwen3.5-0.8b")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", default="cuda")

    # Reservoir config (no WriteHead — read-only)
    p.add_argument("--reservoir_size", type=int, default=10_000,
                   help="Number of ESN reservoir nodes.")
    p.add_argument("--spectral_radius", type=float, default=0.9)
    p.add_argument("--leak_rate", type=float, default=0.5)
    p.add_argument("--input_scaling", type=float, default=1.0)
    p.add_argument("--reservoir_sparsity", type=float, default=0.01)
    p.add_argument("--reservoir_seed", type=int, default=42)

    # Sidecar injection layers
    p.add_argument(
        "--sidecar_layers",
        nargs="*",
        type=int,
        default=None,
        help="Transformer layer indices to inject sidecar (default: every 4th layer).",
    )
    p.add_argument("--num_heads", type=int, default=8,
                   help="Number of attention heads in CrossAttentionSidecar.")
    p.add_argument("--sidecar_dropout", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        nargs="*",
        default=["q_proj", "v_proj"],
        help="LoRA target module names (default: q_proj, v_proj).",
    )

    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--interface_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--dataset_name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_config", default="sample-10BT")

    # Output
    p.add_argument("--output_dir", default="checkpoints/track_a_readonly")
    p.add_argument("--results_file", default="results/track_a/readonly.json")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="lrs-track-a")
    p.add_argument("--wandb_run_name", default="track-a-readonly-qwen3.5-0.8b")

    # Config file (overrides defaults)
    p.add_argument("--config", default=None,
                   help="Path to YAML config file (overrides CLI defaults).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file, returning a flat dict of overrides."""
    try:
        import yaml  # type: ignore[import-untyped]
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed; ignoring --config %s", path)
        return {}


def apply_config_overrides(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Apply YAML config values to args namespace (only known attributes)."""
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def build_dataloader(
    tokenizer: object,
    dataset_name: str,
    dataset_config: str,
    max_seq_length: int,
    batch_size: int,
    seed: int = 42,
) -> object:
    """Build a streaming DataLoader over a HuggingFace text dataset."""
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    def tokenize_and_chunk(example: dict) -> dict:
        text = example.get("text", "")
        ids = tokenizer(  # type: ignore[operator]
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"].squeeze(0)
        return {"input_ids": ids}

    remove_cols = [
        c for c in ["text", "id", "dump", "url", "file_path", "language",
                     "language_score", "token_count", "score", "int_score"]
        if c in (ds.features or {})
    ]
    ds = ds.map(tokenize_and_chunk, remove_columns=remove_cols or None)

    from torch.utils.data import DataLoader  # type: ignore[import]

    def collate(batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            padded[i, : ids.shape[0]] = ids
        labels = padded.clone()
        labels[labels == 0] = -100
        return {"input_ids": padded, "labels": labels}

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# Read-only sidecar module (no WriteHead)
# ---------------------------------------------------------------------------


class ReadOnlySidecarBundle(nn.Module):
    """Bundle of ReadProjection + CrossAttentionSidecar modules for all injected layers.

    Manages one CrossAttentionSidecar per injected transformer layer.
    ReadProjection is used to project reservoir states to the LLM hidden dim
    for a lightweight fallback; the primary injection mechanism is CrossAttentionSidecar.

    NO WriteHead is included — reservoir state is driven solely by input embeddings.
    """

    def __init__(
        self,
        layer_indices: list[int],
        reservoir_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_indices = layer_indices
        self.reservoir_dim = reservoir_dim
        self.hidden_dim = hidden_dim

        from src.reservoir.interface import CrossAttentionSidecar, ReadProjection

        # One CrossAttentionSidecar per injected layer
        self.sidecars = nn.ModuleDict({
            str(idx): CrossAttentionSidecar(
                hidden_dim=hidden_dim,
                reservoir_dim=reservoir_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for idx in layer_indices
        })

        # Shared ReadProjection (optional lightweight read path)
        self.read_proj = ReadProjection(
            reservoir_dim=reservoir_dim,
            hidden_dim=hidden_dim,
        )

    def get_sidecar(self, layer_idx: int) -> nn.Module:
        """Return the CrossAttentionSidecar for a given layer index."""
        return self.sidecars[str(layer_idx)]


# ---------------------------------------------------------------------------
# Reservoir state computation (read-only: from input embeddings only)
# ---------------------------------------------------------------------------


def compute_reservoir_states(
    esn: object,
    embeddings: torch.Tensor,
) -> np.ndarray:
    """Run input embeddings through ESN to produce reservoir state sequence.

    Args:
        esn: ESN instance (must have .step() and .reset()).
        embeddings: Input embeddings, shape (batch, seq_len, embed_dim) or
                    (seq_len, embed_dim). Detached float32 numpy conversion is
                    performed inside this function.

    Returns:
        Reservoir states, shape (batch, seq_len, reservoir_dim) or
        (seq_len, reservoir_dim) matching input batch dimension.
    """
    esn.reset()  # type: ignore[attr-defined]

    emb_np = embeddings.detach().float().cpu().numpy()
    squeeze = emb_np.ndim == 2
    if squeeze:
        emb_np = emb_np[None]  # (1, T, D)

    B, T, D = emb_np.shape
    reservoir_dim = esn.n  # type: ignore[attr-defined]
    states = np.zeros((B, T, reservoir_dim), dtype=np.float32)

    for t in range(T):
        x_t = emb_np[:, t, :]  # (B, D)
        r_t = esn.step(x_t)    # (B, n) or (n,) for B=1
        if r_t.ndim == 1:
            r_t = r_t[None]    # (1, n)
        states[:, t, :] = r_t

    if squeeze:
        states = states[0]  # (T, n)
    return states


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------


class SidecarHookManager:
    """Registers and manages forward hooks for read-only sidecar injection.

    Each hook reads pre-computed reservoir states from a shared store and
    applies CrossAttentionSidecar to modify the layer's output hidden states.
    No write path exists — the reservoir is not updated from LLM hidden states.
    """

    def __init__(
        self,
        model: nn.Module,
        sidecar_bundle: ReadOnlySidecarBundle,
        layer_indices: list[int],
    ) -> None:
        self.sidecar_bundle = sidecar_bundle
        self.layer_indices = layer_indices
        self._handles: list[Any] = []
        # Mutable store for current reservoir states (updated before each forward)
        self._reservoir_states: np.ndarray | None = None

        self._register_hooks(model, layer_indices)

    def _register_hooks(self, model: nn.Module, layer_indices: list[int]) -> None:
        """Register forward hooks on selected transformer layers."""
        transformer_layers = self._get_transformer_layers(model)

        for idx in layer_indices:
            if idx < 0 or idx >= len(transformer_layers):
                logger.warning("Layer index %d out of range (model has %d layers); skipping.",
                               idx, len(transformer_layers))
                continue
            layer = transformer_layers[idx]
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    @staticmethod
    def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
        """Extract the list of transformer decoder layers from the model."""
        # Try common HuggingFace patterns
        for attr_path in ["model.model.layers", "model.layers", "transformer.h", "layers"]:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                return list(obj)
        # Fallback: find nn.ModuleList with the most entries
        best: list[nn.Module] = []
        for _, child in model.named_modules():
            if isinstance(child, nn.ModuleList) and len(list(child)) > len(best):
                best = list(child)
        return best

    def _make_hook(self, layer_idx: int):
        """Return a forward hook closure for a specific layer index."""
        def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
            if self._reservoir_states is None:
                return output
            sidecar = self.sidecar_bundle.get_sidecar(layer_idx)

            # Extract hidden states (first element of tuple outputs)
            if isinstance(output, tuple):
                hidden = output[0]
                modified = sidecar(hidden, self._reservoir_states)
                return (modified,) + output[1:]
            else:
                return sidecar(output, self._reservoir_states)

        return hook

    def set_reservoir_states(self, states: np.ndarray) -> None:
        """Update the current reservoir states before a forward pass."""
        self._reservoir_states = states

    def clear_reservoir_states(self) -> None:
        """Clear stored reservoir states (after forward pass)."""
        self._reservoir_states = None

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Weights & Biases (optional) ---
    if not args.no_wandb:
        try:
            import wandb  # type: ignore[import]
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as exc:
            logger.warning("wandb init failed (%s); continuing without logging.", exc)

    # --- Load base model ---
    logger.info("Loading model: %s", args.model_name)
    from src.models.loader import load_model

    wrapper = load_model(args.model_name, dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    hidden_dim = model.config.hidden_size
    embed_dim = hidden_dim  # input embeddings have same dim as hidden

    # --- Apply LoRA ---
    logger.info("Applying LoRA (rank=%d, alpha=%.1f, targets=%s)",
                args.lora_rank, args.lora_alpha, args.lora_targets)
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]

        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    except ImportError:
        logger.warning("peft not installed — training without LoRA.")

    # Freeze base model (only LoRA + sidecar params will be trained)
    from src.training.lora_trainer import LoRATrainer
    lora_trainer = LoRATrainer()
    lora_trainer.freeze_base_model(model)

    # --- Build ESN reservoir (read-only) ---
    logger.info(
        "Building ESN reservoir: size=%d, spectral_radius=%.2f, leak_rate=%.2f",
        args.reservoir_size, args.spectral_radius, args.leak_rate,
    )
    from src.reservoir.esn import ESN
    from src.types import ReservoirConfig

    reservoir_cfg = ReservoirConfig(
        size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        input_scaling=args.input_scaling,
        sparsity=args.reservoir_sparsity,
        seed=args.reservoir_seed,
    )
    esn = ESN(reservoir_cfg, input_dim=embed_dim)
    logger.info("ESN reservoir built: %d nodes, input_dim=%d", esn.n, embed_dim)

    # --- Determine sidecar layer indices ---
    num_layers = model.config.num_hidden_layers
    if args.sidecar_layers is not None:
        sidecar_layers = args.sidecar_layers
    else:
        # Default: inject every 4th layer (layers 3, 7, 11, ... ≤ num_layers-1)
        sidecar_layers = list(range(3, num_layers, 4))
    logger.info("Sidecar injection at layers: %s (of %d total)", sidecar_layers, num_layers)

    # --- Build ReadOnlySidecarBundle ---
    sidecar_bundle = ReadOnlySidecarBundle(
        layer_indices=sidecar_layers,
        reservoir_dim=args.reservoir_size,
        hidden_dim=hidden_dim,
        num_heads=args.num_heads,
        dropout=args.sidecar_dropout,
    )
    sidecar_bundle = sidecar_bundle.to(device).to(dtype)
    logger.info(
        "SidecarBundle: %d layers, reservoir_dim=%d, hidden_dim=%d",
        len(sidecar_layers), args.reservoir_size, hidden_dim,
    )

    # --- Register forward hooks ---
    hook_manager = SidecarHookManager(model, sidecar_bundle, sidecar_layers)

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    model.train()
    sidecar_bundle.train()
    model.to(device)

    # --- Get embedding layer for reservoir input ---
    embed_layer = model.get_input_embeddings()

    # --- Optimizer ---
    # LoRA params at lora_lr; sidecar params at interface_lr
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    sidecar_params = list(sidecar_bundle.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": sidecar_params, "lr": args.interface_lr},
        ],
        weight_decay=args.weight_decay,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                            total_iters=max(1, args.warmup_steps))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps - args.warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[args.warmup_steps])

    # --- Data ---
    logger.info("Building data loader from %s / %s", args.dataset_name, args.dataset_config)
    loader = build_dataloader(
        tokenizer._tok,  # type: ignore[attr-defined]
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # --- Training loop ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    accum_loss = 0.0
    t_start = time.time()
    optimizer.zero_grad()
    losses: list[float] = []

    for batch in loader:
        if global_step >= args.max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # --- Compute reservoir states from input embeddings (READ-ONLY) ---
        with torch.no_grad():
            embeddings = embed_layer(input_ids)  # (B, T, H)

        # Run each sequence in the batch through the reservoir
        # Using first sequence for single-state injection; for batched we use mean
        # For efficiency: use the mean embedding across the batch as single reservoir input
        # Full per-example reservoir states computed sequence-by-sequence
        batch_size_actual = input_ids.shape[0]
        all_states = []
        for b in range(batch_size_actual):
            esn.reset()
            emb_b = embeddings[b].float().cpu().numpy()  # (T, H)
            states_b = np.zeros((emb_b.shape[0], esn.n), dtype=np.float32)
            for t in range(emb_b.shape[0]):
                states_b[t] = esn.step(emb_b[t])
            all_states.append(states_b)

        # reservoir_states shape: (B, T, reservoir_dim)
        reservoir_states = np.stack(all_states, axis=0)
        hook_manager.set_reservoir_states(reservoir_states)

        # --- Forward pass (hooks inject reservoir states at selected layers) ---
        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

        hook_manager.clear_reservoir_states()

        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % args.grad_accum == 0:
            all_params = lora_params + sidecar_params
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_val = loss.item() * args.grad_accum
        losses.append(loss_val)
        global_step += 1

        if global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = accum_loss * args.grad_accum / args.log_interval
            logger.info(
                "step=%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                global_step,
                avg_loss,
                scheduler.get_last_lr()[0],
                elapsed,
            )
            try:
                import wandb  # type: ignore[import]
                wandb.log({
                    "train/loss": avg_loss,
                    "train/step": global_step,
                    "reservoir/state_norm": float(np.linalg.norm(esn.state)),
                    "reservoir/state_mean": float(esn.state.mean()),
                })
            except Exception:
                pass
            accum_loss = 0.0

        if global_step % args.save_interval == 0:
            ckpt_path = output_dir / f"step_{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving checkpoint to %s", ckpt_path)
            # Save LoRA adapter
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(ckpt_path / "lora_adapter"))
            # Save sidecar bundle
            torch.save(sidecar_bundle.state_dict(), ckpt_path / "sidecar_weights.pt")

    # Final save
    logger.info("Training complete at step %d.  Saving final checkpoint.", global_step)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(final_dir / "lora_adapter"))
    torch.save(sidecar_bundle.state_dict(), final_dir / "sidecar_weights.pt")

    # Clean up hooks
    hook_manager.remove_hooks()

    # --- Compute final perplexity estimate ---
    final_loss = losses[-1] if losses else float("inf")
    import math
    final_ppl = math.exp(min(final_loss, 20.0))

    # --- Record results ---
    results = {
        "model_name": args.model_name,
        "track": "A",
        "mode": "readonly",
        "architecture": {
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "reservoir_size": args.reservoir_size,
            "spectral_radius": args.spectral_radius,
            "leak_rate": args.leak_rate,
            "sidecar_layers": sidecar_layers,
            "write_head": False,
        },
        "training": {
            "steps_trained": global_step,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "interface_lr": args.interface_lr,
        },
        "metrics": {
            "final_train_loss": final_loss,
            "final_train_perplexity": final_ppl,
            "loss_history": losses[::max(1, len(losses) // 100)],  # downsample for JSON
        },
        "output_dir": str(output_dir),
        "args": vars(args),
        "timestamp": time.time(),
    }

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", results_path)

    logger.info(
        "Done.  final_loss=%.4f  final_ppl=%.2f  steps=%d",
        final_loss, final_ppl, global_step,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = _load_yaml_config(args.config)
        apply_config_overrides(args, cfg)
    train(args)
