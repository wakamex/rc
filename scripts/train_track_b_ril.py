#!/usr/bin/env python3
"""Train Track B Config 1: Reservoir Interaction Layer (RIL) insertion.

Architecture (deeper than Track A):
  - Frozen Qwen3.5-0.8B-Base + LoRA adapters (Q/V projections, rank 16)
  - ESN reservoir (best HP from T16 sweep) processes input embeddings
  - ReservoirInteractionLayer (RIL) inserted every 6 transformer blocks
    - RIL applies gated cross-attention + post-norm (deeper than Track A sidecar)
    - Adjacent LayerNorms selectively unfrozen to allow normalization adaptation
  - LoRA + RIL interface params trained; base model frozen otherwise

Track B: DEEPER integration than Track A's bolt-on sidecar.
RIL differs from CrossAttentionSidecar in that it also unfreezes the
LayerNorms adjacent to injection points, giving the model a mechanism to
adapt its normalization scheme around the reservoir signal.

Usage::

    python scripts/train_track_b_ril.py
    python scripts/train_track_b_ril.py --config configs/track_b/ril.yaml
    python scripts/train_track_b_ril.py --max_steps 5000 --output_dir checkpoints/track_b/ril
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    p = argparse.ArgumentParser(description="Train Track B RIL insertion")
    p.add_argument("--model_name", default="qwen3.5-0.8b")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", default="cuda")

    # Reservoir config — best HP from T16 sweep
    p.add_argument("--reservoir_size", type=int, default=10_000)
    p.add_argument("--spectral_radius", type=float, default=0.9)
    p.add_argument("--leak_rate", type=float, default=0.5)
    p.add_argument("--input_scaling", type=float, default=1.0)
    p.add_argument("--reservoir_sparsity", type=float, default=0.01)
    p.add_argument("--reservoir_seed", type=int, default=42)

    # RIL insertion: inject every N transformer blocks
    p.add_argument("--ril_every_n_layers", type=int, default=6,
                   help="Insert RIL after every N-th transformer block (default: 6).")
    p.add_argument(
        "--ril_layers",
        nargs="*",
        type=int,
        default=None,
        help="Explicit RIL injection layer indices (overrides --ril_every_n_layers).",
    )
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ril_dropout", type=float, default=0.0)
    p.add_argument("--unfreeze_adjacent_layernorms", action="store_true", default=True,
                   help="Unfreeze LayerNorms adjacent to RIL injection points.")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", nargs="*", default=["q_proj", "v_proj"])

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
    p.add_argument("--output_dir", default="checkpoints/track_b/ril")
    p.add_argument("--results_file", default="results/track_b/ril.json")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="lrs-track-b")
    p.add_argument("--wandb_run_name", default="track-b-ril-qwen3.5-0.8b")
    p.add_argument("--config", default=None, help="Path to YAML config (overrides CLI defaults).")
    return p.parse_args()


def _load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed; ignoring --config %s", path)
        return {}


def apply_config_overrides(args: argparse.Namespace, config: dict[str, Any]) -> None:
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
# Reservoir Interaction Layer (RIL) — Track B deeper integration
# ---------------------------------------------------------------------------


class ReservoirInteractionLayer(nn.Module):
    """Reservoir Interaction Layer (RIL) for deep reservoir-LLM integration.

    Unlike CrossAttentionSidecar (Track A, bolt-on), RIL is designed for
    deeper integration:
    - Pre-norm applied to LLM hidden states before cross-attention
    - Gated residual: gate ∈ [0,1] learned per-position
    - Post-norm applied after residual addition
    - Adjacent transformer LayerNorms selectively unfrozen externally

    The gated formulation ensures near-identity behavior when the reservoir
    signal is not informative (gate → 0), enabling stable training.
    """

    def __init__(
        self,
        hidden_dim: int,
        reservoir_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.reservoir_dim = reservoir_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Pre-norm (applied to queries before cross-attention)
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention: Q from LLM hidden, K/V from reservoir
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(reservoir_dim, hidden_dim)
        self.v_proj = nn.Linear(reservoir_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating: scalar gate per hidden dimension position
        self.gate_proj = nn.Linear(hidden_dim + reservoir_dim, hidden_dim)

        # Post-norm (applied after gated residual addition)
        self.post_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        reservoir_states: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Apply RIL: gated cross-attention with reservoir states.

        Args:
            hidden: LLM hidden states, shape (batch, seq_len, hidden_dim).
            reservoir_states: Reservoir state sequence,
                shape (batch, seq_len, reservoir_dim) or (batch, reservoir_dim).

        Returns:
            Updated hidden states, same shape as input.
        """
        squeeze_batch = hidden.ndim == 2
        if squeeze_batch:
            hidden = hidden.unsqueeze(0)

        B, T, H = hidden.shape

        # Convert reservoir states (gradient stops at numpy→tensor boundary)
        if isinstance(reservoir_states, np.ndarray):
            r = torch.from_numpy(np.asarray(reservoir_states, dtype=np.float32))
            r = r.to(device=hidden.device, dtype=hidden.dtype)
        else:
            r = reservoir_states.detach().to(device=hidden.device, dtype=hidden.dtype)

        # Ensure shape is (B, S, reservoir_dim)
        if r.ndim == 1:
            r = r.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        elif r.ndim == 2:
            r = r.unsqueeze(0).expand(B, -1, -1)
        S = r.shape[1]

        # Pre-norm on queries
        normed_hidden = self.pre_norm(hidden)

        # Cross-attention: Q from LLM hidden, K/V from reservoir
        Q = self.q_proj(normed_hidden)           # (B, T, H)
        K = self.k_proj(r)                        # (B, S, H)
        V = self.v_proj(r)                        # (B, S, H)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, V)          # (B, heads, T, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        attn_out = self.out_proj(attn_out)         # (B, T, H)

        # Gating: combine hidden + mean reservoir state to compute gate
        # Use mean over reservoir sequence dimension for gate input
        r_mean = r.mean(dim=1, keepdim=True).expand(B, T, -1)  # (B, T, reservoir_dim)
        gate_input = torch.cat([hidden, r_mean], dim=-1)         # (B, T, H + reservoir_dim)
        gate = torch.sigmoid(self.gate_proj(gate_input))         # (B, T, H)

        # Gated residual + post-norm (Track B deeper integration)
        out = self.post_norm(hidden + gate * attn_out)

        if squeeze_batch:
            out = out.squeeze(0)
        return out


# ---------------------------------------------------------------------------
# RIL bundle and hook manager
# ---------------------------------------------------------------------------


class RILBundle(nn.Module):
    """Bundle of ReservoirInteractionLayers for all injected transformer blocks."""

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
        self.rils = nn.ModuleDict({
            str(idx): ReservoirInteractionLayer(
                hidden_dim=hidden_dim,
                reservoir_dim=reservoir_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for idx in layer_indices
        })

    def get_ril(self, layer_idx: int) -> ReservoirInteractionLayer:
        return self.rils[str(layer_idx)]  # type: ignore[return-value]


class RILHookManager:
    """Manages forward hooks for RIL injection into transformer layers."""

    def __init__(
        self,
        model: nn.Module,
        ril_bundle: RILBundle,
        layer_indices: list[int],
    ) -> None:
        self.ril_bundle = ril_bundle
        self.layer_indices = layer_indices
        self._handles: list[Any] = []
        self._reservoir_states: np.ndarray | None = None
        self._register_hooks(model, layer_indices)

    def _register_hooks(self, model: nn.Module, layer_indices: list[int]) -> None:
        transformer_layers = self._get_transformer_layers(model)
        for idx in layer_indices:
            if idx < 0 or idx >= len(transformer_layers):
                logger.warning("RIL layer index %d out of range (model has %d layers); skipping.",
                               idx, len(transformer_layers))
                continue
            layer = transformer_layers[idx]
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    @staticmethod
    def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
        for attr_path in ["model.model.layers", "model.layers", "transformer.h", "layers"]:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                return list(obj)
        best: list[nn.Module] = []
        for _, child in model.named_modules():
            if isinstance(child, nn.ModuleList) and len(list(child)) > len(best):
                best = list(child)
        return best

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
            if self._reservoir_states is None:
                return output
            ril = self.ril_bundle.get_ril(layer_idx)
            if isinstance(output, tuple):
                hidden = output[0]
                modified = ril(hidden, self._reservoir_states)
                return (modified,) + output[1:]
            else:
                return ril(output, self._reservoir_states)
        return hook

    def set_reservoir_states(self, states: np.ndarray) -> None:
        self._reservoir_states = states

    def clear_reservoir_states(self) -> None:
        self._reservoir_states = None

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# LayerNorm unfreezing for adjacent layers
# ---------------------------------------------------------------------------


def unfreeze_adjacent_layernorms(
    model: nn.Module,
    layer_indices: list[int],
) -> list[str]:
    """Selectively unfreeze LayerNorm parameters adjacent to RIL injection points.

    For each injection layer index, finds LayerNorms within ±1 layer
    and re-enables their gradients. This is the "deeper integration"
    compared to Track A (which never unfreezes base model params).

    Returns:
        List of unfrozen parameter names.
    """
    transformer_layers = RILHookManager._get_transformer_layers(model)
    unfrozen: list[str] = []

    for idx in layer_indices:
        # Check layers idx-1, idx, idx+1
        adjacent = {max(0, idx - 1), idx, min(len(transformer_layers) - 1, idx + 1)}
        for adj_idx in adjacent:
            if adj_idx >= len(transformer_layers):
                continue
            layer = transformer_layers[adj_idx]
            for name, param in layer.named_parameters():
                # Unfreeze LayerNorm weight/bias parameters
                if "norm" in name.lower() or "layer_norm" in name.lower():
                    param.requires_grad_(True)
                    unfrozen.append(f"layer.{adj_idx}.{name}")

    logger.info("Unfrozen %d LayerNorm parameter tensors adjacent to RIL injection.", len(unfrozen))
    return unfrozen


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

    if not args.no_wandb:
        try:
            import wandb  # type: ignore[import]
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as exc:
            logger.warning("wandb init failed (%s); continuing without logging.", exc)

    logger.info("Loading model: %s", args.model_name)
    from src.models.loader import load_model
    wrapper = load_model(args.model_name, dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    hidden_dim = model.config.hidden_size
    embed_dim = hidden_dim

    # Apply LoRA
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

    # Freeze base model (only LoRA + RIL + unfrozen LayerNorms trained)
    from src.training.lora_trainer import LoRATrainer
    lora_trainer = LoRATrainer()
    lora_trainer.freeze_base_model(model)

    # Build ESN reservoir
    logger.info("Building ESN: size=%d, spectral_radius=%.2f, leak_rate=%.2f",
                args.reservoir_size, args.spectral_radius, args.leak_rate)
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
    logger.info("ESN built: %d nodes, input_dim=%d", esn.n, embed_dim)

    # Determine RIL injection layers
    num_layers = model.config.num_hidden_layers
    if args.ril_layers is not None:
        ril_layers = args.ril_layers
    else:
        # Every N-th layer: layers N-1, 2N-1, 3N-1, ... (0-indexed)
        n = args.ril_every_n_layers
        ril_layers = list(range(n - 1, num_layers, n))
    logger.info("RIL injection at layers: %s (of %d total)", ril_layers, num_layers)

    # Build RIL bundle
    ril_bundle = RILBundle(
        layer_indices=ril_layers,
        reservoir_dim=args.reservoir_size,
        hidden_dim=hidden_dim,
        num_heads=args.num_heads,
        dropout=args.ril_dropout,
    )
    ril_bundle = ril_bundle.to(device).to(dtype)

    # Selective LayerNorm unfreezing (Track B deeper integration)
    if args.unfreeze_adjacent_layernorms:
        unfreeze_adjacent_layernorms(model, ril_layers)

    # Register RIL hooks
    hook_manager = RILHookManager(model, ril_bundle, ril_layers)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()
    ril_bundle.train()
    model.to(device)

    embed_layer = model.get_input_embeddings()

    # Optimizer: LoRA + unfrozen LayerNorm params at lora_lr, RIL params at interface_lr
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    layernorm_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and "lora_" not in n]
    ril_params = list(ril_bundle.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": layernorm_params, "lr": args.lr * 0.1},  # smaller LR for LayerNorms
            {"params": ril_params, "lr": args.interface_lr},
        ],
        weight_decay=args.weight_decay,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                            total_iters=max(1, args.warmup_steps))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps - args.warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[args.warmup_steps])

    logger.info("Building data loader from %s / %s", args.dataset_name, args.dataset_config)
    loader = build_dataloader(
        tokenizer._tok,  # type: ignore[attr-defined]
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

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

        # Compute reservoir states from input embeddings
        with torch.no_grad():
            embeddings = embed_layer(input_ids)

        batch_size_actual = input_ids.shape[0]
        all_states = []
        for b in range(batch_size_actual):
            esn.reset()
            emb_b = embeddings[b].float().cpu().numpy()
            states_b = np.zeros((emb_b.shape[0], esn.n), dtype=np.float32)
            for t in range(emb_b.shape[0]):
                states_b[t] = esn.step(emb_b[t])
            all_states.append(states_b)

        reservoir_states = np.stack(all_states, axis=0)  # (B, T, reservoir_dim)
        hook_manager.set_reservoir_states(reservoir_states)

        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

        hook_manager.clear_reservoir_states()

        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % args.grad_accum == 0:
            all_params = lora_params + layernorm_params + ril_params
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
            logger.info("step=%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                        global_step, avg_loss, scheduler.get_last_lr()[0], elapsed)
            try:
                import wandb  # type: ignore[import]
                wandb.log({
                    "train/loss": avg_loss,
                    "train/step": global_step,
                    "reservoir/state_norm": float(np.linalg.norm(esn.state)),
                    "train/config": "ril",
                })
            except Exception:
                pass
            accum_loss = 0.0

        if global_step % args.save_interval == 0:
            ckpt_path = output_dir / f"step_{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(ckpt_path / "lora_adapter"))
            torch.save(ril_bundle.state_dict(), ckpt_path / "ril_weights.pt")

    logger.info("Training complete at step %d. Saving final checkpoint.", global_step)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(final_dir / "lora_adapter"))
    torch.save(ril_bundle.state_dict(), final_dir / "ril_weights.pt")

    hook_manager.remove_hooks()

    final_loss = losses[-1] if losses else float("inf")
    final_ppl = math.exp(min(final_loss, 20.0))

    results = {
        "model_name": args.model_name,
        "track": "B",
        "config": "ril",
        "architecture": {
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "reservoir_size": args.reservoir_size,
            "spectral_radius": args.spectral_radius,
            "leak_rate": args.leak_rate,
            "ril_layers": ril_layers,
            "unfreeze_adjacent_layernorms": args.unfreeze_adjacent_layernorms,
            "integration": "deeper_ril",
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
            "loss_history": losses[::max(1, len(losses) // 100)],
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
    logger.info("Done. final_loss=%.4f  final_ppl=%.2f  steps=%d",
                final_loss, final_ppl, global_step)


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = _load_yaml_config(args.config)
        apply_config_overrides(args, cfg)
    train(args)
