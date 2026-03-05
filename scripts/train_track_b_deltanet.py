#!/usr/bin/env python3
"""Train Track B Config 2: DeltaNet block replacement with ESN.

Architecture (deeper than Track A):
  - Frozen Qwen3.5-0.8B-Base + LoRA adapters (Q/V projections, rank 16)
  - 6 of 18 DeltaNet blocks replaced with ESN reservoir modules
    - Every 3rd DeltaNet block is replaced (indices 0, 3, 6, 9, 12, 15 of DeltaNet blocks)
    - ESNReplacementBlock intercepts DeltaNet output via forward hook
    - Learned gating: output = gate * esn_out + (1-gate) * original_deltanet_out
  - LoRA + replacement interface params trained; base model frozen otherwise

Track B Config 2: DEEPER integration via architectural replacement.
The ESN replacement provides a higher-dimensional recurrent state than
DeltaNet's linear attention, potentially capturing richer temporal structure.

Usage::

    python scripts/train_track_b_deltanet.py
    python scripts/train_track_b_deltanet.py --config configs/track_b/deltanet.yaml
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
    p = argparse.ArgumentParser(description="Train Track B DeltaNet block replacement")
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

    # DeltaNet replacement config
    p.add_argument("--replace_every_nth_deltanet", type=int, default=3,
                   help="Replace every N-th DeltaNet block (default: 3 → 6 of 18).")
    p.add_argument("--num_deltanet_blocks", type=int, default=18,
                   help="Expected number of DeltaNet blocks in Qwen3.5 (default: 18).")
    p.add_argument("--replacement_gate_init", type=float, default=0.5,
                   help="Initial gate value for ESN/DeltaNet mix (0=pure DeltaNet, 1=pure ESN).")

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
    p.add_argument("--output_dir", default="checkpoints/track_b/deltanet")
    p.add_argument("--results_file", default="results/track_b/deltanet.json")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="lrs-track-b")
    p.add_argument("--wandb_run_name", default="track-b-deltanet-qwen3.5-0.8b")
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
# ESN Replacement module for DeltaNet blocks
# ---------------------------------------------------------------------------


class ESNReplacementInterface(nn.Module):
    """Trainable interface for ESN-based DeltaNet block replacement.

    When a DeltaNet block's output is intercepted, this module:
    1. Projects the pre-DeltaNet hidden state through the reservoir
    2. Projects ESN states → hidden_dim (ReadProjection)
    3. Uses a learned gate to mix ESN output with original DeltaNet output:
       final = gate * esn_out + (1 - gate) * deltanet_out

    This gated formulation enables:
    - Stable training: gate can start near 0 (pure DeltaNet)
    - Gradual ESN integration as training progresses
    - The model can learn when ESN is beneficial vs. when DeltaNet is better

    Parameters trained: read_proj, gate_proj (and gate_bias).
    """

    def __init__(
        self,
        hidden_dim: int,
        reservoir_dim: int,
        gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reservoir_dim = reservoir_dim

        # Project ESN states → hidden_dim
        self.read_proj = nn.Linear(reservoir_dim, hidden_dim)

        # Layer norm on ESN projection output
        self.esn_norm = nn.LayerNorm(hidden_dim)

        # Gate: sigmoid(linear(hidden)) ∈ [0, 1] per hidden dim
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        # Initialize bias so sigmoid output ≈ gate_init
        gate_bias_init = math.log(gate_init / (1.0 - gate_init + 1e-6))
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    def forward(
        self,
        pre_hidden: torch.Tensor,
        deltanet_output: torch.Tensor,
        reservoir_states: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Mix DeltaNet output with ESN-projected states.

        Args:
            pre_hidden: Input to the DeltaNet block (before DeltaNet),
                shape (batch, seq_len, hidden_dim).
            deltanet_output: Output from the DeltaNet block,
                shape (batch, seq_len, hidden_dim).
            reservoir_states: ESN states aligned with sequence,
                shape (batch, seq_len, reservoir_dim).

        Returns:
            Mixed output, same shape as deltanet_output.
        """
        # Convert reservoir states (gradient stops at boundary)
        if isinstance(reservoir_states, np.ndarray):
            r = torch.from_numpy(np.asarray(reservoir_states, dtype=np.float32))
            r = r.to(device=deltanet_output.device, dtype=deltanet_output.dtype)
        else:
            r = reservoir_states.detach().to(
                device=deltanet_output.device, dtype=deltanet_output.dtype
            )

        # Ensure (B, T, reservoir_dim)
        if r.ndim == 2:
            r = r.unsqueeze(0).expand(deltanet_output.shape[0], -1, -1)

        # Project ESN → hidden_dim
        esn_out = self.esn_norm(self.read_proj(r))  # (B, T, H)

        # Compute gate from deltanet output
        gate = torch.sigmoid(self.gate_proj(deltanet_output))  # (B, T, H)

        # Gated mix
        return gate * esn_out + (1.0 - gate) * deltanet_output


# ---------------------------------------------------------------------------
# DeltaNet block identification and hook-based replacement
# ---------------------------------------------------------------------------


def identify_deltanet_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Identify DeltaNet (linear-attention) blocks in the model.

    Returns list of (module_path, module) tuples for DeltaNet blocks.
    """
    from src.models.loader import DELTANET_PATTERNS

    deltanet_blocks = []
    seen_paths: set[str] = set()

    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(pat in name_lower for pat in DELTANET_PATTERNS):
            # Get the top-level block (e.g. "model.layers.2.deltanet")
            # We want the highest-level DeltaNet module
            parts = name.split(".")
            # Find the deepest DeltaNet-containing parent
            is_parent = any(
                other.startswith(name + ".") for other_name, _ in model.named_modules()
                for other in [other_name]
                if any(pat in other.lower() for pat in DELTANET_PATTERNS)
            )
            if not is_parent and name not in seen_paths:
                seen_paths.add(name)
                deltanet_blocks.append((name, module))

    return deltanet_blocks


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Extract transformer decoder layers from the model."""
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


class DeltaNetReplacementManager:
    """Manages ESN-based interception of selected DeltaNet block outputs.

    Uses forward hooks to:
    1. Capture input to selected DeltaNet blocks
    2. Intercept DeltaNet output
    3. Mix via ESNReplacementInterface with reservoir states

    The block selection strategy: every N-th DeltaNet block by index.
    """

    def __init__(
        self,
        model: nn.Module,
        replacement_interfaces: nn.ModuleDict,
        replace_every_nth: int,
        num_deltanet_blocks: int,
    ) -> None:
        self.model = model
        self.replacement_interfaces = replacement_interfaces
        self.replace_every_nth = replace_every_nth
        self._handles: list[Any] = []
        self._reservoir_states: np.ndarray | None = None
        self._pre_hidden_store: dict[str, torch.Tensor] = {}

        self._selected_block_indices: list[int] = list(
            range(0, num_deltanet_blocks, replace_every_nth)
        )
        logger.info("DeltaNet replacement target indices: %s", self._selected_block_indices)

    def register_hooks(self, transformer_layers: list[nn.Module]) -> None:
        """Register pre/post hooks on transformer layers to intercept DeltaNet blocks.

        For each transformer layer, we hook the layer's forward to:
        - Capture input hidden states (pre-hook)
        - Intercept/modify output (post-hook)
        For layers that contain a "selected" DeltaNet block.
        """
        # Track which transformer layer indices have DeltaNet blocks
        # Heuristic: in Qwen3.5, DeltaNet and full-attention alternate
        # DeltaNet blocks are at even-indexed layers (0, 2, 4, ...) or odd
        # We identify them by checking module names
        deltanet_layer_indices: list[int] = []
        for layer_idx, layer in enumerate(transformer_layers):
            for name, _ in layer.named_modules():
                if any(pat in name.lower() for pat in ["deltanet", "delta_net",
                                                        "linear_attn", "linear_attention"]):
                    deltanet_layer_indices.append(layer_idx)
                    break

        if not deltanet_layer_indices:
            # Fallback: treat alternate layers as DeltaNet (Qwen3.5 hybrid pattern)
            logger.warning("No DeltaNet modules found by name. Falling back to alternate-layer pattern.")
            deltanet_layer_indices = list(range(0, len(transformer_layers), 2))

        logger.info("DeltaNet transformer layer indices: %s (total=%d)",
                    deltanet_layer_indices, len(deltanet_layer_indices))

        # Select which DeltaNet layers to replace
        selected_transformer_indices = [
            deltanet_layer_indices[i]
            for i in self._selected_block_indices
            if i < len(deltanet_layer_indices)
        ]
        logger.info("Replacing transformer layers: %s", selected_transformer_indices)

        for dn_seq_idx, layer_idx in zip(self._selected_block_indices, selected_transformer_indices):
            if str(dn_seq_idx) not in self.replacement_interfaces:
                continue
            layer = transformer_layers[layer_idx]
            key = str(dn_seq_idx)

            # Pre-hook: capture input hidden states
            def make_pre_hook(k: str):
                def pre_hook(module: nn.Module, inputs: tuple) -> None:
                    hidden = inputs[0] if isinstance(inputs, tuple) else inputs
                    if isinstance(hidden, torch.Tensor):
                        self._pre_hidden_store[k] = hidden.detach()
                return pre_hook

            # Post-hook: mix DeltaNet output with ESN output
            def make_post_hook(k: str):
                def post_hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
                    if self._reservoir_states is None or k not in self._pre_hidden_store:
                        return output
                    interface = self.replacement_interfaces[k]
                    pre_hidden = self._pre_hidden_store[k].to(
                        device=next(interface.parameters()).device
                    )
                    if isinstance(output, tuple):
                        deltanet_out = output[0]
                        mixed = interface(pre_hidden, deltanet_out, self._reservoir_states)
                        return (mixed,) + output[1:]
                    else:
                        return interface(pre_hidden, output, self._reservoir_states)
                return post_hook

            h_pre = layer.register_forward_pre_hook(make_pre_hook(key))
            h_post = layer.register_forward_hook(make_post_hook(key))
            self._handles.extend([h_pre, h_post])

    def set_reservoir_states(self, states: np.ndarray) -> None:
        self._reservoir_states = states

    def clear_reservoir_states(self) -> None:
        self._reservoir_states = None
        self._pre_hidden_store.clear()

    def remove_hooks(self) -> None:
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
    logger.info("Applying LoRA (rank=%d, alpha=%.1f)", args.lora_rank, args.lora_alpha)
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

    # Freeze base model
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
    logger.info("ESN built: %d nodes", esn.n)

    # Determine which DeltaNet blocks to replace
    num_dn = args.num_deltanet_blocks
    n_step = args.replace_every_nth_deltanet
    selected_dn_indices = list(range(0, num_dn, n_step))
    logger.info("Replacing DeltaNet blocks at indices: %s (%d of %d)",
                selected_dn_indices, len(selected_dn_indices), num_dn)

    # Build ESNReplacementInterface for each selected DeltaNet block
    replacement_interfaces = nn.ModuleDict({
        str(idx): ESNReplacementInterface(
            hidden_dim=hidden_dim,
            reservoir_dim=args.reservoir_size,
            gate_init=args.replacement_gate_init,
        )
        for idx in selected_dn_indices
    })
    replacement_interfaces = replacement_interfaces.to(device).to(dtype)

    # Get transformer layers and register DeltaNet replacement hooks
    transformer_layers = get_transformer_layers(model)
    logger.info("Model has %d transformer layers", len(transformer_layers))

    replacement_manager = DeltaNetReplacementManager(
        model=model,
        replacement_interfaces=replacement_interfaces,
        replace_every_nth=n_step,
        num_deltanet_blocks=num_dn,
    )
    replacement_manager.register_hooks(transformer_layers)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()
    replacement_interfaces.train()
    model.to(device)

    embed_layer = model.get_input_embeddings()

    # Optimizer
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    interface_params = list(replacement_interfaces.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": interface_params, "lr": args.interface_lr},
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

        reservoir_states = np.stack(all_states, axis=0)
        replacement_manager.set_reservoir_states(reservoir_states)

        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

        replacement_manager.clear_reservoir_states()

        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % args.grad_accum == 0:
            all_params = lora_params + interface_params
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
                    "train/config": "deltanet",
                })
            except Exception:
                pass
            accum_loss = 0.0

        if global_step % args.save_interval == 0:
            ckpt_path = output_dir / f"step_{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(ckpt_path / "lora_adapter"))
            torch.save(replacement_interfaces.state_dict(),
                       ckpt_path / "replacement_interface_weights.pt")

    logger.info("Training complete at step %d. Saving final checkpoint.", global_step)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(final_dir / "lora_adapter"))
    torch.save(replacement_interfaces.state_dict(),
               final_dir / "replacement_interface_weights.pt")

    replacement_manager.remove_hooks()

    final_loss = losses[-1] if losses else float("inf")
    final_ppl = math.exp(min(final_loss, 20.0))

    results = {
        "model_name": args.model_name,
        "track": "B",
        "config": "deltanet",
        "architecture": {
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "reservoir_size": args.reservoir_size,
            "spectral_radius": args.spectral_radius,
            "leak_rate": args.leak_rate,
            "selected_deltanet_blocks": selected_dn_indices,
            "replace_every_nth": n_step,
            "gate_init": args.replacement_gate_init,
            "integration": "deltanet_replacement",
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
