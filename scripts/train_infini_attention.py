#!/usr/bin/env python3
"""Train Infini-attention baseline on Qwen3.5-0.8B-Base.

This script:
  1. Loads Qwen3.5-0.8B-Base via the model loader (T3).
  2. Applies LoRA to the base model parameters.
  3. Adds Infini-attention compressive memory to selected attention layers.
  4. Trains for ~5 K steps on a fine-tuning mix (FineWeb subset by default).
  5. Saves the adapter checkpoint to checkpoints/infini_attention/.

Usage::

    python scripts/train_infini_attention.py
    python scripts/train_infini_attention.py --max_steps 5000 --output_dir checkpoints/infini_attention

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

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Infini-attention baseline")
    p.add_argument("--model_name", default="qwen3.5-0.8b")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", default="cuda")

    # Infini-attention
    p.add_argument(
        "--layer_indices",
        nargs="*",
        type=int,
        default=None,
        help="Transformer layer indices to augment (default: all odd-indexed layers).",
    )
    p.add_argument("--dropout", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--lr", type=float, default=2e-4)
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
    p.add_argument("--output_dir", default="checkpoints/infini_attention")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="lrs-baselines")
    p.add_argument("--wandb_run_name", default="infini-attention-qwen3.5-0.8b")
    return p.parse_args()


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
    """Build a streaming DataLoader over a HuggingFace text dataset.

    Returns an iterable that yields dicts with ``input_ids`` and
    ``labels`` tensors of shape (batch, seq_len).
    """
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

    ds = ds.map(tokenize_and_chunk, remove_columns=["text", "id", "dump", "url",
                                                      "file_path", "language",
                                                      "language_score", "token_count",
                                                      "score", "int_score"])

    from torch.utils.data import DataLoader  # type: ignore[import]

    def collate(batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            padded[i, : ids.shape[0]] = ids
        labels = padded.clone()
        labels[labels == 0] = -100  # ignore padding in loss
        return {"input_ids": padded, "labels": labels}

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # --- Apply LoRA ---
    logger.info("Applying LoRA (rank=%d, alpha=%.1f)", args.lora_rank, args.lora_alpha)
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]
        from src.models.loader import get_lora_targets

        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=get_lora_targets(args.model_name),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    except ImportError:
        logger.warning("peft not installed — training without LoRA.")

    # --- Apply Infini-attention ---
    logger.info("Applying Infini-attention to selected layers")
    from src.models.infini_attention import (
        apply_infini_attention,
        get_infini_trainable_params,
        reset_infini_memory,
    )

    layer_indices = args.layer_indices if args.layer_indices else None
    infini_layers = apply_infini_attention(
        model,
        layer_indices=layer_indices,
        dropout=args.dropout,
    )
    logger.info("Augmented %d layers with Infini-attention: %s", len(infini_layers),
                sorted(infini_layers.keys()))

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()
    model.to(device)

    # --- Optimizer: LoRA + infini-attention params ---
    infini_params = get_infini_trainable_params(model)
    other_params = [p for p in model.parameters() if p.requires_grad and
                    not any(p is ip for ip in infini_params)]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": infini_params, "lr": args.lr * 2},  # slightly higher LR for new modules
        ],
        weight_decay=args.weight_decay,
    )

    # Cosine LR schedule with warmup.
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR  # type: ignore

    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                             total_iters=args.warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps)
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

    for batch in loader:
        if global_step >= args.max_steps:
            break

        # Reset infini-attention memory for each batch (new sequence context).
        reset_infini_memory(model)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = accum_loss / args.log_interval
            logger.info(
                "step=%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                global_step,
                avg_loss,
                scheduler.get_last_lr()[0],
                elapsed,
            )
            try:
                import wandb  # type: ignore[import]

                wandb.log({"train/loss": avg_loss, "train/step": global_step})
            except Exception:
                pass
            accum_loss = 0.0

        if global_step % args.save_interval == 0:
            ckpt_path = output_dir / f"step_{global_step}"
            logger.info("Saving checkpoint to %s", ckpt_path)
            model.save_pretrained(str(ckpt_path))
            tokenizer._tok.save_pretrained(str(ckpt_path))  # type: ignore[attr-defined]

    # Final save
    logger.info("Training complete at step %d.  Saving final checkpoint.", global_step)
    model.save_pretrained(str(output_dir / "final"))
    tokenizer._tok.save_pretrained(str(output_dir / "final"))  # type: ignore[attr-defined]

    # Record training metadata
    meta = {
        "model_name": args.model_name,
        "steps_trained": global_step,
        "output_dir": str(output_dir),
        "args": vars(args),
        "timestamp": time.time(),
    }
    with (output_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done.  Run scripts/eval_infini_attention.py to evaluate.")


if __name__ == "__main__":
    train(parse_args())
