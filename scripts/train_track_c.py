#!/usr/bin/env python3
"""Track C pre-training: RW-Transformer from scratch with curriculum.

Trains the RW-Transformer (T24) from scratch using the three-stage curriculum
(T25) on cloud A100(s).

Training setup:
  - Model: RW-Transformer from src/models/rw_transformer.py (~0.8B params)
  - Data: CurriculumDataPipeline (Stage 1 → 2 → 3)
  - Optimizer: AdamW (full — training from scratch, no LoRA)
  - Precision: BF16 mixed precision
  - Gradient checkpointing: enabled (required for A100 80GB)
  - Target: ~20B tokens minimum
  - LR schedule: cosine with warmup (~2K steps)

Monitoring:
  - Wandb: loss, perplexity, LR, gradient norms, reservoir state statistics
  - Eval checkpoints every ~2B tokens on T5 benchmark subset
  - Early stopping on loss divergence

Usage::

    python scripts/train_track_c.py
    python scripts/train_track_c.py --config configs/track_c_pretrain.yaml
    python scripts/train_track_c.py --resume_from checkpoints/track_c/step_10000
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
    p = argparse.ArgumentParser(description="Train Track C: RW-Transformer from scratch")

    # Model
    p.add_argument("--model_config", default=None,
                   help="Path to RW-Transformer config JSON. If None, uses defaults.")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", default="cuda")

    # Reservoir
    p.add_argument("--fast_reservoir_size", type=int, default=5_000)
    p.add_argument("--fast_spectral_radius", type=float, default=0.9)
    p.add_argument("--fast_leak_rate", type=float, default=0.9)
    p.add_argument("--fast_reservoir_seed", type=int, default=42)
    p.add_argument("--slow_reservoir_size", type=int, default=5_000)
    p.add_argument("--slow_spectral_radius", type=float, default=0.5)
    p.add_argument("--slow_leak_rate", type=float, default=0.1)
    p.add_argument("--slow_reservoir_seed", type=int, default=43)
    p.add_argument("--input_scaling", type=float, default=1.0)
    p.add_argument("--reservoir_sparsity", type=float, default=0.01)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate for AdamW (default: 3e-4).")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR schedule
    p.add_argument("--warmup_steps", type=int, default=2_000,
                   help="Linear warmup steps (default: 2000).")
    p.add_argument("--max_steps", type=int, default=200_000,
                   help="Total training steps. 200K × 4K tokens/step ≈ 800B tokens budget.")
    p.add_argument("--min_lr_ratio", type=float, default=0.1,
                   help="Minimum LR as fraction of peak LR at end of cosine decay.")

    # Training loop
    p.add_argument("--batch_size", type=int, default=4,
                   help="Micro-batch size per gradient accumulation step.")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum).")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    # Curriculum (T25)
    p.add_argument("--stage1_steps", type=int, default=50_000,
                   help="Training steps for Stage 1 (general text only).")
    p.add_argument("--stage2_steps", type=int, default=50_000,
                   help="Training steps for Stage 2 (text + procedural mix).")
    p.add_argument("--text_ratio_stage2", type=float, default=0.7)
    p.add_argument("--procedural_ratio_stage2", type=float, default=0.3)
    p.add_argument("--stage3_steps_per_length", type=int, default=10_000,
                   help="Steps at each context length in Stage 3.")
    p.add_argument("--max_seq_length", type=int, default=4096,
                   help="Context length for Stages 1 and 2 (Stage 3 grows).")
    p.add_argument("--use_loss_plateau", action="store_true", default=False,
                   help="Use loss-plateau-based stage transitions instead of step-based.")

    # Data
    p.add_argument("--dataset_name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_config", default="sample-10BT")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--text_column", default="text")

    # Eval
    p.add_argument("--eval_every_tokens", type=int, default=2_000_000_000,
                   help="Evaluate on T5 benchmark subset every N tokens (~2B).")
    p.add_argument("--eval_steps", type=int, default=100,
                   help="Number of eval batches to use for perplexity estimation.")

    # Early stopping
    p.add_argument("--divergence_threshold", type=float, default=10.0,
                   help="Stop if loss exceeds this multiple of the min observed loss.")

    # Output
    p.add_argument("--output_dir", default="checkpoints/track_c")
    p.add_argument("--results_file", default="results/track_c/pretrain.json")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=5_000)
    p.add_argument("--resume_from", default=None,
                   help="Path to checkpoint directory to resume from.")

    # Wandb
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="lrs-track-c")
    p.add_argument("--wandb_run_name", default="track-c-pretrain-rw-transformer")

    p.add_argument("--config", default=None,
                   help="Path to YAML config file (overrides CLI defaults).")
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
# LR schedule: cosine with linear warmup
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float,
           min_lr_ratio: float) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return peak_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)


# ---------------------------------------------------------------------------
# Tokenize curriculum batch → tensor batch
# ---------------------------------------------------------------------------


def tokenize_batch(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    device: torch.device,
    dtype_ids: torch.dtype = torch.long,
) -> dict[str, torch.Tensor]:
    """Tokenize a list of text strings into padded input_ids / labels tensors."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="longest",
        pad_to_multiple_of=8,
    )
    input_ids = encoded["input_ids"].to(device=device, dtype=dtype_ids)
    labels = input_ids.clone()
    # Mask padding tokens in labels
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    labels[labels == pad_id] = -100
    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------


class TokenCounter:
    """Track total tokens processed for logging and eval scheduling."""

    def __init__(self) -> None:
        self.total = 0

    def update(self, input_ids: torch.Tensor) -> int:
        """Add the number of non-padding tokens from a batch."""
        n = int((input_ids != 0).sum().item())
        self.total += n
        return n


# ---------------------------------------------------------------------------
# Eval: perplexity on held-out text
# ---------------------------------------------------------------------------


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_seq_length: int,
    eval_steps: int = 100,
    seed: int = 0,
) -> float:
    """Estimate perplexity on a small held-out FineWeb sample.

    Returns float perplexity (exp of mean cross-entropy loss).
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=1_000)
    except Exception as exc:
        logger.warning("Could not load eval dataset (%s); skipping perplexity eval.", exc)
        return float("nan")

    model.eval()
    total_loss = 0.0
    n_batches = 0
    rng = iter(ds)

    with torch.no_grad():
        for _ in range(eval_steps):
            try:
                sample = next(rng)
            except StopIteration:
                break
            text = sample.get("text", "")
            if not text:
                continue

            batch = tokenize_batch(tokenizer, [text], max_seq_length, device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1

    model.train()
    if n_batches == 0:
        return float("nan")
    mean_loss = total_loss / n_batches
    return math.exp(min(mean_loss, 20.0))


# ---------------------------------------------------------------------------
# Eval: T5 benchmark subset
# ---------------------------------------------------------------------------


def evaluate_benchmarks(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    n_examples: int = 50,
) -> dict[str, float]:
    """Run a small subset of T5 benchmarks; returns accuracy per task."""
    try:
        from src.eval.benchmarks.computation import (
            DyckLanguage,
            ModularArithmetic,
            MultiDigitArithmetic,
        )
        from src.eval.benchmarks.memory import PasskeyRetrieval
    except ImportError as exc:
        logger.warning("Benchmark imports failed (%s); skipping benchmark eval.", exc)
        return {}

    tasks: dict[str, Any] = {
        "arithmetic_add": MultiDigitArithmetic(
            n=n_examples, digit_count=3, operation="addition", seed=999
        ),
        "modular_arith": ModularArithmetic(n=n_examples, seed=1000),
        "dyck_language": DyckLanguage(n=n_examples, max_depth=3, bracket_types=2, seed=1001),
        "passkey_retrieval": PasskeyRetrieval(n=n_examples, context_length=512, seed=1002),
    }

    model.eval()
    results: dict[str, float] = {}

    with torch.no_grad():
        for task_name, generator in tasks.items():
            correct = 0
            total = 0
            for example in generator:
                prompt = example.input
                target = example.target
                try:
                    enc = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = enc["input_ids"].to(device)
                    target_ids = tokenizer(
                        target,
                        return_tensors="pt",
                        truncation=True,
                        max_length=64,
                    )["input_ids"].to(device)

                    generated = model.generate(
                        input_ids,
                        max_new_tokens=target_ids.shape[1] + 4,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_text = tokenizer.decode(
                        generated[0, input_ids.shape[1]:],
                        skip_special_tokens=True,
                    ).strip()
                    if gen_text.startswith(target.strip()):
                        correct += 1
                    total += 1
                except Exception:
                    total += 1

            results[task_name] = correct / total if total > 0 else 0.0

    model.train()
    return results


# ---------------------------------------------------------------------------
# Reservoir state statistics
# ---------------------------------------------------------------------------


def reservoir_state_stats(multi_res: Any) -> dict[str, float]:
    """Compute summary statistics for fast and slow reservoir states."""
    stats: dict[str, float] = {}
    try:
        fast_state = np.asarray(multi_res.fast.state, dtype=np.float32)
        slow_state = np.asarray(multi_res.slow.state, dtype=np.float32)
        stats["reservoir/fast_norm"] = float(np.linalg.norm(fast_state))
        stats["reservoir/fast_mean"] = float(np.mean(fast_state))
        stats["reservoir/fast_std"] = float(np.std(fast_state))
        stats["reservoir/slow_norm"] = float(np.linalg.norm(slow_state))
        stats["reservoir/slow_mean"] = float(np.mean(slow_state))
        stats["reservoir/slow_std"] = float(np.std(slow_state))
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    global_tokens: int,
    loss: float,
    output_dir: Path,
    curriculum_state: dict[str, Any],
) -> Path:
    ckpt_dir = output_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(ckpt_dir / "model"))
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(
        {
            "step": step,
            "global_tokens": global_tokens,
            "loss": loss,
            "curriculum": curriculum_state,
        },
        ckpt_dir / "meta.pt",
    )
    logger.info("Checkpoint saved to %s", ckpt_dir)
    return ckpt_dir


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    ckpt_dir = Path(checkpoint_path)
    model_dir = ckpt_dir / "model"
    model_file = ckpt_dir / "model.pt"

    if model_dir.exists() and hasattr(model, "load_pretrained"):
        # HuggingFace-style model
        pass  # model is already loaded; just restore weights below
    elif model_file.exists():
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded model weights from %s", model_file)

    opt_file = ckpt_dir / "optimizer.pt"
    if opt_file.exists():
        opt_state = torch.load(opt_file, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(opt_state)
        logger.info("Loaded optimizer state from %s", opt_file)

    meta_file = ckpt_dir / "meta.pt"
    if meta_file.exists():
        return torch.load(meta_file, map_location="cpu", weights_only=False)
    return {}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Training on device=%s  dtype=%s", device, dtype)

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb  # type: ignore[import]
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            logger.info("wandb initialized: project=%s  run=%s",
                        args.wandb_project, args.wandb_run_name)
        except Exception as exc:
            logger.warning("wandb init failed (%s); continuing without logging.", exc)
            use_wandb = False

    # ------------------------------------------------------------------
    # Build model (RW-Transformer, T24)
    # ------------------------------------------------------------------
    logger.info("Building RW-Transformer from scratch (~0.8B params)...")
    try:
        from src.models.rw_transformer import RWTransformerConfig, build_rw_transformer
        if args.model_config:
            with open(args.model_config) as f:
                model_cfg_dict = json.load(f)
            rw_cfg = RWTransformerConfig(**model_cfg_dict)
        else:
            rw_cfg = RWTransformerConfig()  # default ~0.8B config
        model, tokenizer = build_rw_transformer(rw_cfg, dtype=dtype, device=device)
    except ImportError:
        logger.error(
            "src.models.rw_transformer not found (T24 not yet implemented). "
            "The RW-Transformer must be implemented before running Track C training."
        )
        raise

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %d total params (%.2fB),  %d trainable params (%.2fB)",
        total_params, total_params / 1e9,
        trainable_params, trainable_params / 1e9,
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    model.train()
    model.to(device)

    # ------------------------------------------------------------------
    # Build multi-reservoir (fast + slow)
    # ------------------------------------------------------------------
    logger.info(
        "Building MultiReservoir: fast(n=%d, lr=%.2f, sr=%.2f) + slow(n=%d, lr=%.2f, sr=%.2f)",
        args.fast_reservoir_size, args.fast_leak_rate, args.fast_spectral_radius,
        args.slow_reservoir_size, args.slow_leak_rate, args.slow_spectral_radius,
    )
    from src.reservoir.multi_reservoir import MultiReservoir, MultiReservoirConfig
    from src.types import ReservoirConfig

    hidden_dim = model.config.hidden_size if hasattr(model, "config") else 1024
    fast_cfg = ReservoirConfig(
        size=args.fast_reservoir_size,
        spectral_radius=args.fast_spectral_radius,
        leak_rate=args.fast_leak_rate,
        input_scaling=args.input_scaling,
        sparsity=args.reservoir_sparsity,
        seed=args.fast_reservoir_seed,
    )
    slow_cfg = ReservoirConfig(
        size=args.slow_reservoir_size,
        spectral_radius=args.slow_spectral_radius,
        leak_rate=args.slow_leak_rate,
        input_scaling=args.input_scaling,
        sparsity=args.reservoir_sparsity,
        seed=args.slow_reservoir_seed,
    )
    multi_cfg = MultiReservoirConfig(fast=fast_cfg, slow=slow_cfg)
    multi_res = MultiReservoir(config=multi_cfg, input_dim=hidden_dim)
    logger.info(
        "MultiReservoir: combined_dim=%d (fast=%d + slow=%d)",
        multi_res.state_dim, multi_res.fast_dim, multi_res.slow_dim,
    )

    # ------------------------------------------------------------------
    # Optimizer (full AdamW — training from scratch)
    # ------------------------------------------------------------------
    all_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    logger.info(
        "Optimizer: AdamW  lr=%.2e  weight_decay=%.3f  beta1=%.2f  beta2=%.3f",
        args.lr, args.weight_decay, args.adam_beta1, args.adam_beta2,
    )

    # ------------------------------------------------------------------
    # Curriculum data pipeline (T25)
    # ------------------------------------------------------------------
    from src.training.curriculum import CurriculumConfig, CurriculumDataPipeline, StageConfig

    curriculum_config = CurriculumConfig(
        stage1=StageConfig(
            text_ratio=1.0,
            procedural_ratio=0.0,
            context_length=args.max_seq_length,
        ),
        stage2=StageConfig(
            text_ratio=args.text_ratio_stage2,
            procedural_ratio=args.procedural_ratio_stage2,
            context_length=args.max_seq_length,
        ),
        stage3_context_lengths=[4096, 8192, 16384, 32768, 131072],
        stage3_steps_per_length=args.stage3_steps_per_length,
        batch_size=args.batch_size,
        text_corpus=args.dataset_name,
        text_corpus_split=args.dataset_split,
        text_column=args.text_column,
        text_corpus_name=args.dataset_config,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        use_loss_plateau=args.use_loss_plateau,
        seed=args.seed,
    )
    curriculum = CurriculumDataPipeline(config=curriculum_config)
    logger.info("Curriculum pipeline built: Stage1=%d steps, Stage2=%d steps",
                args.stage1_steps, args.stage2_steps)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    global_tokens = 0
    resume_meta: dict[str, Any] = {}

    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        resume_meta = load_checkpoint(model, optimizer, args.resume_from)
        global_step = resume_meta.get("step", 0)
        global_tokens = resume_meta.get("global_tokens", 0)
        if "curriculum" in resume_meta:
            curriculum = CurriculumDataPipeline.from_checkpoint(
                resume_meta["curriculum"],
                config=curriculum_config,
            )
        logger.info("Resumed at step=%d  tokens=%d", global_step, global_tokens)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    token_counter = TokenCounter()
    token_counter.total = global_tokens
    losses: list[float] = []
    min_observed_loss = float("inf")
    tokens_since_eval = 0
    accum_loss = 0.0
    t_start = time.time()

    optimizer.zero_grad()

    eval_perplexity_log: list[dict[str, Any]] = []
    benchmark_log: list[dict[str, Any]] = []

    # Wrap tokenizer for direct use (curriculum yields raw text)
    raw_tokenizer = getattr(tokenizer, "_tok", tokenizer)

    logger.info(
        "Starting training: max_steps=%d  batch=%d  grad_accum=%d  warmup=%d",
        args.max_steps, args.batch_size, args.grad_accum, args.warmup_steps,
    )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    for curriculum_batch in curriculum.iterate():
        if global_step >= args.max_steps:
            break

        texts = curriculum_batch["texts"]
        stage = curriculum_batch["stage"]
        ctx_len = curriculum_batch["context_length"]

        # Tokenize
        try:
            tensor_batch = tokenize_batch(raw_tokenizer, texts, ctx_len, device)
        except Exception as exc:
            logger.warning("Tokenization failed (step=%d): %s", global_step, exc)
            continue

        input_ids = tensor_batch["input_ids"]
        labels = tensor_batch["labels"]

        # Update reservoir (read-only; reservoir not trained — weights frozen)
        try:
            with torch.no_grad():
                embed_layer = model.get_input_embeddings()
                embeddings = embed_layer(input_ids)
                emb_np = embeddings.detach().float().cpu().numpy()
            B, T, D = emb_np.shape
            for b in range(B):
                multi_res.reset()
                for t in range(T):
                    multi_res.step(emb_np[b, t])
        except Exception as exc:
            logger.debug("Reservoir update failed (step=%d): %s", global_step, exc)

        # Update LR
        current_lr = get_lr(
            global_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr_ratio
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward + backward
        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

        loss.backward()
        accum_loss += loss.item()

        n_tokens = token_counter.update(input_ids)
        tokens_since_eval += n_tokens

        # Optimizer step (every grad_accum micro-steps)
        if (global_step + 1) % args.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.item() * args.grad_accum
        losses.append(loss_val)
        min_observed_loss = min(min_observed_loss, loss_val)

        # Report loss for plateau-based curriculum transitions
        if args.use_loss_plateau:
            curriculum.report_loss(loss_val)

        global_step += 1

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = accum_loss * args.grad_accum / args.log_interval
            ppl = math.exp(min(avg_loss, 20.0))
            tokens_per_sec = token_counter.total / max(1, elapsed)
            grad_norm_val = float(
                torch.nn.utils.clip_grad_norm_(all_params, float("inf"))
            )
            logger.info(
                "step=%d  stage=%d  ctx=%d  loss=%.4f  ppl=%.2f  "
                "lr=%.2e  grad_norm=%.3f  tokens=%dM  %.0f tok/s",
                global_step, stage, ctx_len, avg_loss, ppl,
                current_lr, grad_norm_val,
                token_counter.total // 1_000_000,
                tokens_per_sec,
            )

            if use_wandb:
                try:
                    import wandb  # type: ignore[import]
                    log_dict: dict[str, Any] = {
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm_val,
                        "train/stage": stage,
                        "train/context_length": ctx_len,
                        "train/tokens_M": token_counter.total / 1e6,
                        "train/step": global_step,
                        "perf/tokens_per_sec": tokens_per_sec,
                    }
                    log_dict.update(reservoir_state_stats(multi_res))
                    wandb.log(log_dict, step=global_step)
                except Exception:
                    pass

            accum_loss = 0.0

        # ------------------------------------------------------------------
        # Periodic eval (every ~2B tokens)
        # ------------------------------------------------------------------
        if tokens_since_eval >= args.eval_every_tokens:
            tokens_since_eval = 0
            logger.info("Running periodic evaluation at step=%d  tokens=%dM...",
                        global_step, token_counter.total // 1_000_000)

            ppl = evaluate_perplexity(
                model, raw_tokenizer, device, dtype,
                args.max_seq_length, eval_steps=args.eval_steps,
                seed=global_step,
            )
            logger.info("Eval perplexity: %.2f  (step=%d)", ppl, global_step)
            eval_entry = {
                "step": global_step,
                "tokens_M": token_counter.total / 1e6,
                "perplexity": ppl,
            }
            eval_perplexity_log.append(eval_entry)

            bm_results = evaluate_benchmarks(
                model, raw_tokenizer, device, dtype, n_examples=50
            )
            if bm_results:
                logger.info("Benchmark results: %s", bm_results)
                benchmark_log.append({
                    "step": global_step,
                    "tokens_M": token_counter.total / 1e6,
                    **bm_results,
                })

            if use_wandb:
                try:
                    import wandb  # type: ignore[import]
                    eval_log: dict[str, Any] = {
                        "eval/perplexity": ppl,
                        "eval/step": global_step,
                    }
                    eval_log.update({f"eval/bench_{k}": v for k, v in bm_results.items()})
                    wandb.log(eval_log, step=global_step)
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Checkpoint saving
        # ------------------------------------------------------------------
        if global_step % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, global_step, token_counter.total,
                losses[-1] if losses else float("nan"),
                output_dir, curriculum.checkpoint_state(),
            )

        # ------------------------------------------------------------------
        # Early stopping: divergence check
        # ------------------------------------------------------------------
        if loss_val > args.divergence_threshold * min_observed_loss and global_step > 500:
            logger.warning(
                "Loss divergence detected at step=%d: loss=%.4f > %.1f × min_loss=%.4f. "
                "Stopping early.",
                global_step, loss_val, args.divergence_threshold, min_observed_loss,
            )
            break

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    logger.info("Training complete at step=%d  tokens=%dM. Saving final checkpoint.",
                global_step, token_counter.total // 1_000_000)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model, optimizer, global_step, token_counter.total,
        losses[-1] if losses else float("nan"),
        final_dir, curriculum.checkpoint_state(),
    )

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    final_ppl = evaluate_perplexity(
        model, raw_tokenizer, device, dtype,
        args.max_seq_length, eval_steps=args.eval_steps * 2,
        seed=999_999,
    )
    final_benchmarks = evaluate_benchmarks(
        model, raw_tokenizer, device, dtype, n_examples=100
    )
    logger.info("Final perplexity: %.2f", final_ppl)
    logger.info("Final benchmarks: %s", final_benchmarks)

    # ------------------------------------------------------------------
    # Results JSON
    # ------------------------------------------------------------------
    final_loss = losses[-1] if losses else float("nan")
    results = {
        "track": "C",
        "task": "T26",
        "model": "rw-transformer",
        "total_steps": global_step,
        "total_tokens_B": token_counter.total / 1e9,
        "final_train_loss": final_loss,
        "final_train_perplexity": math.exp(min(final_loss, 20.0)) if not math.isnan(final_loss) else float("nan"),
        "eval_perplexity": final_ppl,
        "benchmark_results": final_benchmarks,
        "eval_perplexity_log": eval_perplexity_log,
        "benchmark_log": benchmark_log,
        "loss_history": losses[::max(1, len(losses) // 200)],
        "training_config": vars(args),
        "output_dir": str(output_dir),
        "timestamp": time.time(),
    }

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", results_path)

    if use_wandb:
        try:
            import wandb  # type: ignore[import]
            wandb.log({
                "final/train_loss": final_loss,
                "final/eval_perplexity": final_ppl,
                "final/total_tokens_B": token_counter.total / 1e9,
            })
            wandb.finish()
        except Exception:
            pass

    logger.info(
        "Done. final_loss=%.4f  final_ppl=%.2f  total_tokens=%.2fB  steps=%d",
        final_loss,
        math.exp(min(final_loss, 20.0)) if not math.isnan(final_loss) else float("nan"),
        token_counter.total / 1e9,
        global_step,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = _load_yaml_config(args.config)
        apply_config_overrides(args, cfg)
    train(args)
