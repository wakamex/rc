#!/usr/bin/env python3
"""Train LLaMA-3.2-1B + ESN reservoir (read-only cross-attention sidecar).

T17: Control experiment for the DeltaNet synergy hypothesis.

Same read-only sidecar architecture as T14 (Qwen3.5 + reservoir), but using
LLaMA-3.2-1B (pure softmax attention) as the base model.

Architecture:
- Frozen LLaMA-3.2-1B + LoRA adapters
- ESN reservoir driven by token embeddings (read-only: LLM does not write back)
- CrossAttentionSidecar (from T12) injected at selected decoder layers
- ReadProjection bridges reservoir states to LLM hidden dimension

Hypothesis test (compare with T14 results):
  Δ_Qwen  = T14 - T7   (Qwen + reservoir vs Qwen vanilla)
  Δ_LLaMA = T17 - T10  (LLaMA + reservoir vs LLaMA vanilla)
  Δ_Qwen >> Δ_LLaMA  →  DeltaNet synergy confirmed
  Δ_Qwen ≈  Δ_LLaMA  →  reservoir benefit is architecture-agnostic

Usage::

    python scripts/train_llama_reservoir.py
    python scripts/train_llama_reservoir.py --max_steps 5000 --output_dir checkpoints/llama_reservoir

Output: results/track_a/llama_reservoir.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reservoir sidecar layer wrapper
# ---------------------------------------------------------------------------


class ReservoirSidecarLayer(nn.Module):
    """Wraps a transformer decoder layer with a reservoir cross-attention sidecar.

    After the base decoder layer runs, the CrossAttentionSidecar lets the LLM
    hidden states attend to ESN reservoir states as additional context memory.

    This is *read-only* from the reservoir's perspective: the reservoir is
    driven by token embeddings; the LLM reads from it but does not update it.

    Args:
        base_layer: The original transformer decoder layer to wrap.
        sidecar: CrossAttentionSidecar module (from src.reservoir.interface).
    """

    def __init__(self, base_layer: nn.Module, sidecar: nn.Module) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.sidecar = sidecar
        self._reservoir_states: np.ndarray | None = None

    def set_reservoir_states(self, states: np.ndarray) -> None:
        """Provide reservoir states for the next forward pass.

        Args:
            states: shape ``(batch, seq_len, reservoir_dim)`` or
                    ``(seq_len, reservoir_dim)``.
        """
        self._reservoir_states = states

    def clear_reservoir_states(self) -> None:
        """Remove stored reservoir states (call after forward pass)."""
        self._reservoir_states = None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        base_out = self.base_layer(hidden_states, *args, **kwargs)

        base_is_tuple = isinstance(base_out, tuple)
        if base_is_tuple:
            hidden_out: torch.Tensor = base_out[0]
            rest = base_out[1:]
        else:
            hidden_out = base_out
            rest = ()

        if self._reservoir_states is not None:
            hidden_out = self.sidecar(hidden_out, self._reservoir_states)

        if base_is_tuple:
            return (hidden_out,) + rest
        return hidden_out


# ---------------------------------------------------------------------------
# Reservoir sidecar installation
# ---------------------------------------------------------------------------


def apply_reservoir_sidecar(
    model: nn.Module,
    sidecar_cls: type,
    reservoir_dim: int,
    hidden_dim: int,
    num_heads: int,
    layer_indices: list[int] | None = None,
    dropout: float = 0.0,
) -> dict[int, ReservoirSidecarLayer]:
    """Install ReservoirSidecarLayer wrappers on selected transformer layers.

    Must be called on the *raw* HuggingFace model *before* PEFT wrapping so
    that the wrapped layers appear correctly in the module tree.

    Args:
        model: Raw ``LlamaForCausalLM`` (before ``get_peft_model``).
        sidecar_cls: The ``CrossAttentionSidecar`` class to instantiate.
        reservoir_dim: ESN reservoir state dimension.
        hidden_dim: LLM hidden state dimension.
        num_heads: Number of attention heads for the cross-attention sidecar.
        layer_indices: Decoder layer indices to augment.  Defaults to every
            4th layer (indices 3, 7, 11, 15 for a 16-layer model).
        dropout: Attention dropout in the sidecar.

    Returns:
        Dict mapping each augmented layer index to its ReservoirSidecarLayer.

    Raises:
        ValueError: If transformer layers cannot be located.
    """
    # Locate the ModuleList of transformer decoder layers.
    transformer_layers: nn.ModuleList | None = None
    for attr_path in ("model.layers", "transformer.h", "layers", "model.decoder.layers"):
        obj: Any = model
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            transformer_layers = obj  # type: ignore[assignment]
            break

    if transformer_layers is None:
        raise ValueError(
            "Cannot locate transformer layers in model. "
            "Tried: model.layers, transformer.h, layers, model.decoder.layers"
        )

    n_layers = len(transformer_layers)
    if layer_indices is None:
        # Every 4th layer by default (indices 3, 7, 11, … for 16-layer LLaMA)
        layer_indices = list(range(3, n_layers, 4))

    wrapped: dict[int, ReservoirSidecarLayer] = {}
    for idx in layer_indices:
        if idx >= n_layers:
            logger.warning("Layer index %d >= n_layers %d; skipping.", idx, n_layers)
            continue

        sidecar = sidecar_cls(
            hidden_dim=hidden_dim,
            reservoir_dim=reservoir_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        sidecar_layer = ReservoirSidecarLayer(
            base_layer=transformer_layers[idx],
            sidecar=sidecar,
        )
        transformer_layers[idx] = sidecar_layer
        wrapped[idx] = sidecar_layer

    return wrapped


def set_all_reservoir_states(
    wrapped_layers: dict[int, ReservoirSidecarLayer],
    states: np.ndarray,
) -> None:
    """Broadcast reservoir states to all sidecar layers."""
    for layer in wrapped_layers.values():
        layer.set_reservoir_states(states)


def clear_all_reservoir_states(
    wrapped_layers: dict[int, ReservoirSidecarLayer],
) -> None:
    """Clear reservoir states from all sidecar layers."""
    for layer in wrapped_layers.values():
        layer.clear_reservoir_states()


def get_sidecar_params(
    wrapped_layers: dict[int, ReservoirSidecarLayer],
) -> list[nn.Parameter]:
    """Return all trainable parameters from sidecar modules."""
    params: list[nn.Parameter] = []
    for layer in wrapped_layers.values():
        params.extend(list(layer.sidecar.parameters()))
    return params


# ---------------------------------------------------------------------------
# Reservoir forward pass
# ---------------------------------------------------------------------------


def run_reservoir_on_batch(
    esn: Any,
    embeds: torch.Tensor,
    input_dim: int,
) -> np.ndarray:
    """Run the ESN reservoir over a batch of token embeddings.

    The reservoir is reset at the start of each call, then stepped for every
    token position in the sequence.

    Args:
        esn: ``ESN`` reservoir (will be reset).
        embeds: Token embeddings, shape ``(batch, seq_len, hidden_dim)``.
        input_dim: Number of embedding dims to use as reservoir input.
            Uses the first ``input_dim`` dimensions of the embedding.

    Returns:
        Reservoir state sequence, shape ``(batch, seq_len, reservoir_dim)``.
    """
    B, T, H = embeds.shape
    # Truncate embedding to reservoir input_dim (avoids huge W_in)
    inp_dim = min(input_dim, H)
    embeds_np = embeds[:, :, :inp_dim].detach().cpu().float().numpy()

    esn.reset()
    states: list[np.ndarray] = []
    for t in range(T):
        x_t = embeds_np[:, t, :]  # (B, inp_dim)
        state = esn.step(x_t)     # mutates esn.state; returns (B, n) or (n,)
        states.append(state.copy())

    return np.stack(states, axis=1)  # (B, T, n)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_dataloader(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    max_seq_length: int,
    batch_size: int,
    seed: int = 42,
) -> Any:
    """Build a streaming HuggingFace DataLoader over a text dataset."""
    from datasets import load_dataset  # type: ignore[import]
    from torch.utils.data import DataLoader

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    def tokenize_and_chunk(example: dict) -> dict:
        text = example.get("text", "")
        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"].squeeze(0)
        return {"input_ids": ids}

    # Remove FineWeb-specific columns; ignore missing ones silently.
    fw_columns = ["text", "id", "dump", "url", "file_path", "language",
                  "language_score", "token_count", "score", "int_score"]
    ds = ds.map(tokenize_and_chunk, remove_columns=fw_columns)

    def collate(batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            padded[i, : ids.shape[0]] = ids
        labels = padded.clone()
        labels[labels == 0] = -100  # ignore padding tokens in loss
        return {"input_ids": padded, "labels": labels}

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# Benchmark suite — identical to T10 (eval_llama.py) for direct comparison
# ---------------------------------------------------------------------------


def build_benchmarks(n: int = 200) -> list:
    """Return the full benchmark suite used for T10 / T17 comparison.

    This is identical to the suite in ``scripts/eval_llama.py`` (T10) so that
    T17 - T10 gives a direct per-task measurement of the reservoir benefit.
    """
    from src.eval.benchmarks.computation import (
        DyckLanguage,
        ModularArithmetic,
        MultiDigitArithmetic,
        ProgramTrace,
    )
    from src.eval.benchmarks.emergent import (
        AlgorithmicTransfer,
        CompositionalGeneralization,
        LengthExtrapolation,
    )
    from src.eval.benchmarks.memory import (
        AssociativeRecall,
        PasskeyRetrieval,
        VariableTracking,
    )

    return [
        # Memory benchmarks
        PasskeyRetrieval(n=n, context_length=200, seed=42),
        PasskeyRetrieval(n=n, context_length=500, seed=43),
        VariableTracking(n=n, num_variables=3, num_operations=5, seed=42),
        VariableTracking(n=n, num_variables=5, num_operations=10, seed=43),
        AssociativeRecall(n=n, num_pairs=5, delay_length=30, seed=42),
        AssociativeRecall(n=n, num_pairs=10, delay_length=50, seed=43),
        # Computation benchmarks
        MultiDigitArithmetic(n=n, digit_count=3, operation="addition", seed=42),
        MultiDigitArithmetic(n=n, digit_count=4, operation="addition", seed=43),
        MultiDigitArithmetic(n=n, digit_count=3, operation="multiplication", seed=44),
        ModularArithmetic(n=n, modulus=97, seed=42),
        DyckLanguage(n=n, max_depth=3, bracket_types=1, seed=42),
        DyckLanguage(n=n, max_depth=4, bracket_types=2, seed=43),
        ProgramTrace(n=n, num_steps=4, num_vars=3, seed=42),
        ProgramTrace(n=n, num_steps=6, num_vars=3, seed=43),
        # Emergent benchmarks
        CompositionalGeneralization(n=n, split="train", seed=42),
        CompositionalGeneralization(n=n, split="test", seed=43),
        LengthExtrapolation(n=n, train_length=5, test_multiplier=1.0, seed=42),
        LengthExtrapolation(n=n, train_length=5, test_multiplier=2.0, seed=43),
        LengthExtrapolation(n=n, train_length=5, test_multiplier=4.0, seed=44),
        AlgorithmicTransfer(n=n, family="sorting", split="train", seed=42),
        AlgorithmicTransfer(n=n, family="sorting", split="test", seed=43),
        AlgorithmicTransfer(n=n, family="search", split="train", seed=42),
        AlgorithmicTransfer(n=n, family="search", split="test", seed=43),
    ]


# ---------------------------------------------------------------------------
# Evaluation adapter
# ---------------------------------------------------------------------------


class ReservoirLlamaEvalAdapter:
    """Eval harness adapter for LLaMA + reservoir sidecar.

    Drives the ESN reservoir with prompt embeddings before each ``generate``
    call so the sidecar has reservoir states available during decoding.
    The reservoir is *not* updated during the decoding steps (read-only).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        esn: Any,
        wrapped_layers: dict[int, ReservoirSidecarLayer],
        reservoir_input_dim: int,
        device: torch.device,
        max_new_tokens: int = 64,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.esn = esn
        self.wrapped_layers = wrapped_layers
        self.reservoir_input_dim = reservoir_input_dim
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        """Generate a string continuation from a string prompt."""
        kwargs.pop("seed", None)

        input_ids = self.tokenizer(
            str(prompt),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )["input_ids"].to(self.device)

        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
            states = run_reservoir_on_batch(self.esn, embeds, self.reservoir_input_dim)
            set_all_reservoir_states(self.wrapped_layers, states)

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        clear_all_reservoir_states(self.wrapped_layers)
        new_ids = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        return self.model(input_ids, **kwargs)

    def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
        out = self.model(
            input_ids.to(self.device),
            output_hidden_states=True,
            **kwargs,
        )
        return out.hidden_states[layer]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LLaMA-3.2-1B + ESN reservoir (read-only cross-attention sidecar)"
    )

    p.add_argument("--model_name", default="llama-3.2-1b")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", default="cuda")

    # ESN reservoir hyperparameters (same config as T14 for controlled comparison)
    p.add_argument("--reservoir_size", type=int, default=1000,
                   help="Number of ESN reservoir neurons.")
    p.add_argument("--spectral_radius", type=float, default=0.9)
    p.add_argument("--leak_rate", type=float, default=0.3)
    p.add_argument("--input_scaling", type=float, default=1.0)
    p.add_argument("--reservoir_topology", default="erdos_renyi",
                   choices=["erdos_renyi", "small_world"])
    p.add_argument("--reservoir_sparsity", type=float, default=0.01)
    p.add_argument("--reservoir_input_dim", type=int, default=128,
                   help="Dims of token embeddings used as reservoir input (first N dims).")
    p.add_argument("--reservoir_seed", type=int, default=42)

    # CrossAttentionSidecar configuration
    p.add_argument("--sidecar_num_heads", type=int, default=8)
    p.add_argument("--sidecar_layer_indices", nargs="*", type=int, default=None,
                   help="Decoder layer indices to augment (default: every 4th).")
    p.add_argument("--sidecar_dropout", type=float, default=0.0)

    # LoRA (same hyperparameters as T14)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimiser / schedule (same as T14)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate for LoRA parameters.")
    p.add_argument("--sidecar_lr", type=float, default=1e-3,
                   help="Learning rate for CrossAttentionSidecar parameters.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    # Data (same fine-tuning mix as T14)
    p.add_argument("--dataset_name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_config", default="sample-10BT")

    # Output
    p.add_argument("--output_dir", default="checkpoints/llama_reservoir")
    p.add_argument("--results_output", default="results/track_a/llama_reservoir.json")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="lrs-track-a")
    p.add_argument("--wandb_run_name", default="t17-llama-reservoir")

    # Evaluation
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip benchmark evaluation after training.")
    p.add_argument("--n_eval_examples", type=int, default=200,
                   help="Examples per benchmark task (must match T14 for comparison).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Weights & Biases (optional) ---
    if not args.no_wandb:
        try:
            import wandb  # type: ignore[import]

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except Exception as exc:
            logger.warning("wandb init failed (%s); continuing without logging.", exc)

    # --- Load base model ---
    logger.info("Loading model: %s", args.model_name)
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.loader import get_lora_targets, load_model

    wrapper = load_model(args.model_name, dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer._tok  # type: ignore[attr-defined]

    # Infer LLM hidden dimension from model config
    hidden_dim = int(getattr(model.config, "hidden_size", 2048))
    logger.info("LLaMA hidden_dim=%d", hidden_dim)

    # --- Create ESN reservoir ---
    # Same config as T14 (Qwen + reservoir) for controlled DeltaNet synergy test
    logger.info(
        "Creating ESN reservoir: size=%d, spectral_radius=%.2f, "
        "leak_rate=%.2f, input_dim=%d",
        args.reservoir_size, args.spectral_radius,
        args.leak_rate, args.reservoir_input_dim,
    )
    from src.reservoir.esn import ESN
    from src.types import ReservoirConfig

    esn_config = ReservoirConfig(
        size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        input_scaling=args.input_scaling,
        topology=args.reservoir_topology,
        sparsity=args.reservoir_sparsity,
        seed=args.reservoir_seed,
    )
    esn = ESN(esn_config, input_dim=args.reservoir_input_dim)

    # --- Apply reservoir cross-attention sidecar (BEFORE LoRA wrapping) ---
    logger.info("Applying reservoir cross-attention sidecar")
    from src.reservoir.interface import CrossAttentionSidecar

    wrapped_layers = apply_reservoir_sidecar(
        model=model,
        sidecar_cls=CrossAttentionSidecar,
        reservoir_dim=args.reservoir_size,
        hidden_dim=hidden_dim,
        num_heads=args.sidecar_num_heads,
        layer_indices=args.sidecar_layer_indices,
        dropout=args.sidecar_dropout,
    )
    logger.info(
        "Installed sidecar on %d layers: %s",
        len(wrapped_layers),
        sorted(wrapped_layers.keys()),
    )

    # --- Apply LoRA adapters ---
    logger.info("Applying LoRA (rank=%d, alpha=%.1f)", args.lora_rank, args.lora_alpha)
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]

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

    # Ensure sidecar parameters are trainable (PEFT freezes non-LoRA params)
    for layer in wrapped_layers.values():
        for param in layer.sidecar.parameters():
            param.requires_grad_(True)

    # Move sidecar modules to the target device (they were created on CPU)
    for layer in wrapped_layers.values():
        layer.sidecar.to(device)

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()

    # --- Optimizer with separate learning rates ---
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    ]
    sidecar_params = get_sidecar_params(wrapped_layers)
    other_params = [
        p for p in model.parameters()
        if p.requires_grad
        and not any(p is lp for lp in lora_params)
        and not any(p is sp for sp in sidecar_params)
    ]

    param_groups: list[dict] = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr})
    if sidecar_params:
        param_groups.append({"params": sidecar_params, "lr": args.sidecar_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR  # type: ignore

    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, args.max_steps - args.warmup_steps),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_steps],
    )

    # --- Data ---
    logger.info(
        "Building data loader from %s / %s", args.dataset_name, args.dataset_config
    )
    loader = build_dataloader(
        tokenizer=tokenizer,
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

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 1. Drive the reservoir with token embeddings (no gradient needed)
        with torch.no_grad():
            embeds = model.get_input_embeddings()(input_ids)
        reservoir_states = run_reservoir_on_batch(
            esn, embeds, args.reservoir_input_dim
        )

        # 2. Inject reservoir states into sidecar layers
        set_all_reservoir_states(wrapped_layers, reservoir_states)

        # 3. Forward pass with cross-attention sidecar active
        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

        loss.backward()
        accum_loss += loss.item()

        # 4. Clear reservoir states after backward
        clear_all_reservoir_states(wrapped_layers)

        if (global_step + 1) % args.grad_accum == 0:
            all_trainable = (
                [p for p in model.parameters() if p.requires_grad]
                + [p for p in get_sidecar_params(wrapped_layers) if not p.requires_grad]
            )
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = accum_loss / args.log_interval
            logger.info(
                "step=%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
                global_step, avg_loss, scheduler.get_last_lr()[0], elapsed,
            )
            try:
                import wandb  # type: ignore[import]

                wandb.log({"train/loss": avg_loss, "train/step": global_step})
            except Exception:
                pass
            accum_loss = 0.0

        if global_step % args.save_interval == 0:
            ckpt_path = output_dir / f"step_{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving checkpoint to %s", ckpt_path)
            model.save_pretrained(str(ckpt_path))
            sidecar_state = {
                str(idx): layer.sidecar.state_dict()
                for idx, layer in wrapped_layers.items()
            }
            torch.save(sidecar_state, ckpt_path / "sidecar_weights.pt")

    # --- Final checkpoint ---
    logger.info("Training complete at step %d. Saving final checkpoint.", global_step)
    final_ckpt = output_dir / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_ckpt))
    sidecar_state = {
        str(idx): layer.sidecar.state_dict()
        for idx, layer in wrapped_layers.items()
    }
    torch.save(sidecar_state, final_ckpt / "sidecar_weights.pt")

    # --- Benchmark evaluation ---
    if not args.skip_eval:
        logger.info("Running benchmark evaluation (n=%d per task)…", args.n_eval_examples)
        model.eval()

        eval_adapter = ReservoirLlamaEvalAdapter(
            model=model,
            tokenizer=tokenizer,
            esn=esn,
            wrapped_layers=wrapped_layers,
            reservoir_input_dim=args.reservoir_input_dim,
            device=device,
            max_new_tokens=64,
        )

        benchmarks = build_benchmarks(n=args.n_eval_examples)
        Path(args.results_output).parent.mkdir(parents=True, exist_ok=True)

        from src.eval.harness import EvalConfig, evaluate

        eval_config = EvalConfig(
            batch_size=1,
            num_few_shot=0,
            decode_mode="greedy",
            metrics=["exact_match", "accuracy", "f1"],
            output_file=args.results_output,
            model_name="llama-3.2-1b-reservoir",
            resume=False,
        )

        results = evaluate(eval_adapter, benchmarks, eval_config)  # type: ignore[arg-type]
        logger.info(
            "Evaluation complete. %d results written to %s",
            len(results), args.results_output,
        )
        for r in results:
            logger.info("  %-45s  %s = %.4f", r.task, r.metric, r.value)

    # --- Training metadata ---
    meta = {
        "task": "T17",
        "model_name": args.model_name,
        "steps_trained": global_step,
        "output_dir": str(output_dir),
        "results_output": args.results_output,
        "args": vars(args),
        "timestamp": time.time(),
        "hypothesis": (
            "DeltaNet synergy control — compare Δ_LLaMA=(T17-T10) with Δ_Qwen=(T14-T7). "
            "If Δ_Qwen >> Δ_LLaMA: DeltaNet synergy confirmed. "
            "If Δ_Qwen ≈ Δ_LLaMA: reservoir benefit is architecture-agnostic."
        ),
    }
    with (output_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done. Run eval separately with --skip_eval=False or check %s.",
                args.results_output)


if __name__ == "__main__":
    train(parse_args())
