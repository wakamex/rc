#!/usr/bin/env python3
"""Evaluate Infini-attention baseline and write results to JSON.

This script:
  1. Loads the trained Infini-attention checkpoint (or the base model for a
     quick sanity check with --no_checkpoint).
  2. Runs the full benchmark suite from T5 (memory, computation, emergent).
  3. Writes standardised JSON to results/baselines/infini_attention.json.

Usage::

    # After training:
    python scripts/eval_infini_attention.py \\
        --checkpoint checkpoints/infini_attention/final

    # Quick smoke-test without a trained checkpoint:
    python scripts/eval_infini_attention.py --no_checkpoint --num_examples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Infini-attention baseline")
    p.add_argument("--checkpoint", default=None,
                   help="Path to saved adapter checkpoint (from train_infini_attention.py).")
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Skip checkpoint loading; evaluate base model + infini-attention modules.")
    p.add_argument("--model_name", default="qwen3.5-0.8b")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--layer_indices",
        nargs="*",
        type=int,
        default=None,
        help="Infini-attention layer indices (must match training config).",
    )
    p.add_argument("--num_examples", type=int, default=200,
                   help="Examples per benchmark for evaluation.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="results/baselines/infini_attention.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Eval wrapper that makes ModelWrapperImpl work with the harness
# ---------------------------------------------------------------------------


class InfiniEvalWrapper:
    """Adapts a loaded model + tokenizer for use with src.eval.harness.evaluate.

    The harness calls model.generate(prompt_string, ...).  This wrapper
    tokenizes the prompt, runs generate, and decodes the new tokens.
    """

    def __init__(self, model: object, tokenizer: object, device: torch.device,
                 max_new_tokens: int = 50) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def forward(self, input_ids: torch.Tensor, **kwargs):  # type: ignore[override]
        return self.model(input_ids.to(self.device), **kwargs)

    def generate(self, prompt: str | torch.Tensor, **kwargs) -> str:  # type: ignore[override]
        from src.models.infini_attention import reset_infini_memory
        import torch

        if isinstance(prompt, str):
            input_ids = self.tokenizer(  # type: ignore[operator]
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )["input_ids"].to(self.device)
        else:
            input_ids = prompt.to(self.device)

        # Reset memory before each generation call.
        reset_infini_memory(self.model)  # type: ignore[arg-type]

        with torch.no_grad():
            out = self.model.generate(  # type: ignore[attr-defined]
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,  # type: ignore[attr-defined]
            )
        # Decode only the newly generated tokens.
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[attr-defined]

    def get_hidden(self, input_ids: torch.Tensor, layer: int = -1, **kwargs) -> torch.Tensor:
        out = self.model(  # type: ignore[attr-defined]
            input_ids.to(self.device),
            output_hidden_states=True,
            **kwargs,
        )
        return out.hidden_states[layer]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    logger.info("Loading base model: %s", args.model_name)
    from src.models.loader import load_model

    wrapper = load_model(args.model_name, dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    # --- Apply Infini-attention structure ---
    from src.models.infini_attention import apply_infini_attention

    layer_indices = args.layer_indices if args.layer_indices else None
    infini_layers = apply_infini_attention(model, layer_indices=layer_indices)
    logger.info("Installed Infini-attention on %d layers.", len(infini_layers))

    # --- Load checkpoint ---
    if not args.no_checkpoint and args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        if ckpt.exists():
            logger.info("Loading checkpoint from %s", ckpt)
            try:
                from peft import PeftModel  # type: ignore[import]

                model = PeftModel.from_pretrained(model, str(ckpt))
                model = model.merge_and_unload()
                logger.info("LoRA checkpoint loaded and merged.")
            except Exception as exc:
                logger.warning("Could not load PEFT checkpoint (%s); continuing.", exc)
        else:
            logger.warning("Checkpoint path %s does not exist — evaluating without checkpoint.", ckpt)

    model.eval()

    eval_model = InfiniEvalWrapper(
        model=model,
        tokenizer=tokenizer._tok,  # type: ignore[attr-defined]
        device=device,
    )

    # --- Build benchmark suite ---
    from src.eval.benchmarks.memory import AssociativeRecall, PasskeyRetrieval, VariableTracking
    from src.eval.benchmarks.computation import (
        DyckLanguage,
        ModularArithmetic,
        MultiDigitArithmetic,
    )
    from src.eval.benchmarks.emergent import (
        CompositionalGeneralization,
        LengthExtrapolation,
    )

    n = args.num_examples
    benchmarks = [
        # Memory (primary)
        PasskeyRetrieval(n=n, context_length=100, seed=args.seed),
        PasskeyRetrieval(n=n, context_length=500, seed=args.seed + 1),
        VariableTracking(n=n, num_variables=3, num_operations=5, seed=args.seed),
        VariableTracking(n=n, num_variables=5, num_operations=10, seed=args.seed + 1),
        AssociativeRecall(n=n, num_pairs=5, seed=args.seed),
        AssociativeRecall(n=n, num_pairs=10, seed=args.seed + 1),
        # Computation
        MultiDigitArithmetic(n=n, num_digits=3, seed=args.seed),
        ModularArithmetic(n=n, modulus=97, seed=args.seed),
        DyckLanguage(n=n, max_depth=5, seed=args.seed),
        # Generalisation
        CompositionalGeneralization(n=n, seed=args.seed),
        LengthExtrapolation(n=n, seed=args.seed),
    ]

    from src.eval.harness import EvalConfig, evaluate

    config = EvalConfig(
        batch_size=args.batch_size,
        num_few_shot=0,
        decode_mode="greedy",
        metrics=["exact_match"],
        output_file=args.output,
        model_name=f"infini_attention_{args.model_name}",
        seed=args.seed,
    )

    logger.info("Starting evaluation (%d benchmarks)", len(benchmarks))
    t0 = time.time()
    results = evaluate(eval_model, benchmarks, config)  # type: ignore[arg-type]
    elapsed = time.time() - t0

    logger.info("Evaluation complete in %.1f s.  %d results.", elapsed, len(results))
    for r in results:
        logger.info("  %-40s  %s = %.4f", r.task, r.metric, r.value)

    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main(parse_args())
