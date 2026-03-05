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


# InfiniEvalWrapper moved to src.models.eval_adapter.TextEvalAdapter


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

    from src.models.eval_adapter import TextEvalAdapter
    from src.models.infini_attention import reset_infini_memory

    eval_model = TextEvalAdapter(
        model=model,
        tokenizer=tokenizer._tok,  # type: ignore[attr-defined]
        device=device,
        max_new_tokens=50,
        pre_generate_hook=reset_infini_memory,
    )

    # --- Build benchmark suite ---
    from src.eval.benchmarks.suite import build_benchmark_suite

    benchmarks = build_benchmark_suite(n=args.num_examples)

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
