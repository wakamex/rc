#!/usr/bin/env python3
"""Fast controller sweep: test forgetting controller at each DeltaNet layer.

Runs a short training (1500 steps) + ppl-only eval for each of 18 DeltaNet
layers, at multiple reservoir sizes. Finds the optimal layer for the
forgetting controller (which may differ from the replacement sweep).

Usage:
    python scripts/controller_sweep.py
    python scripts/controller_sweep.py --steps 1500 --reservoir-sizes 100 256 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controller sweep across layers and reservoir sizes")
    p.add_argument("--steps", type=int, default=1500, help="Training steps per config")
    p.add_argument("--reservoir-sizes", type=int, nargs="+", default=[256, 1000])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    return p.parse_args()


def train_and_eval_ppl(
    layer_idx: int,
    reservoir_size: int,
    steps: int,
) -> dict:
    """Train controller at one layer, measure ppl. Returns result dict."""
    output_dir = f"checkpoints/track_b/sweep/dn{layer_idx}_r{reservoir_size}"

    # Train
    train_cmd = [
        "python", "scripts/train_track_b_deltanet.py",
        "--no_wandb",
        "--max_steps", str(steps),
        "--batch_size", "1",
        "--grad_accum", "16",
        "--memory_task_ratio", "0.3",
        "--warmup_steps", str(max(1, steps // 100)),
        "--mode", "controller",
        "--replace_layers", str(layer_idx),
        "--reservoir_size", str(reservoir_size),
        "--max_seq_length", "1024",
        "--output_dir", output_dir,
        "--save_interval", "99999",
        "--log_interval", str(max(1, steps // 5)),
    ]

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    try:
        result = subprocess.run(
            train_cmd, capture_output=True, text=True, env=env, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return {"layer_idx": layer_idx, "reservoir_size": reservoir_size,
                "ppl": float("inf"), "status": "timeout", "train_time": time.time() - t0}
    train_time = time.time() - t0

    if result.returncode != 0:
        # Check if it's OOM
        if "OutOfMemoryError" in (result.stderr or ""):
            return {"layer_idx": layer_idx, "reservoir_size": reservoir_size,
                    "ppl": float("inf"), "status": "OOM", "train_time": train_time}
        return {"layer_idx": layer_idx, "reservoir_size": reservoir_size,
                "ppl": float("inf"), "status": f"crash:{result.returncode}",
                "train_time": train_time, "error": (result.stderr or "")[-500:]}

    # Check hooks fired
    hooks_line = [l for l in result.stderr.split("\n") if "hooks=" in l]
    if hooks_line:
        last_hooks = hooks_line[-1]
        hooks_count = int(last_hooks.split("hooks=")[1].split()[0])
    else:
        hooks_count = 0

    if hooks_count == 0:
        return {"layer_idx": layer_idx, "reservoir_size": reservoir_size,
                "ppl": float("inf"), "status": "no_hooks", "train_time": train_time}

    # Quick ppl eval — just compute perplexity, skip benchmarks
    ppl = compute_ppl_only(output_dir + "/final")
    return {
        "layer_idx": layer_idx,
        "reservoir_size": reservoir_size,
        "ppl": ppl,
        "hooks": hooks_count,
        "status": "ok",
        "train_time": train_time,
    }


def compute_ppl_only(checkpoint_dir: str) -> float:
    """Load checkpoint and compute perplexity only (no benchmark eval)."""
    from src.models.loader import load_model
    from src.reservoir.esn import ESN
    from src.types import ReservoirConfig
    from scripts.train_track_b_deltanet import (
        ESNForgettingController,
        DeltaNetReplacementManager,
        get_transformer_layers,
    )

    ckpt = Path(checkpoint_dir)
    config_path = ckpt / "sidecar_config.json"
    if not config_path.exists():
        return float("inf")

    with open(config_path) as f:
        sc_cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    wrapper = load_model("qwen3.5-0.8b", dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    hidden_dim = model.config.hidden_size

    # Load LoRA
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ckpt / "lora_adapter"))
    model.eval()

    # Build ESN
    reservoir_size = sc_cfg["reservoir_size"]
    selected_dn_indices = sc_cfg["selected_deltanet_blocks"]

    reservoir_cfg = ReservoirConfig(
        size=reservoir_size,
        spectral_radius=sc_cfg.get("spectral_radius", 0.9),
        leak_rate=sc_cfg.get("leak_rate", 0.5),
        input_scaling=sc_cfg.get("input_scaling", 1.0),
        sparsity=sc_cfg.get("reservoir_sparsity", 0.01),
        seed=sc_cfg.get("reservoir_seed", 42),
    )
    esn = ESN(reservoir_cfg, input_dim=hidden_dim)
    if device.type == "cuda":
        esn = esn.to_gpu(str(device))

    # Build controller interfaces
    replacement_interfaces = nn.ModuleDict({
        str(idx): ESNForgettingController(
            hidden_dim=hidden_dim, reservoir_dim=reservoir_size, gate_init=0.9,
        )
        for idx in selected_dn_indices
    })
    weights_path = ckpt / "replacement_interface_weights.pt"
    replacement_interfaces.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    replacement_interfaces = replacement_interfaces.to(device).to(dtype)
    replacement_interfaces.eval()

    # Register hooks
    transformer_layers = get_transformer_layers(model)
    manager = DeltaNetReplacementManager(
        model=model,
        replacement_interfaces=replacement_interfaces,
        replace_every_nth=3,
        num_deltanet_blocks=sc_cfg["num_deltanet_blocks"],
        selected_indices=selected_dn_indices,
    )
    manager.register_hooks(transformer_layers)

    embed_layer = model.get_input_embeddings()

    # Compute perplexity on standard texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, the Riemann hypothesis is a conjecture that the Riemann "
        "zeta function has its zeros only at the negative even integers and complex "
        "numbers with real part equal to one half.",
        "Machine learning is a method of data analysis that automates analytical "
        "model building. It is based on the idea that systems can learn from data, "
        "identify patterns and make decisions with minimal human intervention.",
        "The universe is all of space and time and their contents, including planets, "
        "stars, galaxies, and all other forms of matter and energy.",
        "Python is a high-level, general-purpose programming language. Its design "
        "philosophy emphasizes code readability, using significant indentation.",
    ]

    total_log_prob = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in sample_texts:
            ids = tokenizer.encode(text, padding=False, truncation=True, max_length=512)
            ids = ids.to(device)

            embeddings = embed_layer(ids)
            esn.reset()
            emb_np = embeddings[0].detach().float().cpu().numpy()
            states = esn.forward(emb_np)
            manager.set_reservoir_states(states[None])

            outputs = model(ids, labels=ids)
            manager.clear_reservoir_states()

            n_tokens = ids.shape[-1] - 1
            if n_tokens <= 0:
                continue
            total_log_prob += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    manager.remove_hooks()

    # Clean up GPU memory
    del model, replacement_interfaces, esn, manager
    torch.cuda.empty_cache()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_log_prob / total_tokens)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # All 18 DeltaNet layer indices (in DeltaNet sequence numbering)
    all_dn_indices = list(range(18))

    results = []

    for r_size in args.reservoir_sizes:
        logger.info("=== Reservoir size: %d ===", r_size)
        for dn_idx in all_dn_indices:
            logger.info("  Layer DN#%d (r=%d)...", dn_idx, r_size)
            result = train_and_eval_ppl(dn_idx, r_size, args.steps)
            results.append(result)
            ppl_str = f"{result['ppl']:.2f}" if result['ppl'] < 100 else result.get('status', '?')
            logger.info("    → ppl=%s (%s, %.0fs)",
                        ppl_str, result['status'], result['train_time'])

    # Summary
    print(f"\n{'='*70}")
    print(f"{'DN#':>4} {'Layer':>6} {'r':>6} {'ppl':>8} {'Δppl':>8} {'Status':>8}")
    print(f"{'-'*70}")

    # DeltaNet sequence index → transformer layer index mapping
    dn_to_layer = [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22]
    vanilla_ppl = 6.82

    for r in sorted(results, key=lambda x: (x['reservoir_size'], x['layer_idx'])):
        layer = dn_to_layer[r['layer_idx']] if r['layer_idx'] < len(dn_to_layer) else '?'
        ppl = r['ppl']
        delta = ppl - vanilla_ppl if ppl < 100 else float('inf')
        ppl_str = f"{ppl:.4f}" if ppl < 100 else "inf"
        delta_str = f"{delta:+.4f}" if ppl < 100 else "inf"
        print(f"{r['layer_idx']:>4} {layer:>6} {r['reservoir_size']:>6} {ppl_str:>8} {delta_str:>8} {r['status']:>8}")

    # Best results
    valid = [r for r in results if r['ppl'] < 100]
    if valid:
        best = min(valid, key=lambda r: r['ppl'])
        print(f"\nBest: DN#{best['layer_idx']} r={best['reservoir_size']} → ppl={best['ppl']:.4f}")

        # Top 5
        print("\nTop 5:")
        for i, r in enumerate(sorted(valid, key=lambda x: x['ppl'])[:5]):
            layer = dn_to_layer[r['layer_idx']] if r['layer_idx'] < len(dn_to_layer) else '?'
            print(f"  {i+1}. DN#{r['layer_idx']} (layer {layer}) r={r['reservoir_size']} "
                  f"→ ppl={r['ppl']:.4f} ({r['ppl']-vanilla_ppl:+.4f})")

    # Save results
    out_path = Path("results/track_b/controller_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
