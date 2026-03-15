#!/usr/bin/env python3
"""Evaluate Track B DeltaNet replacement model on the full benchmark suite.

Loads Qwen3.5-0.8B + LoRA adapter + ESN reservoir + replacement interface
weights from a training checkpoint and runs the same benchmarks as Track A
for direct comparison.

Usage:
    python scripts/eval_track_b.py
    python scripts/eval_track_b.py --checkpoint checkpoints/track_b/deltanet/final
    python scripts/eval_track_b.py --n-examples 200 --compare results/baselines/qwen35_vanilla.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.benchmarks.suite import build_benchmark_suite  # noqa: E402
from src.eval.harness import EvalConfig, evaluate  # noqa: E402
from src.models.loader import load_model  # noqa: E402
from src.types import EvalResult  # noqa: E402

# Re-use classes from the training script
from scripts.train_track_b_deltanet import (  # noqa: E402
    ESNReplacementInterface,
    DeltaNetReplacementManager,
    get_transformer_layers,
)


# ---------------------------------------------------------------------------
# Reservoir-aware eval adapter for Track B
# ---------------------------------------------------------------------------


class ReservoirEvalAdapterTrackB:
    """Eval adapter that computes ESN states and uses DeltaNet replacement hooks."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        esn: Any,
        replacement_manager: DeltaNetReplacementManager,
        embed_layer: nn.Module,
        max_new_tokens: int = 64,
        max_input_length: int = 1024,
    ) -> None:
        self._model = model
        self._tok = tokenizer
        self._device = device
        self._esn = esn
        self._replacement_manager = replacement_manager
        self._embed_layer = embed_layer
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self._latencies: list[float] = []

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        kwargs.pop("seed", None)
        t0 = time.perf_counter()

        # Tokenize
        if isinstance(prompt, str):
            input_ids = self._tok.encode(
                prompt, padding=False, truncation=True, max_length=self.max_input_length
            ).to(self._device)
        else:
            input_ids = prompt.to(self._device)

        prompt_len = input_ids.shape[-1]

        # Compute reservoir states from input embeddings
        if self._esn is not None and self._replacement_manager is not None:
            with torch.no_grad():
                embeddings = self._embed_layer(input_ids)  # (1, T, H)

            self._esn.reset()
            emb_np = embeddings[0].detach().float().cpu().numpy()  # (T, H)
            states = self._esn.forward(emb_np)  # (T, reservoir_dim)

            # Set reservoir states: (1, T, reservoir_dim)
            self._replacement_manager.set_reservoir_states(states[None])

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tok.eos_token_id,
                **kwargs,
            )

        if self._replacement_manager is not None:
            self._replacement_manager.clear_reservoir_states()

        elapsed = time.perf_counter() - t0
        self._latencies.append(elapsed)

        new_ids = output_ids[0, prompt_len:]
        return self._tok.decode(new_ids)

    def latency_stats(self) -> dict[str, float]:
        if not self._latencies:
            return {"p50_s": 0.0, "p95_s": 0.0}
        arr = sorted(self._latencies)
        n = len(arr)
        return {
            "p50_s": arr[int(n * 0.50)],
            "p95_s": arr[min(int(n * 0.95), n - 1)],
        }


# ---------------------------------------------------------------------------
# Perplexity (same texts as Track A / vanilla eval)
# ---------------------------------------------------------------------------

_SAMPLE_TEXTS = [
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


def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    replacement_manager: DeltaNetReplacementManager | None,
    esn: Any,
    embed_layer: nn.Module,
    device: torch.device,
) -> float:
    """Compute perplexity with reservoir replacement hooks active."""
    total_log_prob = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in _SAMPLE_TEXTS:
            ids = tokenizer.encode(text, padding=False, truncation=True, max_length=512)
            ids = ids.to(device)

            if esn is not None and replacement_manager is not None:
                embeddings = embed_layer(ids)
                esn.reset()
                emb_np = embeddings[0].detach().float().cpu().numpy()
                states = esn.forward(emb_np)  # (T, reservoir_dim)
                replacement_manager.set_reservoir_states(states[None])

            outputs = model(ids, labels=ids)

            if replacement_manager is not None:
                replacement_manager.clear_reservoir_states()

            n_tokens = ids.shape[-1] - 1
            if n_tokens <= 0:
                continue
            total_log_prob += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_log_prob / total_tokens)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison(
    results: list[EvalResult],
    baseline_path: str | None,
    perplexity: float,
    baseline_ppl: float | None = None,
) -> None:
    baseline_map: dict[str, float] = {}
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            baseline_map[f"{r['task']}::{r['metric']}"] = r["value"]

    print(f"\n{'='*85}")
    print(f"{'Task':<45} {'Metric':<14} {'Track B':>8} {'Vanilla':>8} {'Delta':>8}")
    print(f"{'-'*85}")
    for r in sorted(results, key=lambda x: (x.task, x.metric)):
        key = f"{r.task}::{r.metric}"
        base = baseline_map.get(key)
        delta = f"{r.value - base:+.3f}" if base is not None else "  n/a"
        base_str = f"{base:.3f}" if base is not None else "  n/a"
        print(f"{r.task:<45} {r.metric:<14} {r.value:8.3f} {base_str:>8} {delta:>8}")

    if baseline_ppl is not None:
        ppl_delta = perplexity - baseline_ppl
        print(f"\n  Perplexity: {perplexity:.2f} (vanilla: {baseline_ppl:.2f}, delta: {ppl_delta:+.2f})")
    print(f"{'='*85}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Track B DeltaNet replacement model")
    p.add_argument("--checkpoint", default="checkpoints/track_b/deltanet/final",
                    help="Path to checkpoint directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", default="results/track_b/eval_deltanet.json")
    p.add_argument("--compare", default="results/baselines/qwen35_vanilla.json",
                    help="Baseline results JSON for comparison")
    p.add_argument("--no-sidecar", action="store_true",
                    help="Disable replacement hooks (LoRA-only ablation)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    ckpt = Path(args.checkpoint)

    # --- Load base model ---
    print(f"Loading Qwen3.5-0.8B-Base on {args.device} ({args.dtype})...")
    wrapper = load_model("qwen3.5-0.8b", dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    hidden_dim = model.config.hidden_size

    # --- Load LoRA adapter ---
    lora_path = ckpt / "lora_adapter"
    print(f"Loading LoRA adapter from {lora_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()

    # --- Build ESN reservoir + replacement interfaces ---
    esn = None
    replacement_manager = None
    embed_layer = model.get_input_embeddings()

    if not args.no_sidecar:
        print("Building ESN reservoir + DeltaNet replacement hooks...")
        from src.reservoir.esn import ESN
        from src.types import ReservoirConfig

        # Load sidecar config
        config_path = ckpt / "sidecar_config.json"
        if not config_path.exists():
            print(f"ERROR: {config_path} not found. Cannot load Track B config.")
            sys.exit(1)

        with open(config_path) as f:
            sc_cfg = json.load(f)

        reservoir_size = sc_cfg["reservoir_size"]
        replace_every_nth = sc_cfg["replace_every_nth_deltanet"]
        num_dn = sc_cfg["num_deltanet_blocks"]
        selected_dn_indices = sc_cfg["selected_deltanet_blocks"]
        gate_init = sc_cfg["replacement_gate_init"]

        print(f"  reservoir_size={reservoir_size}, replacing DeltaNet blocks {selected_dn_indices}")

        reservoir_cfg = ReservoirConfig(
            size=reservoir_size,
            spectral_radius=sc_cfg.get("spectral_radius", 0.9),
            leak_rate=sc_cfg.get("leak_rate", 0.5),
            input_scaling=sc_cfg.get("input_scaling", 1.0),
            sparsity=sc_cfg.get("reservoir_sparsity", 0.01),
            seed=sc_cfg.get("reservoir_seed", 42),
        )
        esn_cpu = ESN(reservoir_cfg, input_dim=hidden_dim)
        esn = esn_cpu.to_gpu(args.device) if device.type == "cuda" else esn_cpu

        # Build replacement interfaces
        replacement_interfaces = nn.ModuleDict({
            str(idx): ESNReplacementInterface(
                hidden_dim=hidden_dim,
                reservoir_dim=reservoir_size,
                gate_init=gate_init,
            )
            for idx in selected_dn_indices
        })

        # Load trained weights
        weights_path = ckpt / "replacement_interface_weights.pt"
        print(f"Loading replacement interface weights from {weights_path}...")
        replacement_interfaces.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        replacement_interfaces = replacement_interfaces.to(device).to(dtype)
        replacement_interfaces.eval()

        # Register hooks on transformer layers
        transformer_layers = get_transformer_layers(model)
        replacement_manager = DeltaNetReplacementManager(
            model=model,
            replacement_interfaces=replacement_interfaces,
            replace_every_nth=replace_every_nth,
            num_deltanet_blocks=num_dn,
        )
        replacement_manager.register_hooks(transformer_layers)
    else:
        print("Replacement hooks DISABLED -- LoRA-only ablation")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- Perplexity ---
    print("Computing perplexity...")
    perplexity = compute_perplexity(model, tokenizer, replacement_manager, esn, embed_layer, device)
    print(f"  Perplexity: {perplexity:.2f}")

    # Load baseline perplexity for comparison
    baseline_ppl = None
    if args.compare and Path(args.compare).exists():
        with open(args.compare) as f:
            baseline_ppl = json.load(f).get("perplexity")

    # --- Benchmark suite ---
    model_label = "qwen3.5-0.8b+lora-only" if args.no_sidecar else "qwen3.5-0.8b+track-b-deltanet"
    adapter = ReservoirEvalAdapterTrackB(
        model=model,
        tokenizer=tokenizer,
        device=device,
        esn=esn,
        replacement_manager=replacement_manager,
        embed_layer=embed_layer,
        max_new_tokens=args.max_new_tokens,
    )

    benchmarks = build_benchmark_suite(n=args.n_examples)
    config = EvalConfig(
        batch_size=1,
        num_few_shot=0,
        decode_mode="greedy",
        metrics=["exact_match", "accuracy", "f1"],
        output_file=None,
        model_name=model_label,
    )

    print(f"Running {len(benchmarks)} benchmark tasks ({args.n_examples} examples each)...")
    eval_start = time.perf_counter()
    results = evaluate(adapter, benchmarks, config)
    eval_elapsed = time.perf_counter() - eval_start
    print(f"  Evaluation complete in {eval_elapsed / 60:.1f} min")

    # --- VRAM & latency ---
    vram = {}
    if device.type == "cuda":
        vram = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "peak_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
        print(f"  VRAM: {vram['allocated_mb']:.0f} MB allocated, {vram['peak_mb']:.0f} MB peak")

    latency = adapter.latency_stats()
    print(f"  Latency p50={latency['p50_s']*1000:.1f}ms, p95={latency['p95_s']*1000:.1f}ms")

    # --- Clean up hooks ---
    if replacement_manager is not None:
        replacement_manager.remove_hooks()

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_label,
        "checkpoint": str(ckpt),
        "timestamp": time.time(),
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "n_examples": args.n_examples,
            "max_new_tokens": args.max_new_tokens,
            "reservoir_size": 0 if args.no_sidecar else reservoir_size,
            "sidecar_enabled": not args.no_sidecar,
            "lora_rank": 16,
        },
        "perplexity": perplexity,
        "vram": vram,
        "latency": latency,
        "eval_elapsed_seconds": eval_elapsed,
        "results": [asdict(r) for r in results],
    }

    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # --- Comparison table ---
    print_comparison(results, args.compare, perplexity, baseline_ppl)

    # --- Summary ---
    em_results = [r for r in results if r.metric == "exact_match"]
    if em_results:
        avg_em = sum(r.value for r in em_results) / len(em_results)
        print(f"\nAverage exact-match: {avg_em:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
