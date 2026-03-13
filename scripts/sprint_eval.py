#!/usr/bin/env python3
"""Sprint eval: fast metric for the autoresearch loop.

Runs the 3 reservoir-sensitive benchmarks (ModularArithmetic, LengthExtrapolation,
DyckLanguage) at n=50 + perplexity on 5 standard texts. Prints a summary with the
composite score defined in LOOP.md.

Usage:
    python scripts/sprint_eval.py checkpoints/sprint/final
    python scripts/sprint_eval.py checkpoints/sprint/final --no-sidecar
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.benchmarks.computation import DyckLanguage, ModularArithmetic
from src.eval.benchmarks.emergent import LengthExtrapolation
from src.eval.harness import _normalize, exact_match
from src.models.loader import load_model
from src.reservoir.esn import ESN
from src.types import ReservoirConfig

from scripts.train_track_a_readonly import ReadOnlySidecarBundle, SidecarHookManager

# --- Constants ---
VANILLA_PPL = 6.82
LORA_ONLY_AVG = 0.016
PPL_WEIGHT = 2.0
N_EXAMPLES = 50

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


def compute_perplexity(model, tokenizer, hook_manager, esn, embed_layer, device):
    total_log_prob = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in _SAMPLE_TEXTS:
            ids = tokenizer.encode(text, padding=False, truncation=True, max_length=512)
            ids = ids.to(device)
            if esn is not None and hook_manager is not None:
                embeddings = embed_layer(ids)
                esn.reset()
                emb_np = embeddings[0].detach().float().cpu().numpy()
                states = esn.forward(emb_np)
                hook_manager.set_reservoir_states(states[None])
            outputs = model(ids, labels=ids)
            if hook_manager is not None:
                hook_manager.clear_reservoir_states()
            n_tokens = ids.shape[-1] - 1
            if n_tokens <= 0:
                continue
            total_log_prob += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_log_prob / total_tokens)


def run_benchmark(name, benchmark, model, tokenizer, device, esn, hook_manager, embed_layer):
    hits = 0
    total = 0
    for i, ex in enumerate(benchmark):
        if i >= N_EXAMPLES:
            break
        ids = tokenizer.encode(ex.input, padding=False, truncation=True, max_length=1024)
        ids = ids.to(device)
        with torch.no_grad():
            embeddings = embed_layer(ids)
        if esn is not None and hook_manager is not None:
            esn.reset()
            emb_np = embeddings[0].detach().float().cpu().numpy()
            states = esn.forward(emb_np)
            hook_manager.set_reservoir_states(states[None])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        if hook_manager is not None:
            hook_manager.clear_reservoir_states()
        gen = tokenizer.decode(out[0, ids.shape[-1]:])
        hit = exact_match(gen, ex.target)
        hits += hit
        total += 1
    score = hits / total if total > 0 else 0.0
    return score


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sprint/final"
    no_sidecar = "--no-sidecar" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    t0 = time.perf_counter()

    # Load model + LoRA
    wrapper = load_model("qwen3.5-0.8b", dtype=dtype, device=str(device))
    tokenizer = wrapper.tokenizer
    hidden_dim = wrapper.model.config.hidden_size

    lora_path = f"{ckpt}/lora_adapter"
    if Path(f"{lora_path}/adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(wrapper.model, lora_path)
    else:
        model = wrapper.model
    model.eval()

    # ESN + sidecar
    esn = None
    hook_manager = None
    embed_layer = model.get_input_embeddings()

    if not no_sidecar:
        # Load sidecar config if saved, otherwise use defaults
        import json
        config_path = Path(ckpt) / "sidecar_config.json"
        if config_path.exists():
            with open(config_path) as f:
                sc_cfg = json.load(f)
            reservoir_size = sc_cfg["reservoir_size"]
            sidecar_layers = sc_cfg["sidecar_layers"]
            num_heads = sc_cfg.get("num_heads", 8)
            gate_init = sc_cfg.get("gate_init", 0.0)
            sidecar_type = sc_cfg.get("sidecar_type", "cross_attention")
        else:
            reservoir_size = 10000
            num_layers = model.config.num_hidden_layers
            sidecar_layers = list(range(3, num_layers, 4))
            num_heads = 8
            gate_init = 0.05
            sidecar_type = "cross_attention"

        reservoir_cfg = ReservoirConfig(
            size=reservoir_size, spectral_radius=0.9, leak_rate=0.5,
            input_scaling=1.0, sparsity=0.01, seed=42,
        )
        esn_cpu = ESN(reservoir_cfg, input_dim=hidden_dim)
        esn = esn_cpu.to_gpu(str(device)) if device.type == "cuda" else esn_cpu

        sidecar_weights_path = Path(ckpt) / "sidecar_weights.pt"
        if sidecar_weights_path.exists():
            sidecar_bundle = ReadOnlySidecarBundle(
                layer_indices=sidecar_layers,
                reservoir_dim=reservoir_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.0,
                gate_init=gate_init,
                sidecar_type=sidecar_type,
            )
            # strict=False: older checkpoints lack gate_alpha (added later)
            sidecar_bundle.load_state_dict(torch.load(sidecar_weights_path, map_location=device), strict=False)
            sidecar_bundle = sidecar_bundle.to(device).to(dtype).eval()
            hook_manager = SidecarHookManager(model, sidecar_bundle, sidecar_layers)
        else:
            print(f"WARNING: {sidecar_weights_path} not found, running without sidecar")

    # Perplexity
    ppl = compute_perplexity(model, tokenizer, hook_manager, esn, embed_layer, device)

    # Benchmarks
    modarith = run_benchmark(
        "ModArith",
        ModularArithmetic(n=N_EXAMPLES, modulus=97, seed=42),
        model, tokenizer, device, esn, hook_manager, embed_layer,
    )
    lengthext = run_benchmark(
        "LengthExt",
        LengthExtrapolation(n=N_EXAMPLES, train_length=5, test_multiplier=2.0, seed=43),
        model, tokenizer, device, esn, hook_manager, embed_layer,
    )
    dyck = run_benchmark(
        "Dyck",
        DyckLanguage(n=N_EXAMPLES, max_depth=4, bracket_types=2, seed=43),
        model, tokenizer, device, esn, hook_manager, embed_layer,
    )

    # Cleanup
    if hook_manager is not None:
        hook_manager.remove_hooks()

    # Compute score
    raw_avg = (modarith + lengthext + dyck) / 3.0
    avg_task = raw_avg - LORA_ONLY_AVG
    score = avg_task - PPL_WEIGHT * (ppl - VANILLA_PPL)

    elapsed = time.perf_counter() - t0

    print("---")
    print(f"avg_task:   {avg_task:.4f}")
    print(f"ppl:        {ppl:.2f}")
    print(f"score:      {score:.4f}")
    print(f"modarith:   {modarith:.3f}")
    print(f"lengthext:  {lengthext:.3f}")
    print(f"dyck:       {dyck:.3f}")
    print(f"elapsed_s:  {elapsed:.0f}")


if __name__ == "__main__":
    main()
