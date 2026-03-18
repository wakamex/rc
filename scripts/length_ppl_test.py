#!/usr/bin/env python3
"""Length-dependent perplexity test for the forgetting hypothesis.

Measures perplexity at different context positions (256, 512, 1024, 2048)
to test whether the ESN forgetting controller's benefit increases with
sequence length (stale DeltaNet associations accumulate over time).

Also runs a random-vector ablation: replaces the ESN state with a fixed
random vector to test whether reservoir dynamics matter or if the gate
is just a learned element-wise scaling.

Usage:
    python scripts/length_ppl_test.py --checkpoint checkpoints/track_b/deltanet/final
    python scripts/length_ppl_test.py --checkpoint checkpoints/track_b/best
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.loader import load_model  # noqa: E402
from src.reservoir.esn import ESN  # noqa: E402
from src.types import ReservoirConfig  # noqa: E402

from scripts.train_track_b_deltanet import (  # noqa: E402
    ESNForgettingController,
    ESNReplacementInterface,
    DeltaNetReplacementManager,
    get_transformer_layers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Length-dependent perplexity test")
    p.add_argument("--checkpoint", default="checkpoints/track_b/deltanet/final")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--n-sequences", type=int, default=20,
                   help="Number of long sequences to evaluate")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--positions", type=int, nargs="+", default=[256, 512, 1024, 2048],
                   help="Context positions at which to measure ppl")
    return p.parse_args()


def load_long_sequences(tokenizer, n: int, max_len: int) -> list[torch.Tensor]:
    """Load n sequences of length max_len from FineWeb."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=99, buffer_size=5000)

    sequences = []
    for example in ds:
        if len(sequences) >= n:
            break
        text = example.get("text", "")
        ids = tokenizer._tok(
            text, return_tensors="pt", truncation=True, max_length=max_len
        )["input_ids"].squeeze(0)
        # Only keep sequences that are at least max_len tokens
        if ids.shape[0] >= max_len:
            sequences.append(ids[:max_len])

    print(f"  Loaded {len(sequences)} sequences of length {max_len}")
    return sequences


def compute_ppl_at_positions(
    model: nn.Module,
    sequences: list[torch.Tensor],
    positions: list[int],
    device: torch.device,
    esn: Any | None,
    replacement_manager: DeltaNetReplacementManager | None,
    embed_layer: nn.Module | None,
    use_random_state: bool = False,
    reservoir_size: int = 0,
) -> dict[int, float]:
    """Compute per-position perplexity.

    For each position p, compute the average cross-entropy loss over tokens
    in the window [p-window, p] to get the local perplexity at that depth
    into the sequence.
    """
    model.eval()
    window = 128  # average over this many tokens around each position

    ppl_at_pos: dict[int, list[float]] = {p: [] for p in positions}

    with torch.no_grad():
        for seq_idx, ids in enumerate(sequences):
            ids = ids.unsqueeze(0).to(device)  # (1, T)

            # Compute reservoir states for the full sequence
            if esn is not None and replacement_manager is not None and embed_layer is not None:
                embeddings = embed_layer(ids)
                esn.reset()
                emb_np = embeddings[0].detach().float().cpu().numpy()

                if use_random_state:
                    # Ablation: fixed random vector instead of ESN dynamics
                    rng = np.random.RandomState(42)
                    fixed_state = rng.randn(reservoir_size).astype(np.float32) * 0.3
                    states = np.tile(fixed_state, (emb_np.shape[0], 1))  # (T, R)
                else:
                    states = esn.forward(emb_np)  # (T, R)

                replacement_manager.set_reservoir_states(states[None])

            # Forward pass with labels to get per-token losses
            outputs = model(ids, labels=ids)
            # outputs.loss is the mean over all tokens — we need per-token
            # Re-compute with logits
            logits = outputs.logits  # (1, T, V)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )  # (T-1,)

            if replacement_manager is not None:
                replacement_manager.clear_reservoir_states()

            per_token_loss = per_token_loss.float().cpu().numpy()

            # Extract loss at each position
            for pos in positions:
                if pos > len(per_token_loss):
                    continue
                start = max(0, pos - window)
                end = min(pos, len(per_token_loss))
                if end <= start:
                    continue
                local_loss = per_token_loss[start:end].mean()
                ppl_at_pos[pos].append(float(local_loss))

    # Average and convert to perplexity
    result = {}
    for pos in positions:
        if ppl_at_pos[pos]:
            avg_loss = sum(ppl_at_pos[pos]) / len(ppl_at_pos[pos])
            result[pos] = math.exp(min(avg_loss, 20.0))
    return result


def setup_model_and_hooks(args, ckpt, device, dtype):
    """Load model, LoRA, ESN, and hooks. Returns (model, tokenizer, esn, manager, embed, cfg)."""
    wrapper = load_model("qwen3.5-0.8b", dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    hidden_dim = model.config.hidden_size

    # Load LoRA
    from peft import PeftModel
    lora_path = ckpt / "lora_adapter"
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()

    # Load config
    config_path = ckpt / "sidecar_config.json"
    with open(config_path) as f:
        sc_cfg = json.load(f)

    reservoir_size = sc_cfg["reservoir_size"]
    selected_dn_indices = sc_cfg["selected_deltanet_blocks"]
    gate_init = sc_cfg["replacement_gate_init"]
    mode = sc_cfg.get("integration", "replacement")

    # Build ESN
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

    # Build interfaces
    InterfaceClass = ESNForgettingController if mode == "controller" else ESNReplacementInterface
    iface_gate = 0.9 if mode == "controller" else gate_init
    replacement_interfaces = nn.ModuleDict({
        str(idx): InterfaceClass(
            hidden_dim=hidden_dim, reservoir_dim=reservoir_size, gate_init=iface_gate,
        )
        for idx in selected_dn_indices
    })

    weights_path = ckpt / "replacement_interface_weights.pt"
    replacement_interfaces.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    replacement_interfaces = replacement_interfaces.to(device).to(dtype)
    replacement_interfaces.eval()

    transformer_layers = get_transformer_layers(model)
    replace_every_nth = sc_cfg.get("replace_every_nth_deltanet", 3)
    num_dn = sc_cfg["num_deltanet_blocks"]
    manager = DeltaNetReplacementManager(
        model=model,
        replacement_interfaces=replacement_interfaces,
        replace_every_nth=replace_every_nth,
        num_deltanet_blocks=num_dn,
        selected_indices=selected_dn_indices,
    )
    manager.register_hooks(transformer_layers)

    embed_layer = model.get_input_embeddings()

    return model, tokenizer, esn, manager, embed_layer, sc_cfg


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    ckpt = Path(args.checkpoint)

    print(f"Loading model + checkpoint from {ckpt}...")
    model, tokenizer, esn, manager, embed_layer, sc_cfg = setup_model_and_hooks(
        args, ckpt, device, dtype
    )
    reservoir_size = sc_cfg["reservoir_size"]
    mode = sc_cfg.get("integration", "replacement")

    print(f"Mode: {mode}")
    print(f"Loading {args.n_sequences} long sequences (len={args.max_seq_length})...")
    sequences = load_long_sequences(tokenizer, args.n_sequences, args.max_seq_length)

    if not sequences:
        print("ERROR: no sequences long enough. Try a lower --max-seq-length.")
        return

    # --- Test 1: Controller ppl at different context lengths ---
    print(f"\n{'='*60}")
    print(f"Test 1: Length-dependent perplexity (with ESN controller)")
    print(f"{'='*60}")
    ppl_controller = compute_ppl_at_positions(
        model, sequences, args.positions, device,
        esn, manager, embed_layer,
    )
    for pos, ppl in sorted(ppl_controller.items()):
        print(f"  Position {pos:>5}: ppl = {ppl:.4f}")

    # --- Test 2: Vanilla ppl (disable hooks) ---
    print(f"\n{'='*60}")
    print(f"Test 2: Length-dependent perplexity (vanilla — no ESN)")
    print(f"{'='*60}")
    ppl_vanilla = compute_ppl_at_positions(
        model, sequences, args.positions, device,
        None, None, None,  # no ESN
    )
    for pos, ppl in sorted(ppl_vanilla.items()):
        print(f"  Position {pos:>5}: ppl = {ppl:.4f}")

    # --- Test 3: Random vector ablation ---
    print(f"\n{'='*60}")
    print(f"Test 3: Random vector ablation (fixed z instead of ESN state)")
    print(f"{'='*60}")
    ppl_random = compute_ppl_at_positions(
        model, sequences, args.positions, device,
        esn, manager, embed_layer,
        use_random_state=True, reservoir_size=reservoir_size,
    )
    for pos, ppl in sorted(ppl_random.items()):
        print(f"  Position {pos:>5}: ppl = {ppl:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Summary: ppl ratio (controller/vanilla) by position")
    print(f"{'='*60}")
    print(f"{'Position':>8} {'Vanilla':>10} {'Controller':>12} {'Random':>10} "
          f"{'Ctrl/Van':>10} {'Rand/Van':>10}")
    print(f"{'-'*60}")
    for pos in sorted(set(ppl_controller) & set(ppl_vanilla)):
        van = ppl_vanilla[pos]
        ctrl = ppl_controller[pos]
        rand = ppl_random.get(pos, float('nan'))
        ratio_ctrl = ctrl / van
        ratio_rand = rand / van if not math.isnan(rand) else float('nan')
        print(f"{pos:>8} {van:>10.4f} {ctrl:>12.4f} {rand:>10.4f} "
              f"{ratio_ctrl:>10.4f} {ratio_rand:>10.4f}")

    # Interpretation
    print(f"\n{'='*60}")
    print("Interpretation:")
    ratios = []
    for pos in sorted(set(ppl_controller) & set(ppl_vanilla)):
        ratios.append((pos, ppl_controller[pos] / ppl_vanilla[pos]))
    if len(ratios) >= 2:
        first_ratio = ratios[0][1]
        last_ratio = ratios[-1][1]
        if last_ratio < first_ratio - 0.005:
            print("  SUPPORTS forgetting hypothesis: controller benefit INCREASES with length")
            print(f"  Ratio at {ratios[0][0]}: {first_ratio:.4f} → at {ratios[-1][0]}: {last_ratio:.4f}")
        elif abs(last_ratio - first_ratio) < 0.005:
            print("  NEUTRAL: controller benefit is UNIFORM across lengths")
            print("  May be acting as regularizer / learned gate bias, not temporal forgetting")
        else:
            print("  AGAINST forgetting hypothesis: controller benefit DECREASES with length")

    # Check random ablation
    if ppl_random:
        avg_ctrl = sum(ppl_controller[p] for p in ppl_controller) / len(ppl_controller)
        avg_rand = sum(ppl_random[p] for p in ppl_random) / len(ppl_random)
        avg_van = sum(ppl_vanilla[p] for p in ppl_vanilla) / len(ppl_vanilla)
        if abs(avg_rand - avg_ctrl) < 0.1:
            print("  Random ablation MATCHES controller — reservoir dynamics may not matter")
        elif avg_rand > avg_ctrl:
            print(f"  Random ablation WORSE than controller ({avg_rand:.2f} vs {avg_ctrl:.2f})")
            print("  → ESN temporal dynamics provide the useful forget signal (strong claim)")
        else:
            print(f"  Random ablation BETTER than controller ({avg_rand:.2f} vs {avg_ctrl:.2f})")
            print("  → Unexpected — investigate")


if __name__ == "__main__":
    main()
