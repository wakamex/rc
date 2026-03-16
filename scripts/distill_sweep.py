#!/usr/bin/env python3
"""Phase 1: Distillation sweep — map DeltaNet layer reservoir-compatibility.

For each of 18 DeltaNet (linear_attn) layers in Qwen3.5-0.8B:
1. Capture layer input/output activations on a batch of text
2. Run the same input through an ESN reservoir (r=1000)
3. Fit a ridge regression: ESN states → DeltaNet output
4. Report reconstruction MSE per layer

Low MSE → ESN can replicate the layer → safe candidate for replacement.
High MSE → DeltaNet does something the reservoir can't → don't replace.

Usage:
    python scripts/distill_sweep.py
    python scripts/distill_sweep.py --n-batches 20 --reservoir-size 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.loader import load_model  # noqa: E402
from src.reservoir.esn import ESN  # noqa: E402
from src.types import ReservoirConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distillation sweep: ESN vs DeltaNet per layer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--n-batches", type=int, default=20,
                   help="Number of text batches to collect activations from")
    p.add_argument("--max-seq-length", type=int, default=512,
                   help="Max sequence length per batch")
    p.add_argument("--reservoir-size", type=int, default=1000)
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--ridge-alpha", type=float, default=1.0,
                   help="Ridge regression regularization")
    return p.parse_args()


def get_deltanet_layer_indices(model: torch.nn.Module) -> list[int]:
    """Find transformer layer indices that have linear_attn (DeltaNet)."""
    layers = model.model.layers
    indices = []
    for i, layer in enumerate(layers):
        children = {name for name, _ in layer.named_children()}
        if "linear_attn" in children:
            indices.append(i)
    return indices


def collect_activations(
    model: torch.nn.Module,
    tokenizer: object,
    layer_idx: int,
    device: torch.device,
    n_batches: int,
    max_seq_length: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect input and output activations for a specific layer.

    Returns (inputs_list, outputs_list) where each element is (T, H) numpy array.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=1000)

    layer = model.model.layers[layer_idx]
    inputs_list: list[np.ndarray] = []
    outputs_list: list[np.ndarray] = []

    captured_input = {}
    captured_output = {}

    def pre_hook(module, args):
        hidden = args[0] if isinstance(args, tuple) else args
        if isinstance(hidden, torch.Tensor):
            captured_input["val"] = hidden.detach()

    def post_hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured_output["val"] = out.detach()

    h_pre = layer.register_forward_pre_hook(pre_hook)
    h_post = layer.register_forward_hook(post_hook)

    model.eval()
    batch_count = 0
    with torch.no_grad():
        for example in ds:
            if batch_count >= n_batches:
                break

            text = example.get("text", "")
            if len(text) < 50:
                continue

            ids = tokenizer._tok(
                text, return_tensors="pt", truncation=True, max_length=max_seq_length
            )["input_ids"].to(device)

            if ids.shape[1] < 10:
                continue

            captured_input.clear()
            captured_output.clear()

            model(ids)

            if "val" in captured_input and "val" in captured_output:
                inp = captured_input["val"][0].float().cpu().numpy()  # (T, H)
                out = captured_output["val"][0].float().cpu().numpy()  # (T, H)
                inputs_list.append(inp)
                outputs_list.append(out)
                batch_count += 1

    h_pre.remove()
    h_post.remove()

    return inputs_list, outputs_list


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """Fit ridge regression X @ W = Y, return (W, MSE)."""
    # X: (N, D_in), Y: (N, D_out)
    # W = (X^T X + alpha I)^{-1} X^T Y
    XtX = X.T @ X
    XtX += alpha * np.eye(XtX.shape[0])
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    Y_pred = X @ W
    mse = np.mean((Y - Y_pred) ** 2)
    return W, mse


def main() -> None:
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)

    print(f"Loading Qwen3.5-0.8B-Base on {args.device} ({args.dtype})...")
    wrapper = load_model("qwen3.5-0.8b", dtype=dtype, device=str(device))
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    hidden_dim = model.config.hidden_size

    print(f"Building ESN: r={args.reservoir_size}, spectral_radius={args.spectral_radius}")
    reservoir_cfg = ReservoirConfig(
        size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        leak_rate=0.5,
        input_scaling=1.0,
        sparsity=0.01,
        seed=42,
    )
    esn = ESN(reservoir_cfg, input_dim=hidden_dim)

    dn_indices = get_deltanet_layer_indices(model)
    print(f"Found {len(dn_indices)} DeltaNet layers: {dn_indices}")

    results = []

    for dn_seq, layer_idx in enumerate(dn_indices):
        t0 = time.time()
        print(f"\n--- Layer {layer_idx} (DeltaNet #{dn_seq}) ---")

        # Collect activations
        print(f"  Collecting {args.n_batches} batches of activations...")
        inputs_list, outputs_list = collect_activations(
            model, tokenizer, layer_idx, device,
            n_batches=args.n_batches,
            max_seq_length=args.max_seq_length,
        )

        if not inputs_list:
            print("  WARNING: no activations collected, skipping")
            results.append({"layer_idx": layer_idx, "dn_seq": dn_seq, "mse": float("inf")})
            continue

        # Run ESN on the layer inputs to get reservoir states
        print("  Running ESN on layer inputs...")
        esn_states_list = []
        for inp in inputs_list:
            esn.reset()
            states = esn.forward(inp)  # (T, reservoir_size)
            esn_states_list.append(states)

        # Concatenate all tokens across batches for ridge regression
        all_esn = np.concatenate(esn_states_list, axis=0)     # (N_total, reservoir_size)
        all_outputs = np.concatenate(outputs_list, axis=0)     # (N_total, hidden_dim)
        all_inputs = np.concatenate(inputs_list, axis=0)       # (N_total, hidden_dim)

        # Compute baseline: how much does the layer change the input?
        # (If the layer is mostly identity/residual, even a bad ESN would score well)
        delta = all_outputs - all_inputs
        delta_norm = np.mean(delta ** 2)
        output_norm = np.mean(all_outputs ** 2)

        # Fit ridge regression: ESN states → layer output
        print(f"  Fitting ridge regression ({all_esn.shape[0]} tokens, "
              f"ESN:{all_esn.shape[1]} → output:{all_outputs.shape[1]})...")

        # Predict the full output
        _, mse_full = fit_ridge(all_esn, all_outputs, alpha=args.ridge_alpha)

        # Predict just the delta (output - input) — the layer's contribution
        _, mse_delta = fit_ridge(all_esn, delta, alpha=args.ridge_alpha)

        # Relative MSE (normalized by output variance)
        rel_mse = mse_full / (output_norm + 1e-10)

        elapsed = time.time() - t0

        print(f"  MSE (full output):  {mse_full:.6f}")
        print(f"  MSE (delta only):   {mse_delta:.6f}")
        print(f"  Output norm:        {output_norm:.6f}")
        print(f"  Delta norm:         {delta_norm:.6f}")
        print(f"  Relative MSE:       {rel_mse:.6f}")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            "layer_idx": layer_idx,
            "dn_seq": dn_seq,
            "mse_full": mse_full,
            "mse_delta": mse_delta,
            "output_norm": output_norm,
            "delta_norm": delta_norm,
            "rel_mse": rel_mse,
        })

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Layer':>6} {'DN#':>4} {'MSE(full)':>12} {'MSE(delta)':>12} "
          f"{'Rel MSE':>10} {'Delta/Out':>10} {'Verdict':>12}")
    print(f"{'-'*80}")

    # Sort by MSE for ranking
    ranked = sorted(results, key=lambda r: r.get("rel_mse", float("inf")))

    for r in sorted(results, key=lambda r: r["layer_idx"]):
        rank = next(i for i, x in enumerate(ranked) if x["layer_idx"] == r["layer_idx"])
        rel = r.get("rel_mse", float("inf"))
        delta_ratio = r.get("delta_norm", 0) / (r.get("output_norm", 1) + 1e-10)
        verdict = "EASY" if rank < 6 else ("MEDIUM" if rank < 12 else "HARD")
        print(f"{r['layer_idx']:>6} {r['dn_seq']:>4} {r.get('mse_full', 0):>12.6f} "
              f"{r.get('mse_delta', 0):>12.6f} {rel:>10.6f} {delta_ratio:>10.4f} {verdict:>12}")

    print(f"{'='*80}")

    # Top 3 easiest to replace
    print("\nTop 3 easiest layers to replace (lowest relative MSE):")
    for i, r in enumerate(ranked[:3]):
        print(f"  {i+1}. Layer {r['layer_idx']} (DeltaNet #{r['dn_seq']}): "
              f"rel_mse={r.get('rel_mse', 0):.6f}")

    print("\nTop 3 hardest layers to replace (highest relative MSE):")
    for i, r in enumerate(ranked[-3:]):
        print(f"  {i+1}. Layer {r['layer_idx']} (DeltaNet #{r['dn_seq']}): "
              f"rel_mse={r.get('rel_mse', 0):.6f}")


if __name__ == "__main__":
    main()
