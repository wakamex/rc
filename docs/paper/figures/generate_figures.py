#!/usr/bin/env python3
"""Generate all paper figures.

Produces PDF figures in docs/paper/figures/ for inclusion in main.tex.
Requires: matplotlib, numpy.  No GPU needed.

Usage::

    python docs/paper/figures/generate_figures.py

Outputs
-------
- architecture_sidecar.pdf      Track A sidecar architecture diagram
- architecture_ril.pdf          Track B RIL architecture diagram
- architecture_rw_transformer.pdf  Track C RW-Transformer block diagram
- pareto_frontier.pdf           Quality vs latency Pareto frontier (stub)
- ablation_results.pdf          Ablation bar chart (stub)
- scaling_curves.pdf            Quality vs context length / K sub-steps (stub)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parents[3] / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        **kw,
    )


def _box(ax, x, y, w, h, label, color="lightblue", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, wrap=True)


# ---------------------------------------------------------------------------
# Figure 1: Track A sidecar architecture
# ---------------------------------------------------------------------------

def fig_sidecar_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Track A: Bolt-On Reservoir Sidecar", fontsize=12, fontweight="bold")

    # Frozen LLM column
    _box(ax, 2, 4.5, 2.8, 0.7, "Frozen LLM Layers\n(Qwen3.5-0.8B)", color="#d4e6f1")
    _box(ax, 2, 3.2, 2.8, 0.7, "Hidden States  h_t", color="#d4e6f1")
    _box(ax, 2, 1.8, 2.8, 0.7, "CrossAttention Sidecar\n(K/V from reservoir)", color="#a9cce3")
    _box(ax, 2, 0.6, 2.8, 0.7, "LLM Output Logits", color="#d4e6f1")

    # ESN column
    _box(ax, 7.5, 4.5, 2.5, 0.7, "WriteHead\nh_t → w_t", color="#d5f5e3")
    _box(ax, 7.5, 3.2, 2.5, 0.7, "ESN Reservoir\nr_t = f(r_{t-1}, x_t, w_t)", color="#a9dfbf")
    _box(ax, 7.5, 1.8, 2.5, 0.7, "ReadProjection\nr_t → m_t", color="#d5f5e3")

    # Arrows LLM → ESN
    _arrow(ax, 3.4, 3.2, 6.25, 4.5)  # h_t → WriteHead
    # ESN internal
    _arrow(ax, 7.5, 4.15, 7.5, 3.55)  # WriteHead → ESN
    _arrow(ax, 7.5, 2.85, 7.5, 2.15)  # ESN → ReadProjection
    # ESN → LLM
    _arrow(ax, 6.25, 1.8, 3.4, 1.8)  # ReadProjection → CrossAttn

    # LLM internal
    _arrow(ax, 2, 4.15, 2, 3.55)  # LLM layers → hidden
    _arrow(ax, 2, 2.85, 2, 2.15)  # hidden → CrossAttn (for Q)
    _arrow(ax, 2, 1.45, 2, 0.95)  # CrossAttn → output

    # Labels
    ax.text(4.8, 3.6, "h_t", fontsize=8, color="#555")
    ax.text(6.5, 2.0, "m_t (K/V)", fontsize=8, color="#555")
    ax.text(8.2, 2.5, "r_t", fontsize=8, color="#555")

    # Legend
    frozen_patch = mpatches.Patch(color="#d4e6f1", label="Frozen (not trained)")
    trained_patch = mpatches.Patch(color="#d5f5e3", label="Trained interface")
    esn_patch = mpatches.Patch(color="#a9dfbf", label="Fixed ESN reservoir")
    ax.legend(handles=[frozen_patch, trained_patch, esn_patch],
              loc="lower right", fontsize=8)

    _save(fig, "architecture_sidecar.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Track B RIL architecture
# ---------------------------------------------------------------------------

def fig_ril_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Track B: Reservoir Interaction Layer (RIL)", fontsize=12, fontweight="bold")

    # LLM block
    _box(ax, 2, 6, 2.8, 0.7, "Transformer Layers\n(earlier)", color="#d4e6f1")
    _box(ax, 2, 4.6, 2.8, 0.7, "Hidden States  h_t", color="#d4e6f1")
    _box(ax, 2, 3.0, 2.8, 0.7, "FiLM Modulation\n(gated residual)", color="#a9cce3")
    _box(ax, 2, 1.4, 2.8, 0.7, "Transformer Layers\n(later)", color="#d4e6f1")
    _box(ax, 2, 0.3, 2.8, 0.5, "Output Logits", color="#d4e6f1")

    # Multi-reservoir
    _box(ax, 7.2, 5.5, 2.4, 0.6, "WriteHead  (h_t → w_t)", color="#d5f5e3")
    _box(ax, 6.2, 4.3, 1.8, 0.7, "Fast ESN\nα=0.9, ρ=0.9", color="#82e0aa")
    _box(ax, 8.2, 4.3, 1.8, 0.7, "Slow ESN\nα=0.1, ρ=0.5", color="#a9dfbf")
    _box(ax, 7.2, 3.0, 2.4, 0.7, "Concat [r_fast; r_slow]\n→ ReadProjection → m_t", color="#d5f5e3")

    # Arrows
    _arrow(ax, 2, 5.65, 2, 5.0)
    _arrow(ax, 2, 4.25, 2, 3.35)
    _arrow(ax, 2, 2.65, 2, 1.75)
    _arrow(ax, 2, 1.05, 2, 0.55)
    _arrow(ax, 3.4, 4.6, 6.0, 5.5)   # h_t → WriteHead
    _arrow(ax, 6.4, 5.2, 6.2, 4.65)  # WriteHead → Fast ESN
    _arrow(ax, 8.0, 5.2, 8.2, 4.65)  # WriteHead → Slow ESN
    _arrow(ax, 6.2, 3.95, 7.2, 3.35) # Fast ESN → concat
    _arrow(ax, 8.2, 3.95, 7.2, 3.35) # Slow ESN → concat
    _arrow(ax, 6.0, 3.0, 3.4, 3.0)   # m_t → FiLM

    ax.text(4.8, 4.8, "h_t", fontsize=8, color="#555")
    ax.text(5.5, 3.1, "m_t", fontsize=8, color="#555")

    frozen_patch = mpatches.Patch(color="#d4e6f1", label="Frozen LLM")
    trained_patch = mpatches.Patch(color="#d5f5e3", label="Trained interface")
    fast_patch = mpatches.Patch(color="#82e0aa", label="Fast reservoir")
    slow_patch = mpatches.Patch(color="#a9dfbf", label="Slow reservoir")
    ax.legend(handles=[frozen_patch, trained_patch, fast_patch, slow_patch],
              loc="lower right", fontsize=8)

    _save(fig, "architecture_ril.pdf")


# ---------------------------------------------------------------------------
# Figure 3: RW-Transformer block (Track C)
# ---------------------------------------------------------------------------

def fig_rw_transformer() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Track C: RW-Transformer Decoder Block", fontsize=12, fontweight="bold")

    # Input / output
    _box(ax, 5, 5.3, 3.0, 0.5, "Residual Input  h_t", color="#eaf2ff")
    _box(ax, 5, 0.5, 3.0, 0.5, "Gated Fusion → h_{t+1}", color="#eaf2ff")

    # Three branches
    _box(ax, 1.5, 3.0, 2.2, 0.7, "Attention Branch\n(local + global)", color="#d4e6f1")
    _box(ax, 5.0, 3.0, 2.2, 0.7, "MLP Branch\n(static knowledge)", color="#fdebd0")
    _box(ax, 8.5, 3.0, 2.2, 0.7, "Dual-Reservoir\nWorkspace", color="#d5f5e3")

    # Gating
    _box(ax, 5, 1.6, 6.5, 0.6, "Learned Gated Mixing  (α_attn · A + α_mlp · M + α_res · R)", color="#f5eef8")

    # Arrows input → branches
    for bx in [1.5, 5.0, 8.5]:
        _arrow(ax, 5, 5.05, bx, 3.35)

    # Arrows branches → gating
    for bx in [1.5, 5.0, 8.5]:
        _arrow(ax, bx, 2.65, 5, 1.9)

    # Gating → output
    _arrow(ax, 5, 1.3, 5, 0.75)

    # K sub-step annotation
    ax.annotate(
        "K sub-steps\n(latent reasoning)",
        xy=(8.5, 2.65),
        xytext=(8.5, 1.9),
        ha="center",
        fontsize=7.5,
        color="#1a5276",
        arrowprops=dict(arrowstyle="->", color="#1a5276", lw=0.8),
    )

    attn_p = mpatches.Patch(color="#d4e6f1", label="Attention")
    mlp_p = mpatches.Patch(color="#fdebd0", label="MLP")
    res_p = mpatches.Patch(color="#d5f5e3", label="Reservoir workspace")
    gate_p = mpatches.Patch(color="#f5eef8", label="Gated fusion")
    ax.legend(handles=[attn_p, mlp_p, res_p, gate_p],
              loc="lower left", fontsize=8)

    _save(fig, "architecture_rw_transformer.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Pareto frontier (stub — populated from sweep results)
# ---------------------------------------------------------------------------

def fig_pareto_frontier() -> None:
    pareto_path = RESULTS_DIR / "track_a" / "sweep" / "pareto_frontier.json"

    if pareto_path.exists():
        with open(pareto_path) as f:
            data = json.load(f)
        configs = data.get("configs", [])
        x = [c.get("latency_overhead_pct", 0) for c in configs]
        y = [c.get("passkey_accuracy", 0) for c in configs]
        labels = [c.get("name", "") for c in configs]
    else:
        # Illustrative stub
        rng = np.random.default_rng(42)
        x = rng.uniform(2, 35, 14).tolist()
        y = (0.3 + rng.uniform(0, 0.5, 14)).tolist()
        labels = [f"config-{i}" for i in range(14)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, zorder=3)
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(lbl, (xi, yi), fontsize=6, textcoords="offset points", xytext=(3, 3))

    # Gate A thresholds
    ax.axhline(0.10, color="red", linestyle="--", lw=1, label="Gate A min accuracy")
    ax.axvline(20, color="orange", linestyle="--", lw=1, label="Gate A max overhead")

    ax.set_xlabel("Inference latency overhead (%)", fontsize=11)
    ax.set_ylabel("Passkey retrieval accuracy", fontsize=11)
    ax.set_title("Track A: Quality vs Efficiency Pareto Frontier", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if not pareto_path.exists():
        ax.text(0.5, 0.5, "GPU RUNS PENDING\n(illustrative stub)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="gray", alpha=0.5)

    _save(fig, "pareto_frontier.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Ablation bar chart (stub)
# ---------------------------------------------------------------------------

def fig_ablation_results() -> None:
    ablations = [
        "Full R/W",
        "Read-only",
        "Random state",
        "Single (fast)",
        "Single (slow)",
        r"SR=0.5",
        r"SR=1.1",
    ]
    passkey_delta = [0.0, -0.08, -0.22, -0.05, -0.10, -0.07, -0.15]
    algo_delta = [0.0, -0.12, -0.28, -0.06, -0.08, -0.09, -0.19]

    x = np.arange(len(ablations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, passkey_delta, width, label="Passkey Δ", color="#2980b9", alpha=0.8)
    bars2 = ax.bar(x + width / 2, algo_delta, width, label="Algo Mem Δ", color="#e74c3c", alpha=0.8)

    ax.set_ylabel("Accuracy delta vs full model", fontsize=11)
    ax.set_title("Ablation Study Results", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ablations, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    ax.text(0.5, 0.92, "GPU RUNS PENDING\n(illustrative stub)",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color="gray", alpha=0.5)

    _save(fig, "ablation_results.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Scaling curves
# ---------------------------------------------------------------------------

def fig_scaling_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: quality vs context length
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    ax = axes[0]
    colors = {"Qwen+YaRN": "#2980b9", "Infini-attn": "#e67e22",
              "Track A R/W": "#27ae60", "Track A read-only": "#82e0aa"}
    for model, col in colors.items():
        # Stub: linear degradation for non-reservoir, graceful for reservoir
        if "Track A" in model:
            acc = [0.65 - 0.002 * np.log2(L / 1024) for L in context_lengths]
        else:
            acc = [0.65 - 0.07 * np.log2(L / 1024) for L in context_lengths]
        ax.plot(context_lengths, acc, "o-", label=model, color=col)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)", fontsize=10)
    ax.set_ylabel("Passkey accuracy", fontsize=10)
    ax.set_title("Quality vs Context Length", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.1, "STUB", ha="center", transform=ax.transAxes, color="gray", alpha=0.4, fontsize=16)

    # Right: quality vs K sub-steps (latent reasoning)
    Ks = [1, 2, 4, 8, 16]
    ax = axes[1]
    for task, col in [("Multi-hop QA", "#8e44ad"), ("ProsQA", "#c0392b")]:
        acc = [0.45 + 0.07 * np.log2(K) for K in Ks]
        ax.plot(Ks, acc, "s-", label=task, color=col)
    ax.set_xlabel("K reservoir sub-steps", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Latent Reasoning: Quality vs K Sub-steps", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.1, "STUB", ha="center", transform=ax.transAxes, color="gray", alpha=0.4, fontsize=16)

    fig.suptitle("Scaling Curves (GPU Runs Pending — Illustrative)", fontsize=11, color="gray")
    plt.tight_layout()
    _save(fig, "scaling_curves.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating figures in: {OUT_DIR}")
    fig_sidecar_architecture()
    fig_ril_architecture()
    fig_rw_transformer()
    fig_pareto_frontier()
    fig_ablation_results()
    fig_scaling_curves()
    print("Done.")


if __name__ == "__main__":
    main()
