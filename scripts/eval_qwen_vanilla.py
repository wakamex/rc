"""Evaluate unmodified Qwen3.5-0.8B-Base on the full benchmark suite.

PRIMARY reference point for all subsequent comparisons.

Records:
- Perplexity on held-out text
- Accuracy on all synthetic tasks (memory, computation, emergent from T5)
- Chaotic prediction quality (trajectory prediction vs Lyapunov time from T6)
- Throughput (tokens/sec)
- VRAM usage
- Inference latency (p50, p95)

Output: results/baselines/qwen35_vanilla.json (standardized EvalResult format)
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

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.chaos import generate_trajectory, lyapunov_time, split_trajectory  # noqa: E402
from src.eval.benchmarks.suite import build_benchmark_suite  # noqa: E402
from src.eval.harness import EvalConfig, evaluate  # noqa: E402
from src.models.eval_adapter import TextEvalAdapter  # noqa: E402
from src.models.loader import load_model  # noqa: E402
from src.types import EvalResult  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = str(_REPO_ROOT / "results" / "baselines" / "qwen35_vanilla.json")

# QwenEvalAdapter and build_benchmarks moved to shared modules:
# - src.models.eval_adapter.TextEvalAdapter
# - src.eval.benchmarks.suite.build_benchmark_suite


# ---------------------------------------------------------------------------
# Perplexity on held-out text
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


def compute_text_perplexity(wrapper: Any, texts: list[str], device: torch.device) -> float:
    """Compute average perplexity over a list of texts.

    Uses teacher-forced log-probabilities from the model forward pass.
    """
    tok = wrapper.tokenizer
    total_log_prob = 0.0
    total_tokens = 0

    wrapper.model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tok.encode(text, padding=False, truncation=True, max_length=512)
            ids = ids.to(device)  # (1, T)

            outputs = wrapper.model(ids, labels=ids)
            # outputs.loss is the mean cross-entropy over non-padded tokens
            n_tokens = ids.shape[-1] - 1  # teacher-forcing predicts T-1 tokens
            if n_tokens <= 0:
                continue
            total_log_prob += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    avg_ce = total_log_prob / total_tokens
    return math.exp(avg_ce)


# ---------------------------------------------------------------------------
# Chaos prediction evaluation (T6)
# ---------------------------------------------------------------------------

# Lyapunov times (in simulation time units) for standard parameter sets:
_LYAPUNOV_TIMES = {
    "lorenz63": 1 / 0.905,  # ≈ 1.1 Lyapunov times (λ_max ≈ 0.905)
    "mackey_glass": 1 / 0.007,  # ≈ 143 (τ=17, slow chaos)
}


def _trajectory_to_text(traj: np.ndarray, precision: int = 3) -> str:
    """Serialise a short trajectory segment as a space-separated string."""
    flat = traj.flatten()
    return " ".join(f"{v:.{precision}f}" for v in flat)


def _parse_trajectory_text(text: str, expected_len: int) -> np.ndarray | None:
    """Parse space-separated floats back to array. Returns None on failure."""
    try:
        values = [float(v) for v in text.strip().split()]
        if len(values) < expected_len:
            # Pad with NaN so downstream can detect partial output
            values += [float("nan")] * (expected_len - len(values))
        return np.array(values[:expected_len], dtype=np.float64)
    except ValueError:
        return None


def evaluate_chaos_prediction(
    model_adapter: TextEvalAdapter,
    system: str = "lorenz63",
    n_sequences: int = 20,
    context_steps: int = 50,
    pred_steps: int = 10,
    dt: float = 0.02,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate the model's ability to continue chaotic trajectories.

    The model receives a serialised context trajectory and must predict the
    next `pred_steps` steps.  Quality is measured as normalised RMSE relative
    to the Lyapunov-time baseline (random-walk divergence).

    Returns a dict with:
        rmse        — mean RMSE over all test sequences
        lyapunov_time — 1/λ_max for the system
        skill_score — 1 - rmse/lyapunov_rmse (higher = better; 0 = persistence)
    """
    traj = generate_trajectory(
        system,
        params={"seed": seed},
        T=(n_sequences + 2) * (context_steps + pred_steps) * dt,
        dt=dt,
        transient=50.0,
    )

    lya_time = lyapunov_time(system, dt=dt)
    dim = traj.shape[1]

    rmses: list[float] = []
    rng = np.random.default_rng(seed)

    for i in range(n_sequences):
        start = i * (context_steps + pred_steps)
        ctx_end = start + context_steps
        pred_end = ctx_end + pred_steps

        if pred_end > len(traj):
            break

        ctx_traj = traj[start:ctx_end]
        true_traj = traj[ctx_end:pred_end]

        context_str = _trajectory_to_text(ctx_traj)
        prompt = (
            f"Continue the following {system} trajectory "
            f"(dt={dt}, {pred_steps} more steps):\n{context_str}\nContinuation:"
        )

        raw = model_adapter.generate(prompt, do_sample=False)
        pred = _parse_trajectory_text(raw, pred_steps * dim)

        if pred is None or np.any(np.isnan(pred)):
            # Model failed to produce valid output — use last context point as persistence
            pred_arr = np.tile(ctx_traj[-1], (pred_steps, 1))
        else:
            pred_arr = pred.reshape(pred_steps, dim)

        se = np.mean((pred_arr - true_traj) ** 2, axis=-1)  # (pred_steps,)
        rmse = float(np.sqrt(np.mean(se)))
        rmses.append(rmse)

    mean_rmse = float(np.mean(rmses)) if rmses else float("nan")

    # Lyapunov-time skill: a random-walk predictor error grows as exp(λ * t * dt)
    # Use the mean trajectory std as a scale reference
    splits = split_trajectory(traj, train=0.7, val=0.15, test=0.15)
    test_std = float(np.std(splits["test"]))
    lyapunov_rmse = test_std  # persistence-level error at Lyapunov time

    skill = 1.0 - mean_rmse / lyapunov_rmse if lyapunov_rmse > 0 else float("nan")

    return {
        "rmse": mean_rmse,
        "lyapunov_time": lya_time,
        "skill_score": skill,
    }


# ---------------------------------------------------------------------------
# Throughput & VRAM measurement
# ---------------------------------------------------------------------------


def measure_throughput(
    wrapper: Any,
    tok: Any,
    device: torch.device,
    n_warmup: int = 5,
    n_measure: int = 20,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 64,
) -> dict[str, float]:
    """Measure generation throughput in tokens/sec."""
    input_ids = tok.encode(prompt, padding=False, truncation=True, max_length=512)
    input_ids = input_ids.to(device)

    wrapper.model.eval()
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            wrapper.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                do_sample=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(n_measure):
        with torch.no_grad():
            out = wrapper.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                do_sample=False,
            )
        total_tokens += out.shape[-1] - input_ids.shape[-1]

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "tokens_per_sec": tokens_per_sec,
        "n_measure": n_measure,
        "max_new_tokens": max_new_tokens,
    }


def get_vram_usage(device: torch.device) -> dict[str, float]:
    """Return current and peak VRAM usage in MB."""
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "peak_mb": peak,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-0.8B-Base on the full synthetic benchmark suite."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Model weight dtype (default: fp16)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path for results JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=200,
        help="Examples per benchmark task (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Eval batch size (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per example (default: 64)",
    )
    parser.add_argument(
        "--chaos-sequences",
        type=int,
        default=20,
        help="Number of chaos prediction sequences per system (default: 20)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip benchmark tasks already present in the output file",
    )
    parser.add_argument(
        "--skip-chaos",
        action="store_true",
        help="Skip chaotic prediction evaluation",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print(f"Loading Qwen3.5-0.8B-Base on {args.device} ({args.dtype})…")
    wrapper = load_model("qwen3.5-0.8b", dtype=dtype, device=args.device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_adapter = TextEvalAdapter(wrapper, max_new_tokens=args.max_new_tokens)

    # --- Perplexity on held-out text ---
    print("Computing perplexity on sample texts…")
    perplexity = compute_text_perplexity(wrapper, _SAMPLE_TEXTS, device)
    print(f"  Perplexity: {perplexity:.2f}")

    # --- Main benchmark suite ---
    benchmarks = build_benchmark_suite(n=args.n_examples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(
        batch_size=args.batch_size,
        num_few_shot=0,
        decode_mode="greedy",
        metrics=["exact_match", "accuracy", "f1"],
        output_file=None,  # We save manually below with extra metadata
        model_name="qwen3.5-0.8b",
        resume=args.resume,
    )

    print(f"Running {len(benchmarks)} benchmark tasks…")
    results: list[EvalResult] = evaluate(model_adapter, benchmarks, config)

    # --- Chaos prediction (T6) ---
    chaos_results: dict[str, dict[str, float]] = {}
    if not args.skip_chaos:
        for system in ("lorenz63", "mackey_glass"):
            print(f"Evaluating chaos prediction: {system}…")
            chaos_results[system] = evaluate_chaos_prediction(
                model_adapter,
                system=system,
                n_sequences=args.chaos_sequences,
            )
            cr = chaos_results[system]
            print(
                f"  {system}: RMSE={cr['rmse']:.4f}, "
                f"Lyapunov time={cr['lyapunov_time']:.2f}, "
                f"Skill={cr['skill_score']:.4f}"
            )

    # --- Throughput ---
    print("Measuring throughput…")
    throughput = measure_throughput(
        wrapper, wrapper.tokenizer, device, max_new_tokens=args.max_new_tokens
    )
    print(f"  Throughput: {throughput['tokens_per_sec']:.1f} tokens/sec")

    # --- VRAM usage ---
    vram = get_vram_usage(device)
    print(f"  VRAM: {vram['allocated_mb']:.1f} MB allocated, {vram['peak_mb']:.1f} MB peak")

    # --- Latency stats from inference done during benchmarks ---
    latency = model_adapter.latency_stats()
    print(f"  Latency p50={latency['p50_s']*1000:.1f}ms, p95={latency['p95_s']*1000:.1f}ms")

    # --- Assemble and save final output ---
    import subprocess
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=_REPO_ROOT,
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": "qwen3.5-0.8b",
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "git_hash": git_hash,
        "timestamp": time.time(),
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "n_examples": args.n_examples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "perplexity": perplexity,
        "throughput": throughput,
        "vram": vram,
        "latency": latency,
        "chaos": chaos_results,
        "results": [asdict(r) for r in results],
    }

    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nDone. {len(results)} metric results saved to {args.output}")

    # Print a summary table
    print("\n=== Results Summary ===")
    print(f"{'Task':<45} {'Metric':<15} {'Score':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r.task:<45} {r.metric:<15} {r.value:>7.4f}")

    print("\n=== Performance Summary ===")
    print(f"  Perplexity (sample texts): {perplexity:.2f}")
    print(f"  Throughput: {throughput['tokens_per_sec']:.1f} tokens/sec")
    print(f"  VRAM allocated: {vram['allocated_mb']:.1f} MB")
    print(f"  VRAM peak: {vram['peak_mb']:.1f} MB")
    print(f"  Latency p50: {latency['p50_s']*1000:.1f} ms")
    print(f"  Latency p95: {latency['p95_s']*1000:.1f} ms")

    if chaos_results:
        print("\n=== Chaos Prediction Summary ===")
        for system, cr in chaos_results.items():
            print(
                f"  {system}: RMSE={cr['rmse']:.4f}, "
                f"Lyapunov time={cr['lyapunov_time']:.2f}, "
                f"Skill score={cr['skill_score']:.4f}"
            )


if __name__ == "__main__":
    main()
