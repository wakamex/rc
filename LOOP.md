# Track A Scaling Experiments

Track B is complete (negative result). The project now focuses on characterizing Track A's scaling behavior to build a publishable story: "where and how auxiliary recurrent memory helps transformers."

## Current Status

**Completed sweeps (0.8B):** reservoir size (7 runs), layer count (6 runs), sequence length (3 runs), multi-seed (3 seeds), LLaMA-3.2-1B, Qwen3.5-2B, Qwen3.5-4B (training)

**Best 0.8B config:** r=256, 5 layers [3,7,11,15,23], seq_len=1024 → avg_em=0.347±0.019, ppl=6.77±0.05

**Current work:** 2B/4B hyperparameter sweeps to find optimal sidecar config at larger model sizes. The 0.8B-tuned hyperparameters (gate=0.1, r=256) cause +4% ppl at 2B — needs optimization.

**Active task queue:** See `TASKS.md` for the ordered run list. Work through it sequentially.

## Context

- `RESULTS.md` — Track A results, Track B negative results, scaling sweep data
- `scripts/train_track_a_readonly.py` — Track A training script
- `scripts/eval_track_a.py` — full 23-benchmark eval
- `src/reservoir/interface.py` — GatedLinearSidecar architecture
- `docs/gate_a_report.md` — Track A assessment (4/5 gates passed)

## Workflow

Work directly on `main`. Push commits directly.

| Commit prefix | Checkpoint dir |
|---------------|----------------|
| `CLAUDE:` | `checkpoints/scaling/` |

## Best Config

- **Base model:** Qwen3.5-0.8B-Base (frozen)
- **Sidecar:** GatedLinearSidecar, r=256, 5 layers [3,7,11,15,23]
- **Training:** 5000 steps, batch=1, grad_accum=16, seq_len=1024, lr=2e-4, 30% memory tasks
- **Results:** avg_em=0.374, ppl=6.70 (vanilla: 6.82)

## Completed Scaling Sweeps

### Reservoir Size (r=64 to 4096) — DONE
Peak: r=256 (tasks), r=512 (ppl). Matches Dambre capacity theory.

### Layer Count (1 to 6) — DONE
Peak: 5 layers. Layer 11 critical. Concave curve, clear optimum.

### Sequence Length (512 to 4096) — DONE (seq4096 OOM)
Peak: seq1024. Benefit roughly uniform across lengths — argues against "persistent memory beyond attention window," supports "augments attention at all scales."

## Remaining Experiments

### Multi-Seed Variance (3 seeds)
Best config (r=256, 5 layers, seq1024) at seeds 42, 43, 44. Need mean ± std for paper.

### LLaMA-3.2-1B (~5h)
HF access approved. Pure softmax transformer (no DeltaNet). Tests whether sidecar is architecture-general. Same recipe: r=256, 5 layers at equivalent positions [3, 7, 11, 15], 5000 steps.

### Model Size (optional, cloud)
Qwen3.5-2B or 3B. One data point showing sidecar helps at larger scale.

## The Experiment Loop

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_track_a_readonly.py \
  --no_wandb --max_steps 5000 --batch_size 1 --grad_accum 16 \
  --memory_task_ratio 0.3 --warmup_steps 15 --gate_warmup_steps 10 --gate_warmup_target 0.1 \
  --reservoir_size 256 --sidecar_type gated_linear --gate_init 0.0 \
  --sidecar_layers 3 7 11 15 23 \
  --output_dir checkpoints/scaling/<name> --save_interval 9999 \
  --log_interval 100 > run.log 2>&1
```

Eval: `python scripts/eval_track_a.py --checkpoint checkpoints/scaling/<name>/final`

## Key Constraints

- **VRAM:** 24 GB (RTX 3090). batch_size=1 + grad_accum=16.
- **Gradient checkpointing broken** with sidecar hooks.
- **PYTORCH_ALLOC_CONF=expandable_segments:True** required.
