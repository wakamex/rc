# Track A Scaling Experiments

Track B is complete (negative result). The project now focuses on characterizing Track A's scaling behavior to build a publishable story: "where and how auxiliary recurrent memory helps transformers."

## Objective

**Produce four scaling curves** showing ESN sidecar performance as a function of:
1. Reservoir size (capacity)
2. Layer count (injection depth)
3. Sequence length (context dependence)
4. Base model (architecture generality)

Target figure: four panels, each with ppl and avg_em curves, error bars, clean axes. Caption: "ESN sidecar performance scales predictably with reservoir capacity, injection depth, context length, and model size."

## Context

- `RESULTS.md` — Track A results (incumbent), Track B results (negative)
- `scripts/train_track_a_readonly.py` — Track A training script
- `scripts/sprint_eval.py` — 3-task sprint eval
- `scripts/eval_track_a.py` — full 23-benchmark eval
- `src/reservoir/interface.py` — GatedLinearSidecar architecture
- `docs/gate_a_report.md` — Track A assessment (4/5 gates passed)

## Workflow

Work directly on `main`. Push commits directly.

| Commit prefix | Checkpoint dir |
|---------------|----------------|
| `CLAUDE:` | `checkpoints/scaling/` |

## Current Track A Baseline

- **Architecture:** GatedLinearSidecar at layers [3,7,11,15,23], ESN r=1000
- **Results:** avg_em=0.357, ppl=6.84
- **LoRA-only ablation:** modarith=0.02 vs reservoir 0.14-0.20 (7-10x). Reservoir confirmed to help.

## Scaling Experiments (ordered by impact per compute-hour)

### 1. Second Base Model — LLaMA-3.2-1B (~5h)

**Why highest priority:** If the same recipe works on a pure transformer (no DeltaNet), it proves the approach isn't Qwen-specific. If it works *better* (LLaMA has no recurrent layers → more benefit from external recurrent memory), that's a scaling argument. If it fails, the DeltaNet layers are enabling the sidecar — changes the mechanistic story.

**Method:**
- Load LLaMA-3.2-1B via `load_model("llama-3.2-1b")`
- Same GatedLinearSidecar at equivalent layer positions
- Same training recipe: 5000 steps, 30% memory tasks, gate_warmup=0.1
- Full 23-benchmark eval

**Key question:** Does the ESN help transformers generally, or only Qwen's hybrid architecture?

### 2. Reservoir Size Scaling (~10h, 7 runs)

**Sizes:** r = 64, 128, 256, 512, 1000, 2000, 4096

**Theory:** Jaeger/Dambre capacity curve predicts performance improves with r up to a critical point where the gating projection can't usefully compress the reservoir state, then degrades. We have two data points (r=1000 works, r=10000 too noisy). Need the full curve.

**Method:** For each r, train 5000 steps with standard recipe, run full eval.

**What to plot:** ppl and avg_em vs log(r). Expect concave curve with peak near r=512-1000.

### 3. Layer Count and Position (~9h, 6 runs)

**Configs:** 1, 2, 3, 4, 5, 6 sidecar layers, each at optimal positions.

We know:
- 5 layers [3,7,11,15,23] is current best
- Removing any single layer kills performance (exp44, exp45)

Need the build-up curve: what's the marginal return per additional layer?

**Method:** For each count, pick best positions (start with the most important layers), train, eval.

**Suggested progression:**
- 1 layer: [11] (middle)
- 2 layers: [7, 19]
- 3 layers: [3, 11, 23]
- 4 layers: [3, 7, 15, 23]
- 5 layers: [3, 7, 11, 15, 23] (current best)
- 6 layers: [3, 7, 11, 15, 19, 23]

**What to plot:** ppl and avg_em vs layer count. Look for knee (phase transition) vs concave (diminishing returns).

### 4. Sequence Length Scaling (~8h, 4 runs)

**Lengths:** 512, 1024, 2048, 4096

**Theory:** The reservoir's advantage is persistent memory beyond attention's effective window. If the sidecar's benefit increases with sequence length, the reservoir provides something attention fundamentally can't.

**Method:** Train at each seq length (adjust batch/accum for VRAM), eval at same length.

**What to plot:** ppl ratio (sidecar/vanilla) and avg_em delta vs sequence length. Downward-sloping ratio = reservoir helps more at longer contexts.

### 5. Model Size (optional, needs cloud)

If access to Qwen3.5-2B or 3B is possible, even one run showing the sidecar helps at larger scale is worth it. Reviewers will ask "does this scale?"

## What NOT to spend time on

- Topology experiments (Watts-Strogatz vs Erdős-Rényi) — unlikely to matter for scaling story
- More training recipe tuning — 5000 steps and 30% memory ratio are fine
- Multi-reservoir fast/slow setup — Track B showed forgetting angle doesn't work
- Track B follow-ups — it's a negative result, document it and move on

## The Experiment Loop

**Budget: 5000 steps (~90 min training) + eval (~30 min) = ~2h per config.**

1. **Train:**
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_track_a_readonly.py \
     --no_wandb --max_steps 5000 --batch_size 1 --grad_accum 16 \
     --memory_task_ratio 0.3 --warmup_steps 15 --gate_warmup_steps 10 --gate_warmup_target 0.1 \
     --output_dir checkpoints/scaling/<experiment_name> --save_interval 9999 \
     --log_interval 100 > run.log 2>&1
   ```
2. **Eval:** `python scripts/eval_track_a.py --checkpoint checkpoints/scaling/<experiment_name>/final`
3. **Log:** Update RESULTS.md scaling tables.
4. **Commit:** Include results in message.
5. **Preserve best:** Copy to `checkpoints/scaling/best/` if new best.

## Key Constraints

- **VRAM:** 24 GB (RTX 3090). batch_size=1 + grad_accum=16 is safe default.
- **Gradient checkpointing is broken** with sidecar hooks.
- **PYTORCH_ALLOC_CONF=expandable_segments:True** required.
- **5000 steps baseline.** Only deviate for the sequence length experiment.

## Compute Budget

| Experiment | Runs | Time/run | Total |
|-----------|------|----------|-------|
| LLaMA-3.2-1B | 1 | ~5h | ~5h |
| Reservoir size sweep | 7 | ~1.5h | ~10h |
| Layer count sweep | 6 | ~1.5h | ~9h |
| Sequence length sweep | 4 | ~2h | ~8h |
| **Total** | **18** | | **~32h** |

One weekend on the 3090. All local, no cloud needed except optional model size scaling.
