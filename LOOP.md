# Reservoir Sidecar Experiment Loop

Autonomous experimentation loop for optimizing the ESN reservoir sidecar on Qwen3.5-0.8B.

## Objective

**Maximize `score` where:**

```
avg_task = mean(ModularArithmetic_EM, LengthExtrapolation_EM, DyckLanguage_EM) - 0.016
score    = avg_task - 2 × (ppl - 6.82)
```

- `avg_task` is the reservoir's delta over the LoRA-only baseline (0.016) on 3 structured reasoning tasks that LoRA alone can't solve
- `ppl` is perplexity on 5 standard texts (vanilla baseline: 6.82)
- The penalty is linear and continuous — every +0.1 ppl costs 0.2 avg_task. Lower ppl is always better.
- The no-go gate is ppl ≤ 6.96 (≤2% degradation). At the gate boundary, the ppl penalty is 2 × 0.14 = 0.28 — roughly the current avg_task, so score ≈ 0 there. Above the gate, score goes negative.

**Current incumbent: score = -1.551** (avg_task=0.249, ppl=7.72). The perplexity cost is the primary problem.

## Context

Read these files for full context before your first experiment:

- `RESULTS.md` — what we know so far (durable conclusions, ablation results)
- `src/reservoir/interface.py` — sidecar architecture (CrossAttentionSidecar, FiLMModulation)
- `scripts/train_track_a_readonly.py` — training script (model loading, ESN, hooks, training loop)
- `scripts/sprint_eval.py` — the sprint eval script (3 key benchmarks + perplexity → score)
- `scripts/quick_test.py` — quick AR/VT diagnostic (not used in the loop)

Do NOT read `DIARY.md` — it's a 900-line chronological log. `RESULTS.md` has everything you need.

## Agent Conventions

Multiple agents may run this loop in separate worktrees off the same file. Use your row:

| Agent  | Branch prefix | Commit prefix | Checkpoint dir         |
|--------|---------------|---------------|------------------------|
| Claude | `claude/`     | `CLAUDE:`     | `checkpoints/sprint`   |
| GPT    | `gpt/`        | `GPT:`        | `checkpoints/sprint`   |

Only one agent trains at a time (single GPU, 24GB). Agents run serially, not concurrently.

## Setup

1. **Create branch:** `git checkout -b <your-prefix>autoresearch/<tag>` from main (tag = today's date, e.g. `mar12`).
2. **Verify GPU:** `python -c "import torch; print(torch.cuda.get_device_name())"` — expects RTX 3090.
3. **Verify checkpoint:** `ls checkpoints/track_a_readonly/final/` — needs `lora_adapter/` and `sidecar_weights.pt`.
4. **Create `results.tsv`** with the header row (see Logging below).
5. **Run baseline:** Eval the current checkpoint to establish the sprint baseline:
   ```bash
   python scripts/sprint_eval.py checkpoints/track_a_readonly/final > baseline_eval.log 2>&1
   grep "^avg_task:\|^ppl:\|^score:" baseline_eval.log
   ```

## What You CAN Modify

- `src/reservoir/interface.py` — sidecar architecture is the primary target. Try:
  - Different injection mechanisms (cross-attention vs FiLM vs linear vs MLP)
  - Gate mechanisms (tanh vs sigmoid vs learned schedule vs fixed)
  - Projection dimensions, number of heads
  - Normalization placement
- `scripts/train_track_a_readonly.py` — training hyperparameters:
  - Learning rates, warmup schedule, gate warmup target
  - Memory task ratio (currently 0.3)
  - Number/placement of sidecar layers
  - LoRA rank, alpha, target modules
  - Batch size, grad accumulation, sequence length
- ESN reservoir parameters: size, spectral_radius, leak_rate, sparsity

## What You CANNOT Modify

- `src/eval/benchmarks/` — benchmark definitions are fixed
- `src/eval/harness.py` — evaluation logic is fixed
- `src/reservoir/esn.py` — ESN implementation is fixed
- `src/models/loader.py` — model loading is fixed
- The 3 benchmark tasks and parameters used in sprint eval
- The perplexity computation (same 5 texts, same method)
- Base model (Qwen3.5-0.8B-Base)

## The Experiment Loop

**Sprint budget: 300 training steps (~30 min) + sprint eval (~10 min) = ~40 min total.**

LOOP FOREVER:

1. **Hypothesis:** Write one sentence about what you're testing and why.
2. **Modify code:** Make the change (architecture, hyperparams, etc.).
3. **Commit:** `git commit` with a short description. Use your commit prefix and include the score delta after eval (e.g. `CLAUDE: reduce sidecar to 3 layers (+0.45, +29%, score)`).
4. **Train:**
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_track_a_readonly.py \
     --no_wandb --max_steps 300 --batch_size 1 --grad_accum 16 \
     --memory_task_ratio 0.3 --gate_warmup_steps 50 --gate_warmup_target 0.1 \
     --output_dir checkpoints/sprint --save_interval 9999 \
     --log_interval 50 > run.log 2>&1
   ```
   Adjust args as needed for your experiment. Redirect output — do NOT flood context.
5. **Eval:** `python scripts/sprint_eval.py checkpoints/sprint/final`
6. **Extract:** `grep "^score:\|^avg_task:\|^ppl:" run_eval.log`
7. **Log:** Append to `results.tsv`.
8. **Decide:**
   - If score improved → amend the commit message with the result delta (e.g. `<PREFIX> reduce sidecar to 3 layers (+0.45, +29%, score)`), this is the new baseline.
   - If score is equal or worse → `git reset --hard HEAD~1` to discard.
9. **Repeat.**

If a run crashes, fix the bug and retry once. If the idea is fundamentally broken, log it as a crash and move on.

**Timeout:** If training exceeds 60 minutes, kill it and treat as failure.

## Logging

`results.tsv` — tab-separated, append-only:

```
commit	score	avg_task	ppl	status	description
```

- `commit`: short git hash (7 chars)
- `score`: computed metric
- `avg_task`: mean of 3 task EMs
- `ppl`: perplexity
- `status`: `keep`, `discard`, or `crash`
- `description`: one-line summary of what was tried

Example:
```
commit	score	avg_task	ppl	status	description
a1b2c3d	-1.551	0.249	7.72	keep	baseline (current checkpoint)
b2c3d4e	-1.156	0.164	7.48	keep	reduce reservoir to 1000 nodes
c3d4e5f	0.000	0.000	0.00	crash	FiLM modulation OOM
d4e5f6g	-2.000	0.200	7.92	discard	sigmoid gate worse than tanh
```

Do NOT commit results.tsv — leave it untracked.

## Key Constraints

- **VRAM:** 24 GB (RTX 3090). batch_size=1 + grad_accum=16 is the safe default.
- **Gradient checkpointing is broken** with sidecar hooks — do not enable it.
- **PYTORCH_ALLOC_CONF=expandable_segments:True** required for all training runs.
- **Minimum 300 steps** for any signal on structured reasoning tasks.
- Gates need warmup — without it they never open (chicken-and-egg with tanh gating).

## Research Directions (Ordered by Expected Impact)

1. **Reduce perplexity cost** — this is the blocker. The reservoir adds +0.90 ppl. Ideas:
   - Smaller reservoir (1000 instead of 10000) — less noise injected
   - Fewer sidecar layers (2-3 instead of 6)
   - Lower gate warmup target (0.05 instead of 0.1)
   - FiLM modulation instead of cross-attention (lighter touch)
   - Reduce memory_task_ratio (0.1 instead of 0.3) — less distribution shift
2. **Improve task scores** (once ppl is under control):
   - More sidecar layers at deeper positions
   - Larger LoRA rank
   - Different ESN parameters (higher spectral_radius for longer memory)
3. **Architecture changes:**
   - Replace cross-attention with simple gated linear projection
   - Try FiLMModulation (already implemented in interface.py, never tested)
   - Additive vs multiplicative injection
   - Per-head gating instead of per-layer

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue. Run experiments autonomously until manually interrupted. If you run out of ideas, re-read `RESULTS.md` and `src/reservoir/interface.py` for new angles. Try combining near-misses. Try more radical changes. The loop runs until the human stops you.
