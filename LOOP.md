# Track B Experiment Loop

Autonomous experimentation loop for Track B: DeltaNet block replacement with ESN reservoirs on Qwen3.5-0.8B.

## Objective

**Maximize `avg_em` while keeping `ppl < 7.0` (< +2% over vanilla 6.82).**

```
avg_em = mean of all 23 benchmark exact-match scores
ppl    = perplexity on 5 standard texts (vanilla baseline: 6.82)
```

- Higher avg_em is better
- ppl must stay below 7.0 (Gate B criterion: <2% degradation)
- Track A incumbent: avg_em=0.357, ppl=6.84

## Context

Read these files for full context before your first experiment:

- `RESULTS.md` — Track A final results + Track B experiment log
- `scripts/train_track_b_deltanet.py` — Track B training script (DeltaNet replacement + ESN)
- `scripts/eval_track_b.py` — Track B eval script (full 23-benchmark suite)
- `src/reservoir/interface.py` — sidecar architectures (reference, not directly used in Track B)
- `src/reservoir/deltanet_replace.py` — DeltaNet replacement module
- `docs/gate_a_report.md` — Track A final assessment (what worked, what didn't)

Do NOT read `DIARY.md` — it's a chronological log. `RESULTS.md` has everything you need.

## Workflow

Work directly on `main`. Push commits directly — no side branches.

| Commit prefix | Checkpoint dir                |
|---------------|-------------------------------|
| `CLAUDE:`     | `checkpoints/track_b/deltanet` |

Only one agent trains at a time (single GPU, 24GB).

## Setup

1. **Verify GPU:** `python -c "import torch; print(torch.cuda.get_device_name())"` — expects RTX 3090.
2. **Check RESULTS.md** for the latest Track B experiment results and what's been tried.

## What You CAN Modify

- `scripts/train_track_b_deltanet.py` — training hyperparameters and replacement strategy:
  - Number of DeltaNet blocks replaced (`--replace_every_nth_deltanet`)
  - Gate initialization and warmup schedule
  - Learning rates, warmup schedule
  - Memory task ratio (currently 0.3)
  - LoRA rank, alpha, target modules
  - Batch size, grad accumulation, sequence length
- `scripts/train_track_b_deltanet.py:ESNReplacementInterface` — the replacement interface:
  - Gate mechanism (sigmoid, tanh, etc.)
  - Projection architecture (linear, MLP, etc.)
  - Normalization strategy
- ESN reservoir parameters: size, spectral_radius, leak_rate, sparsity

## What You CANNOT Modify

- `src/eval/benchmarks/` — benchmark definitions are fixed
- `src/eval/harness.py` — evaluation logic is fixed
- `src/reservoir/esn.py` — ESN implementation is fixed
- `src/models/loader.py` — model loading is fixed
- The 23 benchmark tasks and perplexity computation
- Base model (Qwen3.5-0.8B-Base)

## The Experiment Loop

**Budget: 5000 training steps (~90 min) + full eval (~30 min) = ~2 hours total.**

LOOP FOREVER:

1. **Hypothesis:** Write one sentence about what you're testing and why.
2. **Modify code:** Make the change (architecture, hyperparams, etc.).
3. **Train:**
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_track_b_deltanet.py \
     --no_wandb --max_steps 5000 --batch_size 1 --grad_accum 16 \
     --memory_task_ratio 0.3 --warmup_steps 100 --gate_warmup_steps 10 --gate_warmup_target 0.1 \
     --output_dir checkpoints/track_b/deltanet --save_interval 9999 \
     --log_interval 100 > run_track_b.log 2>&1
   ```
   Adjust args as needed. Redirect output — do NOT flood context.
4. **Eval:**
   ```bash
   python scripts/eval_track_b.py --checkpoint checkpoints/track_b/deltanet/final \
     --n-examples 50 --output results/track_b/eval_deltanet.json > eval_track_b.log 2>&1
   ```
5. **Extract:** `tail -30 eval_track_b.log` for the comparison table.
6. **Log:** Update RESULTS.md Track B experiment table.
7. **Commit:** Include results in the commit message.
8. **Preserve best:** If this is the new best, copy checkpoint to `checkpoints/track_b/best/`.
9. **Repeat.**

If a run crashes, fix the bug and retry once. If the idea is fundamentally broken, log it and move on.

**Timeout:** If training exceeds 120 minutes, kill it and treat as failure.

## Key Constraints

- **VRAM:** 24 GB (RTX 3090). batch_size=1 + grad_accum=16 is the safe default.
- **Gradient checkpointing is broken** with forward hooks — do not enable it.
- **PYTORCH_ALLOC_CONF=expandable_segments:True** required for all training runs.
- **5000 steps is the baseline.** Track A found this is the sweet spot. Can try shorter (3000) for quick screening.
- Gates need warmup — without it they never open.
- **Generation fix:** During autoregressive generation, ESN mixing is skipped when seq lengths don't match (KV-cache processes 1 token at a time but reservoir states cover full prompt). This is correct — no new reservoir info for generated tokens.

## Track B Architecture

Qwen3.5-0.8B has a **hybrid architecture**: 18 DeltaNet (linear attention) + 6 full-attention layers in a 3:1 pattern.

**DeltaNet layer indices:** 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
**Full-attention indices:** 3, 7, 11, 15, 19, 23

Track B **replaces** selected DeltaNet blocks with ESN reservoir modules:
- `ESNReplacementInterface`: gated mix of ESN output + original DeltaNet output
- `gate * esn_out + (1-gate) * deltanet_out` — sigmoid gate, per-element
- Original DeltaNet module still runs; ESN output is mixed in via learned gate

## Research Directions (Ordered by Expected Impact)

1. **Reduce replacement aggression** — exp1 replaced 6/18 blocks → ppl=13.76 (+101%). Try:
   - Replace only 1-2 DeltaNet blocks instead of 6
   - Start with gate_init=0.01 (nearly pure DeltaNet to start)
   - Replace only late-layer DeltaNet blocks (layers 16-22)
2. **Hybrid Track A + B** — combine sidecar hooks (Track A) with minimal DeltaNet replacement:
   - Keep the winning GatedLinearSidecar at [3,7,11,15,23] from Track A
   - Add 1-2 DeltaNet replacements at different layers
3. **Alternative replacement strategy:**
   - Instead of full replacement, use ESN as auxiliary input to DeltaNet (additive, not replacement)
   - Multi-reservoir: fast (high leak_rate) + slow (low leak_rate) at different layers
4. **Per-block gate learning:**
   - Different gate_init per replaced block (deeper blocks may need more ESN influence)
   - Learned gate schedule (curriculum from pure DeltaNet → mixed)

## Gate B Success Criteria

From COLLABORATIVE_PROPOSAL.md:
1. ≥20% exact-match gain on long program-trace tasks
2. Better memory-quality-per-byte than RoPE/YaRN-only context extension
3. <2% perplexity degradation (ppl < 6.96)

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue. Run experiments autonomously until manually interrupted. If you run out of ideas, re-read `RESULTS.md` for new angles. Try combining near-misses. Try more radical changes. The loop runs until the human stops you.
