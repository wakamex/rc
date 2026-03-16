# Track B Experiment Loop

Autonomous experimentation loop for Track B: integrating ESN reservoirs with Qwen3.5-0.8B's native DeltaNet layers.

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
- `src/reservoir/interface.py` — sidecar architectures (reference)
- `src/reservoir/deltanet_replace.py` — DeltaNet replacement module
- `docs/gate_a_report.md` — Track A final assessment (what worked, what didn't)

Do NOT read `DIARY.md` — it's a chronological log. `RESULTS.md` has everything you need.

## Workflow

Work directly on `main`. Push commits directly — no side branches.

| Commit prefix | Checkpoint dir                |
|---------------|-------------------------------|
| `CLAUDE:`     | `checkpoints/track_b/deltanet` |

Only one agent trains at a time (single GPU, 24GB).

## Track B Architecture

Qwen3.5-0.8B has a **hybrid architecture**: 18 DeltaNet (linear attention) + 6 full-attention layers in a 3:1 pattern.

**DeltaNet layer indices:** 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
**Full-attention indices:** 3, 7, 11, 15, 19, 23

## Research Plan (Three Phases)

### Phase 1: Distillation Sweep (do this first)

**Goal:** Map which DeltaNet layers are "reservoir-compatible" — can the ESN reproduce what DeltaNet produces?

**Method:** For each of 18 DeltaNet layers independently:
1. Run a batch of data through the frozen model
2. Capture each DeltaNet layer's input and output activations
3. Run the same input through an ESN (r=1000)
4. Train a linear readout (ridge regression) to map ESN states → DeltaNet output
5. Measure reconstruction MSE (distillation loss)

**What this tells us:**
- Low distillation loss → ESN can replicate this layer's function → safe to replace
- High distillation loss → DeltaNet is doing something the reservoir can't → don't replace
- The loss curve across layers maps the difficulty gradient

**This is cheap** — no gradient descent, just linear regression. Should take ~30 min total.

**Script:** `scripts/distill_sweep.py` (to be created)

### Phase 2: Single-Layer Replacement (informed by Phase 1)

**Goal:** Replace the easiest DeltaNet layer (lowest distillation loss from Phase 1) with an ESN and fine-tune.

**Key difference from B1:**
- Replace only ONE layer (B1 replaced 6 → catastrophic ppl blowup)
- Initialize ESN readout from the distillation weights (warm start, not random)
- Much lower gate_init (0.01 — nearly pure DeltaNet to start)

**Method:**
1. Pick the layer with lowest distillation loss from Phase 1
2. Initialize ESNReplacementInterface readout from distillation weights
3. Fine-tune with gate_warmup from 0 → 0.05 over 50 optimizer steps
4. Train 5000 steps, eval on full 23-benchmark suite

**Success criterion:** ppl < 7.0 AND avg_em ≥ Track A (0.357)

**Then iterate:** try 2 layers (the two easiest), try harder layers, etc.

### Phase 3: ESN as Forgetting Controller (the cleaner experiment)

**Goal:** Don't replace DeltaNet at all. Keep it intact, run ESN in parallel, use ESN state to *gate* DeltaNet's memory retention.

**Core hypothesis:** The reservoir's dynamics naturally perform relevance filtering. Inputs that perturb the reservoir state persistently are dynamically important; inputs that decay are forgettable. Use this signal to tell DeltaNet what to keep and what to evict.

**How it works:**
```
deltanet_output = DeltaNet(hidden_states)           # original, untouched
esn_state = ESN.step(hidden_states)                 # parallel reservoir
relevance = sigmoid(linear(esn_state))              # importance signal ∈ [0,1]
output = deltanet_output * relevance                # gate what DeltaNet retains
```

**Why this is different from Track A sidecar:**
- Track A *adds* reservoir info to the residual stream (additive injection)
- Phase 3 *modulates* DeltaNet's own output (multiplicative gating)
- Track A uses reservoir as memory; Phase 3 uses reservoir as memory *controller*

**Why the reservoir is well-suited:**
- Near edge of chaos, reservoir naturally computes implicit importance scores
- Dambre decomposition: tracks both linear memory (what happened) and nonlinear interactions (what patterns are forming)
- Not recency-based like FIFO — a memory from 500 steps ago that's part of an ongoing pattern stays important
- Cheap: ESN update is single matrix multiply + tanh, negligible vs transformer forward pass

**What to measure:**
- Short context tasks (< 1k tokens): expect parity with vanilla
- Long context with mostly relevant info: expect parity
- Long context with high noise/distractor ratio: this is where ESN gating should win
- The gap should widen as memory demands increase

**Spectral radius sweep is critical:**
- Run at multiple spectral radii (0.8, 0.9, 0.95, 0.99, 1.0)
- Optimal for downstream performance may differ from optimal for distillation
- ESN might beat DeltaNet by forgetting things DeltaNet wastefully retains

**Script:** New training script needed — similar to Track A sidecar but with multiplicative gating on DeltaNet output instead of additive injection.

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

## Gate B Success Criteria

From COLLABORATIVE_PROPOSAL.md:
1. ≥20% exact-match gain on long program-trace tasks
2. Better memory-quality-per-byte than RoPE/YaRN-only context extension
3. <2% perplexity degradation (ppl < 6.96)

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue. Run experiments autonomously until manually interrupted. If you run out of ideas, re-read `RESULTS.md` for new angles. Try combining near-misses. Try more radical changes. The loop runs until the human stops you.
