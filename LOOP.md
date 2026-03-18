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

### Phase 1: Distillation Sweep — COMPLETE

Script: `scripts/distill_sweep.py`. Ridge regression ESN→DeltaNet output for all 18 layers.
Result: Layer 10 easiest (rel_mse=0.177), early layers hardest. See RESULTS.md for full table.

### Phase 2: Layer Replacement — COMPLETE (B3 is best)

B2 (1 layer) and B3 (2 layers) both beat Track A. B4 (3 layers) crossed ppl threshold. B5 showed gate_init irrelevant. **Plateau at ~0.368 avg_em with 2 layers.**

### Phase 3: ESN as Forgetting Controller — NEXT

**Goal:** Don't replace DeltaNet. Keep it intact, run ESN in parallel. Use ESN state to *modulate* DeltaNet's output — the reservoir tells the LLM what's worth keeping.

**Core hypothesis:** A reservoir near the edge of chaos naturally performs relevance filtering. Inputs that perturb reservoir state persistently are dynamically important (keep). Inputs that decay are forgettable. This is a byproduct of the dynamics — it's free.

**The theoretical basis (Takens/Dambre):**
- Takens' theorem: ESN state is an implicit nonlinear delay embedding — it reconstructs the input stream's dynamical structure without choosing τ or embedding dimension
- Dambre decomposition: reservoir tracks both linear memory (what happened) and nonlinear interactions (what patterns are forming across time)
- This isn't recency-based like FIFO — a memory from 500 steps ago that's part of an ongoing pattern stays important, while a recent isolated input can be forgotten
- The reservoir computes a continuous, implicit importance score for every piece of information, for free

**How it differs from Track A and Phase 2:**
- Track A: ESN state *added* to residual stream (additive injection, memory store)
- Phase 2: ESN output *replaces* DeltaNet output (substitution)
- **Phase 3: ESN state *modulates* DeltaNet output (multiplicative gating, memory controller)**
- Attention asks "what's relevant to the current query?" — reactive, pairwise
- Reservoir gating asks "what's dynamically alive?" — proactive, holistic, no query needed

**Implementation (~200 lines):**
```
deltanet_output = DeltaNet(hidden_states)        # original, untouched
esn_state = ESN.step(hidden_states)              # parallel reservoir
relevance = sigmoid(linear(esn_state))           # importance signal ∈ [0,1]
output = deltanet_output * relevance             # gate what DeltaNet retains
```

Hook into DeltaNet layers' forward pass. Can reuse the existing `DeltaNetReplacementManager` hook infrastructure, just change the mixing formula from `gate * esn + (1-gate) * deltanet` to `deltanet * relevance_from_esn`.

**Spectral radius sweep is critical:**
- Run at multiple spectral radii (0.8, 0.9, 0.95, 0.99, 1.0)
- The optimal radius for downstream performance may differ from the optimal for distillation
- If the forgetting-controller hypothesis is right, the ESN might beat DeltaNet by forgetting things DeltaNet wastefully retains

**Predictions:**
- Short context tasks (< 1k tokens): expect parity with vanilla
- Long context, mostly relevant info: expect parity
- Long context, high noise/distractor ratio: ESN gating should win
- Gap should widen as memory demands increase

### Future: AttnRes-style Fusion

Instead of fixed injection points or per-layer gating, make the ESN state an additional "source" in depth-wise attention (inspired by AttnRes paper). Each layer learns to attend over both previous-layer outputs AND ESN state. Subsumes all approaches but requires deeper architectural changes.

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
