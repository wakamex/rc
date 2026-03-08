# TODO

## Overnight experiment plan

### ~~Step 0: Fix benchmark seed~~ DONE

Already deterministic — both `AssociativeRecall` and `VariableTracking` default to `seed=42`.
Verified: two runs produce identical targets (AR: [858, 674, 263], VT: [3, 1, 8]).

### ~~Step 1: Run the 1000-step tanh gate baseline~~ DONE

AR=10/10, VT=3/10. Gates still tiny (~0.005). VT improving but gates not opening.
See DIARY.md 02:42 entry.

### ~~Step 2: Higher interface_lr~~ DONE — gates stayed flat

- 2a (iflr=1e-3): AR=9/10, VT=0/10, gates ~0.003-0.007
- 2b (iflr=5e-3): AR=9/10, VT=0/10, gates ~0.001-0.007
- 2c skipped (same mechanism, would give same result)

Conclusion: interface_lr doesn't affect gate learning because gate uses LoRA lr,
and sidecar gradient is scaled by tanh(alpha)~0 regardless.

### Step 3: Linear gate warmup schedule

Gates won't open via gradient alone. Force them open with a schedule:
- Add `--gate_warmup_steps N` to training script
- During steps 0..N: override gate_alpha with linear ramp 0 -> target (e.g. 0.5)
- After step N: let gate_alpha be learnable (initialized at target value)
- This forces gradient to flow through sidecar from early training

Implementation:
1. Add `--gate_warmup_steps` and `--gate_warmup_target` args to train script
2. In training loop, before forward pass, set `sidecar.gate_alpha.data` to
   `min(step/warmup_steps, 1.0) * target` for each sidecar
3. After warmup, stop overriding (let it be learnable)

~~Run 1000 steps with gate_warmup_steps=200, gate_warmup_target=0.5~~ DONE
Result: AR=2/10, VT=6/10. Gates opened to 0.155! VT breakthrough but AR collapsed.
Target=0.5 too aggressive. Loss rose to ppl=44.92.

### ~~Step 3b: Gate warmup sweep~~ DONE

- target=0.1: AR=10/10, VT=6/10 (winner!)
- target=0.2: AR=0/10, VT=2/10 (too aggressive)

### ~~Step 4a+4b: Gate evolution + isolation test~~ DONE

- Gates grow steadily: 0.003 → 0.016 over 500 steps
- VT=0 at step 300, VT=9/10 at step 500 — something critical happens 300-500
- **Sidecar confirmed:** VT=9/10 with sidecar, VT=0/10 without. It's real.
- Minimum useful run: 500 steps (~50 min). 300 is not enough.

### Step 4c: Fewer sidecar layers (optional, ~50 min)

Train 500 steps with 3 layers [7, 15, 23] instead of 6. Tests if fewer layers
with same gate strength work. Needs code change to support custom layer list.
Skip this if we want to go straight to 5k run — sidecar is already confirmed.

### ~~Step 5: 5000-step run~~ DONE

VT degrades with longer pure-FineWeb training. Killed at step ~3500.

| Step | Gate | AR | VT |
|------|------|-----|-----|
| 500 | 0.016 | 10/10 | 9/10 |
| 2000 | 0.062 | 9/10 | 6/10 |
| 3000 | 0.093 | 10/10 | 3/10 |

**Root cause:** FineWeb doesn't reward memory use. Gates grow but sidecar
learns language modeling, not recall.

### Step 6: Mixed-data training (memory tasks + FineWeb) — IN PROGRESS

**Implementation:** DONE
- `src/data/dataloader.py`: `_memory_task_examples()` + `build_mixed_dataloader()`
- `scripts/train_track_a_readonly.py`: `--memory_task_ratio`, `--freeze_gates_at`

**6a (IN PROGRESS):** 500 steps, 10% memory tasks, gate_warmup=0.1.
Compare with pure-FineWeb baseline (VT=9/10 at step 500).

**6a:** DONE. 500 steps, 10% mix: VT=8/10, sidecar confirmed.
**6c:** DONE. 2000 steps, 10% mix: VT non-monotonic 8→6→9→6 (vs pure FineWeb 9→6→3).
  Mixed data prevents VT collapse but doesn't fully stabilize it.

**6b:** DONE. 2000 steps, 30% mix: VT monotonically improving 7→8→8→9!
  **Best result: VT=9/10 at step 2000 (alpha=0.062)**. Pure FineWeb gave 6/10 at same gate.

**6d (NEXT):** Extend 30% mix to 5000 steps. Test if VT continues improving.
  The sidecar is actually learning — longer training should help, not hurt.

### Step 7: Gate freezing experiment (optional)

If mixed-data works, try freezing gates at the sweet spot (alpha=0.016)
to prevent the degradation seen in step 5.

---

## Backlog

- [ ] Run T8 (LLaMA-3.2-1B vanilla) and T9 (Mamba-2 1.3B) baseline evals
- [ ] Fix LR scheduler (step per training step, not per accumulation step)
- [ ] Investigate gradient checkpointing + sidecar hook compatibility
- [ ] Fix gate coupling: add small per-layer noise after warmup ends
- [ ] Put gate_alpha on its own param group with higher lr

---

## Key files

- `src/reservoir/interface.py` — CrossAttentionSidecar (tanh gate)
- `scripts/train_track_a_readonly.py` — training script
- `scripts/quick_test.py` — quick AR/VT evaluation
- `DIARY.md` — experiment log (update after every run)
