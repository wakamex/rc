# Active Task Queue

Sequential runs on single GPU (RTX 3090, 24GB). Each run: train + ppl eval.
Full 23-benchmark eval only on winners.

## Current

- [x] 4B baseline sidecar training (done, gate=0.1, 8 layers, r=256, seq=512, loss=2.06)

## Queue

### 4B Evals
- [x] 4B sidecar eval: avg_em=0.557, ppl=2.32
- [x] 4B vanilla eval: avg_em=0.553, ppl=2.28. **Sidecar is a no-op at 4B (+0.004 avg_em, +1.8% ppl)**

### 2B Gate Warmup Sweep (most likely to fix ppl regression)
- [x] 2B gate_target=0.01: ppl=4.11 (+2.5%), avg_em=0.377 (+14%)
- [x] 2B gate_target=0.03: ppl=4.09 (+2.0%), avg_em=0.378 (+14%). **Best tasks**
- [x] 2B gate_target=0.05: ppl=4.04 (+0.7%), avg_em=0.373 (+12%). **Best ppl**

### 2B Other Sweeps
- [x] 2B 3 layers [3,11,23], gate=0.03: ppl=3.93 (-2.0%!), avg_em=0.367 (+11%). **ppl below vanilla!**
- [x] 2B r=128, gate=0.03, 5 layers: ppl=4.02 (+0.2%), avg_em=0.387 (+17%). **NEW BEST 2B — near-zero ppl cost!**
- [x] 2B r=512, gate=0.03, 5 layers: ppl=4.04 (+0.7%), avg_em=0.358 (+8%). Worse than r=128
- [x] 2B interface_lr=5e-4, gate=0.03, r=256: ppl=3.95 (-1.5%), avg_em=0.354 (+7%). Lower lr hurts tasks

### 4B Sweeps — SKIPPED
Sidecar is a no-op at 4B (+0.004 avg_em, +1.8% ppl). Sweeping hyperparameters won't help — the model is already too capable for r=256 ESN to add value. Would need fundamentally different approach (larger reservoir? different injection mechanism?) to matter at this scale.

### Winners
- [x] Best 2B: r=128, gate=0.03, 5L, ilr=1e-3 → ppl=4.02 (+0.2%), avg_em=0.387 (+17%)
- [x] 4B: sidecar is no-op (+0.7% avg_em, +1.8% ppl). No sweep needed.

## 2B Sweep Complete

All 7 configs tested. Key findings:
- **Smaller reservoir (r=128 vs 256) is better** at 2B — same pattern as 0.8B
- **Lower gate (0.03 vs 0.1) halves ppl cost** while keeping task gains
- **Fewer layers (3L) improves ppl** below vanilla but costs some tasks
- **Lower interface_lr (5e-4) hurts tasks** — interface needs to learn aggressively
- **Best 2B config fixes the +4% ppl problem** completely: now +0.2% at +17% tasks

## Common Training Command

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_track_a_readonly.py \
  --no_wandb --max_steps 5000 --batch_size 1 --grad_accum 16 \
  --memory_task_ratio 0.3 --warmup_steps 15 --gate_warmup_steps 10 \
  --gate_warmup_target <GATE> \
  --reservoir_size <R> --sidecar_type gated_linear --gate_init 0.0 \
  --model_name <MODEL> \
  --sidecar_layers <LAYERS> \
  --max_seq_length <SEQ> \
  --interface_lr <ILR> \
  --output_dir checkpoints/scaling/<NAME> --save_interval 9999 \
  --log_interval 100 > run.log 2>&1
```

## Eval Commands

Sidecar eval:
```bash
python scripts/eval_track_a.py --checkpoint checkpoints/scaling/<NAME>/final \
  --n-examples 50 --model-name <MODEL> --output results/scaling/<NAME>.json > eval_<NAME>.log 2>&1
```

Vanilla eval:
```bash
python scripts/eval_track_a.py --checkpoint checkpoints/scaling/<NAME>/final \
  --n-examples 50 --model-name <MODEL> --no-sidecar --output results/scaling/<NAME>_vanilla.json > eval_<NAME>_vanilla.log 2>&1
```

## Ppl-Only Quick Check

After each training run, compute ppl before deciding whether to do full eval:
```bash
grep "Perplexity:" eval_<NAME>.log
```
If ppl > vanilla + 2%, skip full eval and move to next config.

## Decision Criteria

- **Gate sweep:** pick the gate_target that minimizes |Δppl| while keeping avg_em within 0.01 of best
- **Layer sweep:** if fewer layers reduces ppl cost without killing tasks, use fewer layers at scale
- **Reservoir sweep:** if r/hidden ratio matters, optimal r should scale with hidden_size
- **Interface_lr:** if lower lr helps, the interface is overwriting too aggressively at scale
