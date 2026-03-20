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
- [ ] 2B gate_target=0.01, r=256, 5 layers [3,7,11,15,23], seq=1024
- [ ] 2B gate_target=0.03, r=256, 5 layers [3,7,11,15,23], seq=1024
- [ ] 2B gate_target=0.05, r=256, 5 layers [3,7,11,15,23], seq=1024

### 2B Other Sweeps
- [ ] 2B 3 layers [3,11,23], gate=0.1, r=256, seq=1024
- [ ] 2B r=128, gate=0.1, 5 layers, seq=1024
- [ ] 2B r=512, gate=0.1, 5 layers, seq=1024
- [ ] 2B interface_lr=5e-4, gate=0.1, r=256, 5 layers, seq=1024

### 4B Sweeps — SKIPPED
Sidecar is a no-op at 4B (+0.004 avg_em, +1.8% ppl). Sweeping hyperparameters won't help — the model is already too capable for r=256 ESN to add value. Would need fundamentally different approach (larger reservoir? different injection mechanism?) to matter at this scale.

### Winners
- [ ] Full 23-benchmark eval on best 2B config
- [ ] Full 23-benchmark eval on best 4B config

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
