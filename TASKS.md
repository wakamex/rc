# Active Task Queue

Sequential runs on single GPU (RTX 3090, 24GB). Each run: train + ppl eval.
Full 23-benchmark eval only on winners.

## Current

- [x] 4B baseline sidecar training (running, gate=0.1, 8 layers, r=256, seq=512)

## Queue

### 4B Evals
- [ ] 4B sidecar eval (full 23-benchmark)
- [ ] 4B vanilla eval (--no-sidecar, full 23-benchmark)

### 2B Gate Warmup Sweep (most likely to fix ppl regression)
- [ ] 2B gate_target=0.01, r=256, 5 layers [3,7,11,15,23], seq=1024
- [ ] 2B gate_target=0.03, r=256, 5 layers [3,7,11,15,23], seq=1024
- [ ] 2B gate_target=0.05, r=256, 5 layers [3,7,11,15,23], seq=1024

### 2B Other Sweeps
- [ ] 2B 3 layers [3,11,23], gate=0.1, r=256, seq=1024
- [ ] 2B r=128, gate=0.1, 5 layers, seq=1024
- [ ] 2B r=512, gate=0.1, 5 layers, seq=1024
- [ ] 2B interface_lr=5e-4, gate=0.1, r=256, 5 layers, seq=1024

### 4B Gate Warmup Sweep
- [ ] 4B gate_target=0.01, r=256, 8 layers [3,7,11,15,19,23,27,31], seq=512
- [ ] 4B gate_target=0.03, r=256, 8 layers, seq=512
- [ ] 4B gate_target=0.05, r=256, 8 layers, seq=512

### 4B Other Sweeps
- [ ] 4B 3 layers [3,15,27], gate=0.1, r=256, seq=512
- [ ] 4B r=128, gate=0.1, 8 layers, seq=512
- [ ] 4B r=512, gate=0.1, 8 layers, seq=512
- [ ] 4B interface_lr=5e-4, gate=0.1, r=256, 8 layers, seq=512

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
