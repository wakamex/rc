#!/bin/bash
set -e
PYTHON=".venv/bin/python"

run_experiment() {
    local name="$1" gate_init="$2" interface_lr="$3" use_lora="$4"
    local dir="checkpoints/sweep_${name}"
    local lora_flag=""
    [ "$use_lora" = "no" ] && lora_flag="--no_lora"

    echo ""
    echo "========================================"
    echo "EXP: $name (gate_init=$gate_init, iface_lr=$interface_lr, lora=$use_lora)"
    echo "========================================"

    rm -rf "$dir"
    PYTORCH_ALLOC_CONF=expandable_segments:True $PYTHON scripts/train_track_a_readonly.py \
        --no_wandb \
        --model_name qwen3.5-0.8b \
        --dtype bfloat16 \
        --max_steps 200 \
        --lr 1e-5 \
        --interface_lr "$interface_lr" \
        --lora_rank 16 \
        --lora_alpha 32.0 \
        --gate_init "$gate_init" \
        --batch_size 1 \
        --grad_accum 16 \
        --max_seq_length 2048 \
        --warmup_steps 20 \
        --log_interval 50 \
        --output_dir "$dir" \
        --save_interval 9999 \
        $lora_flag 2>&1 | grep -E "(step=|Done\.|Optimizer)"

    # Check gate values
    echo "--- Gate values ---"
    $PYTHON -c "
import torch
sd = torch.load('$dir/final/sidecar_weights.pt', map_location='cpu')
gates = {k: round(v.item(), 6) for k, v in sd.items() if 'gate' in k}
for k, v in sorted(gates.items()):
    print(f'  {k}: {v}')
"

    # Quick test
    echo "--- Task test ---"
    $PYTHON scripts/quick_test.py "$dir/final" 2>&1 | grep -E "(SCORE|expected)"

    echo "--- $name DONE ---"
}

echo "Start: $(date)"

# Experiment 1: gate_init=0.1, no LoRA (cheapest, tests if gate helps sidecar learn)
run_experiment "g01_nolora" "0.1" "1e-3" "no"

# Experiment 2: gate_init=0.1, with LoRA
run_experiment "g01_lora" "0.1" "5e-4" "yes"

# Experiment 3: gate_init=0.01, with LoRA (gentler)
run_experiment "g001_lora" "0.01" "5e-4" "yes"

# Experiment 4: gate_init=0.5, no LoRA (aggressive, will it survive?)
run_experiment "g05_nolora" "0.5" "1e-3" "no"

echo ""
echo "All done: $(date)"
