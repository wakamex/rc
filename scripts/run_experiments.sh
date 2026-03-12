#!/bin/bash
set -e
PYTHON=".venv/bin/python"

run_experiment() {
    local name="$1" lr="$2" rank="$3"
    local iface_lr=$(python3 -c "print($lr * 5)")
    local dir="checkpoints/exp_${name}"

    echo ""
    echo "========================================"
    echo "EXP: $name (lr=$lr, rank=$rank, 200 steps)"
    echo "========================================"

    $PYTHON scripts/train_track_a_readonly.py \
        --no_wandb \
        --model_name qwen3.5-0.8b \
        --dtype bfloat16 \
        --max_steps 200 \
        --lr "$lr" \
        --interface_lr "$iface_lr" \
        --lora_rank "$rank" \
        --lora_alpha "$(python3 -c "print($rank * 2.0)")" \
        --batch_size 1 \
        --grad_accum 16 \
        --max_seq_length 2048 \
        --warmup_steps 20 \
        --log_interval 50 \
        --output_dir "$dir" \
        --save_interval 9999 2>&1

    echo "--- Testing $name ---"
    $PYTHON scripts/quick_test.py "$dir/final" 2>&1

    echo "--- $name DONE ---"
}

echo "Start: $(date)"
run_experiment "lr1e5_r16" "1e-5" "16"
run_experiment "lr5e6_r16" "5e-6" "16"
run_experiment "lr1e5_r4"  "1e-5" "4"
echo "All done: $(date)"
