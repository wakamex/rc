#!/usr/bin/env bash
# cloud_setup.sh — Environment setup for Track C pre-training on A100 cloud instances.
#
# Supported providers:
#   - RunPod  (https://www.runpod.io)  — recommended spot pricing ~$0.50/h A100
#   - Lambda  (https://lambdalabs.com) — alternative spot pricing
#
# Usage:
#   bash scripts/cloud_setup.sh            # full setup on a fresh instance
#   bash scripts/cloud_setup.sh --test     # verify environment without training
#
# After setup, launch training with:
#   python scripts/train_track_c.py --config configs/track_c_pretrain.yaml
#
# Requirements:
#   - Ubuntu 20.04 / 22.04 with CUDA 12+ driver
#   - Python 3.10+
#   - ~500 GB free disk (checkpoint + data cache)
#   - A100 80GB (single GPU; model fits in BF16 with gradient checkpointing)
#
# Environment variables (set before running):
#   WANDB_API_KEY   — Weights & Biases API key for logging
#   HF_TOKEN        — HuggingFace token (if needed for private datasets)
#   REPO_URL        — Git repo URL to clone (default: local copy assumed)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"
REPO_DIR="${REPO_DIR:-/workspace/rc}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/track_c}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-/workspace/hf_cache}"
LOG_FILE="${LOG_FILE:-/workspace/train_track_c.log}"
TEST_MODE=false

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

for arg in "$@"; do
    case "$arg" in
        --test)  TEST_MODE=true ;;
        --help)
            echo "Usage: bash scripts/cloud_setup.sh [--test]"
            echo "  --test   Verify environment only, do not start training."
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

check_gpu() {
    info "Checking GPU..."
    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi not found. Is the NVIDIA driver installed?"
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ "$gpu_mem" -lt 70000 ]]; then
        warn "GPU has <70 GB VRAM ($gpu_mem MiB). A100 80GB recommended for Track C."
    else
        info "GPU VRAM: ${gpu_mem} MiB — sufficient."
    fi
}

check_disk() {
    info "Checking disk space..."
    local free_gb
    free_gb=$(df -BG /workspace 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo "0")
    if [[ "$free_gb" -lt 200 ]]; then
        warn "Less than 200 GB free disk space ($free_gb GB). Consider using a larger volume."
    else
        info "Disk free: ${free_gb} GB — OK."
    fi
}

install_system_deps() {
    info "Installing system dependencies..."
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        git curl wget build-essential \
        libssl-dev libffi-dev \
        python3-dev python3-venv python3-pip \
        htop tmux nvtop \
        2>/dev/null || warn "Some system packages may not have installed (non-root?)."
}

setup_venv() {
    info "Setting up Python virtual environment at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    "$PIP" install --upgrade pip wheel setuptools -q
    info "Python: $(python --version)"
    info "Pip:    $(pip --version)"
}

install_python_deps() {
    info "Installing Python dependencies..."
    # Core ML stack
    pip install --upgrade \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 \
        -q

    # HuggingFace ecosystem
    pip install --upgrade \
        transformers \
        datasets \
        tokenizers \
        accelerate \
        peft \
        -q

    # Logging and monitoring
    pip install --upgrade \
        wandb \
        tqdm \
        -q

    # Scientific stack
    pip install --upgrade \
        numpy \
        scipy \
        -q

    # YAML config support
    pip install --upgrade pyyaml -q

    info "Python dependencies installed."
}

install_project() {
    info "Installing project (editable)..."
    if [[ -f "$REPO_DIR/pyproject.toml" ]]; then
        pip install -e "$REPO_DIR" -q
        info "Project installed from $REPO_DIR"
    else
        warn "pyproject.toml not found at $REPO_DIR. Skipping project install."
    fi
}

configure_hf_cache() {
    info "Configuring HuggingFace cache at $DATA_CACHE_DIR..."
    mkdir -p "$DATA_CACHE_DIR"
    export HF_HOME="$DATA_CACHE_DIR"
    export HF_DATASETS_CACHE="$DATA_CACHE_DIR/datasets"
    export TRANSFORMERS_CACHE="$DATA_CACHE_DIR/models"

    # Write to shell profile for persistence
    {
        echo "export HF_HOME=$DATA_CACHE_DIR"
        echo "export HF_DATASETS_CACHE=$DATA_CACHE_DIR/datasets"
        echo "export TRANSFORMERS_CACHE=$DATA_CACHE_DIR/models"
    } >> ~/.bashrc

    if [[ -n "${HF_TOKEN:-}" ]]; then
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
            info "HuggingFace login skipped (CLI not yet available)."
    fi
}

configure_wandb() {
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        info "Configuring Weights & Biases..."
        wandb login "$WANDB_API_KEY" --relogin 2>/dev/null || \
            python -c "import wandb; wandb.login(key='$WANDB_API_KEY')" 2>/dev/null || \
            warn "wandb login failed — will attempt at training start."
    else
        warn "WANDB_API_KEY not set. Set it to enable W&B logging."
    fi
}

create_dirs() {
    info "Creating output directories..."
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p /workspace/results/track_c
}

verify_environment() {
    info "Verifying environment..."
    python - <<'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {mem_gb:.1f} GB")
    if mem_gb < 70:
        print("WARNING: GPU has <70 GB VRAM — Track C requires A100 80GB")

import transformers
print(f"transformers: {transformers.__version__}")

import datasets
print(f"datasets: {datasets.__version__}")

try:
    import peft
    print(f"peft: {peft.__version__}")
except ImportError:
    print("WARNING: peft not installed")

try:
    import wandb
    print(f"wandb: {wandb.__version__}")
except ImportError:
    print("WARNING: wandb not installed")

try:
    from src.training.curriculum import CurriculumConfig, CurriculumDataPipeline
    print("src.training.curriculum: OK")
except ImportError as e:
    print(f"WARNING: could not import curriculum ({e})")

try:
    from src.reservoir.multi_reservoir import MultiReservoir
    print("src.reservoir.multi_reservoir: OK")
except ImportError as e:
    print(f"WARNING: could not import MultiReservoir ({e})")

try:
    from src.models.rw_transformer import build_rw_transformer
    print("src.models.rw_transformer: OK")
except ImportError as e:
    print(f"NOTE: rw_transformer not yet available ({e}) — T24 must be implemented first")

print("\nEnvironment check complete.")
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    info "=== Track C Cloud Setup ==="
    info "REPO_DIR=$REPO_DIR"
    info "CHECKPOINT_DIR=$CHECKPOINT_DIR"
    info "TEST_MODE=$TEST_MODE"
    echo

    check_gpu
    check_disk

    # Only install if not just testing
    if [[ "$TEST_MODE" == false ]]; then
        install_system_deps
        setup_venv
        install_python_deps

        # Clone repo if not already present
        if [[ ! -d "$REPO_DIR" ]]; then
            if [[ -n "${REPO_URL:-}" ]]; then
                info "Cloning repo from $REPO_URL..."
                git clone "$REPO_URL" "$REPO_DIR"
            else
                warn "REPO_URL not set and $REPO_DIR not found. Ensure repo is at $REPO_DIR."
            fi
        fi

        install_project
        configure_hf_cache
        configure_wandb
        create_dirs
    fi

    verify_environment

    if [[ "$TEST_MODE" == true ]]; then
        info "=== Environment verification complete. ==="
        exit 0
    fi

    # ---------------------------------------------------------------------------
    # Pre-fetch dataset (optional, speeds up training start)
    # ---------------------------------------------------------------------------
    info "Pre-fetching FineWeb sample (this may take a few minutes)..."
    python - <<'EOF' || warn "Dataset pre-fetch failed; training will fetch on the fly."
from datasets import load_dataset
print("Loading FineWeb sample-10BT (streaming test)...")
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
sample = next(iter(ds))
print(f"OK — first sample text length: {len(sample['text'])} chars")
EOF

    # ---------------------------------------------------------------------------
    # Launch training
    # ---------------------------------------------------------------------------
    info "=== Setup complete. Launching Track C pre-training... ==="
    info "Logs: $LOG_FILE"
    info "Checkpoints: $CHECKPOINT_DIR"
    echo

    # Use tmux to keep the process alive if SSH disconnects
    SESSION="track_c_pretrain"
    if command -v tmux &>/dev/null; then
        info "Starting training in tmux session '$SESSION'..."
        tmux new-session -d -s "$SESSION" \
            "cd $REPO_DIR && \
             source $VENV_DIR/bin/activate && \
             python scripts/train_track_c.py \
               --config configs/track_c_pretrain.yaml \
               --output_dir $CHECKPOINT_DIR \
               --results_file /workspace/results/track_c/pretrain.json \
             2>&1 | tee $LOG_FILE"
        info "Training started in tmux session '$SESSION'."
        info "Attach with: tmux attach -t $SESSION"
        info "Monitor with: tail -f $LOG_FILE"
    else
        warn "tmux not available — running in foreground (beware of SSH disconnects)."
        cd "$REPO_DIR"
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
        python scripts/train_track_c.py \
            --config configs/track_c_pretrain.yaml \
            --output_dir "$CHECKPOINT_DIR" \
            --results_file /workspace/results/track_c/pretrain.json \
            2>&1 | tee "$LOG_FILE"
    fi
}

main "$@"
