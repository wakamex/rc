# Latent Reservoir Scratchpads (LRS)

**Bidirectional Echo State Networks as external working memory and cheap latent reasoning substrates for LLMs.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Overview

The Latent Reservoir Scratchpad (LRS) attaches a fixed-weight [Echo State Network (ESN)](https://en.wikipedia.org/wiki/Echo_state_network) to a Transformer LLM as an external working memory.

Key properties:
- **O(1) memory footprint** regardless of context length (vs O(L) KV-cache)
- **Cheap latent reasoning**: reservoir evolves K sub-steps between tokens via sparse matrix-vector multiplies — ~100x cheaper than Coconut-style hidden-state recirculation
- **Only interface parameters are trained** (ReadProjection, WriteHead, optional LoRA adapters); reservoir weights remain fixed
- **DeltaNet synergy**: Qwen3.5's Gated DeltaNet blocks share a recurrent inductive bias with ESN dynamics, predicted to reduce distributional mismatch

Three integration depths are studied:
- **Track A**: Bolt-on external sidecar (frozen LLM + LoRA + cross-attention sidecar)
- **Track B**: Surgically inserted Reservoir Interaction Layers (RIL) with dual fast/slow timescale reservoirs
- **Track C**: From-scratch RW-Transformer with three parallel branches (attention, MLP, dual-reservoir workspace)

See [`COLLABORATIVE_PROPOSAL.md`](COLLABORATIVE_PROPOSAL.md) for the full research design.
See [`docs/paper/`](docs/paper/) for the draft LaTeX paper and figure scripts.

---

## Installation

### Requirements

- Python 3.10+
- CPU-only evaluation works out of the box; GPU (CUDA) is required for model training

### Quickstart (CPU evaluation)

```bash
git clone https://github.com/[repo]/lrs.git
cd lrs
pip install -e ".[dev]"
```

### GPU training dependencies

```bash
pip install -e .
# Ensure PyTorch CUDA is installed:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Instantiate an ESN reservoir

```python
from src.reservoir import ESN
from src.types import ReservoirConfig

cfg = ReservoirConfig(size=1000, spectral_radius=0.9, leak_rate=0.3)
esn = ESN(cfg, input_dim=64)

import numpy as np
x = np.random.randn(64).astype(np.float32)
state = esn.step(x)           # shape (1000,)
states = esn.forward(np.random.randn(100, 64).astype(np.float32))  # shape (100, 1000)
```

### Run the ESN cross-attention sidecar

```python
import torch
from src.reservoir import ESN, CrossAttentionSidecar
from src.types import ReservoirConfig

cfg = ReservoirConfig(size=512)
esn = ESN(cfg, input_dim=512)

sidecar = CrossAttentionSidecar(hidden_dim=512, reservoir_dim=512, num_heads=8)

hidden = torch.randn(1, 10, 512)          # (batch, seq, hidden)
reservoir_states = esn.forward(           # (seq, 512)
    torch.randn(10, 512).numpy())
out = sidecar(hidden, reservoir_states)   # (1, 10, 512)
```

### Dual fast/slow reservoir

```python
from src.reservoir import MultiReservoir, MultiReservoirConfig
from src.types import ReservoirConfig

cfg = MultiReservoirConfig(
    fast=ReservoirConfig(size=200, spectral_radius=0.9, leak_rate=0.9),
    slow=ReservoirConfig(size=200, spectral_radius=0.5, leak_rate=0.1),
)
mr = MultiReservoir(cfg, input_dim=64)

import numpy as np
state = mr.step(np.random.randn(64).astype(np.float32))  # shape (400,)
```

### Run the benchmark harness (standalone)

```bash
# Zero-shot exact match on all tasks
python scripts/eval_qwen_vanilla.py

# Evaluate the Infini-attention baseline
python scripts/train_infini_attention.py
python scripts/eval_infini_attention.py

# Track A read-only training
python scripts/train_track_a_readonly.py

# Gate A analysis (aggregates all Track A results)
python scripts/gate_a_analysis.py
```

The harness produces self-contained JSON in `results/` with full reproducibility
metadata (git hash, timestamp, config).

---

## Reproducing Paper Results

All Track A evaluations require GPU; the harness itself runs on CPU.

| Task | Script | Output |
|------|--------|--------|
| T7: Qwen vanilla baseline | `python scripts/eval_qwen_vanilla.py` | `results/baselines/qwen_vanilla.json` |
| T8: YaRN context extension | `python scripts/eval_qwen35_yarn.py` | `results/baselines/qwen_yarn.json` |
| T9: Mamba-2 baseline | `python scripts/eval_mamba2.py` | `results/baselines/mamba2.json` |
| T10: LLaMA-3.2 baseline | `python scripts/eval_llama.py` | `results/baselines/llama.json` |
| T11: Infini-attention | `python scripts/train_infini_attention.py && python scripts/eval_infini_attention.py` | `results/baselines/infini_attention.json` |
| T14: Track A read-only | `python scripts/train_track_a_readonly.py` | `results/track_a/readonly.json` |
| T16: HP sweep | `python scripts/sweep_reservoir_hp.py` | `results/track_a/sweep/` |
| T29: Ablations | `python scripts/ablations.py` | `results/ablations/` |
| T29: Efficiency | `python scripts/efficiency_benchmark.py` | `results/efficiency/` |
| Gate A report | `python scripts/gate_a_analysis.py` | `docs/gate_a_report.md` |

### Generate paper figures

```bash
python docs/paper/figures/generate_figures.py
```

### Compile the paper

```bash
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Project Structure

```
lrs/
├── src/
│   ├── reservoir/
│   │   ├── esn.py              ESN implementation (sparse Erdos-Renyi/Watts-Strogatz)
│   │   ├── interface.py        ReadProjection, WriteHead, CrossAttentionSidecar, FiLMModulation
│   │   └── multi_reservoir.py  Dual fast/slow timescale MultiReservoir
│   ├── models/
│   │   ├── loader.py           load_model(), MODEL_REGISTRY, get_lora_targets()
│   │   └── infini_attention.py Infini-attention baseline implementation
│   ├── eval/
│   │   ├── harness.py          evaluate(), EvalConfig, JSON output with reproducibility metadata
│   │   └── benchmarks/         PasskeyRetrieval, VariableTracking, AssociativeRecall, CompGen
│   ├── training/
│   │   ├── lora_trainer.py     LoRATrainer, LoRATrainingConfig
│   │   └── curriculum.py       CurriculumConfig, CurriculumDataPipeline
│   ├── data/
│   │   └── chaos.py            Lorenz-63, Mackey-Glass, Kuramoto-Sivashinsky generators
│   └── types.py                ReservoirConfig, BenchmarkExample, EvalResult, protocols
├── configs/
│   ├── lora_training.yaml      LoRA + optimizer configuration template
│   ├── track_a_readonly.yaml   Track A read-only training config
│   └── sweep/                  Reservoir HP sweep grid configs
├── scripts/
│   ├── train_track_a_readonly.py
│   ├── train_infini_attention.py
│   ├── eval_*.py               Per-baseline evaluation scripts
│   ├── sweep_reservoir_hp.py   Hyperparameter sweep runner
│   ├── ablations.py            Ablation study runner (T29)
│   ├── efficiency_benchmark.py Throughput & latency benchmarks (T29)
│   └── gate_a_analysis.py      Gate A pass/fail report generator
├── docs/
│   ├── gate_a_report.md        Auto-generated Gate A report
│   └── paper/
│       ├── main.tex            Full paper LaTeX source
│       ├── references.bib      Bibliography
│       └── figures/
│           └── generate_figures.py  Figure generation script
├── tests/                      pytest test suite
├── results/                    Evaluation outputs (JSON)
├── COLLABORATIVE_PROPOSAL.md   Full research design and gate criteria
└── pyproject.toml
```

---

## Configuration

### `configs/lora_training.yaml`

Main LoRA + optimizer configuration. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `lora_rank` | 16 | LoRA rank r (4–64) |
| `lora_alpha` | 32 | LoRA scaling factor (typically 2×rank) |
| `lora_lr` | 2e-4 | Learning rate for LoRA adapter params |
| `interface_lr` | 1e-3 | Learning rate for reservoir interface params |
| `num_training_steps` | 5000 | Total training steps |
| `gradient_checkpointing` | true | Enable gradient checkpointing |
| `checkpoint_dir` | `checkpoints/` | Where to save adapter weights |

### `configs/track_a_readonly.yaml`

Track A read-only phase configuration (reservoir hyperparameters + training settings).

### `configs/sweep/`

Grid of 14 YAML files for the reservoir hyperparameter sweep (T16), varying:
- Reservoir size: 500, 2000, 10000, 50000
- Leak rate: 0.1, 0.3, 0.7, 1.0
- Spectral radius: 0.5, 0.9, 0.99, 1.1
- Topology: Erdős–Rényi, Watts-Strogatz small-world

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover ESN dynamics (echo state property, batch step), interface modules,
multi-reservoir, benchmark generators, eval harness, curriculum, and training loop.

---

## Negative Result Commitment

Per the pre-registered proposal, this codebase and paper report all gate outcomes
regardless of direction.  If Gate A criteria are not met, `docs/gate_a_report.md`
and `docs/paper/main.tex` Section 5 document the failure mode and analysis.

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{lrs2026,
  title  = {Latent Reservoir Scratchpads for LLMs: Bidirectional Echo State
            Networks as External Working Memory and Cheap Latent Reasoning Substrates},
  author = {Anonymous},
  year   = {2026},
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
