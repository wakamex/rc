# LRS Project Diary

Progress log tracking what was run, when, and how long it took.

**Hardware:** 1x RTX 3090 (24 GB VRAM), 32 GB RAM, 32 CPU cores

---

## 2026-03-05 (Day 1)

### 00:45–02:00 — Agent swarm completes 30 tasks (~1h15m)

Agent swarm ran 30 tasks across multiple workers, producing the initial codebase:
library code (`src/`), training scripts, eval scripts, configs, and proposals.
316 tests passing. However, the swarm output had systemic quality issues: massive
code duplication across scripts, inconsistent benchmark parameters, fabricated
placeholder results committed as real data, and 6 unmerged PRs.

### 02:35 — Merge 4 PRs with real library code

Merged PRs #17 (RIL module), #18 (DeltaNet replace), #19 (RW-Transformer),
#21 (read/write sidecar script). Closed #20 (LLaMA+RC script) and #26 (paper draft).

### 02:35–08:10 — Clean up agent swarm output (~5.5h across sessions)

8-step cleanup plan executed:
1. Merged 4 PRs, closed 2
2. Extracted shared `TextEvalAdapter` → `src/models/eval_adapter.py`
3. Extracted shared `build_dataloader` → `src/data/dataloader.py`
4. Standardized benchmark suite → `src/eval/benchmarks/suite.py`
5. Removed fabricated results (`results/track_c/latent_reasoning/`, `results/baselines/infini_attention.json`)
6. Fixed gate reports (replaced fabricated content with generation instructions)
7. Fixed silent failures in `harness.py` and `curriculum.py`
8. Replaced 15 sweep configs with generator script

Commits: `fe4357a`, `28e52d4`

### 08:10–12:20 — T7 baseline evaluation (Qwen3.5 vanilla) (~3h50m)

Ran `python scripts/eval_qwen_vanilla.py --n-examples 200`.
Fixed Qwen3.5 multimodal config loading (AutoConfig `text_config` detection)
and duplicate `skip_special_tokens` kwarg in eval adapter.

**Results:** 69 metrics across 23 benchmarks, 200 examples each.
- AssociativeRecall: ~0.77 avg
- VariableTracking: ~0.45
- Everything else: near zero
- Perplexity: 6.82
- Throughput: 17.6 tok/s
- VRAM: 1,443 MB

Output: `results/baselines/qwen35_vanilla.json`
Commit: `6b01e9f`

### 12:20–12:30 — Set up uv venv

Created `.venv` with Python 3.12 via `uv venv`. Installed project + dev deps.
Added `datasets` to `pyproject.toml` via `uv add datasets`. Lockfile generated (`uv.lock`).

### 12:30–12:45 — Debug Track A training launch (4 attempts)

1. **Missing `datasets` library** — fixed with `pip install datasets` (then `uv add`)
2. **`Can't call numpy() on Tensor that requires grad`** — added `.detach()` in `train_track_a_readonly.py:507`
3. **Gradient checkpointing + sidecar hooks conflict** — `CheckpointError: 75 vs 54 tensors`.
   Tried `use_reentrant=False`, still failed. Root cause: sidecar forward hooks modify
   hidden states, checkpointing recomputation doesn't replay them consistently.
   Fix: disabled gradient checkpointing (`configs/track_a_readonly.yaml`)
4. **CUDA OOM at batch_size=4** — used 22.6 GB / 23.5 GB without checkpointing.
   Fix: reduced `batch_size` 4→1, increased `grad_accum` 4→16 (same effective batch=16)

### 12:45–20:59 — Track A read-only sidecar training (~8.2h, 5000 steps)

`python scripts/train_track_a_readonly.py --config configs/track_a_readonly.yaml --no_wandb`

**Config:**
- Model: Qwen3.5-0.8B-Base (frozen) + LoRA (rank=16, alpha=32, q_proj+v_proj)
- ESN reservoir: 10,000 nodes, spectral_radius=0.9, leak_rate=0.5
- Sidecar: CrossAttentionSidecar at layers [3, 7, 11, 15, 19, 23]
- batch_size=1, grad_accum=16, seq_len=2048, lr=2e-4, interface_lr=1e-3
- Data: FineWeb sample-10BT (streaming)
- Gradient checkpointing: OFF (sidecar hook conflict)
- VRAM: ~23.2 GB / 24 GB

**Trainable params:** 638,976 / 753M (0.085%)

**Loss curve:**
| Step | Loss   | LR       | Elapsed |
|------|--------|----------|---------|
|   50 | 10.131 | 7.94e-06 |   277s  |
|  500 |  5.928 | 6.34e-05 |  2825s  |
| 1000 |  5.575 | 1.25e-04 |  6128s  |
| 2000 |  4.017 | 1.89e-04 |     —   |
| 3000 |  2.860 | 1.97e-04 |     —   |
| 4000 |  2.271 | 1.99e-04 |     —   |
| 5000 |  1.513 | 1.99e-04 | 29646s  |

**Final:** loss=1.4533, perplexity=4.28

**Note:** LR scheduler stepping every `grad_accum=16` steps meant warmup took
~1600 training steps instead of 100. Cosine decay barely engaged — LR stayed
near peak (1.99e-4) for most of training. Despite this, loss dropped well.

**Checkpoints saved:** step_1000, step_2000, step_3000, step_4000, step_5000, final
**Results:** `results/track_a/readonly.json`

**Rate:** ~5.9s/step average

---

## TODO (next steps)

- [ ] Run benchmark evaluation on trained Track A model vs T7 baseline
- [ ] Run T8 (LLaMA-3.2-1B vanilla) and T9 (Mamba-2 1.3B) baseline evals
- [ ] Fix LR scheduler (step per training step, not per accumulation step)
- [ ] Investigate gradient checkpointing + sidecar hook compatibility
- [ ] Commit training results and config changes
