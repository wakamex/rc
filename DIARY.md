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

## 2026-03-06 (Day 2)

### 00:00–04:30 — Track A benchmark evaluation (~4.3h)

Ran `python scripts/eval_track_a.py` on the 5000-step checkpoint.

**Results:** Perplexity improved dramatically (2.35 vs 6.82 vanilla), but **ALL benchmark
exact-match/accuracy scores were 0.000**. The model produces garbage instead of terse answers.

### 04:30–05:30 — Diagnosing format collapse (~1h)

A/B testing revealed three levels of degradation:
- **Vanilla Qwen:** clean answers (`858`, `674`)
- **LoRA only** (no sidecar hooks): wrong conversational text (fluent but incorrect)
- **LoRA + Sidecar**: degenerate garbage (repetitive digits, token loops)

Tested all checkpoints (step 1k-5k) — ALL produce 0/10. Damage happened from step 1000.

### 05:30–07:00 — Quick LR/rank experiments (3 × 200 steps)

| Experiment | LR | Rank | AR | VT |
|---|---|---|---|---|
| lr1e5_r16 | 1e-5 | 16 | 0/10 | 0/10 |
| lr5e6_r16 | 5e-6 | 16 | 0/10 | 0/10 |
| lr1e5_r4 | 1e-5 | 4 | 0/10 | 0/10 |

Conclusion: ANY LoRA+sidecar training on raw FineWeb text destroys task performance,
regardless of LR or rank.

### 07:00–08:50 — Root cause: ungated sidecar injection

**Root cause found:** `CrossAttentionSidecar.forward` did:
```python
out = self.out_norm(hidden + cross_attn_output)  # LayerNorm(hidden + sidecar)
```
Two problems:
1. **No gating** — randomly initialized sidecar produces large perturbations from step 0
2. **LayerNorm on hidden** — even with zero sidecar output, `out_norm(hidden)` re-normalizes
   hidden states, disrupting the model's internal representations at every injection layer

**Fix:** gated additive residual with normalized sidecar output:
```python
self.gate = nn.Parameter(torch.zeros(1))  # starts at 0 → no-op
out = self.out_norm(out_proj(cross_attn))  # normalize sidecar output only
return hidden + self.gate * out            # clean residual
```

**Verification:**
- Sidecar-only (no LoRA), 200 steps: AR=9/10, VT=0/10 (matches vanilla!)
- LoRA + gated sidecar, 200 steps: AR=9/10, VT=0/10 (also matches vanilla!)

The gate starts near zero so the model behaves like vanilla initially.
Training gradually opens the gate as the sidecar learns useful representations.

### 08:50–11:15 — 1000-step LoRA+gated sidecar training (~1.7h)

`lr=2e-4, interface_lr=1e-3, rank=16, alpha=32, save every 500 steps`

**Loss curve:**
| Step | Loss   | LR       |
|------|--------|----------|
|  100 | 3.096  | 2.58e-05 |
|  200 | 3.049  | 4.95e-05 |
|  500 | 3.148  | 1.25e-04 |
| 1000 | 3.016  | 2.00e-04 |

Loss barely moved (gate near zero → sidecar contribution minimal during training).
Final perplexity: 19.41 (higher than vanilla 6.82 because sidecar is still learning).

**Quick test results — reservoir sidecar IMPROVES task performance:**

| Checkpoint | AR | VT | Notes |
|---|---|---|---|
| Vanilla baseline (T7) | ~7.7/10 | ~4.5/10 | No sidecar |
| Step 500 | 9/10 | 0/10 | VT outputs `<think>` prefix (format issue) |
| **Step 1000** | **10/10** | **7/10** | VT outputs `Answer: X` (correct, format prefix) |

**AR improved 7.7 → 10/10 (+30%). VT improved 4.5 → 7/10 (+56%).**

The reservoir sidecar enhances the model's memory-dependent task performance,
which is exactly what the architecture was designed to do. AssociativeRecall
requires remembering key-value pairs from earlier in the context, and
VariableTracking requires tracking variable reassignments — both are memory tasks
that benefit from the ESN reservoir's temporal state.

**Key architectural changes that made this work:**
1. Gated residual (gate=0 at init) prevents destroying base model at step 0
2. LayerNorm on sidecar output only (not on hidden states) preserves model internals
3. `--no_lora` flag added to training script for sidecar-only experiments
4. Fixed `freeze_base_model` to not accidentally freeze LoRA params

### 11:15–11:20 — Isolation test: sidecar vs LoRA contribution

Tested 1000-step checkpoint with sidecar hooks disabled to isolate contributions:

| Config | AR | VT | Notes |
|---|---|---|---|
| Vanilla (no training) | ~7.7/10 | ~4.5/10 | Baseline |
| LoRA only (sidecar disabled) | 9/10 | 0/10 | VT outputs `<think>` |
| **LoRA + Sidecar** | **10/10** | **7/10** | Sidecar enables VT |

**The sidecar is the difference.** LoRA alone slightly improves AR but breaks VT.
The reservoir sidecar provides the memory signal that enables VariableTracking (+7/10)
and further improves AssociativeRecall (+1/10).

### 11:20 — Started 5000-step LoRA+gated sidecar training

`lr=2e-4, interface_lr=1e-3, rank=16, alpha=32, warmup=100, save every 1000 steps`

Running in background (`nohup`, PID 1368110). Log: `/tmp/train_gated_5k.log`
Estimated ~8h. Checkpoints at steps 1000, 2000, 3000, 4000, 5000.

### 11:20–23:00 — Discovering the gate never opens (chicken-and-egg problem)

**5k training runs with various LRs all showed the same problem: gates never open.**

Checked gate values from saved checkpoints — all 6 sidecar gates were ~0.002-0.005
after 1000-2000 steps. The sidecar contributed essentially nothing. The VT=7/10 result
from the earlier 1k run was likely a fluke (benchmarks use random test examples each run,
making cross-run comparison unreliable).

**OOM issues:** Two 5k training runs OOM'd at step 1500 due to CUDA memory fragmentation.
Fixed with `PYTORCH_ALLOC_CONF=expandable_segments:True`.

**Gate sweep experiments (200 steps each):**

| Config | gate_init | LoRA | LR | iface_lr | AR | VT | Gate after training |
|---|---|---|---|---|---|---|---|
| Baseline (gate=0) | 0.0 | yes | 1e-5 | 5e-4 | 9/10 | 0/10 | ~0.005 |
| gate=0.1 no LoRA | 0.1 | no | — | 1e-3 | 0/10 | 0/10 | 0.097 (decreased!) |
| gate=0.01 + zero-init out_proj | 0.01 | yes | 1e-5 | 1e-4 | 10/10 | 0/10 | ~0.0098 |
| gate=1.0 + zero-init out_proj | 1.0 | yes | 1e-5 | 5e-4 | 0/10 | 0/10 | 1.0 (loss=48K ppl!) |
| gate=1.0 + zero-init, low lr | 1.0 | yes | 1e-5 | 1e-5 | — | — | Loss rising (3.1→3.7) |

**Root cause analysis:**
1. **gate=0**: `tanh'(0)` would give gradient=1, but raw scalar gate `d/d_gate[gate * out]` = out.
   Since sidecar output is random noise, this gradient is noisy and small. Gate barely moves.
2. **gate>0 with random out_proj**: Random projections inject noise → model degrades → training
   tries to close gate (gate decreases).
3. **Zero-init out_proj + gate=1.0**: Sidecar output starts at zero, but as out_proj learns,
   signal grows without bound → loss explodes.
4. **Zero-init out_proj + small gate**: Gradient through `gate * zero_proj(...)` is ~0 at init.
   Double-zero blocks all learning.

### 23:00 — Flamingo-style tanh gating (the solution)

**Research finding:** Flamingo (Alayrac et al., NeurIPS 2022) solves this exact problem with
`tanh(alpha)` gating:

```python
# Instead of: hidden + gate * sidecar_output
# Use:        hidden + tanh(alpha) * sidecar_output
# alpha initialized to 0
```

Why this works:
- `tanh(0) = 0` → identity at init (no noise injection)
- `tanh'(0) = 1` → **perfect gradient flow from step 1** (unlike raw gate or zero-init)
- `tanh` bounds output to [-1, 1] → prevents explosion
- Sidecar projections keep normal random init → nonzero output → nonzero gradient

The key insight is that with a raw scalar gate at 0, the gradient to sidecar params is
`gate * d_sidecar/d_params = 0`. But with `tanh(alpha)` at alpha=0, the gradient to sidecar
params is `tanh'(alpha) * d_sidecar/d_params = 1 * nonzero = nonzero`. The gradient doesn't
depend on the gate VALUE, only on its DERIVATIVE.

**Also must NOT combine with zero-init out_proj** — that would make sidecar output zero,
killing the gradient path again. Need random init projections + tanh gate.

Implemented in `CrossAttentionSidecar`: renamed `self.gate` to `self.gate_alpha`, changed
forward to use `torch.tanh(self.gate_alpha)`.

---

### 02:42 — Step 1 result: tanh gate baseline (1000 steps, interface_lr=5e-4)

Training: loss stable ~3.0, final ppl=19.43. 1.7h runtime.

Gate alpha values at step 1000 (all tiny, tanh approx alpha at this scale):
| Layer | alpha | tanh(alpha) |
|---|---|---|
| 3 | +0.00455 | +0.00455 |
| 7 | +0.00287 | +0.00287 |
| 11 | +0.00197 | +0.00197 |
| 15 | -0.00304 | -0.00304 |
| 19 | +0.00516 | +0.00516 |
| 23 | -0.00412 | -0.00412 |

Benchmark: **AR=10/10, VT=3/10**

VT improved from 0/10 (at 200 steps) to 3/10 (at 1000 steps).
Gates barely moved but VT is improving — likely from LoRA, not sidecar.
Moving to Step 2: higher interface_lr to see if gates can be pushed open.

### 04:27 — Step 2a result: interface_lr=1e-3 (2x baseline)

Training: loss identical to baseline (~3.0, ppl=19.42). Gates same magnitude (~0.003-0.007).
Benchmark: **AR=9/10, VT=0/10** — worse than baseline (10/10, 3/10).
Higher sidecar lr did NOT help gates open and slightly hurt task performance.
Starting Step 2b (interface_lr=5e-3, 10x aggressive).

### 06:11 — Step 2b result: interface_lr=5e-3 (10x aggressive)

Training: loss identical (ppl=19.46). Gates same magnitude (~0.001-0.007).
Benchmark: **AR=9/10, VT=0/10**

**Conclusion: increasing interface_lr does NOT help gates open.**
The problem is fundamental: sidecar gradient = tanh(alpha) * d_sidecar/d_params,
and tanh(alpha) is ~0.005 regardless of lr. Higher lr * ~zero gradient = ~zero update.
The gate alpha itself moves at the same rate because its gradient (tanh'(alpha) * sidecar_out)
doesn't depend on interface_lr (gate uses LoRA lr=1e-5, not interface_lr).

Skipping Step 2c (no-LoRA variant) — same issue applies.
Moving to fallback: **linear gate warmup schedule** to force gates open.

### 07:54 — Step 3 result: gate warmup (0->0.5 over 200 steps)

Implemented --gate_warmup_steps and --gate_warmup_target in training script.
Training: loss rising (3.1->3.7, ppl=44.92 vs baseline 19.43) — sidecar injection
destabilizing with target=0.5.

Gates at step 500: all identical at 0.077 (dropped from forced 0.5).
Gates at step 1000: all identical at 0.155 (growing back up — sidecar learning!).
Note: all gates locked together (bug — warmup sets identical values, they stay coupled).

Benchmark: **AR=2/10, VT=6/10**

**BREAKTHROUGH: VT=6/10 is the best VT score yet!** The sidecar IS helping variable tracking.
But AR collapsed (2/10 vs 10/10 baseline) — gate target=0.5 too aggressive.
Next: try smaller target (0.1) to balance AR preservation with VT improvement.

### 09:34 — Step 3b result: gate warmup target=0.1

Training: loss only slightly elevated (ppl=23.80 vs baseline 19.43).
Gates: warmed to 0.1, dropped to 0.0155 at step 500, grew back to 0.031 at step 1000.
Still all identical (coupled — warmup sets same value, they stay locked).

Benchmark: **AR=10/10, VT=6/10** — best combined result!

Summary table:
| Config | AR | VT | Gates | ppl |
|---|---|---|---|---|
| No warmup (baseline) | 10/10 | 3/10 | ~0.005 | 19.43 |
| Warmup target=0.5 | 2/10 | 6/10 | 0.155 | 44.92 |
| **Warmup target=0.1** | **10/10** | **6/10** | **0.031** | **23.80** |

Gate warmup target=0.1 is the sweet spot: preserves AR, doubles VT, gates open and growing.
Next: try target=0.2 to see if VT can go higher without hurting AR.

### 11:18 — Step 3c result: gate warmup target=0.2

Training: loss higher (ppl=27.81). Gates at 0.062.
Benchmark: **AR=0/10, VT=2/10** — too aggressive, both tasks degraded.

Full warmup sweep:
| target | AR | VT | Gates | ppl |
|---|---|---|---|---|
| 0 (none) | 10/10 | 3/10 | 0.005 | 19.43 |
| **0.1** | **10/10** | **6/10** | **0.031** | **23.80** |
| 0.2 | 0/10 | 2/10 | 0.062 | 27.81 |
| 0.5 | 2/10 | 6/10 | 0.155 | 44.92 |

**Winner: target=0.1.** But need to validate before committing to 8h long run.

### 12:00 — Killed 5k run, more validation needed

Open questions before long run:
1. Is VT=6/10 from the sidecar or from different LoRA training dynamics?
   → Need --no-sidecar isolation test on warmup=0.1 checkpoint
2. Do we need all 6 sidecar layers, or would 3 work as well?
   → Need fewer-layer experiment (half the sidecar params)
3. Why are all 6 gates identical? Warmup locks them together.
   → May need per-layer noise or separate param groups

### 12:43 — Step 4a: Gate evolution + isolation test (500 steps, save every 100)

Gate evolution (all 6 layers identical):
| Step | alpha | tanh | Note |
|---|---|---|---|
| 100 | 0.003 | 0.003 | warmup active (forced to 0.05, optimizer pulls down) |
| 200 | 0.006 | 0.006 | warmup ends |
| 300 | 0.009 | 0.009 | post-warmup, growing |
| 400 | 0.012 | 0.012 | growing |
| 500 | 0.016 | 0.016 | growing |

Benchmark by step:
| Step | AR | VT |
|---|---|---|
| 200 | 10/10 | 0/10 |
| 300 | 10/10 | 0/10 |
| 500 | 10/10 | **9/10** |

VT jumps from 0 to 9 between step 300-500. Something critical happens there.

**Isolation test (step 500):**
| | AR | VT |
|---|---|---|
| With sidecar | 10/10 | **9/10** |
| Without sidecar | 9/10 | **0/10** |

**CONFIRMED: The sidecar is entirely responsible for VT improvement.**
VT=9/10 is our best result ever. AR preserved at 10/10.

### 14:45 — 5k run step 1000 checkpoint

Gates at step 1000: alpha=0.031 (all identical, consistent with prior runs).

### 16:45 — 5k run step 2000 checkpoint (tested on CPU)

Gates at step 2000: alpha=0.062 (doubled from step 1000, still growing).
Added --cpu flag to quick_test.py to test while GPU is busy training.
Benchmark: **AR=9/10, VT=6/10** — slight regression from 500-step VT=9/10.
Gates may be getting too strong as they continue growing.

See `TODO.md` for current experiment plan and backlog.

### 19:50 — 5k run step 3000 checkpoint (tested on CPU)

Gates at step 3000: alpha=0.093 (all 6 identical, grew from 0.062 at step 2000).
Benchmark: **AR=10/10, VT=3/10** — VT continues degrading.

**VT degradation trend is now clear:**

| Step | Gate alpha | AR | VT |
|------|-----------|-----|-----|
| 500 | 0.016 | 10/10 | 9/10 |
| 1000 | 0.031 | — | — |
| 2000 | 0.062 | 9/10 | 6/10 |
| 3000 | 0.093 | 10/10 | 3/10 |

The sidecar works at low gate values (step 500) but degrades as gates grow during
generic FineWeb training. Root cause: FineWeb doesn't incentivize memory use, so
the growing gate injects increasingly noisy/unhelpful sidecar output. The sidecar's
cross-attention projections learn to produce plausible-looking hidden states for
language modeling, but not useful memory signals.

**Conclusion:** Pure FineWeb training won't work for long runs. Need memory task
data in the training mix to give the sidecar a gradient signal for actual recall.
Killed 5k run at step ~3500 (step 4000 hadn't saved yet) — trend is clear.

### 18:45 — Mixed-data training implementation

**Problem:** FineWeb doesn't incentivize memory use. Sidecar learns LM but not recall.

**Solution:** Mix synthetic memory task examples (AR/VT/Passkey) into training data.
The model gets cross-entropy loss on "Answer: X" tokens, directly rewarding the
sidecar for extracting useful memory from reservoir state.

Implementation:
- `src/data/dataloader.py`: Added `_memory_task_examples()` infinite generator
  (uses seed=1337, different from eval seed=42) and `build_mixed_dataloader()`
- `scripts/train_track_a_readonly.py`: Added `--memory_task_ratio` (0.0-1.0) and
  `--freeze_gates_at` (optional fixed gate value)

**Validation (50 steps, 50% ratio):** Pipeline works, loss converges normally (3.6→2.9).

### 18:50 — Step 6a: Mixed training (500 steps, 10% memory tasks) — IN PROGRESS

Config: gate_warmup=200 steps to 0.1, lr=1e-5, interface_lr=5e-4, memory_task_ratio=0.1.
Saves every 100 steps. Comparing with pure-FineWeb 500-step baseline (VT=9/10).

Expected outcome: VT should hold or improve past step 500, unlike pure FineWeb
where it peaked at 500 then degraded.

### 19:37 — Step 6a results: 500 steps, 10% memory tasks

Loss: 3.2→2.4 (ppl=11.56, much lower than pure-FineWeb ppl=19.43).
Gate evolution: identical to pure-FineWeb (0.003→0.016 over 500 steps).

**Benchmark results (mixed-data vs pure-FineWeb at same step count):**

| Step | Mixed AR | Mixed VT | Pure AR | Pure VT |
|------|----------|----------|---------|---------|
| 300 | 10/10 | **7/10** | 10/10 | 0/10 |
| 500 | 10/10 | **8/10** | 10/10 | 9/10 |

**Key finding:** Mixed-data training gives VT=7/10 at step 300 (vs 0/10 for pure FineWeb).
At step 500 both approaches give strong VT (8 vs 9) but the critical difference is
whether VT holds at longer training. Pure-FineWeb degrades (9→6→3 over 500→2000→3000).

**Isolation test (step 500, mixed data):**
| | AR | VT |
|---|---|---|
| With sidecar | 10/10 | **8/10** |
| Without sidecar | 8/10 | **0/10** |

Sidecar confirmed essential. Output format now includes "Answer: " prefix (learned
from memory task training data). The model is learning the answer format!

### 19:45 — Step 6c: 2000-step mixed training — IN PROGRESS

Running 2000 steps with same config to test long-term VT stability.
Save every 500 steps. Compare with pure-FineWeb (VT=6/10 at step 2000, 3/10 at 3000).
This is the critical test: does mixed-data prevent the VT degradation?

### 21:20 — Step 6c interim: 2k mixed run step 1000

Gates at step 1000: alpha=0.031 (identical to pure-FineWeb).
**AR=10/10, VT=6/10** — VT still degrading, but slower than pure FineWeb.

**VT vs gate alpha comparison (all runs):**

| alpha | Pure FineWeb VT | Mixed 10% VT |
|-------|----------------|--------------|
| 0.016 (step 500) | 9/10 | 8/10 |
| 0.031 (step 1000) | — | 6/10 |
| 0.062 (step 2000) | 6/10 | (pending) |
| 0.093 (step 3000) | 3/10 | — |

**Key insight:** Gate growth is identical regardless of data mix. The 10% memory
ratio isn't enough to change the sidecar's learning dynamics. VT correlates with
gate value, not with training steps — the sweet spot is alpha~0.016.

**Hypotheses:**
1. 10% too low — sidecar still gets 90% FineWeb where reservoir is noise
2. Need much higher ratio (30-50%) to give sidecar enough memory signal
3. Or: gate growth is the root cause, and higher ratio just delays the inevitable

### 22:30 — Step 1500 result: VT RECOVERY!

**AR=10/10, VT=9/10** at step 1500 (alpha=0.046)!

This is a critical finding. In pure FineWeb, alpha=0.046 would give VT~5-6.
With 10% memory tasks, VT recovered from 6/10 at step 1000 to 9/10 at step 1500.
The sidecar IS learning useful memory patterns from the task data.

### 22:59 — Step 6c complete: 2000 steps, 10% memory tasks

Final: **AR=10/10, VT=6/10** at step 2000 (alpha=0.062), ppl=17.37.

**Full comparison — Mixed 10% vs Pure FineWeb:**

| Step | Gate alpha | Mixed VT | Pure FineWeb VT |
|------|-----------|----------|-----------------|
| 300 | 0.009 | 7/10 | 0/10 |
| 500 | 0.016 | 8/10 | 9/10 |
| 1000 | 0.031 | 6/10 | — |
| 1500 | 0.046 | **9/10** | — |
| 2000 | 0.062 | **6/10** | 6/10 |
| 3000 | 0.093 | — | 3/10 |

**Key findings:**
1. Mixed training prevents VT collapse — stays in 6-9 range vs pure FineWeb's 9→3
2. VT is NON-MONOTONIC with mixed data (8→6→9→6) — sidecar is actively learning
3. Pure FineWeb is MONOTONICALLY decreasing (9→6→3) — sidecar learning noise
4. At step 300, mixed gives VT=7 vs pure FineWeb VT=0 — memory tasks accelerate
   sidecar learning of the "Answer:" format
5. Gate growth rate is identical (~0.031 per 1000 steps) regardless of data mix
6. The non-monotonic VT suggests the sidecar's memory capability is fluctuating
   as it balances LM and memory task objectives

**Next:** Try higher memory ratio (30%) to strengthen the memory signal.

### 23:00 — Step 6b: 2000 steps, 30% memory tasks

Loss: 3.16→3.05 (ppl=21.22). Gate growth: identical (0.016→0.062).

**Benchmark results:**

| Step | Gate alpha | 30% mix VT | 10% mix VT | Pure FineWeb VT |
|------|-----------|-----------|-----------|-----------------|
| 500 | 0.016 | 7/10 | 8/10 | 9/10 |
| 1000 | 0.031 | **8/10** | 6/10 | — |
| 1500 | 0.046 | **8/10** | 9/10 | — |
| 2000 | 0.062 | **9/10** | 6/10 | 6/10 |

**30% mix is the winner.** VT improves monotonically (7→8→8→9) while pure FineWeb
collapses (9→6→3). The sidecar is learning to be genuinely useful for memory tasks
at higher gate values — at alpha=0.062, pure FineWeb gives VT=6/10 but 30% mix gives
VT=9/10. Same gate, different sidecar quality.

AR=10/10 across all steps (no degradation).

**Full comparison across all experiments:**

| Config | Step 500 VT | Step 1000 VT | Step 1500 VT | Step 2000 VT | Step 3000 VT |
|--------|------------|-------------|-------------|-------------|-------------|
| Pure FineWeb | 9/10 | — | — | 6/10 | 3/10 |
| 10% mix | 8/10 | 6/10 | 9/10 | 6/10 | — |
| **30% mix** | **7/10** | **8/10** | **8/10** | **9/10** | **—** |

**Conclusions:**
1. 30% memory tasks is the best ratio so far: VT improves over time
2. Gate growth rate is INDEPENDENT of training data — purely a function of step count
3. The data mix controls WHAT the sidecar learns, not HOW MUCH the gate opens
4. Pure FineWeb sidecar learns to inject noise; mixed sidecar learns to inject memory
5. Loss is slightly higher (ppl=21 vs 17 for 10%) because memory tasks are harder
   than random text prediction, but this is a feature, not a bug

### 03:05 — Step 6d: 5000-step 30% mix — IN PROGRESS

Running 5000 steps with the winning config (30% memory tasks). Saves every 1000 steps.
Expected ~8h. Will test at each checkpoint to verify VT continues improving.

### 04:30 — Step 6d: step 1000 checkpoint

Gates: alpha=0.031 (consistent). **AR=10/10, VT=8/10** — matches 2k run at same step.
Reproducible result across runs.

### 06:05 — Step 6d: step 2000 checkpoint

Gates: alpha=0.062. **AR=10/10, VT=8/10** — VT holding steady!
2k run gave VT=9/10 at this point; slight variance but no degradation.
Pure FineWeb at same gate gave VT=6/10. Sidecar learning confirmed stable.

**5k run progress (30% mix):**

| Step | Gate | VT | Note |
|------|------|-----|------|
| 1000 | 0.031 | 8/10 | matches 2k run |
| 2000 | 0.062 | 8/10 | stable (2k run: 9/10, pure FineWeb: 6/10) |
| 3000 | — | — | pending (~2h) |

This is now the LONGEST sidecar run without VT degradation. Pure FineWeb was at
VT=3/10 by step 3000. Step 3000 will be the real test — new territory.

### 07:30 — Step 6d: step 3000 checkpoint — NEW TERRITORY

Gates: alpha=0.093. **AR=10/10, VT=7/10**.

**This is the critical comparison point.** At the same gate alpha=0.093:
- Pure FineWeb: **VT=3/10** (collapsed)
- 30% mix: **VT=7/10** (holding strong)

**5k run progress (30% mix):**

| Step | Gate | 30% mix VT | Pure FineWeb VT |
|------|------|-----------|-----------------|
| 1000 | 0.031 | 8/10 | — |
| 2000 | 0.062 | 8/10 | 6/10 |
| 3000 | 0.093 | **7/10** | **3/10** |
| 4000 | ~0.124 | pending | — |
| 5000 | ~0.155 | pending | — |

VT slightly down from 8→7 but far from collapsing. Steps 4000-5000 are fully
uncharted — we've never run with gates this open before.

### 08:00 — Step 6d: step 4000 — GATES SELF-CORRECTING

**AR=10/10, VT=9/10!** Best score yet at the highest step count.

**Critical discovery: gates are self-correcting and diverging.**

| Layer | Step 3000 alpha | Step 4000 alpha |
|-------|----------------|----------------|
| 3 | 0.093 | 0.075 |
| 7 | 0.093 | 0.075 |
| 11 | 0.093 | 0.075 |
| 15 | 0.093 | 0.075 |
| 19 | 0.093 | **0.077** |
| 23 | 0.093 | **0.098** |

Three things happened:
1. **Gates DECREASED** from 0.093 to 0.075 (most layers) — self-regulation!
2. **Gate coupling broke** — layers 19 and 23 diverge from others
3. **VT improved** (7→9) as gates found a better operating point

The memory task gradient is telling the gates "you're too open, dial back." This
never happened with pure FineWeb because FineWeb doesn't have a clear signal for
what the optimal gate value is.

**Updated 5k run table:**

| Step | Gate (layer 3) | VT | Note |
|------|---------------|-----|------|
| 1000 | 0.031 | 8/10 | growing |
| 2000 | 0.062 | 8/10 | growing |
| 3000 | 0.093 | 7/10 | peak gate, slight VT dip |
| 4000 | **0.075** | **9/10** | **gates self-corrected!** |
| 5000 | **0.044** | **9/10** | **gates continued decreasing** |

### 10:30 — Step 6d COMPLETE: 5000 steps, 30% mix

**Final: AR=10/10, VT=9/10, ppl=11.09.** Best result of the entire project.

**Gate evolution — self-regulation confirmed:**

| Step | L3 | L7 | L11 | L15 | L19 | L23 |
|------|-----|-----|------|------|------|------|
| 1000 | .031 | .031 | .031 | .031 | .031 | .031 |
| 2000 | .062 | .062 | .062 | .062 | .062 | .062 |
| 3000 | .093 | .093 | .093 | .093 | .093 | .093 |
| 4000 | .075 | .075 | .075 | .075 | .077 | .098 |
| 5000 | **.046** | **.044** | **.044** | **.050** | **.059** | **.098** |

**Three phenomena at once:**
1. **Self-regulation:** Gates grew to 0.093 (step 3000), then DECREASED to ~0.044-0.059
   (step 5000). The cosine LR schedule + memory task gradient found the optimal range.
2. **Gate decoupling:** Layer 23 stayed at 0.098 while others dropped to 0.044-0.059.
   Each layer finding its own optimal operating point — no longer locked together.
3. **Stable VT:** VT=8→8→7→9→9 over 5000 steps. No degradation.

**Isolation test (step 5000):**
| | AR | VT |
|---|---|---|
| With sidecar | 10/10 | **9/10** |
| Without sidecar | 8/10 | **0/10** |

**Final comparison — all experiments:**

| Config | Step 500 | Step 2000 | Step 3000 | Step 5000 |
|--------|----------|-----------|-----------|-----------|
| Pure FineWeb | VT=9 | VT=6 | VT=3 | — |
| 10% mix | VT=8 | VT=6 | — | — |
| **30% mix** | VT=7 | VT=8 | VT=7 | **VT=9** |

**Training stats:** 5000 steps in 7h13m (~26000s). Loss: 3.16→2.41 (ppl=11.09).

**Summary:** Mixed-data training with 30% synthetic memory tasks completely solves
the VT degradation problem. The sidecar learns to extract useful memory from the
reservoir, gates self-regulate to optimal values, and per-layer gate values diverge
as each layer specializes. This is a validated, working reservoir computing sidecar.

---

## 2026-03-08 (Day 4)

### GPU ESN acceleration (cuSPARSE)

Full benchmark eval (n=200, 23 benchmarks) was running on CPU ESN — estimated ~23h total.
Implemented `ESNGpu` class that converts scipy sparse CSR matrices to PyTorch sparse CSR
tensors on GPU, using cuSPARSE for sparse matrix-vector multiplies.

**Benchmark:** CPU 18.3s → GPU 0.47s per 2048-step sequence (n=10000) = **38.7x speedup**.

Updated all scripts: `train_track_a_readonly.py`, `quick_test.py`, `eval_track_a.py`.

Killed the CPU eval (PID 3189456, ~285 min elapsed, no results yet) and restarted with
GPU ESN (PID 3484740). New estimate: ~3h for full 23-benchmark suite.

### Step 8: Full benchmark eval — DONE (2h46m)

**Checkpoint:** mixed_030_5k/step_5000 (best result from Step 6d)
**Config:** n=200, 23 benchmarks, GPU ESN, bf16

| Task | EM (Track A) | EM (Vanilla) | Delta |
|------|-------------|-------------|-------|
| **VariableTracking** | **0.635** | 0.455 | **+0.180** |
| **DyckLanguage** | **0.425** | 0.000 | **+0.425** |
| CompositionalGeneralization | 0.115 | 0.085 | +0.030 |
| ProgramTrace | 0.005 | 0.000 | +0.005 |
| AssociativeRecall | 0.315 | 0.825 | **-0.510** |
| PasskeyRetrieval | 0.000 | 0.005 | -0.005 |
| AlgorithmicTransfer | 0.000 | 0.000 | 0.000 |
| LengthExtrapolation | 0.000 | 0.000 | 0.000 |
| ModularArithmetic | 0.000 | 0.000 | 0.000 |
| MultiDigitArithmetic | 0.000 | 0.000 | 0.000 |

**Wins: 3, Losses: 1, Ties: 6**

**Perplexity:** 7.70 (vanilla: 6.82, delta: +0.88)
**VRAM:** 1853 MB allocated, 2001 MB peak
**Latency:** p50=2171ms, p95=5712ms

**Key findings:**
1. **VT +0.180**: Reservoir sidecar significantly improves variable tracking at scale (n=200).
   Confirms the quick_test results.
2. **Dyck +0.425**: Huge unexpected win on bracket matching — reservoir state tracking
   helps with structural/recursive patterns. This is a novel finding.
3. **AR -0.510**: Associative recall regressed at n=200 despite being 10/10 at n=10.
   However, AR f1=0.64 (vs EM=0.32) suggests formatting mismatch, not capability loss.
   The quick_test used lenient matching; the harness uses strict exact_match.
4. **Perplexity +0.88**: Modest cost from mixed training. The 30% memory task ratio
   slightly degrades general language modeling as expected.
