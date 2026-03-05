# LRS Implementation Plan

**Project:** Latent Reservoir Scratchpads for LLMs
**Spec:** [`COLLABORATIVE_PROPOSAL.md`](COLLABORATIVE_PROPOSAL.md)
**Original proposals:** [`docs/proposals/`](docs/proposals/)

---

## 1. Compute Constraints

### 1.1 Local Hardware

| Resource | Spec | Implication |
|----------|------|-------------|
| GPU | 1× RTX 3090 (24 GB VRAM) | All local GPU tasks are **serialized** |
| CPU | Unlimited async workers | All CPU-only tasks run in **full parallel** |
| RAM | Assumed ≥ 32 GB | Reservoir matrices (sparse, ≤ 100 MB) fit easily |
| Storage | Assumed ≥ 500 GB | Pre-training corpus (~100–200 GB) fits |

### 1.2 What Fits on the 3090

| Workload | VRAM Estimate | Fits? |
|----------|---------------|-------|
| Qwen3.5-0.8B inference (FP16) | ~2 GB | Yes |
| Qwen3.5-0.8B LoRA fine-tune (BF16, grad ckpt) | ~6–8 GB | Yes |
| Qwen3.5-0.8B full fine-tune (BF16, grad ckpt) | ~10–14 GB | Yes |
| LLaMA-3.2-1B LoRA fine-tune | ~6–8 GB | Yes |
| Mamba-2 1.3B inference | ~3 GB | Yes |
| ESN reservoir (50K nodes, 1% sparse) | ~100 MB | Yes |
| Model + reservoir + LoRA training combined | ~8–10 GB | Yes |
| Track C from-scratch 0.8B (AdamW full) | ~18–22 GB | Tight — use cloud |

### 1.3 What Needs Cloud

| Workload | Why | Estimated Cloud Hours |
|----------|-----|----------------------|
| Track C pre-training (0.8B from scratch) | AdamW optimizer states + speed | ~500 h A100 |
| Track C latent reasoning sweeps | Parallel K-sweep configs | ~50 h A100 |
| HP sweep overflow (optional) | Speed up Track A wall-clock | ~40 h A100 |
| **Total cloud** | | **~590 h** |

At RunPod A100 spot (~$0.50/h): **~$295**. At on-demand (~$0.80/h): **~$472**. Both well under $1k.

### 1.4 Data Requirements

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| Fine-tuning mix | Track A/B adapter training | ~500 M tokens | FineWeb subset + synthetic |
| Synthetic benchmarks | Evaluation across all tracks | Generated on-the-fly | T5 generators |
| Chaotic systems | Lorenz, Mackey-Glass, KS trajectories | ~1 GB | T6 generators |
| Pre-training corpus | Track C from-scratch | ~20–50 B tokens (~100 GB) | FineWeb / RedPajama |
| Eval benchmarks | GSM8K, ARC, Dyck, passkey, etc. | ~1 GB | Public datasets |

---

## 2. Dependency Graph

```
            ┌─ T1: Scaffolding ──────────────────────────────────────────────────────┐
            ├─ T2: ESN Library ──────────────────────────────────────────────────────┐│
            ├─ T3: Model Utils ─────────────────────────────────────────────────────┐││
            ├─ T4: Eval Harness ───────────────────────────────────────────────────┐│││
            ├─ T5: Benchmark Suite ───────────────────────────────────────────────┐││││
            └─ T6: Chaos Pipeline ───────────────────────────────────────────────┐│││││
              FOUNDATION — all CPU, all parallel                                 ││││││
                                                                                 ││││││
              ┌──────────────────────────────────────────────────────────────────┘│││││
              │  ┌────────────────────────────────────────────────────────────────┘││││
              │  │  ┌──────────────────────────────────────────────────────────────┘│││
              │  │  │  ┌────────────────────────────────────────────────────────────┘││
              │  │  │  │  ┌──────────────────────────────────────────────────────────┘│
              │  │  │  │  │  ┌────────────────────────────────────────────────────────┘
              ▼  ▼  ▼  ▼  ▼  ▼
         ┌─────────────────────────┐    ┌─────────────────────────┐
         │  BASELINES [GPU-serial] │    │ TRACK A MODULES [CPU]   │
         │  T7:  Qwen3.5 vanilla   │    │ T12: Sidecar module     │
         │  T8:  RoPE/YaRN ext.    │    │ T13: LoRA pipeline      │
         │  T9:  Mamba-2 1.3B      │    │                         │
         │  T10: LLaMA-3.2-1B     │    │  depends: T2, T3        │
         │  T11: Infini-attn      │    └────────────┬────────────┘
         │                         │                 │
         │  depends: T3, T4, T5    │                 │
         └─────┬──────────┬───────┘                 │
               │          │                          │
          (T7 done)  (T10 done)                     │
               │          │                          │
               │          │    ┌─────────────────────┘
               ▼          │    ▼
     ┌─────────────────┐  │  (T7 + T12 + T13 ready)
     │ T14: Read-only  │◄─┼──┘
     │ sidecar [GPU]   │  │
     └────────┬────────┘  │
              │           │
              ▼           │
     ┌─────────────────┐  │
     │ T15: Read/write │  │
     │ sidecar [GPU]   │  │
     └────────┬────────┘  │
              │           │
         ┌────┴────┐      │
         ▼         │      ▼
┌──────────────┐   │   ┌─────────────────────┐
│ T16: Hyperp. │   │   │ T17: LLaMA+RC ctrl  │
│ sweep [GPU]  │   │   │ [GPU]               │
└───────┬──────┘   │   │ depends: T10,T12,T13│
        │          │   └──────────┬──────────┘
        │          │              │
        └────┬─────┘──────────────┘
             │    + T7–T11 all done
             ▼
    ╔════════════════════╗
    ║  T18: GATE A       ║
    ║  pass/fail decision║
    ╚════════╤═══════════╝
             │
             │ pass ──────────────────────────────────────────────────┐
             ▼                                                        │
    ┌─────────────────────────┐                                       │
    │ TRACK B MODULES [CPU]   │                                       │
    │ T19: RIL module         │                                       │
    │ T20: DeltaNet replace   │                                       │
    │ T21: Multi-reservoir    │                                       │
    │ depends: T2, T3, T12    │                                       │
    └────────────┬────────────┘                                       │
                 │                                                    │
                 ▼                                                    │
    ┌─────────────────────────┐                                       │
    │ T22: Track B training   │                                       │
    │ + evaluation [GPU]      │                                       │
    │ depends: T18, T19–T21   │                                       │
    └────────────┬────────────┘                                       │
                 │                                                    │
                 ▼                                                    │
    ╔════════════════════╗                                            │
    ║  T23: GATE B       ║                                            │
    ╚════════╤═══════════╝                                            │
             │                                                        │
             │ pass                                                   │
             ▼                                                        │
    ┌─────────────────────────┐                                       │
    │ TRACK C MODULES [CPU]   │                                       │
    │ T24: RW-Transformer     │                                       │
    │ T25: Training curriculum │                                       │
    │ depends: T2, T5, T21    │                                       │
    └────────────┬────────────┘                                       │
                 │                                                    │
                 ▼                                                    │
    ┌─────────────────────────┐                                       │
    │ T26: Track C pre-train  │                                       │
    │ [CLOUD GPU]             │                                       │
    └────────────┬────────────┘                                       │
                 │                                                    │
                 ▼                                                    │
    ┌─────────────────────────┐                                       │
    │ T27: Latent reasoning   │                                       │
    │ K sub-steps [GPU/CLOUD] │                                       │
    └────────────┬────────────┘                                       │
                 │                                                    │
                 ▼                                                    │
    ╔════════════════════╗                                            │
    ║  T28: GATE C       ║                                            │
    ╚════════╤═══════════╝                                            │
             │                                                        │
             └────────────────────┬───────────────────────────────────┘
                                  │  (from best completed gate)
                                  ▼
                     ┌─────────────────────────┐
                     │ T29: Ablations +        │
                     │ Coconut comparison [GPU]│
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ T30: Paper + release    │
                     │ [CPU]                   │
                     └─────────────────────────┘
```

---

## 3. Task Breakdown

### Foundation (T1–T6) — All CPU, All Parallel

---

**T1 — Project scaffolding & shared interfaces**
- **Scope:** `pyproject.toml` with torch/transformers/reservoirpy deps, `src/` package layout, `tests/`, pre-commit (ruff, mypy), CI skeleton. **Shared protocols and types** that downstream tasks code against: `ReservoirConfig` dataclass, `Reservoir` protocol (step/reset), `ModelWrapper` protocol (forward/generate/get_hidden), `BenchmarkExample` dataclass (input/target/metadata), `EvalResult` dataclass (task/metric/value/config), benchmark `Generator` protocol, `DataPipeline` protocol (iterate/split).
- **Depends on:** nothing
- **Blocks:** everything (indirectly)
- **Compute:** CPU only
- **Acceptance:** `pip install -e .` works, `pytest` runs, linter passes, all protocols importable from `src.types`.

---

**T2 — ESN reservoir core library**
- **Scope:** `src/reservoir/esn.py` — ESN class with configurable size (500–50K), spectral radius, leak rate, input scaling, sparse topology (Erdős–Rényi, small-world). State evolution via `step(x_t, w_t) → r_t`. Sparse matrix storage. Batched forward for GPU tensors.
- **Depends on:** T1
- **Blocks:** T12, T19, T21, T24
- **Independent from:** T3, T4, T5, T6
- **Compute:** CPU only (reservoir math is sparse mat-vec, trivial)
- **GPU hours:** 0
- **Acceptance:** Unit tests confirm echo state property (spectral radius < 1 ⟹ state fading), <1 ms per step at 50K nodes on CPU, state norm stays bounded over 10K steps across parameter regimes.

---

**T3 — Model loading & tokenizer utilities**
- **Scope:** `src/models/loader.py` — unified interface to load Qwen3.5-0.8B-Base, LLaMA-3.2-1B, Mamba-2 1.3B in FP16/BF16 on 3090. Tokenizer wrappers. Layer-name inspection utilities (for LoRA target selection, DeltaNet block identification in Qwen3.5).
- **Depends on:** T1
- **Blocks:** T7–T11, T12, T13, T20
- **Independent from:** T2, T4, T5, T6
- **Compute:** CPU only (downloads models, no training)
- **GPU hours:** 0 (brief smoke test on GPU)
- **Acceptance:** All three model families load on 3090, forward pass produces logits, Qwen3.5 DeltaNet layers identified by name.

---

**T4 — Evaluation harness**
- **Scope:** `src/eval/harness.py` — run any benchmark suite against any model, output standardized JSON. Support few-shot prompting, greedy/sampling decode, per-task metrics (exact match, perplexity, accuracy). Batch evaluation for throughput.
- **Depends on:** T1
- **Blocks:** T7–T11, T18, T23, T28
- **Independent from:** T2, T3, T5, T6
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Harness runs a dummy benchmark against a dummy model, produces valid JSON results.

---

**T5 — Synthetic benchmark suite**
- **Scope:** `src/eval/benchmarks/` — generators for:
  - **Memory:** passkey retrieval (configurable depth), variable tracking across distractors, associative recall with delayed query.
  - **Computation:** multi-digit addition/multiplication with carry, modular arithmetic, Dyck language recognition, synthetic program execution traces.
  - **Emergent:** compositional generalization (held-out operators), length extrapolation (train short, test long), algorithmic transfer.
  - All generators produce `(input, target, metadata)` tuples with configurable difficulty.
- **Depends on:** T1
- **Blocks:** T7–T11, T14, T25
- **Independent from:** T2, T3, T4, T6
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Each generator produces 1K+ valid examples, difficulty scales with parameters, format compatible with T4 harness.

---

**T6 — Chaotic systems data pipeline**
- **Scope:** `src/data/chaos.py` — generate trajectories for Lorenz-63, Mackey-Glass (variable τ), Kuramoto-Sivashinsky. Train/val/test splits. Normalization. Lyapunov time computation for evaluation horizon calibration.
- **Depends on:** T1
- **Blocks:** T7 (chaotic eval only)
- **Independent from:** T2, T3, T4, T5
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Generated trajectories match known Lyapunov exponents (Lorenz-63: λ ≈ 0.906), splits are non-overlapping, data loads in <1 s.

---

### Baselines (T7–T11) — GPU-Serialized

These establish the comparison points required before Gate A.

---

**T7 — Qwen3.5-0.8B-Base vanilla baselines**
- **Scope:** Evaluate unmodified Qwen3.5-0.8B-Base on full benchmark suite (T5 + T6). Record: perplexity on held-out text, accuracy on all synthetic tasks, throughput, VRAM. This is the primary reference point.
- **Depends on:** T3, T4, T5, T6
- **Blocks:** T14, T18
- **Independent from:** T8–T11, T12, T13
- **Compute:** GPU
- **GPU hours:** ~3 h
- **Acceptance:** Complete results JSON for all benchmark categories, baseline numbers recorded.

---

**T8 — Qwen3.5-0.8B + RoPE/YaRN context extension**
- **Scope:** Apply YaRN-style RoPE scaling to extend Qwen3.5-0.8B context. Evaluate on long-context benchmarks (passkey at 128K+, variable tracking at long range). This is Control 1 from the proposal.
- **Depends on:** T3, T4, T5
- **Blocks:** T18
- **Independent from:** T7, T9–T13
- **Compute:** GPU
- **GPU hours:** ~3 h
- **Acceptance:** Long-context eval results at 32K, 64K, 128K+ positions.

---

**T9 — Mamba-2 1.3B native SSM baseline**
- **Scope:** Evaluate Mamba-2 (1.3B) on full benchmark suite. This is Control 2 — tests whether SSMs natively solve the memory gap without reservoir augmentation.
- **Depends on:** T3, T4, T5
- **Blocks:** T18
- **Independent from:** T7, T8, T10–T13
- **Compute:** GPU
- **GPU hours:** ~5 h
- **Acceptance:** Full results JSON, direct comparison table vs T7.

---

**T10 — LLaMA-3.2-1B pure-attention baseline**
- **Scope:** Evaluate LLaMA-3.2-1B (pure softmax attention, no DeltaNet) on full benchmark suite. Architecture control for the DeltaNet synergy hypothesis.
- **Depends on:** T3, T4, T5
- **Blocks:** T17, T18
- **Independent from:** T7–T9, T11–T13
- **Compute:** GPU
- **GPU hours:** ~3 h
- **Acceptance:** Full results JSON.

---

**T11 — Infini-attention / Titans baseline**
- **Scope:** Implement compressive memory baseline (Infini-attention or Titans/ATLAS style test-time memory) on Qwen3.5-0.8B. Train adapter + memory module. Evaluate. This is Control 3 — the strongest non-reservoir memory baseline.
- **Depends on:** T3, T4, T5
- **Blocks:** T18
- **Independent from:** T7–T10, T12, T13
- **Compute:** GPU
- **GPU hours:** ~15 h (implementation + training + eval)
- **Acceptance:** Trained model, eval results on memory benchmarks, comparison vs T7.

---

### Track A: Bolt-On Sidecar (T12–T18)

---

**T12 — Read/Write interface & cross-attention sidecar module**
- **Scope:** `src/reservoir/interface.py` — ReadProjection (reservoir → LLM dim), WriteHead (LLM hidden → reservoir input), CrossAttentionSidecar (reservoir states as KV, LLM hidden as Q, inserted at configurable layers). FiLM-style residual modulation as alternative injection. All modules work with frozen LLM + LoRA.
- **Depends on:** T2, T3
- **Blocks:** T14, T17, T19, T20
- **Independent from:** T4–T11, T13
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Modules instantiate, forward pass produces correct shapes, gradient flows through interface but stops at reservoir boundary.

---

**T13 — LoRA + adapter training pipeline**
- **Scope:** `src/training/lora_trainer.py` — training loop for frozen LLM + LoRA + reservoir interface. AdamW optimizer (LoRA + interface params only). Gradient checkpointing. Mixed precision. Wandb logging. Configurable: LoRA rank, target modules, learning rate schedule, dataset mixing.
- **Depends on:** T3
- **Blocks:** T14, T17
- **Independent from:** T2, T4–T12
- **Compute:** CPU only (builds pipeline, no training)
- **GPU hours:** 0
- **Acceptance:** Pipeline runs a 10-step smoke test on GPU without OOM, loss decreases, checkpoints save/load correctly.

---

**T14 — Track A read-only sidecar training** ⚡ GPU
- **Scope:** Train Qwen3.5-0.8B-Base with a read-only reservoir sidecar (no write head). Reservoir processes input embeddings and provides memory via cross-attention. Train LoRA + ReadProjection on fine-tuning mix. Evaluate on full benchmark suite.
- **Depends on:** T5, T7, T12, T13
- **Blocks:** T15
- **Independent from:** T8–T11, T17
- **Compute:** GPU
- **GPU hours:** ~30 h (5K steps + eval)
- **Acceptance:** Model trains without divergence, perplexity degradation < 2% vs T7, benchmark results recorded.

---

**T15 — Track A full read/write sidecar training** ⚡ GPU
- **Scope:** Add WriteHead to T14's read-only model. Train with curriculum: start from T14 checkpoint (read-only pretrained), gradually enable write head (gated, increasing from 0→1 over first 1K steps). Evaluate on full suite. Compare read-only vs read/write.
- **Depends on:** T14
- **Blocks:** T16, T18
- **Independent from:** T8–T11, T17
- **Compute:** GPU
- **GPU hours:** ~30 h
- **Acceptance:** Write head produces non-trivial signals (write vector norm > 0, entropy > threshold), read/write model meets or beats read-only on memory benchmarks, perplexity degradation < 2%.

---

**T16 — Reservoir hyperparameter sweep** ⚡ GPU
- **Scope:** Sweep key reservoir hyperparameters using best config from T15 as starting point. Sequential sweeps (not full grid):
  1. Reservoir size: 500, 2K, 10K, 50K (4 runs)
  2. Spectral radius: 0.5, 0.9, 0.99, 1.1 (4 runs)
  3. Leak rate: 0.1, 0.3, 0.7, 1.0 (4 runs)
  4. Topology: Erdős–Rényi vs small-world (2 runs)
  5. Best combo fine-tune (1 run)
  - Each run: 1–2K steps with early stopping, eval on memory + computation subsets.
- **Depends on:** T15
- **Blocks:** T18
- **Independent from:** T17
- **Compute:** GPU (optionally offload to cloud for speed)
- **GPU hours:** ~80 h (15 runs × ~5 h)
- **Acceptance:** Pareto frontier identified (reservoir size vs quality vs latency), best config documented, all results in wandb.

---

**T17 — LLaMA-3.2-1B + reservoir (DeltaNet synergy control)** ⚡ GPU
- **Scope:** Apply the same read-only sidecar architecture (T12) to LLaMA-3.2-1B (pure softmax attention). Train LoRA + ReadProjection with same data and hyperparameters as T14. Compare gains vs T14 to test whether Qwen3.5's DeltaNet layers provide synergistic compatibility with reservoir states.
- **Depends on:** T10, T12, T13
- **Blocks:** T18
- **Independent from:** T14, T15, T16
- **Compute:** GPU
- **GPU hours:** ~30 h
- **Acceptance:** Results directly comparable to T14 (same eval suite), DeltaNet synergy hypothesis has quantitative answer (Δ between Qwen gain and LLaMA gain).

---

**T18 — Gate A evaluation & decision report**
- **Scope:** Compile all Track A results. Evaluate against Gate A thresholds:
  1. ≥ 10% gain on long-context retrieval at ≥ 128K context equivalent vs T7
  2. ≥ 15% gain on algorithmic memory tasks vs T7
  3. ≥ 10% gain on compositional generalization vs T7
  4. ≤ 20% inference latency overhead vs T7
  5. < 2% perplexity degradation vs T7
  - Compare against all baselines (T8–T11). Document DeltaNet synergy result (T17 vs T14).
- **Depends on:** T7, T8, T9, T10, T11, T15, T16, T17
- **Blocks:** T22 (Track B training)
- **Compute:** CPU (analysis + reporting, brief GPU for any missing evals)
- **GPU hours:** ~5 h
- **Acceptance:** Written report with pass/fail on each gate criterion, recommendation to proceed or stop, all numbers reproducible from logged artifacts.

---

### Track B: Inserted Layers (T19–T23) — Gated on Gate A Pass

---

**T19 — Reservoir Interaction Layer (RIL) module**
- **Scope:** `src/reservoir/ril.py` — module that inserts between transformer blocks: reads reservoir state, computes write signal from block output, updates reservoir, returns modulated hidden state via gated residual. Configurable insertion points (every k layers). Compatible with Qwen3.5 block structure.
- **Depends on:** T2, T12
- **Blocks:** T22
- **Independent from:** T20, T21
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Module inserts into Qwen3.5 without shape errors, forward pass works, gradient flows correctly through RIL interface.

---

**T20 — DeltaNet block replacement module**
- **Scope:** `src/reservoir/deltanet_replace.py` — replace selected Gated DeltaNet blocks in Qwen3.5 with ESN reservoir modules. The reservoir serves as a "richer" recurrent module with higher-dimensional state. Surrounding full-attention and FFN layers unchanged. Configurable: which DeltaNet blocks to replace (some or all of the 18 DeltaNet layers).
- **Depends on:** T3, T12
- **Blocks:** T22
- **Independent from:** T19, T21
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Model with replaced blocks produces valid output, parameter count accounting correct, DeltaNet vs ESN A/B swap works.

---

**T21 — Multi-reservoir (fast/slow) module**
- **Scope:** `src/reservoir/multi_reservoir.py` — parallel reservoir configuration with different timescales:
  - **Fast reservoir:** high leak rate (α ≈ 0.7–1.0), spectral radius near 1.0, for computational scratch-space.
  - **Slow reservoir:** low leak rate (α ≈ 0.1–0.3), spectral radius near 0.5, for long-term contextual memory.
  - Shared read interface (LLM cross-attends to concatenated reservoir states).
  - Independent write heads per reservoir.
- **Depends on:** T2
- **Blocks:** T22, T24
- **Independent from:** T19, T20
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Dual-reservoir state evolves at different timescales (verified by autocorrelation of state trajectories), read interface produces correct shapes.

---

**T22 — Track B training & evaluation** ⚡ GPU
- **Scope:** Train three Track B configurations on Qwen3.5-0.8B-Base:
  1. **RIL insertion** (T19): reservoir interaction layers every 6 blocks
  2. **DeltaNet replacement** (T20): replace 6/18 DeltaNet blocks with ESN
  3. **Multi-reservoir RIL** (T21 + T19): fast/slow reservoirs with RIL insertion
  - Training: LoRA + RIL/replacement interface params + selective unfreezing of adjacent LayerNorms. Same fine-tuning data as Track A. Evaluate each on full suite.
- **Depends on:** T18 (Gate A pass), T19, T20, T21
- **Blocks:** T23
- **Compute:** GPU
- **GPU hours:** ~120 h (3 configs × 40 h each)
- **Acceptance:** All three configs train stably, results comparable across configs, best config identified.

---

**T23 — Gate B evaluation & decision report**
- **Scope:** Evaluate Track B against Gate B thresholds:
  1. ≥ 20% exact-match gain on long program-trace tasks vs T7
  2. Better memory-quality-per-byte than RoPE/YaRN (T8)
  3. < 2% perplexity degradation vs T7
  - Compare Track B best vs Track A best. Document which insertion strategy works best and whether multi-reservoir outperforms single.
- **Depends on:** T22
- **Blocks:** T26 (Track C training)
- **Compute:** CPU + GPU
- **GPU hours:** ~10 h
- **Acceptance:** Written report with gate pass/fail, Track A vs B comparison, architectural recommendation.

---

### Track C: From Scratch (T24–T28) — Gated on Gate B Pass

---

**T24 — RW-Transformer architecture definition**
- **Scope:** `src/models/rw_transformer.py` — define the from-scratch hybrid architecture:
  - Decoder block with three parallel branches: (1) local/global attention, (2) MLP, (3) bidirectional multi-reservoir workspace (fast + slow).
  - Gated residual mixing layer to fuse branches.
  - ~0.8B trainable parameters (transformer + interfaces), reservoir weights frozen.
  - Same tokenizer as Qwen3.5 (248K vocab) for fair comparison.
- **Depends on:** T2, T21
- **Blocks:** T26
- **Independent from:** T25
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Model instantiates at ~0.8B params, forward pass produces logits, architecture diagram documented.

---

**T25 — Training curriculum & data pipeline**
- **Scope:** `src/training/curriculum.py` — implement the three-stage curriculum:
  1. **Stage 1:** standard next-token prediction on general text
  2. **Stage 2:** mix in procedural objectives (arithmetic traces, symbolic state tracking, Dyck)
  3. **Stage 3:** length curriculum (4K → 32K → 128K+ context)
  - Data loading with streaming, dynamic mixing ratios, auxiliary reservoir-state prediction loss (optional).
- **Depends on:** T5
- **Blocks:** T26
- **Independent from:** T24
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Data pipeline produces batches at each curriculum stage, mixing ratios configurable, streaming works for large corpus.

---

**T26 — Track C pre-training** ☁️ CLOUD
- **Scope:** Pre-train the RW-Transformer (T24) from scratch using the curriculum (T25) on cloud A100(s). Full AdamW, gradient checkpointing, mixed precision. Target: ~20B tokens minimum.
- **Depends on:** T23 (Gate B pass), T24, T25
- **Blocks:** T27
- **Compute:** Cloud GPU (A100 80GB)
- **Cloud hours:** ~500 h
- **Estimated cost:** $250–400
- **Acceptance:** Training loss converges, perplexity within +3% of Qwen3.5-0.8B baseline (T7) on held-out text.

---

**T27 — Latent reasoning experiments (K sub-steps)** ⚡ GPU/☁️ CLOUD
- **Scope:** Test multi-sub-step reservoir evolution between tokens:
  - **K sweep:** K = 1, 2, 4, 8, 16 sub-steps between token generations
  - **Halting strategies:** fixed K vs convergence-based (‖r^(k) − r^(k−1)‖ < ε) vs learned halting (PonderNet-style)
  - **Comparison:** Reservoir latent reasoning vs Coconut-style hidden-state recirculation (implement Coconut baseline on Qwen3.5-0.8B for fair comparison)
  - Evaluate on ProsQA, multi-hop deduction, and Track C gate benchmarks.
  - Measure reasoning accuracy vs sub-step count and vs compute budget.
- **Depends on:** T26
- **Blocks:** T28
- **Compute:** GPU (local for small K) + Cloud (large K sweep)
- **GPU hours:** ~40 h local + ~50 h cloud
- **Acceptance:** K scaling curve documented, comparison vs Coconut at matched compute budget, best halting strategy identified.

---

**T28 — Gate C evaluation & decision report**
- **Scope:** Evaluate Track C against Gate C thresholds:
  1. Match baseline perplexity within +3% (vs T7)
  2. ≥ 25% gain on long-horizon memory tasks vs T7
  3. O(1) memory scaling for workspace state, better inference memory efficiency than dense attention at long horizons
  - Full comparison table: Track A vs Track B vs Track C vs all baselines.
- **Depends on:** T27
- **Blocks:** T30
- **Compute:** CPU + GPU
- **GPU hours:** ~10 h
- **Acceptance:** Comprehensive comparison table, gate pass/fail, scaling analysis.

---

### Analysis & Deliverables (T29–T30)

---

**T29 — Ablation studies & efficiency benchmarking** ⚡ GPU
- **Scope:** Run ablations on best-performing track:
  - Read-only vs write-only vs full read/write
  - Single vs multi-reservoir
  - Frozen vs partially trainable input projections
  - Spectral radius regimes
  - **Randomized dynamics control:** replace ESN with random stateless projection to verify recurrent dynamics (not just extra parameters) are the source of gains
  - **Efficiency:** throughput (tokens/sec), p50/p95 latency, VRAM at matched quality, quality-per-FLOP, quality-per-byte
  - **Coconut comparison** (if not already done in T27): Coconut vs reservoir latent reasoning at matched compute
- **Depends on:** T18 (minimum — runs after best completed gate)
- **Blocks:** T30
- **Compute:** GPU
- **GPU hours:** ~60 h
- **Acceptance:** All ablation results in standardized format, efficiency numbers documented, randomized control confirms (or refutes) that recurrent dynamics matter.

---

**T30 — Paper draft & open-source release**
- **Scope:** Write up results, prepare figures, clean codebase for release. Includes negative-result report if any gate failed (per proposal commitment). Open-source all code, configs, and benchmark harness.
- **Depends on:** T29
- **Compute:** CPU only
- **GPU hours:** 0
- **Acceptance:** Draft paper with all figures, code repo documented and releasable, benchmark harness independently runnable.

---

## 4. Parallelism Map

### 4.1 Phase Timeline

```
Phase 0 (Wk 1–2):    T1 + T2 + T3 + T4 + T5 + T6    (6 CPU workers)
Phase 1 (Wk 2–3):    T7* + T12 + T13                  (1 GPU + 2 CPU workers)
Phase 2 (Wk 3–6):    T14* → T15* → T16* + T17*        (GPU-serialized + CPU ahead-work)
                      T8* + T9* + T10* + T11*           (GPU, interleaved with above)
Phase 3 (Wk 6–7):    T18 + T19 + T20 + T21             (CPU workers, Gate A eval on GPU)
Phase 4 (Wk 7–9):    T22*                               (GPU, Track B training)
Phase 5 (Wk 9–10):   T23 + T24 + T25                   (CPU workers, Gate B eval on GPU)
Phase 6 (Wk 10–12):  T26☁ + T27*                        (CLOUD + local GPU)
Phase 7 (Wk 12–13):  T28 + T29*                         (Gate C + ablations on GPU)
Phase 8 (Wk 13–14):  T30                                (CPU only, paper)

* = GPU task    ☁ = cloud task
```

### 4.2 Optimal GPU Schedule (1× RTX 3090)

The GPU is the serialization bottleneck. Optimal ordering minimizes idle time and unblocks dependent tasks earliest:

```
GPU Timeline (sequential, ~450 hours):
──────────────────────────────────────────────────────────────────────────
│ T7  │    T14         │T10│    T15         │ T8│T9 │    T17         │
│ 3h  │    30h         │3h │    30h         │ 3h│5h │    30h         │
──────────────────────────────────────────────────────────────────────────
│T11     │       T16              │ T18│
│15h     │       80h              │ 5h │
──────────────────────────────────────────── ← Gate A (~204 h)
│          T22                        │T23│
│          120h                       │10h│
──────────────────────────────────────────── ← Gate B (~334 h)
│    T27        │ T28│     T29           │
│    40h        │ 10h│     60h           │
──────────────────────────────────────────── ← Done (~444 h)

Cloud (parallel with local GPU after Gate B):
│              T26: 500h on A100              │
│  + T27 overflow: 50h on A100                │
```

### 4.3 CPU Worker Utilization During GPU Phases

| GPU Phase | CPU Workers Doing |
|-----------|-------------------|
| T7 (baseline) | Finishing T2, T4, T5, T6 if not done; starting T12, T13 |
| T14–T15 (Track A training) | T12, T13 complete → start speculative T19–T21 (Track B modules) |
| T16–T17 (sweep + control) | T19–T21 complete → start speculative T24–T25 (Track C modules) |
| T22 (Track B training) | T24, T25 finalize; data pipeline for Track C |
| T26 (cloud, Track C) | Analyze Track B results; prepare ablation configs |

**Key insight:** CPU workers build 1–2 phases ahead. Track B modules (T19–T21) are developed while Track A trains. Track C modules (T24–T25) are developed while Track B trains. If a gate fails, the speculative CPU work is discarded — cost is zero (just developer time).

---

## 5. Cloud Budget Allocation

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| Track C pre-training (T26) | 500 h | $0.50/h (A100 spot) | $250 |
| Latent reasoning sweep (T27) | 50 h | $0.50/h | $25 |
| HP sweep overflow (T16, optional) | 40 h | $0.50/h | $20 |
| LLaMA control (T17, optional offload) | 30 h | $0.50/h | $15 |
| **Subtotal (spot pricing)** | **620 h** | | **$310** |
| Buffer for on-demand / restarts | ~200 h | $0.80/h | $160 |
| **Total budget** | | | **$470** |

Remaining from $1,000 budget: **$530** for contingency (larger sweeps, scale-up experiments on Qwen3.5-2B, or extended Track C training).

**Provider recommendation:** RunPod or Lambda for A100 spot instances. Both support PyTorch + CUDA out of the box. Prefer single-GPU instances (A100 80GB) for simplicity — the 0.8B model doesn't need multi-GPU.

---

## 6. Critical Path Analysis

### 6.1 Critical Path (Minimum Time to Each Gate)

```
Gate A critical path:
  T1(0) → T3(0) → T7(3h) → T14(30h) → T15(30h) → T16(80h) → T18(5h)
  Total: ~148 h GPU + foundation CPU time
  Wall-clock: ~2 weeks (foundation) + ~10 days GPU = ~3.5 weeks

  Parallel requirement for T18: T8(3h) + T9(5h) + T10(3h) + T11(15h) + T17(30h)
  These 56h of GPU work must also complete before T18.
  With interleaving: T7 → T14 → T10 → T15 → T8 → T9 → T17 → T11 → T16 → T18
  Total GPU: ~204 h → ~13 days at 16h/day → ~4.5 weeks from start

Gate B critical path:
  Gate A (204h) → T22(120h) → T23(10h)
  Total: ~334 h GPU → ~21 days at 16h/day → ~7.5 weeks from start

Gate C critical path:
  Gate B (334h local) + T26(500h cloud, parallel) → T27(40h local + 50h cloud) → T28(10h)
  Cloud starts after Gate B, takes ~3 days wall-clock on 8× A100 or ~6 days on 4× A100.
  Total wall-clock: ~9 weeks from start

Final (ablations + paper):
  Gate C + T29(60h) + T30(CPU)
  Total: ~10–11 weeks from start
```

### 6.2 Early Termination Scenarios

| Scenario | When | What Happens | Salvage Value |
|----------|------|--------------|---------------|
| Gate A fails (all criteria) | Week ~5 | Stop. Write negative-result report. | Reservoir library, eval harness, baselines — all reusable. DeltaNet synergy answer is publishable. |
| Gate A partial pass | Week ~5 | Proceed to Track B with adjusted expectations. | Partial gains documented, subset of benchmarks showed improvement. |
| Gate B fails | Week ~8 | Stop Tracks B/C. Deepen Track A ablations. | Track A results + comparison = solid paper. |
| Gate B partial pass | Week ~8 | Attempt Track C on reduced scale. | Two tracks of results. |
| Gate C fails | Week ~10 | Write comprehensive negative-result report. | Three tracks of data, architectural comparison = strong negative-result paper. |

### 6.3 GPU Utilization Target

- **Target:** ≥ 90% GPU utilization during training phases (Weeks 3–9)
- **Method:** Queue next GPU task before current one finishes (have configs ready). CPU workers prepare all data/configs ahead of GPU need.
- **Monitoring:** Track GPU idle hours in a simple log. If idle > 4h, something is wrong with the pipeline.

---

## 7. Repo Structure

```
/code/rc/
├── COLLABORATIVE_PROPOSAL.md          # Active spec (kept in root)
├── IMPLEMENTATION_PLAN.md             # This document
├── docs/
│   └── proposals/
│       ├── CLAUDE_PROPOSAL.md         # Original Claude proposal
│       ├── GEMINI_PROPOSAL.md         # Original Gemini proposal
│       └── CODEX_PROPOSAL.md         # Original Codex proposal
├── latentspace.md                     # Latent reasoning research notes
├── src/
│   ├── reservoir/
│   │   ├── esn.py                     # T2: ESN core
│   │   ├── interface.py               # T12: Read/Write/CrossAttn
│   │   ├── ril.py                     # T19: Reservoir Interaction Layer
│   │   ├── deltanet_replace.py        # T20: DeltaNet replacement
│   │   └── multi_reservoir.py         # T21: Fast/slow reservoirs
│   ├── models/
│   │   ├── loader.py                  # T3: Model loading
│   │   └── rw_transformer.py          # T24: From-scratch architecture
│   ├── training/
│   │   ├── lora_trainer.py            # T13: LoRA pipeline
│   │   └── curriculum.py             # T25: Training curriculum
│   ├── eval/
│   │   ├── harness.py                 # T4: Evaluation harness
│   │   └── benchmarks/               # T5: Benchmark generators
│   │       ├── memory.py
│   │       ├── computation.py
│   │       └── emergent.py
│   └── data/
│       └── chaos.py                   # T6: Chaotic systems
├── configs/                           # Experiment configs (YAML)
├── tests/                             # Unit tests
└── scripts/                           # Training/eval launch scripts
```
