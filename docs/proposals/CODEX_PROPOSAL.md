# Reservoir Workspace for LLMs
## A Three-Track Proposal for Memory and Computation (Bolt-On, Inserted Layers, From Scratch)

**Author:** Codex
**Date:** March 4, 2026

## 1. Thesis
Use fixed-dynamics reservoirs as a **continuous latent workspace** that LLMs can read and write during generation. The objective is not to replace attention, but to offload two hard functions:
1. long-horizon state retention under strict memory budgets
2. intermediate-state computation for multi-step symbolic tasks
3. emergence of new capabilities under compositional and length extrapolation stress

This proposal executes in three tracks with explicit go/no-go gates:
1. **Track A (Bolt-On):** reservoir sidecar added to a pretrained LLM with PEFT
2. **Track B (Inserted Layers):** reservoir interaction layers inserted into an existing LLM
3. **Track C (From Scratch):** hybrid attention + reservoir model trained end-to-end

## 2. What This Proposal Changes vs Existing Drafts
The two existing drafts are directionally strong, but this version adds missing rigor:
1. **Hard baselines** against modern long-context methods (Infini-attention, Mamba-2, Titans/ATLAS, retrieval baselines), not only vanilla transformers.
2. **Concrete gates** with quantitative thresholds to decide whether each track should continue.
3. **Training clarity**: gradients can flow through fixed reservoir state transitions to interface weights; reservoir weights remain frozen.
4. **No overclaiming**: fixed-size reservoir gives bounded-state compression, not guaranteed unlimited memory.
5. **Task separation**: distinct experiments for memory retention vs procedural computation.

## 3. Core Architecture
At token step `t`:

`r_t = (1 - alpha) r_{t-1} + alpha * phi(W_r r_{t-1} + W_x x_t + W_w w_t + b)`

Where:
- `r_t` is reservoir state (`D_r`)
- `x_t` is current model input embedding
- `w_t` is write signal from transformer hidden state
- `W_r` is fixed sparse recurrent matrix (spectral radius sweep around edge-of-chaos)
- `W_x`, `W_w` are fixed random input projections (optionally train tiny low-rank corrections)
- `phi` is `tanh` or clipped GELU

Read path:
- `m_t = ReadProj(Pool(r_t))`
- `m_t` is injected into the LLM via cross-attention KV adapters or residual FiLM-style modulation.

Write path:
- `w_t = WriteHead(h_t)` from selected layer hidden state `h_t`.

Memory principle:
- Keep reservoir state fixed-size per sequence (constant-state inference memory for the workspace itself).
- Measure whether this compressed state preserves task-relevant information better than extending KV cache at equal memory budget.

## 4. Track A: Bolt-On Reservoir Sidecar (Lowest Risk)
### 4.1 Setup
- Primary model: `Qwen/Qwen3.5-0.8B-Base`.
- Secondary model: `Qwen/Qwen3.5-0.8B` (post-trained comparison).
- Control model: one similarly sized plain decoder-only baseline to isolate architecture effects.
- Freeze base weights.
- Train only:
  - read projection
  - write head
  - LoRA on selected attention/output projections

### 4.2 Injection Modes
1. **Prefix memory tokens:** map reservoir readout into virtual tokens prepended per segment.
2. **Cross-attention memory slots:** lightweight cross-attention to reservoir slots.
3. **Residual modulation:** affine modulation of intermediate activations.

### 4.3 Success Gate A
Proceed only if all hold vs strong baseline at matched latency budget:
1. `>= 10%` gain on long-context retrieval/needle tasks at `>= 128k` context equivalent.
2. `>= 15%` gain on algorithmic memory tasks (copy/reverse/variable tracking).
3. `>= 10%` gain on compositional generalization splits (held-out operator/length combinations).
4. `<= 20%` inference latency overhead.

## 5. Track B: Inserted Reservoir Interaction Layers (Medium Risk, Higher Upside)
### 5.1 Reservoir Interaction Layer (RIL)
Insert one RIL every `k` transformer blocks:
1. read reservoir state into current block
2. compute write signal from block output
3. update reservoir once per token
4. return modulated hidden state through residual path

### 5.2 Multi-Reservoir Design
- **Fast reservoir:** short-term computational workspace
- **Slow reservoir:** longer-horizon contextual trace
- Optional third associative reservoir for key-value style traces

### 5.3 Training
- Train inserted RIL parameters + selective unfreezing of nearby norms/projections.
- Stabilizers: reservoir state norm clipping, Jacobian penalty on write head, spectral regularization for effective contractivity.

### 5.4 Success Gate B
Proceed to broad benchmarking only if:
1. `>= 20%` exact-match gain on long arithmetic/program-trace tasks.
2. Better memory-quality-per-byte than RoPE/YaRN-only context extension.
3. `>= 15%` gain on emergent-composition evaluation (zero/few-shot).
4. No catastrophic degradation (`< 2%`) on general language perplexity benchmark.

## 6. Track C: From-Scratch Hybrid (Highest Risk, Highest Ceiling)
### 6.1 Model Family: RW-Transformer
A decoder architecture with three parallel branches per block:
1. local/global attention branch
2. MLP branch
3. reservoir workspace branch

Fuse branches via gated residual mixing.

### 6.2 Training Curriculum
1. Pretrain on standard next-token objective.
2. Mix in procedural objectives (algorithmic traces, arithmetic, symbolic state tracking).
3. Length curriculum (4k -> 32k -> 128k+ equivalent sequence dependency).
4. Optional auxiliary reservoir-state prediction loss to encourage meaningful write/read usage.

### 6.3 Success Gate C
1. Match baseline perplexity within `+3%` on standard corpora.
2. Beat baseline by `>= 25%` on long-horizon memory/computation suite.
3. Beat baseline by `>= 20%` on emergent capability suite.
4. Achieve better inference memory efficiency than dense-attention baseline at long horizon.

## 7. Evaluation Program
### 7.1 Memory Benchmarks
- passkey/needle-in-haystack variants
- variable tracking across distractors
- associative recall with delayed query
- long-document multi-hop QA with controlled evidence positions

### 7.2 Computation Benchmarks
- multi-digit addition/multiplication with carry diagnostics
- modular arithmetic and parity
- synthetic program execution traces
- formal-language tasks (Dyck, stack-like dependencies)

### 7.3 Emergent Capability Benchmarks
- length extrapolation with held-out lengths and distractor structure
- compositional generalization with unseen operator combinations
- algorithmic transfer (train on traces, test on altered control flow)
- latent state intervention tests to verify causal role of reservoir state

### 7.4 Realistic Benchmarks
- long-context reasoning suites

### 7.5 Ablations
- read-only vs write-only vs full read/write
- single vs multi-reservoir
- frozen vs partially trainable input projections
- different spectral radius/leak regimes

### 7.6 Efficiency Metrics
- throughput (tokens/sec)
- p50/p95 step latency
- VRAM usage at matched quality
- quality-per-FLOP and quality-per-byte

## 8. Baselines (Required)
Compare against:
1. Same LLM without reservoir
2. Context-extension methods (RoPE scaling / YaRN or equivalent)
3. Retrieval-augmented memory baseline
4. State-space alternatives (Mamba/Mamba-2 class)
5. Recent long-memory test-time methods (Infini-attention, Titans-like/ATLAS-like where applicable)

## 9. Risks and Mitigations
1. **Write path learns noisy control**
   - Mitigation: gated write sparsity, norm budget, entropy regularization, curriculum from read-only to read/write.
2. **Reservoir collapses to unused feature**
   - Mitigation: dropout on direct residual path, auxiliary tasks that require delayed state recall.
3. **Training instability at long horizons**
   - Mitigation: contractive settings, state normalization, truncated gradient windows for interfaces.
4. **No benefit over simpler methods**
   - Mitigation: early gates and terminate failing track quickly.

## 10. Execution Plan (12 Months)
1. **Months 1-2:** infrastructure, baselines, synthetic task suite, Qwen3.5 0.8B Base/Post-trained integration.
2. **Months 3-5:** Track A full sweep, Gate A decision.
3. **Months 6-8:** Track B implementation/sweep, Gate B decision.
4. **Months 9-12:** Track C prototype if B passes strongly; otherwise deepen B on emergent-capability domains.
5. **Optional Month 12 sanity check (not milestone):** small chaotic forecasting probe for representation diagnostics only.

## 11. Deliverables
1. Reproducible code for all three tracks.
2. Benchmark harness focused on memory/computation decoupling.
3. Negative-result report if reservoir workspace fails gates (important for field clarity).
4. Final paper-quality report with scaling laws for reservoir size/timescale vs task class.

## 12. Recommended Immediate Start
Start with **Track A read-only then read/write sidecar** on `Qwen/Qwen3.5-0.8B-Base`, then replicate on `Qwen/Qwen3.5-0.8B`. This gives the fastest signal on emergent RC+LLM gains before committing to architectural surgery.

## 13. Key References
- Qwen3.5 repository (release notes, tooling): https://github.com/QwenLM/Qwen3.5
- Qwen3.5-0.8B-Base model card: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- Qwen3.5-0.8B model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Reservoir Transformers (ACL 2021): https://aclanthology.org/2021.acl-long.331/
- Reservoir Transformer at Infinite Horizon (arXiv 2024): https://arxiv.org/abs/2402.09573
- Reservoir Computing as a Language Model (arXiv 2025): https://arxiv.org/abs/2507.15779
- Echo State Transformer: When chaos brings memory to attention (arXiv 2025): https://arxiv.org/abs/2507.02917
- Syntactic Learnability of Echo-State Neural Language Models at Scale (arXiv 2025): https://arxiv.org/abs/2503.01724
- Mamba (arXiv 2023): https://arxiv.org/abs/2312.00752
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (arXiv 2024): https://arxiv.org/abs/2405.21060
- Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention (arXiv 2024): https://arxiv.org/abs/2404.07143
- Titans: Learning to Memorize at Test Time (arXiv 2025): https://arxiv.org/abs/2501.00663
- ATLAS: Learning to Optimally Memorize the Context at Test Time (arXiv 2025): https://arxiv.org/abs/2505.23735
- Emerging opportunities and challenges for the future of reservoir computing (Nature Communications 2024): https://www.nature.com/articles/s41467-024-46701-5
