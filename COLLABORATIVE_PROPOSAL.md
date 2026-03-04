# Latent Reservoir Scratchpads (LRS): A Three-Track Research Program for Bidirectional Memory and Computation in LLMs

**Collaborators:** Claude, Gemini, Codex
**Date:** March 4, 2026

## 1. Executive Summary
Large Language Models (LLMs) struggle with infinite-horizon memory and multi-step symbolic reasoning due to the stateless nature of their forward pass and the computational cost of expanding the KV-cache. We propose the **Latent Reservoir Scratchpad (LRS)**: a bidirectional dynamical system—specifically, an Echo State Network (ESN)—integrated directly into the Transformer architecture. This reservoir acts as a fixed-size, continuous-state working memory that the LLM can both read from and write to. By combining the strengths of our three original proposals, this document outlines a rigorous three-track research program with quantitative success gates, compute-matched baselines against state-of-the-art context extension methods, and a targeted focus on the Qwen3.5 model family to test the DeltaNet synergy hypothesis. The Gated DeltaNet architecture utilizes a delta rule for hidden state updates, analogous to recurrent reservoir state updates, providing a shared recurrent inductive bias that minimizes distributional mismatch between standard attention and continuous dynamical variables.

## 2. Core Mathematical Framework
The LRS is a high-dimensional dynamical system parallel to or outside the standard attention stream. At each token step $t$:

**1. Write Path & Reservoir Update:**
The Transformer generates a write signal $w_t$ from its hidden state $h_t$: $w_t = \text{WriteHead}(h_t)$.
The reservoir state $r_t \in \mathbb{R}^{D_r}$ evolves:
$$r_t = (1 - \alpha) r_{t-1} + \alpha \cdot \phi(W_{res} r_{t-1} + W_{in} x_t + W_w w_t + b_{res})$$
*Where $\alpha$ is the leak rate, $W_{res}$ is a fixed sparse matrix with spectral radius $\rho \approx 1$, and $W_{in}, W_w$ are fixed random projections.*

**2. Read Path & Integration:**
The reservoir state is projected back into the Transformer's dimension:
$$m_t = \text{ReadProj}(\text{Pool}(r_t))$$
This $m_t$ is injected into the LLM via cross-attention slots or residual FiLM-style modulation.

**Crucially, only the interface parameters (ReadProj, WriteHead) and selective Transformer parameters are trained. The reservoir weights ($W_{res}$, $W_{in}$, $W_w$) remain fixed, providing bounded-state compression.**

## 3. The Three-Track Execution Plan
We adopt a progressive, gated approach to minimize risk and establish clear empirical foundations. 

**Progression: Single to Multi-Reservoir:** We explicitly mandate a strict single-reservoir topology for Track A to isolate the baseline benefits of continuous dynamical state. Multi-reservoir topologies (e.g., parallel fast/slow tracks) are strictly gated and only introduced in Tracks B and C after base capabilities are confirmed.

### Track A: Bolt-On Reservoir Sidecar (Lowest Risk)
- **Concept:** Attach a single reservoir as an external preprocessor to a frozen LLM (Qwen3.5-0.8B-Base). Train only adapter layers (Read/Write projections + LoRA).
- **Integration:** Reservoir states are accessed via a lightweight cross-attention sidecar. Training progresses iteratively from a read-only baseline to full bidirectional read/write access.
- **Success Gate A:** Proceed to Track B only if:
  1. $\ge 10\%$ gain on long-context retrieval at $\ge 128k$ context equivalent compared to baseline at matched latency.
  2. $\ge 15\%$ gain on algorithmic memory tasks (e.g., variable tracking, multi-digit arithmetic).
  3. $\ge 10\%$ gain on compositional generalization splits (held-out operators/lengths).
  4. $\le 20\%$ inference latency overhead.
  5. No catastrophic degradation ($< 2\%$) on general language perplexity.

### Track B: Inserted Reservoir Interaction Layers (Medium Risk, Higher Upside)
- **Concept:** Surgically insert the LRS into the mid-to-late layers of the LLM. Introduces the multi-reservoir progression (e.g., fast short-term vs. slow long-term reservoirs).
- **Integration (The DeltaNet Hypothesis):** Replace specific Gated DeltaNet blocks in Qwen3.5 with LRS modules to test architectural synergy with recurrent state-space layers. 
- **Success Gate B:** Proceed to Track C only if:
  1. $\ge 20\%$ exact-match gain on long program-trace tasks.
  2. Better memory-quality-per-byte than RoPE/YaRN-only context extension.
  3. No catastrophic degradation ($< 2\%$) on general language perplexity.

### Track C: From-Scratch Hybrid (Highest Risk, Highest Ceiling)
- **Concept:** Train a new ~0.8B parameter hybrid model (RW-Transformer) end-to-end. 
- **Expanded Architecture:** The decoder block features three parallel branches: 
  1) Local/global attention branch for exact pattern matching.
  2) Standard MLP branch for static knowledge retrieval.
  3) Bidirectional multi-reservoir workspace branch containing at least two continuous states: a high-leak "fast" reservoir for computational scratch-space, and a low-leak "slow" reservoir for contextual history. 
  These branches are fused via a learned gated residual mixing layer.
- **Integration:** Full end-to-end curriculum training starting with next-token prediction, progressing to procedural/symbolic traces, and concluding with length extrapolation (4k $\rightarrow$ 128k+). The write path is optimized via straight-through estimators and REINFORCE-style policy gradients, allowing gradients to guide the WriteHead despite the non-differentiable frozen reservoir.
- **Success Gate C:** 
  1. Match baseline perplexity within $+3\%$.
  2. Beat dense-attention baselines by $\ge 25\%$ on long-horizon memory tasks.
  3. Achieve strict $O(1)$ memory scaling for the workspace state with better inference memory efficiency than dense-attention baseline at long horizons.

## 4. Evaluation & Baselines
We commit to rigorous baselines, prioritizing emergent capabilities and working memory at compute-matched and memory-matched scales.

### 4.1 Controls & Baselines Matrix

| Configuration | Base Model | Memory Component | Training Cost | Primary Question |
|---------------|------------|------------------|---------------|------------------|
| **Control 1** | Qwen3.5-0.8B (Frozen) | RoPE / YaRN Extension | Zero | Base model max context capability? |
| **Control 2** | Mamba-2 (1.3B) | Native SSM State | Zero | Do SSMs natively solve the memory gap? |
| **Control 3** | Qwen3.5-0.8B + LoRA | Infini-attention / Titans | Low | Test-time memory vs. fixed reservoir? |
| **Track A** | Qwen3.5-0.8B + LoRA | Single External ESN | Low | Bolt-on viability with minimal training? |
| **Track B** | Qwen3.5-0.8B Modified | Internal Multi-ESN | Medium | Does internal insertion beat external bolt-on? |
| **Track C** | Custom RW-Transformer | Multi-Branch Multi-ESN | High | Ceiling performance trained from scratch? |
| **Arch Control** | LLaMA-3.2-1B + LoRA | Single External ESN | Low | Is DeltaNet-RC synergy real vs. pure softmax attention? |

### 4.2 Key Benchmark Domains
1. **Emergent Capabilities (Primary):** Compositional generalization with unseen operators, formal-language tasks (Dyck), and out-of-distribution length extrapolation.
2. **Logic & Computation:** GSM8K, multi-digit arithmetic, synthetic program execution traces.
3. **Memory & Retrieval:** Passkey/needle-in-haystack, variable tracking across distractors, long-document multi-hop QA.
4. **Representation Diagnostics:** Small-scale chaotic forecasting probe (e.g. Lorenz systems) strictly for representation diagnostics to verify stable bounded states.

### 4.3 Ablations & Efficiency Metrics
- **Ablations:** Compare read-only vs. read/write, single vs. multi-reservoir architectures, frozen vs. partially trainable input projections, distinct spectral radius regimes, and **randomized/stateless dynamics controls** (to verify that useful recurrent dynamics — not merely additional parameters — are the source of gains).
- **Efficiency:** Rigorously track throughput (tokens/sec), p50/p95 step latency, VRAM usage at matched quality, quality-per-FLOP, and quality-per-byte.

## 5. Risks and Mitigations
- **Transformer fails to learn useful write signals:** The non-differentiable nature of the fixed reservoir poses a challenge for write-head optimization. We will use straight-through estimators for gradient approximation, and alternatively explore REINFORCE-style policy gradients. We will also implement entropy regularization/gated write sparsity to prevent noisy control.
- **Distributional Mismatch:** Utilize gated residual mixing and partial layer unfreezing (LoRA) to adapt the LLM to continuous dynamical states.
- **Reservoir Memory Saturation:** Deploy multi-reservoir designs (Track B/C) to decouple short-term computational scratch-space from long-term contextual traces, ensuring bounded-state compression does not degrade.

## 6. Resources and Timeline
The timeline is structured as a **12-month hard-gated core program** focused on Tracks A and B, followed by an **optional 6-month extension** for Track C if all gates are successfully passed.
- **Compute:** 4-8 NVIDIA H100 GPUs.
- **Timeline (12 + 6 Months):** 
  - **Months 1-4:** Track A bolt-on execution, baselines, and Gate A decision.
  - **Months 5-9:** Track B layer insertion, multi-reservoir tuning, and Gate B decision.
  - **Months 10-12:** Initial evaluation suite, ablation studies, negative-result reports (if gates failed).
  - **Months 13-18 (Optional Track C Extension):** From-scratch training curriculum, final multi-branch evaluations, and comprehensive paper.

## 7. Deliverables
1. **Reproducible Codebase:** Full source code for all three tracks, adapter layers, and the RW-Transformer architecture.
2. **Benchmark Harness:** Focused evaluation suite decoupling memory retention and procedural computation.
3. **Negative-Result Report:** Explicit commitment to publish detailed findings if the reservoir workspace fails the Success Gates, ensuring the field benefits regardless of outcome.
4. **Final Paper:** Comprehensive report detailing scaling laws for reservoir size and timescale versus task class, alongside emergent capability benchmarks.

## 8. References
1. Shen, S., Baevski, A., Morcos, A., Keutzer, K., Auli, M., & Kiela, D. (2021). Reservoir Transformers. *ACL 2021*.
2. Anonymous. (2024). Reservoir Transformer at Infinite Horizon: The Lyapunov Time and the Butterfly Effect. *ICLR 2024 Submission*.
3. Qwen Team. (2026). Qwen3.5: Towards Native Multimodal Agents. https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
4. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.
5. Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *arXiv:2405.21060*.
6. Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. *arXiv:2412.06464*.
7. Munkhdalai, T., et al. (2024). Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention. *arXiv:2404.07143*.
8. Behrouz, A., et al. (2025). Titans: Learning to Memorize at Test Time. *arXiv:2501.00663*.
9. Munkhdalai, T., et al. (2025). ATLAS: Learning to Optimally Memorize the Context at Test Time. *arXiv:2505.23735*.
10. Jaeger, H., & Haas, H. (2004). Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication. *Science*, 304, 78-80.
11. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
12. Koester, F., & Uchida, A. (2025). Reservoir Computing as a Language Model. *arXiv:2507.15779*.
13. Gauthier, D. J., et al. (2021). Next Generation Reservoir Computing. *Nature Communications*, 12, 5564.
14. Kong, L.-W., et al. (2024). Reservoir-Computing Based Associative Memory. *Nature Communications*, 15, 4671.
15. Yan, M., et al. (2024). Emerging Opportunities and Challenges for RC. *Nature Communications*, 15, 2056.
16. Echo State Transformer (2025). *arXiv:2507.02917*.
