# Reservoir-Augmented Large Language Models: Dynamical Scratch-Space for Enhanced Memory and Computation

**A Research Proposal**

---

## 1. Executive Summary

Large Language Models (LLMs) exhibit remarkable capabilities in language understanding and generation, yet remain fundamentally limited by finite context windows, weak arithmetic and multi-step reasoning, and the absence of persistent working memory during inference. Separately, reservoir computing (RC) — a framework in which a fixed, recurrent nonlinear dynamical system projects inputs into high-dimensional state spaces — has demonstrated exceptional efficiency for temporal sequence processing, chaotic system forecasting, and associative memory of complex dynamical attractors.

This proposal outlines a research program to systematically investigate the integration of reservoir computing modules into LLM architectures, progressing from read-only preprocessing to a novel **bidirectional dynamical scratch-space** in which the transformer and reservoir form a coupled system. We aim to demonstrate that reservoir augmentation can improve LLM memory capacity, multi-step reasoning, and temporal prediction, while maintaining computational efficiency.

---

## 2. Background and Motivation

### 2.1 Limitations of Current LLMs

Despite rapid scaling, transformer-based LLMs face persistent challenges:

- **Fixed context windows**: Transformers process a bounded number of tokens. Information beyond this window is lost entirely, limiting performance on tasks requiring long-term temporal context.
- **Weak procedural computation**: LLMs struggle with arithmetic, symbolic manipulation, and multi-step algorithmic reasoning because these tasks require precise intermediate state tracking that autoregressive token prediction was not designed for.
- **No persistent working memory**: Each forward pass is stateless. The model has no mechanism to maintain, update, or query a running computation buffer across generation steps, unlike biological cognition which relies heavily on working memory.

### 2.2 Reservoir Computing: Complementary Strengths

Reservoir computing offers properties that directly address these gaps:

- **Fading memory (echo state property)**: Reservoir states naturally encode a decaying trace of recent inputs, functioning as an automatic temporal buffer without explicit memory management.
- **Nonlinear high-dimensional projection**: The reservoir's recurrent dynamics mix inputs nonlinearly, generating rich feature representations that can capture complex temporal dependencies.
- **Extreme training efficiency**: Only the readout layer is trained (typically via linear regression), making RC orders of magnitude cheaper to train than comparable recurrent networks.
- **Proven chaotic system performance**: RC has demonstrated state-of-the-art results in model-free prediction of chaotic attractors, achieving 5–6 Lyapunov times of accurate forecasting, with hybrid schemes extending this to 12+ Lyapunov times.

### 2.3 Existing Hybrid Work

Prior research has explored reservoir-transformer combinations along two axes:

**Reservoir as replacement layers.** Shen et al. (ACL 2021) demonstrated that randomly initialized, frozen nonlinear layers can substitute for trained transformer layers in an alternating pattern, improving wall-clock convergence time while maintaining competitive performance on translation and language modeling. This approach treats reservoirs as cheap computational fillers — the reservoir has no persistent temporal state and serves only as a stateless nonlinear projection.

**Reservoir as input preprocessor.** The "Reservoir Transformer at Infinite Horizon" (ICLR 2024 submission) uses an ensemble of reservoirs to compress arbitrarily long time series histories into fixed-dimensional representations, which are then processed by a transformer. This achieved up to 89.43% error reduction over baseline transformers on chaotic prediction tasks. Here the reservoir has temporal state but the information flow is unidirectional — the transformer cannot influence the reservoir's dynamics.

**Reservoir as standalone language model.** Köster and Uchida (2025) directly compared standalone reservoir computing against transformers for character-level language modeling. Transformers achieved superior prediction quality, while reservoirs were dramatically faster to train and run — highlighting that the paradigms have complementary strengths rather than one dominating the other.

### 2.4 The Missing Piece: Bidirectional Dynamical Scratch-Space

No existing work implements a system where the LLM can **both read from and write to** a reservoir during multi-step inference. We propose that this bidirectional coupling — treating the reservoir as a dynamical scratch-space — represents a qualitatively new architectural paradigm that can unlock capabilities inaccessible to either component alone.

---

## 3. Research Objectives

This program pursues four primary objectives, structured as progressive phases that systematically explore three distinct integration strategies:

1. **Phase 1 — Bolt-On Reservoir Augmentation**: Attach reservoir modules to frozen pretrained LLMs as external preprocessors, training only lightweight adapter layers. This is the lowest-cost, fastest-to-iterate approach and establishes baselines for reservoir–LLM interaction.

2. **Phase 2 — Reservoir Layer Insertion into Existing LLMs**: Surgically insert reservoir layers into the internal architecture of a pretrained LLM (between or alongside existing transformer blocks), requiring partial fine-tuning to adapt surrounding layers. This tests whether reservoirs can serve as persistent dynamical state within the model's own computation graph.

3. **Phase 3 — Reservoir-Native Training from Scratch**: Train a new model from scratch with reservoir modules as first-class architectural components, including the bidirectional scratch-space with read/write channels. This removes distributional mismatch constraints and reveals the full potential of the hybrid architecture.

4. **Phase 4 — Task-Specific Applications and Comparative Analysis**: Apply the best-performing architectures from Phases 1–3 to targeted domains — chaotic system forecasting, multi-step arithmetic/reasoning, and long-horizon planning — and rigorously compare all three integration strategies against each other and against standard LLM baselines.

---

## 4. Technical Approach

### 4.0 Candidate Base Models

A critical experimental design choice is selecting base LLMs at appropriate scales. We propose a multi-scale evaluation strategy centered on the recently released **Qwen3.5 Small Series** (March 2026), with comparisons to other open-weight families.

#### 4.0.1 Primary: Qwen3.5-0.8B-Base

The Qwen3.5-0.8B-Base is an exceptionally strong candidate for this research for several reasons:

**Hybrid architecture with built-in linear attention.** Qwen3.5-0.8B uses a Gated DeltaNet hybrid architecture with a 3:1 ratio of linear attention (Gated DeltaNet) to full softmax attention blocks. Its hidden layout is `6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` with 24 total layers. The linear attention blocks already function somewhat analogously to reservoir-like state-space dynamics — they maintain constant memory complexity via gated decay mechanisms. This means the model is architecturally predisposed to work with recurrent-style state representations, potentially reducing the distributional mismatch problem that plagues bolt-on approaches with pure-attention models.

**Massive native context.** The model supports 262,144 tokens natively and up to 1,010,000 tokens with extension — making it ideal for testing whether reservoir augmentation provides benefits *beyond* what an already-long-context model can achieve through attention alone.

**Small enough for rapid iteration.** At 0.8B parameters with a hidden dimension of 1024, this model can be fine-tuned (even fully) on a single GPU. This is essential for a research program that needs to explore a large space of reservoir configurations and integration strategies. The reservoir's state dimensionality (500–50,000 nodes) is on the same order as the LLM's hidden dimension, simplifying interface design.

**Base model available.** The Base variant (as opposed to Instruct) is released under Apache 2.0, specifically intended for fine-tuning and research. This gives us full flexibility for architectural modifications without fighting instruction-tuned behavior.

**DeltaNet–Reservoir synergy hypothesis.** The Gated DeltaNet blocks use a delta rule for hidden state updates — mathematically related to the echo state network update rule. We hypothesize that the reservoir's state will be more naturally "legible" to DeltaNet attention blocks than to standard softmax attention, since both operate on recurrent state representations. This is a testable hypothesis unique to this model family.

#### 4.0.2 Scale-Up Candidates

To study how reservoir augmentation interacts with model capacity, we will additionally test:

- **Qwen3.5-2B**: Same architecture family, 2.5× larger. Tests whether reservoir benefits diminish as the LLM's own capacity increases.
- **Qwen3.5-9B**: The reasoning-optimized variant trained with Scaled RL. Tests whether reservoir augmentation is complementary to or redundant with RL-enhanced reasoning.
- **LLaMA-3.2-1B / 3B**: Pure softmax-attention architecture, serving as a control to isolate whether the DeltaNet–reservoir synergy hypothesis holds. If reservoir augmentation works significantly better with Qwen3.5's hybrid architecture than with LLaMA's pure attention, this validates the architectural compatibility argument.

#### 4.0.3 Base vs. Instruct

We will primarily use Base models for Phases 1–3, since architectural modifications are cleaner without instruction-tuning artifacts. For Phase 4 task evaluations, we will compare reservoir-augmented Base models against both reservoir-augmented Instruct models and unmodified Instruct baselines to disentangle the effects of reservoir augmentation from instruction tuning.

### 4.1 Phase 1: Bolt-On Reservoir Augmentation

This phase attaches reservoir modules externally to a frozen pretrained LLM, training only lightweight adapters.

#### 4.1.1 Architecture

The simplest integration places one or more echo state networks (ESNs) upstream of a frozen pretrained LLM. The reservoir consumes a raw input sequence (token embeddings, time series values, or sensor data) and evolves its recurrent state. At each timestep, the reservoir state vector is mapped through a learned projection layer into the LLM's embedding space.

```
Input Sequence → [Reservoir (frozen)] → [Projection (trained)] → [LLM (frozen or LoRA)]
```

We will test two bolt-on configurations:

**Prepended reservoir tokens.** The reservoir state at each timestep is projected into the LLM's embedding space and prepended as additional "virtual tokens" to the input sequence, similar to soft prompt tuning. The LLM attends to these virtual tokens alongside the real input.

**Cross-attention sidecar.** A small cross-attention module is inserted at selected layers of the frozen LLM. The LLM's hidden states serve as queries; the reservoir states (mapped through a learned projection) serve as keys and values. This is more expressive but requires slightly more invasive modification.

#### 4.1.2 Adapter Strategy

We will evaluate three levels of LLM adaptation:

- **Projection only**: Train a linear or small MLP projection from reservoir state dimensionality to LLM embedding dimensionality. LLM weights are entirely frozen.
- **Projection + LoRA**: Additionally apply low-rank adaptation to the LLM's attention layers, enabling it to learn reservoir-aware attention patterns with minimal parameter overhead.
- **Projection + partial fine-tuning**: Unfreeze the LLM's first few layers to allow deeper distributional adaptation to reservoir-derived inputs.

#### 4.1.3 Reservoir Configuration

We will systematically explore:

- **Reservoir size**: 500 to 50,000 nodes, characterizing scaling behavior.
- **Spectral radius**: Sweep from 0.1 to 1.5, bracketing the edge-of-chaos regime where computational capacity is typically maximized.
- **Input scaling**: Control the degree of nonlinear mixing applied to inputs.
- **Leak rate**: Tune the effective memory timescale of the reservoir.
- **Topology**: Compare sparse random (Erdős–Rényi), small-world, scale-free, and deterministic (e.g., logistic map-based) reservoir topologies.

#### 4.1.4 Evaluation

- **Long-context language modeling**: Measure perplexity on documents exceeding the LLM's native context window, comparing reservoir-augmented vs. standard positional encoding extensions (RoPE scaling, ALiBi).
- **Chaotic time series**: Forecast Lorenz-63, Mackey-Glass, and Kuramoto-Sivashinsky systems, comparing reservoir-augmented LLM against standalone RC, standalone transformer, and the Infinite Horizon architecture.
- **Efficiency**: Measure wall-clock training time, inference latency, and memory footprint.

### 4.2 Phase 2: Reservoir Layer Insertion into Existing LLMs

This phase goes beyond external attachment to embed reservoir modules *within* the transformer's layer stack, creating persistent dynamical state that the model can access at intermediate computation stages.

#### 4.2.1 Architecture

We explore three insertion strategies:

**Interleaved reservoir layers (Shen et al. variant, extended).** Following the Reservoir Transformer approach but with a critical modification: our inserted reservoir layers maintain persistent recurrent state across tokens, unlike Shen et al.'s stateless random projections. After every $k$ transformer blocks, we insert a reservoir interaction layer where:
1. The current hidden state is projected into the reservoir's input space and fed as driving input
2. The reservoir evolves one step
3. The reservoir state is projected back and added (via a learned gating mechanism) to the hidden state

This creates a recurrent "backbone" threaded through the transformer's feedforward computation.

**Parallel reservoir stream.** Rather than inserting sequentially, we run the reservoir as a parallel stream alongside the transformer layers. At designated synchronization points (every $k$ layers), the two streams exchange information via learned cross-projections. This avoids disrupting the transformer's residual stream while still providing dynamical state.

**Reservoir replacement of DeltaNet blocks (Qwen3.5-specific).** For Qwen3.5 models, we exploit the existing hybrid architecture by replacing some or all of the Gated DeltaNet linear attention blocks with ESN reservoir modules. Since DeltaNet blocks already function as recurrent state-space layers, the reservoir serves as a "richer" recurrent module with higher-dimensional state. The surrounding full-attention blocks and FFN layers remain unchanged. This is the most architecturally natural insertion point.

#### 4.2.2 Training Strategy

Layer insertion requires partial fine-tuning since the LLM's internal representations will be disrupted. We adopt a staged approach:

1. **Freeze the reservoir** (as always) and the majority of transformer layers.
2. **Train the interface projections** (reservoir input/output projections and gating mechanisms).
3. **LoRA-adapt adjacent transformer layers** (the 1–2 layers immediately before and after each insertion point) to accommodate the new information flow.
4. **Optional: brief full fine-tuning** with a very low learning rate to allow global adaptation.

We will carefully measure performance at each stage to determine the minimum fine-tuning required.

#### 4.2.3 Evaluation

In addition to the Phase 1 benchmarks, Phase 2 adds:

- **Representation analysis**: Use probing classifiers on intermediate hidden states to test whether reservoir-inserted layers carry different (and complementary) information compared to the layers they augment.
- **Gradient flow analysis**: Verify that inserting reservoir modules does not disrupt gradient flow through the remaining trained layers (monitoring gradient norms and training stability).
- **Comparison with state-space model baselines**: Since the reservoir insertion is functionally similar to injecting Mamba or S4 layers into a transformer, we compare against these established hybrid architectures.

### 4.3 Phase 3: Reservoir-Native Training from Scratch

This phase removes all constraints of pretrained weight compatibility and trains a new model with the bidirectional reservoir scratch-space as a first-class component.

#### 4.3.1 Architecture

We propose a coupled transformer-reservoir system with explicit read and write channels, designed and trained together from initialization:

```
                    ┌──────────────────────────────────┐
                    │        Reservoir Module(s)        │
                    │   Fixed recurrent dynamics, no    │
                    │   backpropagation through state   │
                    └──────┬──────────────┬─────────────┘
                      Read │              │ Write
                      (r_t)│              │ (w_t)
                           ▼              ▲
                    ┌──────────────────────────────────┐
                    │         Transformer LLM           │
                    │                                    │
                    │  Cross-attention over r_t          │
                    │  Write head produces w_t           │
                    │  Standard autoregressive decode    │
                    └──────────────────────────────────┘
```

**Read interface**: At each decoding step $t$, the reservoir state $r_t \in \mathbb{R}^{D_r}$ is exposed to the transformer via a cross-attention mechanism. The transformer's queries attend over a set of learned "slots" derived from the reservoir state (analogous to how Perceiver/Perceiver IO attends over latent arrays). This allows the transformer to selectively extract relevant information from the reservoir's high-dimensional dynamical trace.

**Write interface**: A dedicated write head (a small MLP attached to the transformer's final hidden state) produces a write vector $w_t \in \mathbb{R}^{D_{in}}$ that is injected into the reservoir as external input at the next timestep. This allows the transformer to **steer** the reservoir's dynamics — pushing it toward attractor basins that encode useful intermediate computations.

**Reservoir evolution**: Between steps, the reservoir evolves according to:

$$r_{t+1} = (1 - \alpha) \cdot r_t + \alpha \cdot f(W_{in} \cdot [x_t; w_t] + W_{res} \cdot r_t)$$

where $x_t$ is the current token embedding, $w_t$ is the transformer's write vector, $\alpha$ is the leak rate, $f$ is a nonlinear activation (typically $\tanh$), and $W_{in}$, $W_{res}$ are the fixed input and reservoir weight matrices.

#### 4.3.2 Training Strategy (From-Scratch)

A critical design decision is what gets trained. The reservoir weights ($W_{in}$, $W_{res}$) remain fixed by definition — this is a core RC principle we preserve even in from-scratch training. The trainable components are:

- The **read projection** (reservoir state → cross-attention keys/values)
- The **write head** (transformer hidden state → reservoir input)
- **LoRA adapters** on the transformer's attention layers
- Optionally, a **nonlinear readout** on the reservoir (small MLP) following the approach from the Infinite Horizon paper

We will investigate two training regimes:

- **Transformer-only gradients**: All transformer parameters, interface projections, and read/write heads are trained end-to-end via standard backpropagation. The reservoir is treated as a fixed "environment" that the transformer learns to interact with. Gradients stop at the reservoir boundary.
- **Straight-through estimation for write optimization**: For the write pathway, gradients cannot flow through the fixed reservoir dynamics. We will explore straight-through estimators, REINFORCE-style policy gradient for the write head, and evolution strategies as alternatives for optimizing the write signal.

**From-scratch model sizing.** To enable direct comparison with the bolt-on (Phase 1) and insertion (Phase 2) experiments on Qwen3.5-0.8B, we will train from-scratch models at approximately matching total parameter count (~0.8B trainable parameters in the transformer, plus the fixed reservoir). We will use the same tokenizer (Qwen3.5's 248K vocabulary) to eliminate tokenization as a confound.

#### 4.3.3 Multi-Reservoir Configurations

Drawing on the ensemble reservoir approach, we will explore:

- **Parallel reservoirs with different timescales**: A fast reservoir ($\alpha$ near 1.0, spectral radius near 1.0) for short-term arithmetic scratch-space, and a slow reservoir ($\alpha$ near 0.1, spectral radius near 0.5) for long-term contextual memory. The transformer attends to both.
- **Hierarchical reservoirs**: A small, fast reservoir feeds into a larger, slower reservoir, creating a multi-scale temporal buffer. The transformer can read from any level.
- **Task-specific reservoir banks**: Different reservoirs optimized (via hyperparameter selection, not gradient training) for different functions — one for temporal pattern detection, one for associative memory, one for accumulating sequential computation results.

#### 4.3.4 Evaluation

- **Synthetic reasoning tasks**: Multi-step arithmetic (addition of large numbers, multiplication), variable tracking (assign and recall values across long sequences), and algorithmic execution (sorting, searching).
- **Associative memory**: Store and retrieve key-value pairs injected at arbitrary positions in long sequences, measuring recall accuracy as a function of sequence length and number of stored items.
- **Ablation studies**: Systematically remove the read channel, write channel, or both to quantify the contribution of each.
- **Dynamical analysis**: Characterize the reservoir state trajectories during successful vs. failed reasoning, examining whether the transformer learns to use distinct attractor basins for different computations.

### 4.4 Phase 4: Domain Applications and Cross-Strategy Comparison

This phase applies the best architectures from Phases 1–3 to real-world domains and performs rigorous cross-comparison of all three integration strategies.

#### 4.4.1 Integration Strategy Comparison Matrix

For each downstream task, we evaluate:

| Configuration | Base Model | RC Integration | Training Cost | Key Question |
|---------------|-----------|----------------|---------------|--------------|
| Baseline | Qwen3.5-0.8B (unmodified) | None | Zero | How well does the base model perform? |
| Bolt-on (Phase 1) | Qwen3.5-0.8B (frozen + LoRA) | External preprocessor | Low | Can RC help a pretrained model without retraining? |
| Layer insertion (Phase 2) | Qwen3.5-0.8B (partially fine-tuned) | Internal interleaved/parallel | Medium | Does internal RC state outperform external attachment? |
| From-scratch (Phase 3) | Custom ~0.8B transformer | Bidirectional scratch-space | High | What is the ceiling when there are no compatibility constraints? |
| Scale control | Qwen3.5-2B / 9B (frozen + LoRA) | External preprocessor | Low | Do RC benefits diminish with larger models? |
| Architecture control | LLaMA-3.2-1B (frozen + LoRA) | External preprocessor | Low | Is DeltaNet–RC synergy real? |

#### 4.4.2 Chaotic System Forecasting

Apply the bidirectional architecture to predict:

- Lorenz-63 and Lorenz-96 systems (varying dimensionality)
- Mackey-Glass delay differential equation (varying delay parameter $\tau$ to control chaotic complexity)
- Kuramoto-Sivashinsky equation (spatiotemporal chaos)
- Real-world chaotic data: weather prediction, financial time series

The key hypothesis is that the bidirectional architecture outperforms both the read-only reservoir preprocessor and standalone transformers because the transformer can actively modulate the reservoir's dynamics to maintain prediction-relevant state information beyond the reservoir's natural fading memory horizon.

#### 4.4.3 Enhanced LLM Reasoning

Evaluate on established benchmarks:

- **GSM8K / MATH**: Grade-school and competition mathematics requiring multi-step arithmetic and reasoning.
- **ARC (Abstraction and Reasoning Corpus)**: Pattern recognition and procedural reasoning.
- **BigBench-Hard**: Diverse tasks selected for difficulty, including tracking shuffled objects, logical deduction, and multi-step algorithms.
- **Long-context QA**: Tasks requiring integration of information scattered across very long documents.

#### 4.4.4 Robotics and Control

Time-series-native applications where the reservoir's dynamical properties are most naturally exploited:

- Model predictive control with learned dynamics models
- Online adaptation to distributional shift in sensor data
- Multi-agent coordination requiring persistent state tracking

---

## 5. Expected Contributions

1. **First bidirectional reservoir-transformer architecture** with explicit read/write channels, establishing a new paradigm for dynamical working memory in LLMs.

2. **Systematic comparison of three integration strategies** — bolt-on attachment, layer insertion, and from-scratch training — providing practical guidance on cost-benefit tradeoffs for reservoir augmentation of existing vs. new models.

3. **Training methodology** for optimizing transformer-to-reservoir write signals without backpropagation through the reservoir, applicable to any fixed-dynamics external module.

4. **Multi-reservoir design principles** characterizing how reservoir hyperparameters (timescale, spectral radius, topology) should be configured for different cognitive functions (short-term computation vs. long-term memory vs. associative recall).

5. **DeltaNet–reservoir synergy analysis**: First empirical study of whether hybrid linear-attention architectures (Gated DeltaNet) are more compatible with reservoir augmentation than pure softmax-attention transformers, with implications for future hybrid architecture design.

6. **Open-source implementation** of all architectures, training pipelines, and evaluation benchmarks, including reservoir integration modules compatible with the Hugging Face Transformers ecosystem.

---

## 6. Timeline

| Phase | Duration | Milestones |
|-------|----------|------------|
| **Phase 1**: Bolt-on augmentation | Months 1–5 | Adapter training pipeline on Qwen3.5-0.8B-Base; reservoir hyperparameter sweep; long-context and chaotic forecasting baselines; LLaMA control experiments |
| **Phase 2**: Layer insertion | Months 4–10 | Interleaved, parallel, and DeltaNet-replacement architectures; partial fine-tuning methodology; representation and gradient flow analysis |
| **Phase 3**: From-scratch training | Months 8–16 | Bidirectional scratch-space architecture; write-head optimization methodology; multi-reservoir configurations; synthetic task evaluation |
| **Phase 4**: Domain applications | Months 12–22 | Cross-strategy comparison matrix; chaotic forecasting benchmarks; reasoning benchmarks (GSM8K, ARC, BigBench-Hard); robotics proof-of-concept |
| **Analysis and writing** | Months 18–24 | Dynamical analysis of learned reservoir usage; DeltaNet–RC synergy analysis; ablation studies; paper preparation and open-source release |

---

## 7. Required Resources

- **Compute**: 4–8 A100 or H100 GPUs for LLM fine-tuning and from-scratch training experiments. Reservoir simulation itself is CPU-efficient and can run on standard hardware. Phase 1 bolt-on experiments on Qwen3.5-0.8B can run on a single GPU.
- **Base models**: Primary — Qwen3.5-0.8B-Base (Apache 2.0, Hugging Face). Scale-up — Qwen3.5-2B-Base and Qwen3.5-9B-Base. Architecture control — LLaMA-3.2-1B and LLaMA-3.2-3B.
- **Reservoir software**: ReservoirPy (Python) for ESN implementation, with custom extensions for the bidirectional interface and layer-insertion modules.
- **Benchmarks**: Standard datasets (GSM8K, ARC, BigBench-Hard, LoCoMo, LongBench v2) plus custom chaotic system generators.
- **Estimated total GPU-hours**: ~15,000 hours across all phases (Phase 1: ~1,000h; Phase 2: ~3,000h; Phase 3: ~8,000h; Phase 4: ~3,000h).

---

## 8. Risks and Mitigation

**Risk: The transformer fails to learn useful write signals.**
The write channel optimization is the most technically uncertain component, since gradients cannot flow through the fixed reservoir. *Mitigation*: We will implement multiple optimization strategies (straight-through estimators, REINFORCE, evolution strategies) and also explore a simpler "echo" write mode where the transformer's hidden state is directly injected without a learned mapping, relying on the reservoir's natural dynamics to produce useful transformations.

**Risk: Distributional mismatch between reservoir states and LLM expectations.**
Pretrained LLMs expect token embeddings with specific statistical properties. *Mitigation*: Phase 1 explicitly studies adapter strategies to bridge this gap before introducing the more complex bidirectional architecture.

**Risk: Reservoir memory capacity is insufficient for complex reasoning.**
Standard ESNs have limited memory capacity that scales linearly with reservoir size. *Mitigation*: Multi-reservoir configurations with different timescales, plus exploration of next-generation RC architectures (e.g., nonlinear vector autoregression-based NGRC) that achieve comparable performance with far fewer nodes.

**Risk: Added latency from reservoir simulation negates efficiency gains.**
*Mitigation*: Reservoir evolution is a single matrix-vector multiply per timestep — orders of magnitude cheaper than a transformer forward pass. The overhead should be negligible. For large-scale deployment, reservoir simulation can be parallelized or implemented on analog/photonic hardware.

**Risk: DeltaNet–reservoir synergy hypothesis does not hold.**
The hypothesis that Qwen3.5's Gated DeltaNet layers are more compatible with reservoir states than standard softmax attention may prove unfounded. *Mitigation*: This is why we include LLaMA-3.2 as an architecture control. Even if the synergy hypothesis fails, the core research on bolt-on, insertion, and from-scratch integration strategies remains valid and informative. A null result on DeltaNet synergy is itself a publishable finding that informs hybrid architecture design.

---

## 9. Related Work and Differentiation

| Approach | Read from RC | Write to RC | RC has temporal state | Trains RC weights | Integration mode |
|----------|:---:|:---:|:---:|:---:|:---|
| Reservoir Transformers (Shen et al. 2021) | ✓ | ✗ | ✗ | ✗ | Layer replacement |
| Reservoir Transformer at Infinite Horizon | ✓ | ✗ | ✓ | ✗ | Bolt-on preprocessor |
| Köster & Uchida 2025 | N/A | N/A | ✓ | ✗ | Standalone |
| Memory-Augmented LLMs (MAuLLM, ChatDB) | ✓ | ✓ | ✗ (discrete) | N/A | Bolt-on memory |
| **This proposal (Phase 1)** | **✓** | **✗** | **✓** | **✗** | **Bolt-on preprocessor** |
| **This proposal (Phase 2)** | **✓** | **✓ (implicit)** | **✓** | **✗** | **Layer insertion** |
| **This proposal (Phase 3)** | **✓** | **✓ (explicit)** | **✓** | **✗** | **From-scratch** |

Our approach is unique in combining bidirectional information flow with a temporally persistent, continuous dynamical system that requires no gradient-based training of its internal dynamics.

---

## 10. References

1. Shen, S., Baevski, A., Morcos, A., Keutzer, K., Auli, M., & Kiela, D. (2021). Reservoir Transformers. *Proceedings of ACL 2021*, 4294–4309.
2. Köster, F. & Uchida, A. (2025). Reservoir Computing as a Language Model. *arXiv:2507.15779*.
3. Anonymous. (2024). Reservoir Transformer at Infinite Horizon: The Lyapunov Time and the Butterfly Effect. *ICLR 2024 Submission*.
4. Jaeger, H. & Haas, H. (2004). Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication. *Science*, 304, 78–80.
5. Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach. *Physical Review Letters*, 120, 024102.
6. Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021). Next Generation Reservoir Computing. *Nature Communications*, 12, 5564.
7. Kong, L.-W., Weng, T., Lu, B., Wang, X., Zhu, J., & Lai, Y.-C. (2024). Reservoir-Computing Based Associative Memory and Itinerancy for Complex Dynamical Attractors. *Nature Communications*, 15, 4671.
8. Yan, M., Huang, C., Bienstman, P., Tiño, P., Lin, W., & Sun, J. (2024). Emerging Opportunities and Challenges for the Future of Reservoir Computing. *Nature Communications*, 15, 2056.
9. Schuurmans, D. (2023). Memory Augmented Large Language Models are Computationally Universal. *arXiv:2301.04589*.
10. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
11. Qwen Team. (2026). Qwen3.5: Towards Native Multimodal Agents. *https://qwen.ai/blog?id=qwen3.5*. Models available at *https://huggingface.co/Qwen/Qwen3.5-0.8B-Base*.
12. Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. *arXiv:2412.06464*.
