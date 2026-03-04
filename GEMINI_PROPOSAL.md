# Research Proposal: Latent Reservoir Scratchpads (LRS)
## Enhancing LLM Working Memory and Arithmetic Logic via Hybrid Dynamical Systems

**Date:** March 4, 2026  
**Subject:** Architectural Innovation in Large Language Models (LLMs)  
**Focus:** Reservoir Computing, Transformers, Chaotic Systems, Symbolic Computation

---

## 1. Executive Summary
This research proposes a hybrid architecture, the **Latent Reservoir Scratchpad (LRS)**, which integrates an Echo State Network (ESN) as a non-trainable, high-dimensional "working memory" within a Transformer block. Unlike standard KV-caching, the LRS acts as a continuous-state dynamical system that maintains temporal context and intermediate computational states, specifically targeting improvements in multi-step arithmetic and long-context reasoning.

---

## 2. Research Objectives
1.  **Memory Persistence:** Evaluate if a reservoir can maintain "hidden" variables across sequence lengths that exceed the Transformer’s native context window.
2.  **Computational Offloading:** Determine if symbolic tasks (e.g., multi-digit multiplication, parity checks) can be mapped to the high-dimensional nonlinear space of a reservoir more efficiently than through standard self-attention.
3.  **Stability in Chaos:** Test the model's ability to forecast chaotic attractors (e.g., Lorenz system) where standard Transformers typically suffer from accumulation of error (drift).

---

## 3. Proposed Architecture: The LRS Layer
The LRS will be implemented as a parallel module within the mid-to-late layers of a decoder-only Transformer (e.g., Llama-3.2-1B or Phi-3).

### 3.1 Architectural Schematic


### 3.2 Mathematical Framework
At each timestep $t$, the Transformer hidden state $h_t$ is injected into the reservoir:

1.  **Reservoir Update:** $$x_t = (1 - \alpha)x_{t-1} + \alpha \cdot \tanh(W_{res}x_{t-1} + W_{in}h_t + b_{res})$$  
    *Where $\alpha$ is the leaking rate and $W_{res}$ is a fixed, sparse matrix with a spectral radius $\rho \approx 1$.*

2.  **Latent Readout:** $$y_t = W_{out}x_t$$  
    *Only $W_{out}$ is trained via standard backpropagation or ridge regression.*

3.  **Integration:** The final output is a gated fusion: $H_{out} = \text{LayerNorm}(h_t + \sigma(g) \cdot y_t)$.

---

## 4. Methodology & Implementation
### 4.1 Development Stack
* **Framework:** PyTorch (Core tensor operations).
* **Libraries:** `EsnTorch` for reservoir dynamics; `Hugging Face Transformers` for base model weights.
* **Optimization:** Since the reservoir weights are fixed, we will focus on **Parameter-Efficient Fine-Tuning (PEFT)** for the readout layers ($W_{out}$) and the injection matrices ($W_{in}$).

### 4.2 Experimental Phases
| Phase | Task | Dataset | Metric |
| :--- | :--- | :--- | :--- |
| **I: Baseline** | Sequence Recall | Random Bit Strings | Bit-level accuracy @ 10k tokens |
| **II: Logic** | N-step Arithmetic | GSM8K / Synthetic Math | Step-wise correctness |
| **III: Dynamics** | Chaotic Forecasting | Lorenz / Kuramoto-Sivashinsky | Lyapunov Time Horizon |

---

## 5. Hypotheses
* **H1:** The reservoir will mitigate "Attention Glitch" in arithmetic by providing a stable, recurrent representation of carrying digits.
* **H2:** The LRS will require significantly less VRAM than an equivalent-sized KV-cache for sequences longer than 32k tokens.
* **H3:** The fixed dynamics of the reservoir will act as a "regularizer," preventing the model from diverging during long-term time-series forecasting.

---

## 6. Resources Required
* **Compute:** 4x NVIDIA H100 (for rapid iteration on 1B-3B parameter models).
* **Data:** Synthetic symbolic traces and high-frequency sensor data from nonlinear systems.

---

## 7. Expected Contributions
This research aims to provide a path toward **O(1) Memory Scaling** for specific computational tasks, moving away from the "memory-as-a-lookup-table" (Attention) toward "memory-as-a-dynamical-state" (Reservoir).
