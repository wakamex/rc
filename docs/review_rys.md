# Review: RYS (Repeat Your Self) — Layer Duplication for Improved Reasoning

**Source:** https://github.com/dnhkng/dnhkng.github.io/blob/main/_posts/2026-03-10-rys.md
**Author:** David Noel Ng (dnhkng)
**Reviewed:** 2026-03-22

## Summary

dnhkng took the #1 spot on the HuggingFace Open LLM Leaderboard by duplicating a block of 7 middle layers in Qwen2-72B — no training, no weight changes, just running layers 45–51 twice. The method (RYS) discovers "LLM Neuroanatomy": transformers develop functionally distinct regions (encoding → reasoning circuits → decoding) during pre-training. Block-level duplication works but single-layer duplication doesn't, providing empirical evidence for circuit-level organization in the residual stream.

## Key Findings

1. **Middle layers form coherent reasoning circuits** that can be re-executed. Their outputs are "compatible" with their own inputs — the loop improves performance because the circuit refines its own intermediate representations.

2. **Layer-duplication heatmaps** across architectures (Llama-3-70B, Phi-3-medium, GLM-4.7, Qwen3-30B-A3B) reveal consistent functional anatomy: encoding region (early), reasoning circuits (middle), decoding region (late). Different models show different specific boundaries but the general pattern holds.

3. **Block-level sensitivity**: duplicating a contiguous block of ~7 layers helps, but single-layer duplication doesn't. This suggests tightly coupled multi-layer circuits, not independently useful layers.

4. **VRAM-free**: pointer-based duplication shares weights between original and duplicated layers. No extra VRAM, just more compute + KV cache.

## Strengths

- The heatmap methodology is the real contribution — a cheap, general tool for probing transformer functional anatomy
- Cross-architecture consistency supports the "functional anatomy" framing
- Good scientific narrative: observation → hypothesis → probe → sweep → validation
- Practical: immediately usable with any model

## Weaknesses

- "Topped the leaderboard" framing oversells — the intellectual contribution is the heatmap methodology
- Base64 "clue" is weaker than presented (Base64 is heavily represented in training data)
- No ablation on proxy tasks — different probes might yield different optimal (i,j) configs
- "Circuit" used loosely (not in the mechanistic interpretability sense)
- Statistical rigor thin: 16 math questions, no confidence intervals over 3,241 configurations
- Missing comparison to depth-scaling literature (Universal Transformers, CALM, adaptive depth)

## Relevance to Our RC Project

### Direct connections

1. **Why Track B failed**: If DeltaNet layers are part of tightly coupled circuits (as RYS suggests), replacing individual layers with ESN modules disrupts circuit integrity. This explains why even single-layer replacement (B7) hurts — you can't swap one piece of a multi-layer circuit without breaking the whole thing.

2. **Sidecar injection placement**: The heatmaps can identify circuit *boundaries* — the junctions between functional regions. On larger models with more differentiated anatomy, concentrating sidecar injection at these boundaries (instead of uniform every-4th-layer) might make the sidecar relevant at scale. This could explain the 4B no-op result: uniform injection misses the model's actual functional structure.

3. **Scaling the ESN's contribution**: RYS-style layer repetition naturally amplifies the sidecar's impact. On a second pass through a circuit block, the ESN accumulates more state, effectively doubling its signal without increasing reservoir size. The experiment: does RYS + sidecar produce superlinear gains?

### Actionable experiments

**Cheap (hours):** Run the layer-duplication sweep on Qwen3.5-0.8B (~300 configs, seconds each). Produces a heatmap of our target model's functional anatomy. Compare with/without sidecar to see if the sidecar changes which circuits benefit from repetition.

**Medium (1-2 days):** Per-layer sidecar contribution probe. For each of 24 layers, attach sidecar at *only that layer*, measure ppl. This creates a sidecar-specific version of the heatmap showing where reservoir injection has maximum impact. Cross-reference with the RYS heatmap to see if injection points correspond to circuit boundaries.

**Ambitious (1 week):** Learned adaptive injection. Instead of fixed injection points, train a lightweight controller that decides per-input which layers get sidecar signal. The RYS heatmap provides ground truth for what "good injection" looks like. The ESN state could drive this controller — its recurrent state tracks computational progress through the network and could learn to detect when a circuit boundary is approaching.

### Key insight for scaling

The 4B sidecar failure might not be about reservoir size — it might be about injection placement. On 0.8B, everything is entangled, so uniform injection helps everywhere a little. On 4B, the model has clearer functional anatomy, and injecting at the wrong points is noise. The heatmap tells you where the right points are.
