"""Model loading & tokenizer utilities for LRS project.

Provides a unified interface to load Qwen3.5-0.8B-Base (hybrid DeltaNet +
full-attention), LLaMA-3.2-1B (pure softmax attention), and Mamba-2 1.3B
(SSM architecture) in FP16/BF16.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.types import ModelWrapper as ModelWrapperProtocol  # noqa: F401 (imported for protocol conformance check)

# ---------------------------------------------------------------------------
# Model registry: short name → HuggingFace model ID
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B-Base",  # hybrid DeltaNet + full-attention
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",  # pure softmax attention
    "mamba2-1.3b": "state-spaces/mamba2-1.3b-hf",  # SSM architecture
}

# Patterns that identify DeltaNet (linear attention) layers in Qwen3.5
DELTANET_PATTERNS: list[str] = ["deltanet", "delta_net", "linear_attn", "linear_attention"]

# Standard full-attention patterns (used for filtering)
ATTENTION_PATTERNS: list[str] = ["attn", "attention", "self_attn"]

# LoRA target module names per model family
LORA_TARGETS: dict[str, list[str]] = {
    "qwen3.5-0.8b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama-3.2-1b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mamba2-1.3b": ["in_proj", "out_proj", "x_proj", "dt_proj"],
}

# Bytes per parameter for dtype estimation
_BYTES_PER_PARAM: dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

# Approximate total parameter counts (billions) per model
_MODEL_PARAMS_B: dict[str, float] = {
    "qwen3.5-0.8b": 0.8,
    "llama-3.2-1b": 1.2,
    "mamba2-1.3b": 1.3,
}


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


def estimate_vram(model_name: str, dtype: torch.dtype = torch.float16) -> int:
    """Estimate VRAM requirement in bytes before loading.

    Accounts for model weights plus ~20% overhead for activations and
    KV-cache at inference time.

    Args:
        model_name: Short model key (e.g. "qwen3.5-0.8b").
        dtype: Target dtype for the model weights.

    Returns:
        Estimated bytes of VRAM needed.
    """
    params_b = _MODEL_PARAMS_B.get(model_name.lower(), 1.0)
    bytes_per_param = _BYTES_PER_PARAM.get(dtype, 2)
    weight_bytes = math.ceil(params_b * 1e9 * bytes_per_param)
    overhead = math.ceil(weight_bytes * 0.2)
    return weight_bytes + overhead


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------


class TokenizerWrapper:
    """Consistent tokenizer API across all supported model families."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model_name: str) -> None:
        self._tok = tokenizer
        self.model_name = model_name
        # Ensure pad token is defined (required for batched inference)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    def encode(self, text: str | list[str], **kwargs: Any) -> torch.Tensor:
        """Tokenize text and return input_ids tensor (batch, seq_len)."""
        out = self._tok(text, return_tensors="pt", **kwargs)
        return out["input_ids"]

    def decode(self, token_ids: torch.Tensor, **kwargs: Any) -> str | list[str]:
        """Decode token ids to string(s)."""
        if token_ids.dim() == 1:
            return self._tok.decode(token_ids, skip_special_tokens=True, **kwargs)
        return self._tok.batch_decode(token_ids, skip_special_tokens=True, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying tokenizer."""
        return self._tok(*args, **kwargs)


# ---------------------------------------------------------------------------
# ModelWrapper implementation
# ---------------------------------------------------------------------------


class ModelWrapperImpl:
    """Wrapped language model satisfying the ModelWrapper protocol.

    Implements forward / generate / get_hidden as required by src.types.ModelWrapper.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: TokenizerWrapper,
        model_name: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device

    # -- ModelWrapper protocol methods ----------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        """Run a forward pass and return model outputs (with .logits)."""
        return self.model(input_ids.to(self.device), **kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate token sequences starting from input_ids."""
        return self.model.generate(input_ids.to(self.device), **kwargs)

    def get_hidden(self, input_ids: torch.Tensor, layer: int = -1, **kwargs: Any) -> torch.Tensor:
        """Return hidden states at the specified transformer layer.

        Args:
            input_ids: Token ids (batch, seq_len).
            layer: Layer index. 0 = embedding output, -1 = final layer.

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim).
        """
        outputs = self.model(
            input_ids.to(self.device),
            output_hidden_states=True,
            **kwargs,
        )
        return outputs.hidden_states[layer]

    # -- Convenience properties ------------------------------------------------

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype


# ---------------------------------------------------------------------------
# Layer inspection utilities
# ---------------------------------------------------------------------------


def list_layer_names(model: PreTrainedModel) -> list[str]:
    """Return all named module paths in the model (depth-first)."""
    return [name for name, _ in model.named_modules()]


def get_deltanet_layers(model: PreTrainedModel) -> list[str]:
    """Identify DeltaNet (linear attention) layer names in a Qwen3.5 model.

    Scans module names for any of the patterns in DELTANET_PATTERNS.
    These are the recurrent linear-attention blocks that alternate with
    full softmax-attention blocks in the Qwen3.5 hybrid architecture.

    Returns:
        Sorted list of module path strings that are (part of) DeltaNet blocks.
    """
    deltanet = []
    for name, _ in model.named_modules():
        name_lower = name.lower()
        if any(pat in name_lower for pat in DELTANET_PATTERNS):
            deltanet.append(name)
    return deltanet


def get_full_attention_layers(model: PreTrainedModel) -> list[str]:
    """Identify full softmax-attention layer names (excluding DeltaNet).

    Returns:
        Sorted list of module path strings that belong to standard attention
        blocks (e.g. self_attn, MultiheadAttention) but are NOT DeltaNet layers.
    """
    full_attn = []
    for name, _ in model.named_modules():
        name_lower = name.lower()
        is_attn = any(pat in name_lower for pat in ATTENTION_PATTERNS)
        is_deltanet = any(pat in name_lower for pat in DELTANET_PATTERNS)
        if is_attn and not is_deltanet:
            full_attn.append(name)
    return full_attn


def get_lora_targets(model_name: str) -> list[str]:
    """Return recommended LoRA target module names for the given model family.

    Args:
        model_name: Short model key.

    Returns:
        List of sub-module names to pass as `target_modules` to PEFT LoraConfig.
        Falls back to a standard attention projection set for unknown models.
    """
    key = model_name.lower()
    return LORA_TARGETS.get(key, ["q_proj", "k_proj", "v_proj", "o_proj"])


# ---------------------------------------------------------------------------
# Hidden state extraction utility
# ---------------------------------------------------------------------------


def get_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    layer_indices: list[int],
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Extract hidden states at multiple layer indices in a single forward pass.

    Args:
        model: The language model (must support output_hidden_states=True).
        input_ids: Token ids of shape (batch, seq_len).
        layer_indices: Layer indices to extract (0 = embedding, -1 = last layer,
            positive integers = specific layers).
        device: Device to move inputs to. Defaults to the model's device.

    Returns:
        Mapping from layer index to hidden state tensor (batch, seq_len, hidden_dim).
        Tensors are detached from the computation graph.
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple: (n_layers + 1,) of (B, T, H)
    return {idx: hidden_states[idx].detach() for idx in layer_indices}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
    trust_remote_code: bool = True,
    model_id: str | None = None,
) -> ModelWrapperImpl:
    """Load a model and return a ModelWrapperImpl.

    Supported short names: "qwen3.5-0.8b", "llama-3.2-1b", "mamba2-1.3b".
    Pass ``model_id`` to override the HuggingFace model ID used for loading.

    Args:
        model_name: Short name key identifying the model family.
        dtype: Weight dtype — torch.float16 or torch.bfloat16.
        device: Target device ("cuda", "cpu", "cuda:0", or torch.device).
        trust_remote_code: Allow custom model code from HuggingFace Hub.
        model_id: Override the HuggingFace model ID. Uses MODEL_REGISTRY by default.

    Returns:
        ModelWrapperImpl satisfying the ModelWrapper protocol.

    Raises:
        ValueError: If model_name is not in MODEL_REGISTRY and model_id is not given.
    """
    key = model_name.lower()
    hf_id = model_id or MODEL_REGISTRY.get(key)
    if hf_id is None:
        raise ValueError(
            f"Unknown model name: {model_name!r}. "
            f"Known models: {list(MODEL_REGISTRY.keys())}. "
            "Pass model_id= to use a custom HuggingFace ID."
        )

    if isinstance(device, str):
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    tok_wrapper = TokenizerWrapper(tokenizer, key)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=str(device),
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    return ModelWrapperImpl(model=model, tokenizer=tok_wrapper, model_name=key, device=device)
