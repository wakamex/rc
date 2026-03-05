"""Smoke tests for src/models/loader.py.

All tests mock HuggingFace model/tokenizer loading so they run in CI
without downloading any model weights or requiring a GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.loader import (
    DELTANET_PATTERNS,
    MODEL_REGISTRY,
    ModelWrapperImpl,
    TokenizerWrapper,
    estimate_vram,
    get_deltanet_layers,
    get_full_attention_layers,
    get_hidden_states,
    get_lora_targets,
    list_layer_names,
    load_model,
)
from src.types import ModelWrapper


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 32000
_HIDDEN_SIZE = 64
_SEQ_LEN = 8
_BATCH = 1


def _make_mock_model(num_layers: int = 4, hidden_size: int = _HIDDEN_SIZE) -> MagicMock:
    """Build a MagicMock that mimics a HuggingFace CausalLM.

    Even-indexed transformer layers are DeltaNet; odd-indexed are full-attention.
    """
    model = MagicMock()
    model.config.hidden_size = hidden_size
    model.config.num_hidden_layers = num_layers

    # Construct realistic named_modules output (Qwen3.5-style hybrid)
    layer_entries: list[tuple[str, MagicMock]] = [
        ("", model),
        ("model", MagicMock()),
        ("model.embed_tokens", MagicMock()),
    ]
    for i in range(num_layers):
        if i % 2 == 0:  # DeltaNet block
            layer_entries += [
                (f"model.layers.{i}", MagicMock()),
                (f"model.layers.{i}.deltanet", MagicMock()),
                (f"model.layers.{i}.deltanet.q_proj", MagicMock()),
                (f"model.layers.{i}.mlp", MagicMock()),
            ]
        else:  # full softmax-attention block
            layer_entries += [
                (f"model.layers.{i}", MagicMock()),
                (f"model.layers.{i}.self_attn", MagicMock()),
                (f"model.layers.{i}.self_attn.q_proj", MagicMock()),
                (f"model.layers.{i}.mlp", MagicMock()),
            ]
    layer_entries.append(("lm_head", MagicMock()))
    model.named_modules.return_value = layer_entries

    # Parameters: needed for .device / .dtype lookups
    param = torch.zeros(1, dtype=torch.float16)
    model.parameters = lambda: iter([param])

    # Forward pass output (used by model(input_ids, ...))
    logits = torch.randn(_BATCH, _SEQ_LEN, _VOCAB_SIZE)
    hidden = tuple(
        torch.randn(_BATCH, _SEQ_LEN, hidden_size) for _ in range(num_layers + 1)
    )
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_output.hidden_states = hidden
    model.return_value = mock_output

    # Generate output
    model.generate.return_value = torch.randint(0, _VOCAB_SIZE, (_BATCH, _SEQ_LEN + 4))

    return model


def _make_mock_tokenizer() -> MagicMock:
    """Build a MagicMock that mimics a HuggingFace tokenizer."""
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    tok.vocab_size = _VOCAB_SIZE
    # tok(text, return_tensors="pt") → dict with "input_ids"
    tok.return_value = {"input_ids": torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)}
    tok.decode.return_value = "hello world"
    tok.batch_decode.return_value = ["hello world"]
    return tok


def _make_wrapper(num_layers: int = 4, hidden_size: int = _HIDDEN_SIZE) -> ModelWrapperImpl:
    model = _make_mock_model(num_layers=num_layers, hidden_size=hidden_size)
    tok = TokenizerWrapper(_make_mock_tokenizer(), "qwen3.5-0.8b")
    return ModelWrapperImpl(model=model, tokenizer=tok, model_name="qwen3.5-0.8b", device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# estimate_vram
# ---------------------------------------------------------------------------


def test_estimate_vram_qwen_fp16():
    vram = estimate_vram("qwen3.5-0.8b", torch.float16)
    # 0.8B params × 2 bytes + 20% overhead ≈ 1.92 GB
    assert vram > 1_500_000_000
    assert vram < 3_000_000_000


def test_estimate_vram_llama_fp16():
    vram = estimate_vram("llama-3.2-1b", torch.float16)
    assert vram > 2_000_000_000  # 1.2B params × 2 + overhead > 2 GB


def test_estimate_vram_mamba_bf16():
    vram = estimate_vram("mamba2-1.3b", torch.bfloat16)
    assert vram > 2_500_000_000  # 1.3B params × 2 + overhead > 2.5 GB


def test_estimate_vram_bf16_same_as_fp16():
    # Both use 2 bytes per parameter
    v_fp16 = estimate_vram("llama-3.2-1b", torch.float16)
    v_bf16 = estimate_vram("llama-3.2-1b", torch.bfloat16)
    assert v_fp16 == v_bf16


def test_estimate_vram_unknown_model_uses_default():
    # Should not raise; falls back to 1.0B default
    vram = estimate_vram("unknown-model", torch.float16)
    assert vram > 0


# ---------------------------------------------------------------------------
# TokenizerWrapper
# ---------------------------------------------------------------------------


def test_tokenizer_wrapper_vocab_size():
    wrapper = TokenizerWrapper(_make_mock_tokenizer(), "llama-3.2-1b")
    assert wrapper.vocab_size == _VOCAB_SIZE


def test_tokenizer_wrapper_eos_token_id():
    wrapper = TokenizerWrapper(_make_mock_tokenizer(), "llama-3.2-1b")
    assert wrapper.eos_token_id == 2


def test_tokenizer_wrapper_model_name():
    wrapper = TokenizerWrapper(_make_mock_tokenizer(), "llama-3.2-1b")
    assert wrapper.model_name == "llama-3.2-1b"


def test_tokenizer_wrapper_sets_pad_token_when_none():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "</s>"
    tok.vocab_size = _VOCAB_SIZE
    tok.eos_token_id = 2
    TokenizerWrapper(tok, "llama-3.2-1b")
    assert tok.pad_token == "</s>"


def test_tokenizer_wrapper_encode_calls_tokenizer():
    mock_tok = _make_mock_tokenizer()
    wrapper = TokenizerWrapper(mock_tok, "llama-3.2-1b")
    result = wrapper.encode("hello world")
    mock_tok.assert_called_once_with("hello world", return_tensors="pt")
    assert result.shape == (_BATCH, _SEQ_LEN)


def test_tokenizer_wrapper_decode_single():
    wrapper = TokenizerWrapper(_make_mock_tokenizer(), "llama-3.2-1b")
    ids = torch.ones(_SEQ_LEN, dtype=torch.long)
    out = wrapper.decode(ids)
    assert isinstance(out, str)


def test_tokenizer_wrapper_decode_batch():
    wrapper = TokenizerWrapper(_make_mock_tokenizer(), "llama-3.2-1b")
    ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    out = wrapper.decode(ids)
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Layer inspection: list_layer_names
# ---------------------------------------------------------------------------


def test_list_layer_names_returns_list():
    model = _make_mock_model()
    names = list_layer_names(model)
    assert isinstance(names, list)
    assert len(names) > 0


def test_list_layer_names_includes_root():
    model = _make_mock_model()
    names = list_layer_names(model)
    assert "" in names  # root module is always first


def test_list_layer_names_includes_lm_head():
    model = _make_mock_model()
    assert "lm_head" in list_layer_names(model)


# ---------------------------------------------------------------------------
# Layer inspection: get_deltanet_layers
# ---------------------------------------------------------------------------


def test_get_deltanet_layers_returns_list():
    model = _make_mock_model(num_layers=4)
    result = get_deltanet_layers(model)
    assert isinstance(result, list)


def test_get_deltanet_layers_finds_deltanet_modules():
    model = _make_mock_model(num_layers=4)
    result = get_deltanet_layers(model)
    # Layers 0 and 2 are DeltaNet in the mock (even indices)
    assert len(result) > 0
    assert all("deltanet" in name.lower() for name in result)


def test_get_deltanet_layers_excludes_full_attention():
    model = _make_mock_model(num_layers=4)
    result = get_deltanet_layers(model)
    assert not any("self_attn" in name and "deltanet" not in name.lower() for name in result)


# ---------------------------------------------------------------------------
# Layer inspection: get_full_attention_layers
# ---------------------------------------------------------------------------


def test_get_full_attention_layers_returns_list():
    model = _make_mock_model(num_layers=4)
    result = get_full_attention_layers(model)
    assert isinstance(result, list)


def test_get_full_attention_layers_finds_self_attn():
    model = _make_mock_model(num_layers=4)
    result = get_full_attention_layers(model)
    # Layers 1 and 3 are full-attention in the mock (odd indices)
    assert len(result) > 0
    assert any("self_attn" in name for name in result)


def test_get_full_attention_layers_excludes_deltanet():
    model = _make_mock_model(num_layers=4)
    result = get_full_attention_layers(model)
    assert not any(pat in name.lower() for name in result for pat in DELTANET_PATTERNS)


def test_deltanet_and_full_attn_are_disjoint():
    model = _make_mock_model(num_layers=4)
    deltanet = set(get_deltanet_layers(model))
    full_attn = set(get_full_attention_layers(model))
    assert deltanet.isdisjoint(full_attn)


# ---------------------------------------------------------------------------
# Layer inspection: get_lora_targets
# ---------------------------------------------------------------------------


def test_get_lora_targets_qwen():
    targets = get_lora_targets("qwen3.5-0.8b")
    assert "q_proj" in targets
    assert "v_proj" in targets
    assert "o_proj" in targets


def test_get_lora_targets_llama():
    targets = get_lora_targets("llama-3.2-1b")
    assert "q_proj" in targets
    assert "gate_proj" in targets


def test_get_lora_targets_mamba():
    targets = get_lora_targets("mamba2-1.3b")
    assert "in_proj" in targets
    assert "out_proj" in targets


def test_get_lora_targets_unknown_returns_default():
    targets = get_lora_targets("unknown-model")
    assert isinstance(targets, list)
    assert len(targets) > 0
    assert "q_proj" in targets


# ---------------------------------------------------------------------------
# get_hidden_states
# ---------------------------------------------------------------------------


def test_get_hidden_states_returns_requested_indices():
    model = _make_mock_model(num_layers=4, hidden_size=_HIDDEN_SIZE)
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    result = get_hidden_states(model, input_ids, layer_indices=[0, -1])
    assert 0 in result
    assert -1 in result


def test_get_hidden_states_correct_shape():
    model = _make_mock_model(num_layers=4, hidden_size=_HIDDEN_SIZE)
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    result = get_hidden_states(model, input_ids, layer_indices=[0, -1])
    assert result[0].shape == (_BATCH, _SEQ_LEN, _HIDDEN_SIZE)
    assert result[-1].shape == (_BATCH, _SEQ_LEN, _HIDDEN_SIZE)


def test_get_hidden_states_detached():
    model = _make_mock_model(num_layers=4, hidden_size=_HIDDEN_SIZE)
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    result = get_hidden_states(model, input_ids, layer_indices=[0])
    assert not result[0].requires_grad


def test_get_hidden_states_multiple_layers():
    model = _make_mock_model(num_layers=4, hidden_size=_HIDDEN_SIZE)
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    result = get_hidden_states(model, input_ids, layer_indices=[0, 1, 2])
    assert len(result) == 3


# ---------------------------------------------------------------------------
# ModelWrapperImpl
# ---------------------------------------------------------------------------


def test_model_wrapper_forward_shape():
    wrapper = _make_wrapper()
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    outputs = wrapper.forward(input_ids)
    assert outputs.logits.shape == (_BATCH, _SEQ_LEN, _VOCAB_SIZE)


def test_model_wrapper_generate_shape():
    wrapper = _make_wrapper()
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    generated = wrapper.generate(input_ids, max_new_tokens=4)
    assert generated.shape[0] == _BATCH
    assert generated.shape[1] > _SEQ_LEN  # at least one new token


def test_model_wrapper_get_hidden_shape():
    wrapper = _make_wrapper(hidden_size=_HIDDEN_SIZE)
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    hidden = wrapper.get_hidden(input_ids, layer=0)
    assert hidden.shape == (_BATCH, _SEQ_LEN, _HIDDEN_SIZE)


def test_model_wrapper_dtype():
    wrapper = _make_wrapper()
    assert wrapper.dtype == torch.float16


def test_model_wrapper_satisfies_protocol():
    wrapper = _make_wrapper()
    assert isinstance(wrapper, ModelWrapper)


# ---------------------------------------------------------------------------
# load_model (mocked via patch)
# ---------------------------------------------------------------------------


@patch("src.models.loader.AutoModelForCausalLM.from_pretrained")
@patch("src.models.loader.AutoTokenizer.from_pretrained")
def test_load_model_qwen(mock_tok_cls, mock_model_cls):
    mock_model_cls.return_value = _make_mock_model()
    mock_tok_cls.return_value = _make_mock_tokenizer()

    wrapper = load_model("qwen3.5-0.8b", dtype=torch.float16, device="cpu")

    assert isinstance(wrapper, ModelWrapperImpl)
    assert wrapper.model_name == "qwen3.5-0.8b"
    mock_model_cls.return_value.eval.assert_called_once()


@patch("src.models.loader.AutoModelForCausalLM.from_pretrained")
@patch("src.models.loader.AutoTokenizer.from_pretrained")
def test_load_model_llama(mock_tok_cls, mock_model_cls):
    mock_model_cls.return_value = _make_mock_model()
    mock_tok_cls.return_value = _make_mock_tokenizer()

    wrapper = load_model("llama-3.2-1b", dtype=torch.bfloat16, device="cpu")

    assert isinstance(wrapper, ModelWrapperImpl)
    assert wrapper.model_name == "llama-3.2-1b"
    # Verify dtype was passed through
    _, kwargs = mock_model_cls.call_args
    assert kwargs.get("torch_dtype") == torch.bfloat16


@patch("src.models.loader.AutoModelForCausalLM.from_pretrained")
@patch("src.models.loader.AutoTokenizer.from_pretrained")
def test_load_model_mamba(mock_tok_cls, mock_model_cls):
    mock_model_cls.return_value = _make_mock_model()
    mock_tok_cls.return_value = _make_mock_tokenizer()

    wrapper = load_model("mamba2-1.3b", dtype=torch.float16, device="cpu")

    assert isinstance(wrapper, ModelWrapperImpl)
    assert wrapper.model_name == "mamba2-1.3b"


def test_load_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        load_model("definitely-not-a-real-model-xyz")


@patch("src.models.loader.AutoModelForCausalLM.from_pretrained")
@patch("src.models.loader.AutoTokenizer.from_pretrained")
def test_load_model_custom_id(mock_tok_cls, mock_model_cls):
    mock_model_cls.return_value = _make_mock_model()
    mock_tok_cls.return_value = _make_mock_tokenizer()

    load_model("qwen3.5-0.8b", model_id="my-org/my-custom-model", device="cpu")

    # Both tokenizer and model should use the custom ID
    mock_tok_cls.assert_called_once()
    assert mock_tok_cls.call_args[0][0] == "my-org/my-custom-model"
    mock_model_cls.assert_called_once()
    assert mock_model_cls.call_args[0][0] == "my-org/my-custom-model"


@patch("src.models.loader.AutoModelForCausalLM.from_pretrained")
@patch("src.models.loader.AutoTokenizer.from_pretrained")
def test_load_model_registry_has_all_families(mock_tok_cls, mock_model_cls):
    """MODEL_REGISTRY covers all three supported model families."""
    assert "qwen3.5-0.8b" in MODEL_REGISTRY
    assert "llama-3.2-1b" in MODEL_REGISTRY
    assert "mamba2-1.3b" in MODEL_REGISTRY


@patch("src.models.loader.AutoModelForCausalLM.from_pretrained")
@patch("src.models.loader.AutoTokenizer.from_pretrained")
def test_load_model_forward_produces_logits(mock_tok_cls, mock_model_cls):
    mock_model_cls.return_value = _make_mock_model()
    mock_tok_cls.return_value = _make_mock_tokenizer()

    wrapper = load_model("llama-3.2-1b", device="cpu")
    input_ids = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.long)
    outputs = wrapper.forward(input_ids)
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[-1] == _VOCAB_SIZE
