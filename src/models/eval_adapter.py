"""Shared text-level eval adapter for the benchmark harness.

Wraps any ModelWrapperImpl (or raw model + tokenizer pair) to provide
string-in / string-out generate() as required by src.eval.harness.evaluate.
"""

from __future__ import annotations

import time
from typing import Any

import torch


class TextEvalAdapter:
    """Wraps a model + tokenizer for the eval harness (string → string).

    Accepts either a ModelWrapperImpl (has .model, .tokenizer, .device) or
    explicit model/tokenizer/device arguments for models loaded outside the
    standard loader (e.g. YaRN, infini-attention).

    Args:
        wrapper: A ModelWrapperImpl instance. Mutually exclusive with
            model/tokenizer/device.
        model: Raw HuggingFace model. Used when wrapper is None.
        tokenizer: Raw HuggingFace tokenizer. Used when wrapper is None.
        device: Torch device. Used when wrapper is None.
        max_new_tokens: Maximum tokens to generate per call.
        max_input_length: Maximum input token length for truncation.
        pre_generate_hook: Optional callable(model) invoked before each
            generate() call (e.g. reset_infini_memory).
    """

    def __init__(
        self,
        wrapper: Any = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        device: torch.device | str | None = None,
        max_new_tokens: int = 64,
        max_input_length: int = 1024,
        pre_generate_hook: Any = None,
    ) -> None:
        if wrapper is not None:
            self._model = wrapper.model
            self._tok = wrapper.tokenizer
            self._device = wrapper.device
        else:
            if model is None or tokenizer is None:
                raise ValueError("Provide either wrapper or model+tokenizer+device")
            self._model = model
            self._tok = tokenizer
            self._device = torch.device(device) if device is not None else torch.device("cpu")

        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self._pre_generate_hook = pre_generate_hook
        self._latencies: list[float] = []

    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        return self._model(input_ids.to(self._device), **kwargs)

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        """Accept a string prompt, return a string continuation."""
        kwargs.pop("seed", None)

        t0 = time.perf_counter()

        if isinstance(prompt, str):
            # Try .encode() first (ModelWrapperImpl tokenizer), fall back to __call__
            if hasattr(self._tok, "encode") and callable(self._tok.encode):
                try:
                    input_ids = self._tok.encode(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_length,
                    )
                    # Some tokenizers return a tensor directly, others a dict
                    if isinstance(input_ids, dict):
                        input_ids = input_ids["input_ids"]
                except TypeError:
                    # ModelWrapperImpl tokenizer .encode() doesn't take return_tensors
                    input_ids = self._tok.encode(
                        prompt,
                        padding=False,
                        truncation=True,
                        max_length=self.max_input_length,
                    )
            else:
                encoded = self._tok(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                )
                input_ids = encoded["input_ids"]
            input_ids = input_ids.to(self._device)
        else:
            input_ids = prompt.to(self._device)

        prompt_len = input_ids.shape[-1]

        if self._pre_generate_hook is not None:
            self._pre_generate_hook(self._model)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tok.eos_token_id,
                **kwargs,
            )

        elapsed = time.perf_counter() - t0
        self._latencies.append(elapsed)

        new_ids = output_ids[0, prompt_len:]
        return self._tok.decode(new_ids)

    def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
        outputs = self._model(
            input_ids.to(self._device),
            output_hidden_states=True,
            **kwargs,
        )
        return outputs.hidden_states[layer]

    def latency_stats(self) -> dict[str, float]:
        """Return p50 and p95 latency in seconds."""
        if not self._latencies:
            return {"p50_s": 0.0, "p95_s": 0.0}
        arr = sorted(self._latencies)
        n = len(arr)
        p50 = arr[int(n * 0.50)]
        p95 = arr[min(int(n * 0.95), n - 1)]
        return {"p50_s": p50, "p95_s": p95}
