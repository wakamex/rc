"""Tests for src/eval/harness.py."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterator
from typing import Any

import pytest

from src.eval.harness import (
    EvalConfig,
    compute_perplexity,
    evaluate,
    exact_match,
    token_f1,
)
from src.types import BenchmarkExample, EvalResult


# ---------------------------------------------------------------------------
# Dummy implementations for testing
# ---------------------------------------------------------------------------


class DummyModel:
    """Minimal ModelWrapper that echoes the input or returns a fixed string."""

    def __init__(self, responses: dict[str, str] | None = None, echo: bool = False) -> None:
        self.responses = responses or {}
        self.echo = echo

    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        return None

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        text = str(input_ids)
        if self.echo:
            return text
        # Strip the "Output:" suffix and look up a response
        for key, val in self.responses.items():
            if key in text:
                return val
        return ""

    def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
        return None


class DummyBenchmark:
    """Minimal Generator yielding a fixed list of BenchmarkExample items."""

    def __init__(self, examples: list[BenchmarkExample], name: str = "dummy") -> None:
        self._examples = examples
        self.name = name

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------


def test_exact_match_equal():
    assert exact_match("hello", "hello") == 1.0


def test_exact_match_case_insensitive():
    assert exact_match("Hello", "hello") == 1.0


def test_exact_match_whitespace():
    assert exact_match("  hello  ", "hello") == 1.0


def test_exact_match_different():
    assert exact_match("hello", "world") == 0.0


def test_token_f1_perfect():
    assert token_f1("a b c", "a b c") == 1.0


def test_token_f1_partial():
    score = token_f1("a b c", "a b d")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    assert token_f1("x y z", "a b c") == 0.0


def test_token_f1_empty_both():
    assert token_f1("", "") == 1.0


def test_token_f1_empty_one():
    assert token_f1("hello", "") == 0.0


def test_compute_perplexity_uniform():
    # log(1) = 0 → perplexity = exp(0) = 1
    lp = [0.0] * 5
    assert math.isclose(compute_perplexity(lp), 1.0)


def test_compute_perplexity_empty():
    assert compute_perplexity([]) == float("inf")


# ---------------------------------------------------------------------------
# Harness integration tests
# ---------------------------------------------------------------------------


def _make_examples(pairs: list[tuple[str, str]]) -> list[BenchmarkExample]:
    return [BenchmarkExample(input=inp, target=tgt) for inp, tgt in pairs]


def test_harness_runs_and_returns_eval_results():
    """Harness runs a dummy benchmark and returns a non-empty list of EvalResult."""
    examples = _make_examples([("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
    model = DummyModel(responses={"q1": "a1", "q2": "a2", "q3": "a3"})
    benchmark = DummyBenchmark(examples, name="test_bench")
    config = EvalConfig(batch_size=2, metrics=["exact_match"])

    results = evaluate(model, [benchmark], config)

    assert len(results) > 0
    assert all(isinstance(r, EvalResult) for r in results)


def test_harness_exact_match_perfect():
    """Model that always returns correct answers → exact_match == 1.0."""
    examples = _make_examples([("q1", "a1"), ("q2", "a2")])
    model = DummyModel(responses={"q1": "a1", "q2": "a2"})
    benchmark = DummyBenchmark(examples, name="em_bench")
    config = EvalConfig(metrics=["exact_match"])

    results = evaluate(model, [benchmark], config)
    em_results = [r for r in results if r.metric == "exact_match"]
    assert len(em_results) == 1
    assert em_results[0].value == 1.0


def test_harness_exact_match_zero():
    """Model that always returns wrong answers → exact_match == 0.0."""
    examples = _make_examples([("q1", "a1"), ("q2", "a2")])
    model = DummyModel(responses={"q1": "WRONG", "q2": "WRONG"})
    benchmark = DummyBenchmark(examples, name="em_zero")
    config = EvalConfig(metrics=["exact_match"])

    results = evaluate(model, [benchmark], config)
    em_results = [r for r in results if r.metric == "exact_match"]
    assert len(em_results) == 1
    assert em_results[0].value == 0.0


def test_harness_accuracy_partial():
    """Model correct on half the examples → accuracy == 0.5."""
    examples = _make_examples([("q1", "a1"), ("q2", "a2")])
    model = DummyModel(responses={"q1": "a1", "q2": "WRONG"})
    benchmark = DummyBenchmark(examples, name="partial")
    config = EvalConfig(metrics=["accuracy"])

    results = evaluate(model, [benchmark], config)
    acc_results = [r for r in results if r.metric == "accuracy"]
    assert len(acc_results) == 1
    assert math.isclose(acc_results[0].value, 0.5)


def test_harness_f1_metric():
    """F1 metric is computed and in [0, 1]."""
    examples = _make_examples([("q1", "the cat sat"), ("q2", "the dog ran")])
    model = DummyModel(responses={"q1": "the cat", "q2": "the dog"})
    benchmark = DummyBenchmark(examples, name="f1_bench")
    config = EvalConfig(metrics=["f1"])

    results = evaluate(model, [benchmark], config)
    f1_results = [r for r in results if r.metric == "f1"]
    assert len(f1_results) == 1
    assert 0.0 <= f1_results[0].value <= 1.0


def test_harness_multiple_metrics():
    """Multiple metrics produce multiple EvalResult entries per task."""
    examples = _make_examples([("q1", "answer")])
    model = DummyModel(responses={"q1": "answer"})
    benchmark = DummyBenchmark(examples, name="multi_metric")
    config = EvalConfig(metrics=["exact_match", "f1", "accuracy"])

    results = evaluate(model, [benchmark], config)
    assert len(results) == 3
    metrics_found = {r.metric for r in results}
    assert metrics_found == {"exact_match", "f1", "accuracy"}


def test_harness_multiple_benchmarks():
    """Harness evaluates multiple benchmarks and returns results for each."""
    b1 = DummyBenchmark(_make_examples([("q1", "a1")]), name="bench1")
    b2 = DummyBenchmark(_make_examples([("q2", "a2")]), name="bench2")
    model = DummyModel(responses={"q1": "a1", "q2": "a2"})
    config = EvalConfig(metrics=["exact_match"])

    results = evaluate(model, [b1, b2], config)
    tasks = {r.task for r in results}
    assert "bench1" in tasks
    assert "bench2" in tasks


def test_harness_result_schema():
    """EvalResult fields match the expected schema."""
    examples = _make_examples([("q", "a")])
    model = DummyModel(responses={"q": "a"})
    benchmark = DummyBenchmark(examples, name="schema_test")
    config = EvalConfig(metrics=["exact_match"], model_name="test-model")

    results = evaluate(model, [benchmark], config)
    assert len(results) == 1
    r = results[0]
    assert r.task == "schema_test"
    assert r.metric == "exact_match"
    assert isinstance(r.value, float)
    assert isinstance(r.config, dict)
    assert r.config["model_name"] == "test-model"
    assert "git_hash" in r.config
    assert "timestamp" in r.config


def test_harness_json_output(tmp_path):
    """JSON output file is valid and contains all required fields."""
    examples = _make_examples([("q1", "a1"), ("q2", "a2")])
    model = DummyModel(responses={"q1": "a1", "q2": "a2"})
    benchmark = DummyBenchmark(examples, name="json_bench")
    output_file = str(tmp_path / "results.json")
    config = EvalConfig(metrics=["exact_match"], output_file=output_file, model_name="my-model")

    evaluate(model, [benchmark], config)

    with open(output_file) as f:
        data = json.load(f)

    assert "model_name" in data
    assert "git_hash" in data
    assert "timestamp" in data
    assert "config" in data
    assert "results" in data
    assert data["model_name"] == "my-model"
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 1

    result_entry = data["results"][0]
    assert "task" in result_entry
    assert "metric" in result_entry
    assert "value" in result_entry
    assert "config" in result_entry


def test_harness_zero_shot_prompt():
    """Zero-shot: prompt ends with 'Output:' and no examples are prepended."""
    prompts_seen: list[str] = []

    class RecordingModel:
        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            return None

        def generate(self, input_ids: Any, **kwargs: Any) -> Any:
            prompts_seen.append(str(input_ids))
            return "irrelevant"

        def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
            return None

    examples = _make_examples([("my question", "answer")])
    benchmark = DummyBenchmark(examples, name="prompt_test")
    config = EvalConfig(num_few_shot=0, metrics=["exact_match"])

    evaluate(RecordingModel(), [benchmark], config)
    assert len(prompts_seen) == 1
    assert "Output:" in prompts_seen[0]
    # No previous examples should appear
    assert prompts_seen[0].count("Input:") == 1


def test_harness_few_shot_prompt():
    """k-shot: k example input/output pairs appear before the query."""
    prompts_seen: list[str] = []

    class RecordingModel:
        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            return None

        def generate(self, input_ids: Any, **kwargs: Any) -> Any:
            prompts_seen.append(str(input_ids))
            return "irrelevant"

        def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
            return None

    # 2 shots + 2 eval examples = 4 total
    examples = _make_examples([
        ("shot1", "s_ans1"),
        ("shot2", "s_ans2"),
        ("eval1", "e_ans1"),
        ("eval2", "e_ans2"),
    ])
    benchmark = DummyBenchmark(examples, name="fewshot_test")
    config = EvalConfig(num_few_shot=2, metrics=["exact_match"])

    evaluate(RecordingModel(), [benchmark], config)
    # Two eval examples generate two prompts
    assert len(prompts_seen) == 2
    for prompt in prompts_seen:
        # Both shot examples appear in every prompt
        assert "shot1" in prompt
        assert "s_ans1" in prompt
        assert "shot2" in prompt
        assert "s_ans2" in prompt


def test_harness_resume(tmp_path):
    """Resume mode skips already-evaluated tasks."""
    call_count = [0]

    class CountingModel:
        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            return None

        def generate(self, input_ids: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            return "answer"

        def get_hidden(self, input_ids: Any, layer: int = -1, **kwargs: Any) -> Any:
            return None

    examples = _make_examples([("q", "answer")])
    output_file = str(tmp_path / "resume_test.json")

    # First run
    b = DummyBenchmark(examples, name="resume_bench")
    cfg = EvalConfig(metrics=["exact_match"], output_file=output_file)
    evaluate(CountingModel(), [b], cfg)
    first_count = call_count[0]

    # Second run with resume=True — should not call model.generate again
    b2 = DummyBenchmark(examples, name="resume_bench")
    cfg2 = EvalConfig(metrics=["exact_match"], output_file=output_file, resume=True)
    results = evaluate(CountingModel(), [b2], cfg2)

    assert call_count[0] == first_count  # no new calls
    assert len(results) > 0


def test_harness_batch_size_respected():
    """Harness produces the same results regardless of batch_size."""
    examples = _make_examples([("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")])
    model = DummyModel(responses={"q1": "a1", "q2": "a2", "q3": "a3", "q4": "a4"})

    results_bs1 = evaluate(
        model,
        [DummyBenchmark(examples, name="b")],
        EvalConfig(batch_size=1, metrics=["exact_match"]),
    )
    results_bs4 = evaluate(
        model,
        [DummyBenchmark(examples, name="b")],
        EvalConfig(batch_size=4, metrics=["exact_match"]),
    )

    assert results_bs1[0].value == results_bs4[0].value


def test_harness_empty_benchmark():
    """Empty benchmark produces no results."""
    model = DummyModel()
    benchmark = DummyBenchmark([], name="empty")
    config = EvalConfig(metrics=["exact_match"])

    results = evaluate(model, [benchmark], config)
    assert results == []


def test_harness_perplexity_skipped():
    """Perplexity metric is silently skipped (no crash)."""
    examples = _make_examples([("q", "a")])
    model = DummyModel(responses={"q": "a"})
    benchmark = DummyBenchmark(examples, name="ppl_test")
    config = EvalConfig(metrics=["perplexity"])

    results = evaluate(model, [benchmark], config)
    # Perplexity is skipped — no results
    assert results == []
