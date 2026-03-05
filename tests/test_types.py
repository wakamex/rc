"""Smoke tests: ensure all shared types are importable and instantiable."""

from src.types import (
    BenchmarkExample,
    DataPipeline,
    EvalResult,
    Generator,
    ModelWrapper,
    Reservoir,
    ReservoirConfig,
)


def test_reservoir_config_defaults():
    cfg = ReservoirConfig()
    assert cfg.size == 1000
    assert 0 < cfg.spectral_radius < 1.5
    assert 0 < cfg.leak_rate <= 1.0


def test_reservoir_config_custom():
    cfg = ReservoirConfig(size=500, spectral_radius=0.5, leak_rate=0.1, topology="small_world")
    assert cfg.size == 500
    assert cfg.topology == "small_world"


def test_benchmark_example():
    ex = BenchmarkExample(input="hello", target="world")
    assert ex.input == "hello"
    assert ex.target == "world"
    assert ex.metadata == {}


def test_eval_result():
    res = EvalResult(task="passkey", metric="accuracy", value=0.95)
    assert res.task == "passkey"
    assert res.value == 0.95
    assert res.config == {}


def test_protocols_are_runtime_checkable():
    # These should not raise; they are Protocol classes marked @runtime_checkable
    assert issubclass(Reservoir, Reservoir)
    assert issubclass(ModelWrapper, ModelWrapper)
    assert issubclass(Generator, Generator)
    assert issubclass(DataPipeline, DataPipeline)
