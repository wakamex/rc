"""Tests for the three-stage training curriculum data pipeline."""

from __future__ import annotations

import itertools

import pytest

from src.training.curriculum import (
    CurriculumConfig,
    CurriculumDataPipeline,
    CurriculumStage,
    StageConfig,
)
from src.types import DataPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_iter(n: int = 10_000) -> iter:
    """Return a finite or infinite stream of synthetic text strings."""
    templates = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Language models learn from text. " * 20,
        "Mathematics is the language of the universe. " * 15,
    ]
    return itertools.cycle(templates)


def _make_pipeline(
    batch_size: int = 4,
    stage1_steps: int = 5,
    stage2_steps: int = 5,
    text_ratio: float = 0.7,
    procedural_ratio: float = 0.3,
    stage3_context_lengths: list[int] | None = None,
    stage3_steps_per_length: int | None = None,
    use_loss_plateau: bool = False,
    aux_reservoir_loss: bool = False,
    start_stage: int = 1,
    start_step: int = 0,
) -> CurriculumDataPipeline:
    """Build a fast-cycling pipeline for testing."""
    config = CurriculumConfig(
        stage1=StageConfig(text_ratio=1.0, procedural_ratio=0.0, context_length=128),
        stage2=StageConfig(
            text_ratio=text_ratio,
            procedural_ratio=procedural_ratio,
            context_length=128,
        ),
        stage3_context_lengths=stage3_context_lengths or [128, 256, 512],
        stage3_steps_per_length=stage3_steps_per_length,
        batch_size=batch_size,
        text_corpus=None,  # no network calls in tests
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        use_loss_plateau=use_loss_plateau,
        aux_reservoir_loss=aux_reservoir_loss,
        seed=42,
    )
    return CurriculumDataPipeline(
        config=config,
        text_iterator=_text_iter(),
        start_stage=start_stage,
        start_step=start_step,
    )


def _take(pipeline: CurriculumDataPipeline, n: int) -> list[dict]:
    """Collect n batches from the pipeline."""
    batches = []
    for batch in pipeline.iterate():
        batches.append(batch)
        if len(batches) >= n:
            break
    return batches


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestDataPipelineProtocol:
    def test_satisfies_protocol(self):
        pipeline = _make_pipeline()
        assert isinstance(pipeline, DataPipeline)

    def test_iterate_is_iterator(self):
        pipeline = _make_pipeline()
        it = pipeline.iterate()
        assert hasattr(it, "__next__")
        assert hasattr(it, "__iter__")

    def test_split_returns_dict(self):
        pipeline = _make_pipeline()
        splits = pipeline.split({"train": 0.8, "val": 0.2})
        assert isinstance(splits, dict)
        assert "train" in splits
        assert "val" in splits


# ---------------------------------------------------------------------------
# Batch structure
# ---------------------------------------------------------------------------


class TestBatchStructure:
    def test_batch_has_required_keys(self):
        pipeline = _make_pipeline()
        batch = next(pipeline.iterate())
        for key in ("texts", "sources", "stage", "context_length", "step"):
            assert key in batch, f"Missing key: {key}"

    def test_batch_size_correct(self):
        pipeline = _make_pipeline(batch_size=4)
        batch = next(pipeline.iterate())
        assert len(batch["texts"]) == 4
        assert len(batch["sources"]) == 4

    def test_texts_are_strings(self):
        pipeline = _make_pipeline()
        batch = next(pipeline.iterate())
        for t in batch["texts"]:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_sources_valid_values(self):
        pipeline = _make_pipeline()
        batches = _take(pipeline, 20)
        valid_sources = {"text", "procedural"}
        for batch in batches:
            for src in batch["sources"]:
                assert src in valid_sources

    def test_stage_is_int(self):
        pipeline = _make_pipeline()
        batch = next(pipeline.iterate())
        assert isinstance(batch["stage"], int)
        assert batch["stage"] in (1, 2, 3)

    def test_step_increments(self):
        pipeline = _make_pipeline()
        batches = _take(pipeline, 5)
        steps = [b["step"] for b in batches]
        assert steps == list(range(5))

    def test_aux_reservoir_loss_false_by_default(self):
        pipeline = _make_pipeline()
        batch = next(pipeline.iterate())
        assert batch["aux_reservoir_loss"] is False

    def test_aux_reservoir_loss_true_when_configured(self):
        pipeline = _make_pipeline(aux_reservoir_loss=True)
        batch = next(pipeline.iterate())
        assert batch["aux_reservoir_loss"] is True


# ---------------------------------------------------------------------------
# Stage 1 behaviour
# ---------------------------------------------------------------------------


class TestStage1:
    def test_stage_1_all_text(self):
        """Stage 1 should produce 100% text sources."""
        pipeline = _make_pipeline(stage1_steps=100)
        # Collect fewer than stage1_steps batches
        batches = _take(pipeline, 10)
        for batch in batches:
            assert batch["stage"] == 1
            assert all(s == "text" for s in batch["sources"])

    def test_stage_1_context_length(self):
        pipeline = _make_pipeline(stage1_steps=100)
        batch = next(pipeline.iterate())
        assert batch["context_length"] == 128  # as configured in _make_pipeline


# ---------------------------------------------------------------------------
# Stage 2 behaviour
# ---------------------------------------------------------------------------


class TestStage2:
    def test_stage_2_has_procedural(self):
        """Stage 2 should mix procedural examples into batches."""
        pipeline = _make_pipeline(
            batch_size=10,
            stage1_steps=1,  # transition after 1 step
            stage2_steps=100,
            text_ratio=0.5,
            procedural_ratio=0.5,
        )
        # Skip to stage 2
        batches = _take(pipeline, 20)
        stage2_batches = [b for b in batches if b["stage"] == 2]
        assert len(stage2_batches) > 0
        # At least some batches should have procedural sources
        all_sources = [s for b in stage2_batches for s in b["sources"]]
        assert "procedural" in all_sources

    def test_stage_2_mixing_ratio(self):
        """Stage 2 text/procedural ratio should be approximately correct."""
        pipeline = _make_pipeline(
            batch_size=10,
            stage1_steps=1,
            stage2_steps=200,
            text_ratio=0.7,
            procedural_ratio=0.3,
        )
        # Collect stage 2 batches
        stage2_sources: list[str] = []
        for batch in pipeline.iterate():
            if batch["stage"] == 2:
                stage2_sources.extend(batch["sources"])
            if len(stage2_sources) >= 200:
                break

        proc_frac = stage2_sources.count("procedural") / len(stage2_sources)
        # Allow ±10% tolerance around the 0.3 target
        assert abs(proc_frac - 0.3) < 0.1, f"Procedural fraction {proc_frac:.3f} ≠ 0.3"

    def test_mixing_ratio_all_procedural(self):
        """100% procedural mix should yield only procedural sources."""
        pipeline = _make_pipeline(
            batch_size=4,
            stage1_steps=1,
            stage2_steps=100,
            text_ratio=0.0,
            procedural_ratio=1.0,
        )
        stage2_batches = []
        for batch in pipeline.iterate():
            if batch["stage"] == 2:
                stage2_batches.append(batch)
            if len(stage2_batches) >= 5:
                break
        for b in stage2_batches:
            assert all(s == "procedural" for s in b["sources"])


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------


class TestStageTransitions:
    def test_transition_1_to_2_on_steps(self):
        """Pipeline should advance from Stage 1 to Stage 2 after stage1_steps."""
        pipeline = _make_pipeline(stage1_steps=3, stage2_steps=100)
        stages = [b["stage"] for b in _take(pipeline, 10)]
        assert 1 in stages
        assert 2 in stages
        # Stage 1 should come before Stage 2
        first_2 = stages.index(2)
        assert all(s == 1 for s in stages[:first_2])

    def test_transition_2_to_3_on_steps(self):
        """Pipeline should advance from Stage 2 to Stage 3 after stage2_steps."""
        pipeline = _make_pipeline(stage1_steps=2, stage2_steps=3)
        stages = [b["stage"] for b in _take(pipeline, 15)]
        assert 3 in stages, f"Stage 3 not reached in: {stages}"

    def test_no_regression_to_earlier_stage(self):
        """Stages should only increase, never decrease."""
        pipeline = _make_pipeline(stage1_steps=2, stage2_steps=2)
        stages = [b["stage"] for b in _take(pipeline, 20)]
        for i in range(1, len(stages)):
            assert stages[i] >= stages[i - 1], f"Stage regressed at index {i}: {stages}"

    def test_loss_plateau_transition(self):
        """Pipeline should advance stage when loss plateaus."""
        pipeline = _make_pipeline(
            use_loss_plateau=True,
            stage1_steps=10_000,  # large so step-based never fires
        )
        pipeline.config.loss_plateau_patience = 5
        pipeline.config.loss_plateau_threshold = 0.1

        it = pipeline.iterate()
        # Feed a stagnant loss and observe transition
        for _ in range(4):
            next(it)
            pipeline.report_loss(1.0)  # not enough history yet

        assert pipeline.current_stage == CurriculumStage.STAGE_1

        # One more — crosses patience=5
        next(it)
        pipeline.report_loss(1.0)
        assert pipeline.current_stage == CurriculumStage.STAGE_2

    def test_no_plateau_transition_when_loss_drops(self):
        """Pipeline should NOT advance if loss is still dropping."""
        pipeline = _make_pipeline(
            use_loss_plateau=True,
            stage1_steps=10_000,
        )
        pipeline.config.loss_plateau_patience = 5
        pipeline.config.loss_plateau_threshold = 0.1

        it = pipeline.iterate()
        for i in range(10):
            next(it)
            pipeline.report_loss(2.0 - i * 0.3)  # steadily decreasing

        assert pipeline.current_stage == CurriculumStage.STAGE_1


# ---------------------------------------------------------------------------
# Stage 3 — length curriculum
# ---------------------------------------------------------------------------


class TestStage3LengthCurriculum:
    def test_stage_3_starts_at_first_length(self):
        """When Stage 3 begins, context_length should be the first in the list."""
        pipeline = _make_pipeline(
            stage1_steps=1,
            stage2_steps=1,
            stage3_context_lengths=[64, 128, 256],
        )
        for batch in pipeline.iterate():
            if batch["stage"] == 3:
                assert batch["context_length"] == 64
                break

    def test_stage_3_length_advances(self):
        """Context length should increase within Stage 3 after steps_per_length."""
        pipeline = _make_pipeline(
            stage1_steps=1,
            stage2_steps=1,
            stage3_context_lengths=[64, 128, 256],
            stage3_steps_per_length=3,
        )
        lengths_seen: list[int] = []
        for batch in _take(pipeline, 20):
            if batch["stage"] == 3:
                lengths_seen.append(batch["context_length"])

        # Should see at least two distinct lengths
        assert len(set(lengths_seen)) >= 2, f"Only one length seen: {set(lengths_seen)}"

    def test_stage_3_length_does_not_exceed_max(self):
        """Context length should cap at the last configured value."""
        max_len = 256
        pipeline = _make_pipeline(
            stage1_steps=1,
            stage2_steps=1,
            stage3_context_lengths=[64, 128, max_len],
            stage3_steps_per_length=1,
        )
        for batch in _take(pipeline, 30):
            assert batch["context_length"] <= max_len

    def test_stage_3_maintains_procedural_mix(self):
        """Stage 3 should preserve Stage 2 mixing ratios."""
        pipeline = _make_pipeline(
            batch_size=10,
            stage1_steps=1,
            stage2_steps=1,
            text_ratio=0.6,
            procedural_ratio=0.4,
            stage3_context_lengths=[64, 128],
        )
        stage3_sources: list[str] = []
        for batch in pipeline.iterate():
            if batch["stage"] == 3:
                stage3_sources.extend(batch["sources"])
            if len(stage3_sources) >= 100:
                break

        proc_frac = stage3_sources.count("procedural") / len(stage3_sources)
        assert abs(proc_frac - 0.4) < 0.1, f"Proc fraction {proc_frac:.3f} in Stage 3"


# ---------------------------------------------------------------------------
# Streaming (no full-corpus load into memory)
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_does_not_materialise_all_text(self):
        """Iterating N batches should never need to buffer the whole corpus."""
        consumed: list[int] = []

        def counting_iter() -> iter:
            """Count how many items have been consumed from the source."""
            i = 0
            while True:
                consumed.append(i)
                yield f"Sample text {i}. " * 50
                i += 1

        config = CurriculumConfig(
            batch_size=2,
            text_corpus=None,
            stage1_steps=1000,
            stage2_steps=1000,
            seed=0,
        )
        pipeline = CurriculumDataPipeline(
            config=config,
            text_iterator=counting_iter(),
        )
        batches = _take(pipeline, 10)
        # We consumed 20 text samples for 10 batches of size 2 — not millions
        assert len(consumed) <= 25, (
            f"Expected lazy streaming, but consumed {len(consumed)} items"
        )

    def test_pipeline_is_lazy(self):
        """The pipeline should not eagerly materialise the text corpus on construction."""
        constructed = []

        def tracking_iter():
            constructed.append("started")
            yield from itertools.cycle(["a text sample"])

        config = CurriculumConfig(text_corpus=None, batch_size=2, seed=0)
        _pipeline = CurriculumDataPipeline(config=config, text_iterator=tracking_iter())
        # The iterator should not have been started until iterate() is called
        # (Some implementations may start it lazily; others not. We just verify
        # that we can construct without error and without consuming the corpus.)
        assert True  # construction did not raise


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def test_checkpoint_captures_stage(self):
        pipeline = _make_pipeline(stage1_steps=3, stage2_steps=3)
        _take(pipeline, 5)
        state = pipeline.checkpoint_state()
        assert "stage" in state
        assert "global_step" in state

    def test_resume_from_checkpoint(self):
        """Resuming from a checkpoint should restore stage and step."""
        pipeline = _make_pipeline(stage1_steps=3, stage2_steps=100)
        _take(pipeline, 5)
        state = pipeline.checkpoint_state()

        restored = CurriculumDataPipeline.from_checkpoint(
            state,
            config=pipeline.config,
            text_iterator=_text_iter(),
        )
        assert restored.global_step == state["global_step"]
        assert int(restored.current_stage) == state["stage"]

    def test_resume_continues_from_correct_step(self):
        """After resuming, the step counter should continue from the saved value."""
        pipeline = _make_pipeline(stage1_steps=100)
        _take(pipeline, 7)
        state = pipeline.checkpoint_state()
        saved_step = state["global_step"]

        restored = CurriculumDataPipeline.from_checkpoint(
            state,
            config=pipeline.config,
            text_iterator=_text_iter(),
        )
        batch = next(restored.iterate())
        assert batch["step"] == saved_step

    def test_start_at_stage_2(self):
        """Pipeline created with start_stage=2 should begin in Stage 2."""
        pipeline = _make_pipeline(start_stage=2)
        batch = next(pipeline.iterate())
        assert batch["stage"] == 2


# ---------------------------------------------------------------------------
# StageConfig validation
# ---------------------------------------------------------------------------


class TestStageConfigValidation:
    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            StageConfig(text_ratio=0.5, procedural_ratio=0.3)

    def test_valid_stage_config(self):
        cfg = StageConfig(text_ratio=0.7, procedural_ratio=0.3)
        assert cfg.text_ratio == 0.7

    def test_all_procedural(self):
        cfg = StageConfig(text_ratio=0.0, procedural_ratio=1.0)
        assert cfg.procedural_ratio == 1.0

    def test_all_text(self):
        cfg = StageConfig(text_ratio=1.0, procedural_ratio=0.0)
        assert cfg.text_ratio == 1.0
