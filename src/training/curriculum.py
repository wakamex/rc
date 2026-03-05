"""Three-stage training curriculum for Track C pre-training."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from src.eval.benchmarks.computation import (
    DyckLanguage,
    ModularArithmetic,
    MultiDigitArithmetic,
    ProgramTrace,
)
from src.types import DataPipeline


# ---------------------------------------------------------------------------
# Stage identifiers
# ---------------------------------------------------------------------------


class CurriculumStage(IntEnum):
    """The three curriculum stages."""

    STAGE_1 = 1  # General text, 4K context, standard next-token prediction
    STAGE_2 = 2  # Mixed text + procedural, 4K context
    STAGE_3 = 3  # Length curriculum with Stage 2 mixing ratios


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""

    text_ratio: float = 1.0
    """Fraction of batch examples drawn from the general text corpus."""

    procedural_ratio: float = 0.0
    """Fraction of batch examples drawn from procedural tasks."""

    context_length: int = 4096
    """Maximum token context length for this stage."""

    max_steps: int | None = None
    """Maximum training steps for this stage. None means no hard limit."""

    def __post_init__(self) -> None:
        if abs(self.text_ratio + self.procedural_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"text_ratio + procedural_ratio must equal 1.0, "
                f"got {self.text_ratio} + {self.procedural_ratio}"
            )


@dataclass
class CurriculumConfig:
    """Full configuration for the three-stage training curriculum."""

    # Stage 1: standard next-token prediction on general text
    stage1: StageConfig = field(
        default_factory=lambda: StageConfig(
            text_ratio=1.0,
            procedural_ratio=0.0,
            context_length=4096,
        )
    )

    # Stage 2: mixed text + procedural tasks
    stage2: StageConfig = field(
        default_factory=lambda: StageConfig(
            text_ratio=0.7,
            procedural_ratio=0.3,
            context_length=4096,
        )
    )

    # Stage 3: length curriculum — uses Stage 2 mixing ratios, grows context
    stage3_context_lengths: list[int] = field(
        default_factory=lambda: [4096, 8192, 16384, 32768, 131072]
    )
    """Sequence of context lengths to cycle through in Stage 3."""

    stage3_steps_per_length: int | None = None
    """Steps at each context length before advancing. None = manual control."""

    # Batch configuration
    batch_size: int = 8

    # Text corpus — HuggingFace dataset name, or None to use synthetic text
    text_corpus: str | None = "HuggingFaceFW/fineweb"
    text_corpus_split: str = "train"
    text_column: str = "text"
    text_corpus_name: str | None = "sample-10BT"
    """datasets config name (e.g. 'sample-10BT' for FineWeb)."""

    # Stage transition thresholds (step-based)
    stage1_steps: int = 10_000
    """Steps in Stage 1 before transitioning to Stage 2."""

    stage2_steps: int = 10_000
    """Steps in Stage 2 before transitioning to Stage 3."""

    # Loss-plateau-based transition (alternative to step-based)
    use_loss_plateau: bool = False
    loss_plateau_patience: int = 500
    loss_plateau_threshold: float = 0.01

    # Procedural data settings
    procedural_n: int = 100_000
    """Number of examples per procedural generator before cycling."""

    # Auxiliary reservoir-state prediction loss
    aux_reservoir_loss: bool = False

    # RNG seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infinite_procedural(rng: random.Random, n: int, seed_base: int) -> Iterator[str]:
    """Yield procedural training examples (arithmetic, Dyck, state-tracking) indefinitely."""
    generators = [
        MultiDigitArithmetic(n=n, digit_count=4, operation="addition", seed=seed_base),
        MultiDigitArithmetic(n=n, digit_count=3, operation="multiplication", seed=seed_base + 1),
        ModularArithmetic(n=n, seed=seed_base + 2),
        DyckLanguage(n=n, max_depth=4, bracket_types=2, seed=seed_base + 3),
        ProgramTrace(n=n, num_steps=6, num_vars=3, seed=seed_base + 4),
    ]
    # Flatten all generators into one pool and cycle
    pool: list[str] = []
    for gen in generators:
        for ex in gen:
            pool.append(f"{ex.input}\nAnswer: {ex.target}")
    rng.shuffle(pool)
    yield from itertools.cycle(pool)


def _synthetic_text_stream(rng: random.Random) -> Iterator[str]:
    """Generate synthetic placeholder text for testing (no corpus download needed)."""
    templates = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "In the beginning there was light and darkness. " * 20,
        "Mathematical reasoning requires careful step-by-step thinking. " * 20,
        "Language models learn from vast amounts of text data. " * 20,
        "The cat sat on the mat and looked out the window. " * 20,
    ]
    idx = 0
    while True:
        yield templates[idx % len(templates)]
        idx += 1


def _fineweb_stream(config: CurriculumConfig) -> Iterator[str]:
    """Stream text from a HuggingFace dataset. Falls back to synthetic on import error."""
    try:
        from datasets import load_dataset  # type: ignore[import]

        kwargs: dict[str, Any] = {
            "split": config.text_corpus_split,
            "streaming": True,
        }
        if config.text_corpus_name is not None:
            kwargs["name"] = config.text_corpus_name

        ds = load_dataset(config.text_corpus, **kwargs)
        for item in ds:
            text = item.get(config.text_column, "")
            if text:
                yield text
    except Exception:
        # Fallback: synthetic stream so pipeline is always usable without network
        rng = random.Random(0)
        yield from _synthetic_text_stream(rng)


# ---------------------------------------------------------------------------
# Sub-pipeline for split() support
# ---------------------------------------------------------------------------


class _SlicePipeline:
    """Wraps a parent pipeline and samples only a fraction of its batches."""

    def __init__(self, parent: CurriculumDataPipeline, ratio: float) -> None:
        self._parent = parent
        self._ratio = ratio

    def iterate(self, split: str = "train") -> Iterator[Any]:
        rng = random.Random(self._parent.config.seed + hash(split))
        for batch in self._parent.iterate(split=split):
            if rng.random() < self._ratio:
                yield batch

    def split(self, ratios: dict[str, float]) -> dict[str, DataPipeline]:
        return self._parent.split(ratios)


# ---------------------------------------------------------------------------
# Main CurriculumDataPipeline
# ---------------------------------------------------------------------------


class CurriculumDataPipeline:
    """Three-stage training curriculum data pipeline for Track C pre-training.

    Implements the DataPipeline protocol from src.types.

    Stages:
        1 — Standard next-token prediction on general text (4K context).
        2 — Procedural objective mixing: 70% text / 30% procedural (4K).
        3 — Length curriculum 4K → 8K → 16K → 32K → 128K, Stage 2 mix.

    Each batch is a ``dict`` with the following keys:
        - ``"texts"``: list[str] — one entry per batch element
        - ``"sources"``: list[str] — ``"text"`` or ``"procedural"``
        - ``"stage"``: int — current curriculum stage (1, 2, or 3)
        - ``"context_length"``: int — active context length this step
        - ``"step"``: int — global step counter (incremented per batch)
        - ``"aux_reservoir_loss"``: bool — whether auxiliary loss is active
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        *,
        text_iterator: Iterator[str] | None = None,
        start_stage: int = 1,
        start_step: int = 0,
    ) -> None:
        """
        Args:
            config: Curriculum configuration. Defaults to ``CurriculumConfig()``.
            text_iterator: Override the text corpus with a custom iterator
                (useful for testing without downloading a dataset).
            start_stage: Resume from this stage (1, 2, or 3).
            start_step: Resume from this global step count.
        """
        self.config = config or CurriculumConfig()
        self._text_iterator = text_iterator
        self._rng = random.Random(self.config.seed)
        self._current_stage = CurriculumStage(start_stage)
        self._global_step = start_step
        self._stage_step = 0  # steps within the current stage
        self._loss_history: list[float] = []
        self._stage3_length_idx = 0
        self._stage3_length_step = 0  # steps at the current Stage 3 length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        """Currently active curriculum stage."""
        return self._current_stage

    @property
    def global_step(self) -> int:
        """Total number of batches yielded so far."""
        return self._global_step

    @property
    def current_context_length(self) -> int:
        """Active context length for the current stage / sub-stage."""
        if self._current_stage == CurriculumStage.STAGE_1:
            return self.config.stage1.context_length
        if self._current_stage == CurriculumStage.STAGE_2:
            return self.config.stage2.context_length
        # Stage 3
        idx = min(self._stage3_length_idx, len(self.config.stage3_context_lengths) - 1)
        return self.config.stage3_context_lengths[idx]

    def report_loss(self, loss: float) -> None:
        """Report the current training loss for loss-plateau-based stage transitions.

        Args:
            loss: The most recent training loss value.
        """
        self._loss_history.append(loss)
        if self.config.use_loss_plateau:
            self._maybe_transition_on_plateau()

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the pipeline state for checkpointing."""
        return {
            "global_step": self._global_step,
            "stage": int(self._current_stage),
            "stage_step": self._stage_step,
            "stage3_length_idx": self._stage3_length_idx,
            "stage3_length_step": self._stage3_length_step,
        }

    @classmethod
    def from_checkpoint(
        cls,
        state: dict[str, Any],
        config: CurriculumConfig | None = None,
        text_iterator: Iterator[str] | None = None,
    ) -> CurriculumDataPipeline:
        """Restore a pipeline from a saved checkpoint state.

        Args:
            state: Dict previously returned by ``checkpoint_state()``.
            config: Pipeline configuration. If None, uses defaults.
            text_iterator: Optional text source override.

        Returns:
            A new ``CurriculumDataPipeline`` at the checkpointed position.
        """
        pipeline = cls(
            config=config,
            text_iterator=text_iterator,
            start_stage=state["stage"],
            start_step=state["global_step"],
        )
        pipeline._stage_step = state.get("stage_step", 0)
        pipeline._stage3_length_idx = state.get("stage3_length_idx", 0)
        pipeline._stage3_length_step = state.get("stage3_length_step", 0)
        return pipeline

    # ------------------------------------------------------------------
    # DataPipeline protocol
    # ------------------------------------------------------------------

    def iterate(self, split: str = "train") -> Iterator[Any]:
        """Yield batches following the three-stage curriculum.

        Each batch is a ``dict``; see the class docstring for the schema.

        Args:
            split: Data split to iterate. Only ``"train"`` uses the full
                curriculum; other values (``"val"``, ``"test"``) yield
                Stage 1 batches from the text corpus only.

        Yields:
            Batch dicts.
        """
        text_src = self._build_text_source()
        proc_src = _infinite_procedural(self._rng, self.config.procedural_n, self.config.seed)

        if split != "train":
            # Non-training splits: plain text only, no stage advancement
            yield from self._val_iterate(text_src, split=split)
            return

        while True:
            stage_cfg = self._stage_config()
            ctx_len = self.current_context_length
            batch = self._build_batch(text_src, proc_src, stage_cfg, ctx_len)
            yield batch

            self._global_step += 1
            self._stage_step += 1
            self._maybe_advance_stage3_length()
            self._maybe_transition_on_steps()

    def split(self, ratios: dict[str, float]) -> dict[str, DataPipeline]:
        """Return sub-pipelines for each named split based on sampling ratios.

        The ratios should sum to ≤ 1.0. Each sub-pipeline independently
        samples from this pipeline's iterate stream.

        Args:
            ratios: Mapping of split name to sampling fraction.

        Returns:
            Dict of split name → sub-pipeline.
        """
        total = sum(ratios.values())
        if total > 1.0 + 1e-6:
            raise ValueError(f"Ratios sum to {total:.3f} > 1.0")
        return {name: _SlicePipeline(self, r) for name, r in ratios.items()}  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_text_source(self) -> Iterator[str]:
        """Return the text source iterator (custom or corpus-based)."""
        if self._text_iterator is not None:
            return self._text_iterator
        if self.config.text_corpus is not None:
            return _fineweb_stream(self.config)
        return _synthetic_text_stream(self._rng)

    def _stage_config(self) -> StageConfig:
        """Return the StageConfig for the current stage."""
        if self._current_stage == CurriculumStage.STAGE_1:
            return self.config.stage1
        if self._current_stage == CurriculumStage.STAGE_2:
            return self.config.stage2
        # Stage 3 reuses Stage 2 mixing ratios
        return StageConfig(
            text_ratio=self.config.stage2.text_ratio,
            procedural_ratio=self.config.stage2.procedural_ratio,
            context_length=self.current_context_length,
        )

    def _build_batch(
        self,
        text_src: Iterator[str],
        proc_src: Iterator[str],
        stage_cfg: StageConfig,
        ctx_len: int,
    ) -> dict[str, Any]:
        """Build one training batch according to the current stage config."""
        texts: list[str] = []
        sources: list[str] = []

        n_text = round(self.config.batch_size * stage_cfg.text_ratio)
        n_proc = self.config.batch_size - n_text

        for _ in range(n_text):
            texts.append(next(text_src))
            sources.append("text")

        for _ in range(n_proc):
            texts.append(next(proc_src))
            sources.append("procedural")

        # Shuffle within batch
        combined = list(zip(texts, sources))
        self._rng.shuffle(combined)
        texts, sources = zip(*combined) if combined else ([], [])

        return {
            "texts": list(texts),
            "sources": list(sources),
            "stage": int(self._current_stage),
            "context_length": ctx_len,
            "step": self._global_step,
            "aux_reservoir_loss": self.config.aux_reservoir_loss,
        }

    def _val_iterate(self, text_src: Iterator[str], split: str) -> Iterator[Any]:
        """Yield plain text batches for validation/test splits (no stage advancement)."""
        for _ in itertools.count():
            texts = [next(text_src) for _ in range(self.config.batch_size)]
            yield {
                "texts": texts,
                "sources": ["text"] * self.config.batch_size,
                "stage": int(self._current_stage),
                "context_length": self.config.stage1.context_length,
                "step": self._global_step,
                "aux_reservoir_loss": False,
            }

    def _maybe_transition_on_steps(self) -> None:
        """Advance stage when step thresholds are reached (step-based mode)."""
        if self.config.use_loss_plateau:
            return
        if (
            self._current_stage == CurriculumStage.STAGE_1
            and self._stage_step >= self.config.stage1_steps
        ):
            self._advance_stage()
        elif (
            self._current_stage == CurriculumStage.STAGE_2
            and self._stage_step >= self.config.stage2_steps
        ):
            self._advance_stage()

    def _maybe_transition_on_plateau(self) -> None:
        """Advance stage when training loss plateaus (loss-based mode)."""
        patience = self.config.loss_plateau_patience
        threshold = self.config.loss_plateau_threshold
        if len(self._loss_history) < patience:
            return
        recent = self._loss_history[-patience:]
        delta = max(recent) - min(recent)
        if delta < threshold and self._current_stage != CurriculumStage.STAGE_3:
            self._advance_stage()

    def _advance_stage(self) -> None:
        """Move to the next curriculum stage and reset the stage-local step counter."""
        if self._current_stage == CurriculumStage.STAGE_1:
            self._current_stage = CurriculumStage.STAGE_2
        elif self._current_stage == CurriculumStage.STAGE_2:
            self._current_stage = CurriculumStage.STAGE_3
        # Stage 3 is the final stage — no further transition
        self._stage_step = 0

    def _maybe_advance_stage3_length(self) -> None:
        """Within Stage 3, advance to the next context length when threshold is reached."""
        if self._current_stage != CurriculumStage.STAGE_3:
            return
        spl = self.config.stage3_steps_per_length
        if spl is None:
            return
        self._stage3_length_step += 1
        if self._stage3_length_step >= spl:
            self._stage3_length_idx = min(
                self._stage3_length_idx + 1,
                len(self.config.stage3_context_lengths) - 1,
            )
            self._stage3_length_step = 0
