"""Shared streaming dataloader for HuggingFace text datasets."""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch


def build_dataloader(
    tokenizer: object,
    dataset_name: str,
    dataset_config: str,
    max_seq_length: int,
    batch_size: int,
    seed: int = 42,
) -> object:
    """Build a streaming DataLoader over a HuggingFace text dataset.

    Returns an iterable that yields dicts with ``input_ids`` and
    ``labels`` tensors of shape (batch, seq_len).
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    def tokenize_and_chunk(example: dict) -> dict:
        text = example.get("text", "")
        ids = tokenizer(  # type: ignore[operator]
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"].squeeze(0)
        return {"input_ids": ids}

    remove_cols = [
        c for c in ["text", "id", "dump", "url", "file_path", "language",
                     "language_score", "token_count", "score", "int_score"]
        if c in (ds.features or {})
    ]
    ds = ds.map(tokenize_and_chunk, remove_columns=remove_cols or None)

    from torch.utils.data import DataLoader  # type: ignore[import]

    def collate(batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            padded[i, : ids.shape[0]] = ids
        labels = padded.clone()
        labels[labels == 0] = -100
        return {"input_ids": padded, "labels": labels}

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# Memory task data generator for mixed training
# ---------------------------------------------------------------------------


def _memory_task_examples(seed: int = 1337, include_computation: bool = False) -> Iterator[str]:
    """Infinite iterator of synthetic task prompt+answer strings.

    Uses AssociativeRecall, VariableTracking, and PasskeyRetrieval with
    varying parameters. Optionally includes computation/emergent tasks
    (ModularArithmetic, DyckLanguage, LengthExtrapolation).
    Uses a different seed than eval (eval=42) to avoid train/eval overlap.
    """
    from src.eval.benchmarks.memory import (
        AssociativeRecall,
        PasskeyRetrieval,
        VariableTracking,
    )

    task_types = ["ar", "vt", "pk"]

    if include_computation:
        from src.eval.benchmarks.computation import DyckLanguage, ModularArithmetic
        from src.eval.benchmarks.emergent import LengthExtrapolation
        task_types.extend(["modarith", "dyck", "lengthext"])

    rng = random.Random(seed)
    task_seed = seed

    while True:
        task_seed += 1
        # Vary task parameters for diversity
        task_type = rng.choice(task_types)

        if task_type == "ar":
            num_pairs = rng.randint(3, 8)
            delay = rng.randint(15, 50)
            gen = AssociativeRecall(n=1, num_pairs=num_pairs,
                                   delay_length=delay, seed=task_seed)
        elif task_type == "vt":
            num_vars = rng.randint(2, 5)
            num_ops = rng.randint(3, 8)
            gen = VariableTracking(n=1, num_variables=num_vars,
                                   num_operations=num_ops, seed=task_seed)
        elif task_type == "pk":
            ctx_len = rng.randint(100, 300)
            gen = PasskeyRetrieval(n=1, context_length=ctx_len, seed=task_seed)
        elif task_type == "modarith":
            modulus = rng.choice([23, 37, 53, 71, 97])
            gen = ModularArithmetic(n=1, modulus=modulus, seed=task_seed)
        elif task_type == "dyck":
            max_depth = rng.randint(2, 4)
            bracket_types = rng.randint(1, 2)
            gen = DyckLanguage(n=1, max_depth=max_depth,
                               bracket_types=bracket_types, seed=task_seed)
        elif task_type == "lengthext":
            train_len = rng.randint(3, 6)
            mult = rng.choice([1.0, 1.5, 2.0])
            gen = LengthExtrapolation(n=1, train_length=train_len,
                                      test_multiplier=mult, seed=task_seed)

        for ex in gen:
            # Use benchmark's native prompt format directly.
            yield f"{ex.input}\n{ex.target}"


def build_mixed_dataloader(
    tokenizer: object,
    dataset_name: str,
    dataset_config: str,
    max_seq_length: int,
    batch_size: int,
    memory_task_ratio: float = 0.1,
    seed: int = 42,
    memory_seed: int = 1337,
    include_computation: bool = False,
) -> object:
    """Build a dataloader that mixes FineWeb with synthetic tasks.

    Args:
        memory_task_ratio: Fraction of batches that are synthetic tasks (0.0-1.0).
        memory_seed: Seed for task generation (must differ from eval seed=42).
        include_computation: If True, include ModularArithmetic, DyckLanguage,
            LengthExtrapolation in the synthetic task mix.
    """
    from datasets import load_dataset  # type: ignore[import]
    from torch.utils.data import DataLoader, IterableDataset  # type: ignore[import]

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    def tokenize_text(text: str) -> torch.Tensor:
        ids = tokenizer(  # type: ignore[operator]
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"].squeeze(0)
        return ids

    class MixedDataset(IterableDataset):
        def __init__(self):
            self.fineweb_iter = None
            self.memory_iter = _memory_task_examples(seed=memory_seed, include_computation=include_computation)
            self.rng = random.Random(seed)

        def __iter__(self):
            # Lazily create fineweb iterator
            self.fineweb_iter = iter(ds)
            return self

        def __next__(self) -> dict:
            use_memory = self.rng.random() < memory_task_ratio

            if use_memory:
                text = next(self.memory_iter)
                ids = tokenize_text(text)
            else:
                example = next(self.fineweb_iter)  # type: ignore[arg-type]
                text = example.get("text", "")
                ids = tokenize_text(text)

            return {"input_ids": ids}

    def collate(batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            padded[i, : ids.shape[0]] = ids
        labels = padded.clone()
        labels[labels == 0] = -100
        return {"input_ids": padded, "labels": labels}

    return DataLoader(MixedDataset(), batch_size=batch_size, collate_fn=collate)
