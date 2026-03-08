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


def _memory_task_examples(seed: int = 1337) -> Iterator[str]:
    """Infinite iterator of synthetic memory task prompt+answer strings.

    Uses AssociativeRecall, VariableTracking, and PasskeyRetrieval with
    varying parameters. Uses a different seed than eval (eval=42) to
    avoid train/eval overlap.
    """
    from src.eval.benchmarks.memory import (
        AssociativeRecall,
        PasskeyRetrieval,
        VariableTracking,
    )

    rng = random.Random(seed)
    task_seed = seed

    while True:
        task_seed += 1
        # Vary task parameters for diversity
        task_type = rng.choice(["ar", "vt", "pk"])

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
        else:
            ctx_len = rng.randint(100, 300)
            gen = PasskeyRetrieval(n=1, context_length=ctx_len, seed=task_seed)

        for ex in gen:
            # Format as prompt + answer for causal LM training
            yield f"{ex.input}\nAnswer: {ex.target}"


def build_mixed_dataloader(
    tokenizer: object,
    dataset_name: str,
    dataset_config: str,
    max_seq_length: int,
    batch_size: int,
    memory_task_ratio: float = 0.1,
    seed: int = 42,
    memory_seed: int = 1337,
) -> object:
    """Build a dataloader that mixes FineWeb with synthetic memory tasks.

    Args:
        memory_task_ratio: Fraction of batches that are memory tasks (0.0-1.0).
        memory_seed: Seed for memory task generation (must differ from eval seed=42).
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
            self.memory_iter = _memory_task_examples(seed=memory_seed)
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
