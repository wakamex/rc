"""Shared streaming dataloader for HuggingFace text datasets."""

from __future__ import annotations

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
