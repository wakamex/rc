"""Emergent capability benchmark generators: compositional generalization, length extrapolation, algorithmic transfer."""

from __future__ import annotations

import random
from collections.abc import Iterator

from src.types import BenchmarkExample

# ---------------------------------------------------------------------------
# Compositional Generalization
# ---------------------------------------------------------------------------

_ALL_OPERATORS = ["+", "-", "*", "//", "%"]


class CompositionalGeneralization:
    """Train on a subset of operators, test on held-out combinations.

    Each example presents a multi-step expression using a specific set of
    operators.  The generator is parameterised as 'train' or 'test' split,
    where the test split uses held-out operator combinations not seen in train.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        num_operands: int = 3,
        value_range: int = 20,
        split: str = "train",
        held_out_ops: list[str] | None = None,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples.
            num_operands: How many operands per expression.
            value_range: Operands drawn from [1, value_range].
            split: 'train' or 'test'.
            held_out_ops: Operators reserved for the test split.
                          Default: ['*', '//'] are held out.
            seed: RNG seed.
        """
        self.n = n
        self.num_operands = num_operands
        self.value_range = value_range
        self.split = split
        self.held_out_ops = held_out_ops or ["*", "//"]
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def _safe_eval(self, a: int, op: str, b: int) -> int | None:
        """Evaluate a op b safely; return None on error (e.g. div by zero)."""
        try:
            if op in ("//", "%") and b == 0:
                return None
            return int(eval(f"{a} {op} {b}"))  # noqa: S307
        except Exception:
            return None

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)
        train_ops = [op for op in _ALL_OPERATORS if op not in self.held_out_ops]

        if self.split == "train":
            available_ops = train_ops
        else:
            available_ops = self.held_out_ops if self.held_out_ops else _ALL_OPERATORS

        count = 0
        attempts = 0
        max_attempts = self.n * 20

        while count < self.n and attempts < max_attempts:
            attempts += 1
            operands = [rng.randint(1, self.value_range) for _ in range(self.num_operands)]
            ops = [rng.choice(available_ops) for _ in range(self.num_operands - 1)]

            # Build expression left to right, tracking the running value
            value = operands[0]
            parts = [str(operands[0])]
            valid = True
            for op, operand in zip(ops, operands[1:]):
                result = self._safe_eval(value, op, operand)
                if result is None:
                    valid = False
                    break
                value = result
                parts.append(op)
                parts.append(str(operand))

            if not valid:
                continue

            expr = " ".join(parts)
            prompt = f"Evaluate: {expr}\n\nAnswer with just the number."
            target = str(value)

            yield BenchmarkExample(
                input=prompt,
                target=target,
                metadata={
                    "task": "compositional_generalization",
                    "expression": expr,
                    "operators": ops,
                    "result": value,
                    "split": self.split,
                    "num_operands": self.num_operands,
                    "index": count,
                },
            )
            count += 1


# ---------------------------------------------------------------------------
# Length Extrapolation
# ---------------------------------------------------------------------------


class LengthExtrapolation:
    """Generate sequences at train length, test at 2x-10x length.

    The task is to compute the sum of a sequence of integers.  At train length
    the sequences are short; at test lengths they are proportionally longer.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        train_length: int = 5,
        test_multiplier: float = 1.0,
        value_range: int = 10,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples.
            train_length: Number of elements at the base (train) length.
            test_multiplier: Multiplier applied to train_length for this split.
                             1.0 = train, 2.0 = 2x, ..., 10.0 = 10x.
            value_range: Elements drawn from [1, value_range].
            seed: RNG seed.
        """
        self.n = n
        self.train_length = train_length
        self.test_multiplier = test_multiplier
        self.value_range = value_range
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)
        seq_length = max(1, int(self.train_length * self.test_multiplier))

        for i in range(self.n):
            sequence = [rng.randint(1, self.value_range) for _ in range(seq_length)]
            total = sum(sequence)

            seq_str = ", ".join(map(str, sequence))
            prompt = f"Compute the sum of: {seq_str}\n\nAnswer with just the number."

            yield BenchmarkExample(
                input=prompt,
                target=str(total),
                metadata={
                    "task": "length_extrapolation",
                    "sequence": sequence,
                    "sum": total,
                    "seq_length": seq_length,
                    "train_length": self.train_length,
                    "test_multiplier": self.test_multiplier,
                    "index": i,
                },
            )


# ---------------------------------------------------------------------------
# Algorithmic Transfer
# ---------------------------------------------------------------------------

_SORT_ALGOS = ["bubble_sort", "selection_sort"]
_SEARCH_ALGOS = ["linear_search", "binary_search"]


class AlgorithmicTransfer:
    """Train on one algorithm, test on a related algorithm.

    Two task families are supported:
      - sorting: bubble_sort vs selection_sort (both output a sorted list)
      - search: linear_search vs binary_search (both find an element's index)

    The prompt describes the algorithm by name and gives the input.  The
    target is always the correct output, regardless of algorithm name.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        family: str = "sorting",
        split: str = "train",
        seq_length: int = 6,
        value_range: int = 20,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples.
            family: 'sorting' or 'search'.
            split: 'train' uses the first algorithm; 'test' uses the second.
            seq_length: Length of the input sequence.
            value_range: Values drawn from [1, value_range].
            seed: RNG seed.
        """
        if family not in ("sorting", "search"):
            raise ValueError("family must be 'sorting' or 'search'")
        self.n = n
        self.family = family
        self.split = split
        self.seq_length = seq_length
        self.value_range = value_range
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def _sorting_example(
        self, seq: list[int], algo: str
    ) -> tuple[str, str]:
        sorted_seq = sorted(seq)
        seq_str = ", ".join(map(str, seq))
        result_str = ", ".join(map(str, sorted_seq))
        prompt = (
            f"Apply {algo.replace('_', ' ')} to sort the following list: [{seq_str}]\n\n"
            f"Answer with the sorted list in the format: [a, b, c, ...]"
        )
        return prompt, f"[{result_str}]"

    def _search_example(
        self, seq: list[int], target: int, algo: str
    ) -> tuple[str, str]:
        try:
            idx = seq.index(target)
        except ValueError:
            idx = -1
        seq_str = ", ".join(map(str, seq))
        prompt = (
            f"Use {algo.replace('_', ' ')} to find the index of {target} in [{seq_str}]\n\n"
            f"Answer with just the index (0-based), or -1 if not found."
        )
        return prompt, str(idx)

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)

        if self.family == "sorting":
            algo = _SORT_ALGOS[0] if self.split == "train" else _SORT_ALGOS[1]
        else:
            algo = _SEARCH_ALGOS[0] if self.split == "train" else _SEARCH_ALGOS[1]

        for i in range(self.n):
            seq = [rng.randint(1, self.value_range) for _ in range(self.seq_length)]

            if self.family == "sorting":
                prompt, target = self._sorting_example(seq, algo)
                meta: dict = {
                    "task": "algorithmic_transfer",
                    "family": self.family,
                    "algorithm": algo,
                    "sequence": seq,
                    "split": self.split,
                    "index": i,
                }
            else:
                search_target = rng.choice(seq + [rng.randint(1, self.value_range)])
                prompt, target = self._search_example(seq, search_target, algo)
                meta = {
                    "task": "algorithmic_transfer",
                    "family": self.family,
                    "algorithm": algo,
                    "sequence": seq,
                    "search_target": search_target,
                    "split": self.split,
                    "index": i,
                }

            yield BenchmarkExample(input=prompt, target=target, metadata=meta)
