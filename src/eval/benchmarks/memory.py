"""Memory benchmark generators: passkey retrieval, variable tracking, associative recall."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from src.types import BenchmarkExample

# ---------------------------------------------------------------------------
# Distractor text helpers
# ---------------------------------------------------------------------------

_DISTRACTOR_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
    "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
    "aliquip", "ex", "ea", "commodo", "consequat", "duis", "aute", "irure",
    "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
    "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
    "proident", "sunt", "culpa", "qui", "officia", "deserunt", "mollit",
]


def _make_distractor(length: int, seed: int | None = None) -> str:
    """Generate a string of distractor words."""
    rng = random.Random(seed)
    words = [rng.choice(_DISTRACTOR_WORDS) for _ in range(length)]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Passkey Retrieval
# ---------------------------------------------------------------------------


class PasskeyRetrieval:
    """Hide a passkey at configurable depth in distractor text, query at end.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        context_length: int = 200,
        passkey_position: float | None = None,
        distractor_type: str = "words",
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples to generate.
            context_length: Number of distractor words surrounding the passkey.
            passkey_position: Fraction in [0, 1] where the passkey is placed.
                              None means randomised per example.
            distractor_type: 'words' (random words) or 'numbers' (random digits).
            seed: RNG seed for reproducibility.
        """
        self.n = n
        self.context_length = context_length
        self.passkey_position = passkey_position
        self.distractor_type = distractor_type
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def _make_distractor_chunk(self, length: int, rng: random.Random) -> str:
        if self.distractor_type == "numbers":
            return " ".join(str(rng.randint(1000, 9999)) for _ in range(length))
        return " ".join(rng.choice(_DISTRACTOR_WORDS) for _ in range(length))

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)
        for i in range(self.n):
            passkey = str(rng.randint(10000, 99999))
            if self.passkey_position is None:
                pos = rng.random()
            else:
                pos = float(self.passkey_position)

            before = max(1, int(pos * self.context_length))
            after = max(0, self.context_length - before)

            before_text = self._make_distractor_chunk(before, rng)
            after_text = self._make_distractor_chunk(after, rng)

            if after_text:
                text = (
                    f"{before_text} "
                    f"The passkey is {passkey}. "
                    f"{after_text}"
                )
            else:
                text = f"{before_text} The passkey is {passkey}."

            prompt = text + "\n\nWhat is the passkey?"

            yield BenchmarkExample(
                input=prompt,
                target=passkey,
                metadata={
                    "task": "passkey_retrieval",
                    "passkey": passkey,
                    "position": pos,
                    "context_length": self.context_length,
                    "distractor_type": self.distractor_type,
                    "index": i,
                },
            )


# ---------------------------------------------------------------------------
# Variable Tracking
# ---------------------------------------------------------------------------


class VariableTracking:
    """Define variables, perform assignments across distractors, query final value.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        num_variables: int = 3,
        num_operations: int = 5,
        distractor_length: int = 20,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples to generate.
            num_variables: How many variable names to use.
            num_operations: Number of assignment operations.
            distractor_length: Words of distractor text between operations.
            seed: RNG seed.
        """
        self.n = n
        self.num_variables = num_variables
        self.num_operations = num_operations
        self.distractor_length = distractor_length
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[BenchmarkExample]:
        var_names = [chr(ord("a") + k) for k in range(self.num_variables)]
        rng = random.Random(self.seed)

        for i in range(self.n):
            state: dict[str, int] = {}
            lines: list[str] = []

            # Initialise all variables
            for v in var_names:
                val = rng.randint(0, 9)
                state[v] = val
                lines.append(f"Let {v} = {val}.")

            lines.append(_make_distractor(self.distractor_length, rng.randint(0, 10**6)))

            # Perform assignments
            ops: list[dict[str, Any]] = []
            for _ in range(self.num_operations):
                target_var = rng.choice(var_names)
                op = rng.choice(["assign", "add", "sub"])
                if op == "assign":
                    new_val = rng.randint(0, 9)
                    state[target_var] = new_val
                    lines.append(f"Set {target_var} = {new_val}.")
                elif op == "add":
                    addend = rng.randint(1, 5)
                    state[target_var] += addend
                    lines.append(f"Increment {target_var} by {addend}.")
                else:
                    sub = rng.randint(1, 5)
                    state[target_var] -= sub
                    lines.append(f"Decrement {target_var} by {sub}.")
                ops.append({"op": op, "var": target_var})
                lines.append(_make_distractor(self.distractor_length, rng.randint(0, 10**6)))

            query_var = rng.choice(var_names)
            prompt = "\n".join(lines) + f"\n\nWhat is the final value of {query_var}?"
            target = str(state[query_var])

            yield BenchmarkExample(
                input=prompt,
                target=target,
                metadata={
                    "task": "variable_tracking",
                    "query_var": query_var,
                    "final_state": dict(state),
                    "num_variables": self.num_variables,
                    "num_operations": self.num_operations,
                    "distractor_length": self.distractor_length,
                    "index": i,
                },
            )


# ---------------------------------------------------------------------------
# Associative Recall
# ---------------------------------------------------------------------------


class AssociativeRecall:
    """Present key-value pairs, add distractor, query by key.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        num_pairs: int = 5,
        delay_length: int = 30,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples to generate.
            num_pairs: Number of key-value pairs presented.
            delay_length: Words of distractor between pairs and query.
            seed: RNG seed.
        """
        self.n = n
        self.num_pairs = num_pairs
        self.delay_length = delay_length
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)

        for i in range(self.n):
            # Generate unique keys and values
            keys = rng.sample(_DISTRACTOR_WORDS, min(self.num_pairs, len(_DISTRACTOR_WORDS)))
            pairs: dict[str, str] = {}
            for k in keys:
                val = str(rng.randint(100, 999))
                pairs[k] = val

            lines = ["Here are some key-value pairs:"]
            for k, v in pairs.items():
                lines.append(f"  {k}: {v}")

            lines.append("")
            lines.append(_make_distractor(self.delay_length, rng.randint(0, 10**6)))

            query_key = rng.choice(list(pairs.keys()))
            prompt = "\n".join(lines) + f"\n\nWhat is the value for key '{query_key}'?"

            yield BenchmarkExample(
                input=prompt,
                target=pairs[query_key],
                metadata={
                    "task": "associative_recall",
                    "query_key": query_key,
                    "num_pairs": self.num_pairs,
                    "delay_length": self.delay_length,
                    "pairs": dict(pairs),
                    "index": i,
                },
            )
