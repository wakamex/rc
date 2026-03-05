"""Computation benchmark generators: arithmetic, modular arithmetic, Dyck language, program traces."""

from __future__ import annotations

import random
from collections.abc import Iterator

from src.types import BenchmarkExample

# ---------------------------------------------------------------------------
# Multi-digit Addition / Multiplication
# ---------------------------------------------------------------------------


class MultiDigitArithmetic:
    """Multi-digit addition or multiplication with carry.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        digit_count: int = 4,
        operation: str = "addition",
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples to generate.
            digit_count: Number of digits in each operand.
            operation: 'addition' or 'multiplication'.
            seed: RNG seed.
        """
        if operation not in ("addition", "multiplication"):
            raise ValueError("operation must be 'addition' or 'multiplication'")
        self.n = n
        self.digit_count = digit_count
        self.operation = operation
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)
        lo = 10 ** (self.digit_count - 1)
        hi = 10**self.digit_count - 1

        for i in range(self.n):
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)

            if self.operation == "addition":
                result = a + b
                op_sym = "+"
            else:
                result = a * b
                op_sym = "*"

            prompt = f"Compute {a} {op_sym} {b}. Answer with just the number."
            target = str(result)

            yield BenchmarkExample(
                input=prompt,
                target=target,
                metadata={
                    "task": "multi_digit_arithmetic",
                    "operation": self.operation,
                    "a": a,
                    "b": b,
                    "result": result,
                    "digit_count": self.digit_count,
                    "index": i,
                },
            )


# ---------------------------------------------------------------------------
# Modular Arithmetic
# ---------------------------------------------------------------------------


class ModularArithmetic:
    """Modular arithmetic: a op b mod m.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        operand_size: int = 100,
        modulus: int = 97,
        operations: list[str] | None = None,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples to generate.
            operand_size: Upper bound for operands (exclusive).
            modulus: The modulus m.
            operations: List of ops to sample from: 'add', 'sub', 'mul'. Default all three.
            seed: RNG seed.
        """
        self.n = n
        self.operand_size = operand_size
        self.modulus = modulus
        self.operations = operations or ["add", "sub", "mul"]
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)
        op_symbols = {"add": "+", "sub": "-", "mul": "*"}

        for i in range(self.n):
            a = rng.randint(0, self.operand_size - 1)
            b = rng.randint(0, self.operand_size - 1)
            op = rng.choice(self.operations)
            sym = op_symbols[op]

            if op == "add":
                result = (a + b) % self.modulus
            elif op == "sub":
                result = (a - b) % self.modulus
            else:
                result = (a * b) % self.modulus

            prompt = f"Compute ({a} {sym} {b}) mod {self.modulus}. Answer with just the number."
            target = str(result)

            yield BenchmarkExample(
                input=prompt,
                target=target,
                metadata={
                    "task": "modular_arithmetic",
                    "operation": op,
                    "a": a,
                    "b": b,
                    "modulus": self.modulus,
                    "result": result,
                    "operand_size": self.operand_size,
                    "index": i,
                },
            )


# ---------------------------------------------------------------------------
# Dyck Language Recognition
# ---------------------------------------------------------------------------

_OPEN_CLOSE = {"(": ")", "[": "]", "{": "}"}
_OPEN = list(_OPEN_CLOSE.keys())
_CLOSE = list(_OPEN_CLOSE.values())


class DyckLanguage:
    """Balanced parentheses / Dyck language recognition.

    Generates both valid (balanced) and invalid (unbalanced) strings,
    asking the model to classify them.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        max_depth: int = 4,
        length: int = 10,
        bracket_types: int = 1,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples.
            max_depth: Maximum nesting depth for valid strings.
            length: Approximate number of bracket pairs in valid strings.
            bracket_types: How many bracket types to use (1-3).
            seed: RNG seed.
        """
        self.n = n
        self.max_depth = max_depth
        self.length = length
        self.bracket_types = min(max(bracket_types, 1), 3)
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def _gen_valid(self, rng: random.Random) -> str:
        """Generate a random valid Dyck string."""
        opens = _OPEN[: self.bracket_types]
        stack: list[str] = []
        result: list[str] = []
        remaining = self.length * 2  # total tokens

        while remaining > 0:
            can_open = len(stack) < self.max_depth and remaining > len(stack)
            can_close = len(stack) > 0

            if can_open and can_close:
                choice = rng.choice(["open", "close"])
            elif can_open:
                choice = "open"
            elif can_close:
                choice = "close"
            else:
                break

            if choice == "open":
                b = rng.choice(opens)
                stack.append(b)
                result.append(b)
            else:
                b = stack.pop()
                result.append(_OPEN_CLOSE[b])

            remaining -= 1

        # Close any remaining
        while stack:
            result.append(_OPEN_CLOSE[stack.pop()])

        return "".join(result)

    def _corrupt(self, s: str, rng: random.Random) -> str:
        """Create an unbalanced version by swapping or removing a bracket."""
        chars = list(s)
        if not chars:
            return "()"[::-1]  # trivially invalid
        idx = rng.randrange(len(chars))
        method = rng.choice(["swap", "remove", "wrong_close"])
        if method == "swap" and len(chars) >= 2:
            j = rng.randrange(len(chars))
            chars[idx], chars[j] = chars[j], chars[idx]
        elif method == "remove":
            chars.pop(idx)
        else:
            opens = _OPEN[: self.bracket_types]
            chars[idx] = rng.choice(opens)
        return "".join(chars)

    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)

        for i in range(self.n):
            is_valid = rng.random() < 0.5
            valid_str = self._gen_valid(rng)

            if is_valid:
                s = valid_str
                label = "yes"
            else:
                s = self._corrupt(valid_str, rng)
                # verify it's actually invalid
                if self._check_valid(s):
                    label = "yes"
                    is_valid = True
                else:
                    label = "no"

            prompt = (
                f"Is the following bracket sequence balanced? "
                f"Answer 'yes' or 'no'.\n\n{s}"
            )

            yield BenchmarkExample(
                input=prompt,
                target=label,
                metadata={
                    "task": "dyck_language",
                    "sequence": s,
                    "is_valid": is_valid,
                    "max_depth": self.max_depth,
                    "bracket_types": self.bracket_types,
                    "index": i,
                },
            )

    @staticmethod
    def _check_valid(s: str) -> bool:
        stack: list[str] = []
        for ch in s:
            if ch in _OPEN_CLOSE:
                stack.append(ch)
            elif ch in _CLOSE:
                if not stack or _OPEN_CLOSE[stack[-1]] != ch:
                    return False
                stack.pop()
        return len(stack) == 0


# ---------------------------------------------------------------------------
# Synthetic Program Execution Traces
# ---------------------------------------------------------------------------


class ProgramTrace:
    """Synthetic program execution traces (variable assignments and arithmetic).

    Generates a short program and asks the model to report the value of a
    variable at the end.

    Implements the Generator protocol.
    """

    def __init__(
        self,
        n: int = 1000,
        num_steps: int = 6,
        num_vars: int = 3,
        value_range: int = 20,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            n: Number of examples.
            num_steps: Number of program statements.
            num_vars: Number of variable slots.
            value_range: Literals are drawn from [0, value_range).
            seed: RNG seed.
        """
        self.n = n
        self.num_steps = num_steps
        self.num_vars = num_vars
        self.value_range = value_range
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[BenchmarkExample]:
        var_names = [f"x{k}" for k in range(self.num_vars)]
        rng = random.Random(self.seed)

        for i in range(self.n):
            state: dict[str, int] = {v: 0 for v in var_names}
            stmts: list[str] = []

            for _ in range(self.num_steps):
                target_var = rng.choice(var_names)
                kind = rng.choice(["literal", "copy", "add", "sub"])

                if kind == "literal":
                    val = rng.randint(0, self.value_range - 1)
                    state[target_var] = val
                    stmts.append(f"{target_var} = {val}")
                elif kind == "copy":
                    src = rng.choice(var_names)
                    state[target_var] = state[src]
                    stmts.append(f"{target_var} = {src}")
                elif kind == "add":
                    src = rng.choice(var_names)
                    literal = rng.randint(0, self.value_range // 2)
                    state[target_var] = state[src] + literal
                    stmts.append(f"{target_var} = {src} + {literal}")
                else:
                    src = rng.choice(var_names)
                    literal = rng.randint(0, self.value_range // 2)
                    state[target_var] = state[src] - literal
                    stmts.append(f"{target_var} = {src} - {literal}")

            query_var = rng.choice(var_names)
            program_text = "\n".join(stmts)
            prompt = (
                f"Execute the following program and report the final value of {query_var}:\n\n"
                f"{program_text}\n\n"
                f"What is {query_var}?"
            )
            target = str(state[query_var])

            yield BenchmarkExample(
                input=prompt,
                target=target,
                metadata={
                    "task": "program_trace",
                    "query_var": query_var,
                    "final_state": dict(state),
                    "num_steps": self.num_steps,
                    "num_vars": self.num_vars,
                    "index": i,
                },
            )
