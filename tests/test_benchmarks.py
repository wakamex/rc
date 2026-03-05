"""Tests for synthetic benchmark generators."""

from __future__ import annotations

import pytest

from src.types import BenchmarkExample, Generator
from src.eval.benchmarks.memory import AssociativeRecall, PasskeyRetrieval, VariableTracking
from src.eval.benchmarks.computation import (
    DyckLanguage,
    ModularArithmetic,
    MultiDigitArithmetic,
    ProgramTrace,
)
from src.eval.benchmarks.emergent import (
    AlgorithmicTransfer,
    CompositionalGeneralization,
    LengthExtrapolation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect(gen, limit: int = 50) -> list[BenchmarkExample]:
    """Collect up to `limit` examples from a generator."""
    examples = []
    for ex in gen:
        examples.append(ex)
        if len(examples) >= limit:
            break
    return examples


def check_protocol(gen) -> None:
    """Assert the generator satisfies the Generator protocol."""
    assert isinstance(gen, Generator), f"{type(gen)} does not satisfy Generator protocol"
    assert isinstance(len(gen), int)
    assert len(gen) >= 0


def check_examples(examples: list[BenchmarkExample], task: str) -> None:
    """Common checks for a list of benchmark examples."""
    assert len(examples) > 0, "No examples generated"
    for ex in examples:
        assert isinstance(ex, BenchmarkExample)
        assert isinstance(ex.input, str) and ex.input
        assert isinstance(ex.target, str) and ex.target
        assert isinstance(ex.metadata, dict)
        assert ex.metadata.get("task") == task


# ---------------------------------------------------------------------------
# Memory benchmarks
# ---------------------------------------------------------------------------


class TestPasskeyRetrieval:
    def test_protocol(self):
        gen = PasskeyRetrieval(n=1000)
        check_protocol(gen)

    def test_len(self):
        gen = PasskeyRetrieval(n=1000)
        assert len(gen) == 1000

    def test_generates_1k(self):
        gen = PasskeyRetrieval(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_example_structure(self):
        gen = PasskeyRetrieval(n=10)
        examples = collect(gen)
        check_examples(examples, "passkey_retrieval")

    def test_passkey_in_input(self):
        gen = PasskeyRetrieval(n=20)
        for ex in gen:
            passkey = ex.metadata["passkey"]
            assert passkey in ex.input
            assert ex.target == passkey

    def test_fixed_position(self):
        gen = PasskeyRetrieval(n=10, passkey_position=0.1)
        examples = collect(gen)
        for ex in examples:
            assert ex.metadata["position"] == 0.1

    def test_number_distractor(self):
        gen = PasskeyRetrieval(n=10, distractor_type="numbers")
        examples = collect(gen)
        check_examples(examples, "passkey_retrieval")

    def test_reproducible(self):
        ex1 = list(PasskeyRetrieval(n=5, seed=1))
        ex2 = list(PasskeyRetrieval(n=5, seed=1))
        assert [e.input for e in ex1] == [e.input for e in ex2]

    def test_different_seeds(self):
        ex1 = list(PasskeyRetrieval(n=5, seed=1))
        ex2 = list(PasskeyRetrieval(n=5, seed=2))
        assert [e.input for e in ex1] != [e.input for e in ex2]


class TestVariableTracking:
    def test_protocol(self):
        gen = VariableTracking(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = VariableTracking(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_example_structure(self):
        gen = VariableTracking(n=10)
        examples = collect(gen)
        check_examples(examples, "variable_tracking")

    def test_target_is_correct_value(self):
        gen = VariableTracking(n=20, seed=99)
        for ex in gen:
            query_var = ex.metadata["query_var"]
            final_state = ex.metadata["final_state"]
            assert ex.target == str(final_state[query_var])

    def test_configurable_params(self):
        gen = VariableTracking(n=10, num_variables=5, num_operations=10, distractor_length=5)
        examples = collect(gen)
        check_examples(examples, "variable_tracking")
        for ex in examples:
            assert ex.metadata["num_variables"] == 5
            assert ex.metadata["num_operations"] == 10


class TestAssociativeRecall:
    def test_protocol(self):
        gen = AssociativeRecall(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = AssociativeRecall(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_example_structure(self):
        gen = AssociativeRecall(n=10)
        examples = collect(gen)
        check_examples(examples, "associative_recall")

    def test_target_in_pairs(self):
        gen = AssociativeRecall(n=20, seed=7)
        for ex in gen:
            query_key = ex.metadata["query_key"]
            pairs = ex.metadata["pairs"]
            assert ex.target == pairs[query_key]

    def test_query_key_in_input(self):
        gen = AssociativeRecall(n=20)
        for ex in gen:
            assert ex.metadata["query_key"] in ex.input


# ---------------------------------------------------------------------------
# Computation benchmarks
# ---------------------------------------------------------------------------


class TestMultiDigitArithmetic:
    def test_protocol(self):
        gen = MultiDigitArithmetic(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = MultiDigitArithmetic(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_addition_correct(self):
        gen = MultiDigitArithmetic(n=20, digit_count=3, operation="addition", seed=0)
        for ex in gen:
            a, b = ex.metadata["a"], ex.metadata["b"]
            assert int(ex.target) == a + b

    def test_multiplication_correct(self):
        gen = MultiDigitArithmetic(n=20, digit_count=2, operation="multiplication", seed=0)
        for ex in gen:
            a, b = ex.metadata["a"], ex.metadata["b"]
            assert int(ex.target) == a * b

    def test_invalid_operation(self):
        with pytest.raises(ValueError):
            MultiDigitArithmetic(operation="division")

    def test_configurable_digits(self):
        gen = MultiDigitArithmetic(n=10, digit_count=6)
        examples = collect(gen)
        for ex in examples:
            a = ex.metadata["a"]
            assert 100000 <= a <= 999999


class TestModularArithmetic:
    def test_protocol(self):
        gen = ModularArithmetic(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = ModularArithmetic(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_results_correct(self):
        gen = ModularArithmetic(n=30, operand_size=50, modulus=13, seed=3)
        op_map = {"add": "+", "sub": "-", "mul": "*"}
        for ex in gen:
            a = ex.metadata["a"]
            b = ex.metadata["b"]
            op = ex.metadata["operation"]
            m = ex.metadata["modulus"]
            expected = eval(f"({a} {op_map[op]} {b}) % {m}")  # noqa: S307
            assert int(ex.target) == expected

    def test_target_in_range(self):
        gen = ModularArithmetic(n=30, modulus=17)
        for ex in gen:
            assert 0 <= int(ex.target) < 17


class TestDyckLanguage:
    def test_protocol(self):
        gen = DyckLanguage(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = DyckLanguage(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_example_structure(self):
        gen = DyckLanguage(n=10)
        examples = collect(gen)
        check_examples(examples, "dyck_language")

    def test_target_is_yes_or_no(self):
        gen = DyckLanguage(n=30)
        for ex in gen:
            assert ex.target in ("yes", "no")

    def test_valid_sequences_are_balanced(self):
        gen = DyckLanguage(n=30, seed=0)
        for ex in gen:
            seq = ex.metadata["sequence"]
            is_valid = ex.metadata["is_valid"]
            assert DyckLanguage._check_valid(seq) == is_valid

    def test_multiple_bracket_types(self):
        gen = DyckLanguage(n=20, bracket_types=3)
        examples = collect(gen)
        check_examples(examples, "dyck_language")


class TestProgramTrace:
    def test_protocol(self):
        gen = ProgramTrace(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = ProgramTrace(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_example_structure(self):
        gen = ProgramTrace(n=10)
        examples = collect(gen)
        check_examples(examples, "program_trace")

    def test_target_matches_state(self):
        gen = ProgramTrace(n=20, seed=5)
        for ex in gen:
            query_var = ex.metadata["query_var"]
            final_state = ex.metadata["final_state"]
            assert ex.target == str(final_state[query_var])


# ---------------------------------------------------------------------------
# Emergent benchmarks
# ---------------------------------------------------------------------------


class TestCompositionalGeneralization:
    def test_protocol(self):
        gen = CompositionalGeneralization(n=1000)
        check_protocol(gen)

    def test_generates_1k_train(self):
        gen = CompositionalGeneralization(n=1000, split="train")
        examples = list(gen)
        assert len(examples) == 1000

    def test_generates_1k_test(self):
        gen = CompositionalGeneralization(n=1000, split="test")
        examples = list(gen)
        assert len(examples) == 1000

    def test_example_structure(self):
        gen = CompositionalGeneralization(n=10)
        examples = collect(gen)
        check_examples(examples, "compositional_generalization")

    def test_train_uses_train_ops(self):
        held_out = ["*", "//"]
        gen = CompositionalGeneralization(n=30, held_out_ops=held_out, split="train")
        for ex in gen:
            for op in ex.metadata["operators"]:
                assert op not in held_out

    def test_test_uses_held_out_ops(self):
        held_out = ["*", "//"]
        gen = CompositionalGeneralization(n=30, held_out_ops=held_out, split="test")
        for ex in gen:
            ops = ex.metadata["operators"]
            assert all(op in held_out for op in ops)

    def test_result_correct(self):
        gen = CompositionalGeneralization(n=20, split="train", seed=1)
        for ex in gen:
            assert ex.target == str(ex.metadata["result"])


class TestLengthExtrapolation:
    def test_protocol(self):
        gen = LengthExtrapolation(n=1000)
        check_protocol(gen)

    def test_generates_1k(self):
        gen = LengthExtrapolation(n=1000)
        examples = list(gen)
        assert len(examples) == 1000

    def test_train_length(self):
        gen = LengthExtrapolation(n=20, train_length=5, test_multiplier=1.0)
        for ex in gen:
            assert len(ex.metadata["sequence"]) == 5

    def test_2x_length(self):
        gen = LengthExtrapolation(n=20, train_length=5, test_multiplier=2.0)
        for ex in gen:
            assert len(ex.metadata["sequence"]) == 10

    def test_10x_length(self):
        gen = LengthExtrapolation(n=20, train_length=5, test_multiplier=10.0)
        for ex in gen:
            assert len(ex.metadata["sequence"]) == 50

    def test_sum_correct(self):
        gen = LengthExtrapolation(n=20, seed=42)
        for ex in gen:
            assert int(ex.target) == sum(ex.metadata["sequence"])


class TestAlgorithmicTransfer:
    def test_protocol(self):
        gen = AlgorithmicTransfer(n=1000)
        check_protocol(gen)

    def test_generates_1k_train(self):
        gen = AlgorithmicTransfer(n=1000, split="train")
        examples = list(gen)
        assert len(examples) == 1000

    def test_generates_1k_test(self):
        gen = AlgorithmicTransfer(n=1000, split="test")
        examples = list(gen)
        assert len(examples) == 1000

    def test_sorting_train_algo(self):
        gen = AlgorithmicTransfer(n=10, family="sorting", split="train")
        for ex in gen:
            assert ex.metadata["algorithm"] == "bubble_sort"

    def test_sorting_test_algo(self):
        gen = AlgorithmicTransfer(n=10, family="sorting", split="test")
        for ex in gen:
            assert ex.metadata["algorithm"] == "selection_sort"

    def test_sorting_result_correct(self):
        gen = AlgorithmicTransfer(n=20, family="sorting", split="train")
        for ex in gen:
            seq = ex.metadata["sequence"]
            expected = f"[{', '.join(map(str, sorted(seq)))}]"
            assert ex.target == expected

    def test_search_train_algo(self):
        gen = AlgorithmicTransfer(n=10, family="search", split="train")
        for ex in gen:
            assert ex.metadata["algorithm"] == "linear_search"

    def test_search_test_algo(self):
        gen = AlgorithmicTransfer(n=10, family="search", split="test")
        for ex in gen:
            assert ex.metadata["algorithm"] == "binary_search"

    def test_search_result_correct(self):
        gen = AlgorithmicTransfer(n=20, family="search", split="train", seed=7)
        for ex in gen:
            seq = ex.metadata["sequence"]
            t = ex.metadata["search_target"]
            try:
                expected_idx = seq.index(t)
            except ValueError:
                expected_idx = -1
            assert int(ex.target) == expected_idx

    def test_invalid_family(self):
        with pytest.raises(ValueError):
            AlgorithmicTransfer(family="hashing")
