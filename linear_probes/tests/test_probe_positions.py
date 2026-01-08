"""Tests for utils/probe_positions.py"""
import pytest
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.probe_positions import (
    get_magnitude_bin, get_coarse_bin, get_difficulty_bin, get_step_position,
    get_A1_labels, get_A2_label, check_operation_correctness,
    get_last_position, get_equals_positions,
)


class TestGetMagnitudeBin:
    """Tests for get_magnitude_bin function."""

    def test_negative(self):
        assert get_magnitude_bin(-5) == 0

    def test_small(self):
        """Values 0-10 should return bin 1."""
        assert get_magnitude_bin(0) == 1
        assert get_magnitude_bin(5) == 1
        assert get_magnitude_bin(9.9) == 1

    def test_medium(self):
        """Values 10-100 should return bin 2."""
        assert get_magnitude_bin(10) == 2
        assert get_magnitude_bin(50) == 2
        assert get_magnitude_bin(99) == 2

    def test_large(self):
        """Values 100-1000 should return bin 3."""
        assert get_magnitude_bin(100) == 3
        assert get_magnitude_bin(500) == 3
        assert get_magnitude_bin(999) == 3

    def test_xlarge(self):
        """Values 1000-10000 should return bin 4."""
        assert get_magnitude_bin(1000) == 4
        assert get_magnitude_bin(5000) == 4
        assert get_magnitude_bin(9999) == 4

    def test_huge(self):
        """Values 10000+ should return bin 5."""
        assert get_magnitude_bin(10000) == 5
        assert get_magnitude_bin(50000) == 5
        assert get_magnitude_bin(1000000) == 5


class TestGetCoarseBin:
    """Tests for get_coarse_bin function."""

    def test_small(self):
        assert get_coarse_bin(5) == 0
        assert get_coarse_bin(-5) == 0  # Uses abs()

    def test_medium(self):
        assert get_coarse_bin(50) == 1

    def test_large(self):
        assert get_coarse_bin(500) == 2

    def test_xlarge(self):
        assert get_coarse_bin(5000) == 3


class TestGetDifficultyBin:
    """Tests for get_difficulty_bin function."""

    def test_one_operation(self):
        assert get_difficulty_bin(1) == 0

    def test_two_operations(self):
        assert get_difficulty_bin(2) == 1

    def test_three_operations(self):
        assert get_difficulty_bin(3) == 2

    def test_four_operations(self):
        assert get_difficulty_bin(4) == 3

    def test_five_plus_capped(self):
        """5+ operations should all return 4 (capped)."""
        assert get_difficulty_bin(5) == 4
        assert get_difficulty_bin(10) == 4
        assert get_difficulty_bin(100) == 4


class TestGetStepPosition:
    """Tests for get_step_position function."""

    def test_first(self):
        assert get_step_position(0, 3) == 0

    def test_middle(self):
        assert get_step_position(1, 3) == 1

    def test_last(self):
        assert get_step_position(2, 3) == 2

    def test_single_step_is_first(self):
        """Single step matches 'first' condition (idx==0), returns 0."""
        assert get_step_position(0, 1) == 0

    def test_two_steps(self):
        assert get_step_position(0, 2) == 0  # first
        assert get_step_position(1, 2) == 2  # last


class TestGetA1Labels:
    """Tests for get_A1_labels function."""

    def test_all_operations(self):
        ops_by_type = {'add': 1, 'sub': 1, 'mult': 1, 'div': 1}
        labels = get_A1_labels(ops_by_type)
        assert labels == [1, 1, 1, 1]

    def test_add_only(self):
        ops_by_type = {'add': 2, 'sub': 0, 'mult': 0, 'div': 0}
        labels = get_A1_labels(ops_by_type)
        assert labels == [1, 0, 0, 0]

    def test_mult_div(self):
        ops_by_type = {'add': 0, 'sub': 0, 'mult': 1, 'div': 1}
        labels = get_A1_labels(ops_by_type)
        assert labels == [0, 0, 1, 1]

    def test_empty(self):
        ops_by_type = {}
        labels = get_A1_labels(ops_by_type)
        assert labels == [0, 0, 0, 0]


class TestGetA2Label:
    """Tests for get_A2_label function."""

    def test_maps_to_difficulty_bin(self):
        assert get_A2_label(1) == 0
        assert get_A2_label(3) == 2
        assert get_A2_label(5) == 4


class TestCheckOperationCorrectness:
    """Tests for check_operation_correctness function."""

    def test_correct_add(self):
        op = {'operand1': 5, 'operand2': 3, 'result': 8, 'operator': 'add'}
        assert check_operation_correctness(op) is True

    def test_incorrect_add(self):
        op = {'operand1': 5, 'operand2': 3, 'result': 9, 'operator': 'add'}
        assert check_operation_correctness(op) is False

    def test_correct_sub(self):
        op = {'operand1': 10, 'operand2': 3, 'result': 7, 'operator': 'sub'}
        assert check_operation_correctness(op) is True

    def test_correct_mult(self):
        op = {'operand1': 4, 'operand2': 5, 'result': 20, 'operator': 'mult'}
        assert check_operation_correctness(op) is True

    def test_correct_div(self):
        op = {'operand1': 20, 'operand2': 4, 'result': 5, 'operator': 'div'}
        assert check_operation_correctness(op) is True

    def test_div_by_zero(self):
        op = {'operand1': 10, 'operand2': 0, 'result': 0, 'operator': 'div'}
        assert check_operation_correctness(op) is False

    def test_float_tolerance(self):
        """Small float differences should be tolerated."""
        op = {'operand1': 10, 'operand2': 3, 'result': 3.333, 'operator': 'div'}
        assert check_operation_correctness(op) is True


class TestGetLastPosition:
    """Tests for get_last_position function."""

    def test_single_position(self):
        assert get_last_position([5]) == 5

    def test_multiple_positions(self):
        assert get_last_position([5, 6, 7]) == 7

    def test_empty_list(self):
        assert get_last_position([]) == -1

    def test_invalid_marker(self):
        assert get_last_position([-1]) == -1


class TestGetEqualsPositions:
    """Tests for get_equals_positions function."""

    def test_finds_equals(self):
        tokens = ['5', '+', '3', '=', '8']
        positions = get_equals_positions(tokens)
        assert positions == [3]

    def test_multiple_equals(self):
        tokens = ['5', '=', '5', 'and', '3', '=', '3']
        positions = get_equals_positions(tokens)
        assert positions == [1, 5]

    def test_no_equals(self):
        tokens = ['hello', 'world']
        positions = get_equals_positions(tokens)
        assert positions == []

    def test_with_prefix(self):
        """Test with tokenizer prefix like Ġ."""
        tokens = ['5', 'Ġ+', 'Ġ3', 'Ġ=', 'Ġ8']
        positions = get_equals_positions(tokens)
        assert positions == [3]
