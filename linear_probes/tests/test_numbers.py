"""Tests for utils/numbers.py"""
import pytest
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.numbers import parse_number, extract_final_answer


class TestParseNumber:
    """Tests for parse_number function."""

    def test_integer(self):
        assert parse_number("42") == 42.0

    def test_with_comma(self):
        assert parse_number("1,000") == 1000.0

    def test_large_with_commas(self):
        assert parse_number("1,234,567") == 1234567.0

    def test_negative(self):
        assert parse_number("-5") == -5.0

    def test_decimal(self):
        assert parse_number("3.14") == 3.14

    def test_negative_decimal(self):
        assert parse_number("-5.5") == -5.5

    def test_with_plus_sign(self):
        assert parse_number("+10") == 10.0


class TestExtractFinalAnswer:
    """Tests for extract_final_answer function."""

    def test_gsm8k_format(self):
        """Test #### pattern (GSM8K format)."""
        text = "So the total is #### 42"
        assert extract_final_answer(text) == 42.0

    def test_gsm8k_no_space(self):
        """Test ####N pattern without space."""
        text = "The answer is ####42"
        assert extract_final_answer(text) == 42.0

    def test_natural_language(self):
        """Test 'The answer is X' pattern."""
        text = "Therefore, the answer is 100."
        assert extract_final_answer(text) == 100.0

    def test_natural_language_colon(self):
        """Test 'The answer is: X' pattern."""
        text = "The answer is: 50"
        assert extract_final_answer(text) == 50.0

    def test_fallback_last_number(self):
        """Test fallback to last number in text."""
        text = "First we get 10, then 20, finally 30."
        assert extract_final_answer(text) == 30.0

    def test_no_numbers(self):
        """Test text with no numbers."""
        text = "No numbers here at all."
        assert extract_final_answer(text) is None

    def test_empty_string(self):
        """Test empty string."""
        assert extract_final_answer("") is None

    def test_with_comma_number(self):
        """Test number with comma separator."""
        text = "The total is 1,500 dollars."
        assert extract_final_answer(text) == 1500.0

    def test_decimal_answer(self):
        """Test decimal number in answer."""
        text = "The answer is 3.14159"
        assert extract_final_answer(text) == 3.14159

    def test_gsm8k_takes_priority(self):
        """Test that #### pattern takes priority over 'answer is' pattern."""
        text = "The answer is 100. #### 42"
        assert extract_final_answer(text) == 42.0
