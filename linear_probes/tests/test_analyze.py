"""Tests for 02_analyze_responses.py functions."""
import pytest
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the script using importlib since filename starts with number
import importlib.util
spec = importlib.util.spec_from_file_location(
    "analyze_responses",
    Path(__file__).parent.parent / "02_analyze_responses.py"
)
analyze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_module)

extract_operations = analyze_module.extract_operations
normalize_token = analyze_module.normalize_token
find_number_in_tokens = analyze_module.find_number_in_tokens
find_operator_in_tokens = analyze_module.find_operator_in_tokens


class TestExtractOperations:
    """Tests for extract_operations function."""

    def test_add_spaced(self):
        ops = extract_operations("5 + 3 = 8")
        assert len(ops) == 1
        assert ops[0]['operand1'] == 5.0
        assert ops[0]['operator'] == 'add'
        assert ops[0]['operand2'] == 3.0
        assert ops[0]['result'] == 8.0
        assert ops[0]['is_correct'] is True

    def test_add_compact(self):
        ops = extract_operations("5+3=8")
        assert len(ops) == 1
        assert ops[0]['operator'] == 'add'

    def test_mult_x(self):
        """Test multiplication with 'x' notation."""
        ops = extract_operations("5 x 3 = 15")
        assert len(ops) == 1
        assert ops[0]['operator'] == 'mult'
        assert ops[0]['result'] == 15.0

    def test_mult_asterisk(self):
        ops = extract_operations("5 * 3 = 15")
        assert len(ops) == 1
        assert ops[0]['operator'] == 'mult'

    def test_sub(self):
        ops = extract_operations("10 - 3 = 7")
        assert len(ops) == 1
        assert ops[0]['operator'] == 'sub'

    def test_div(self):
        ops = extract_operations("20 / 4 = 5")
        assert len(ops) == 1
        assert ops[0]['operator'] == 'div'

    def test_unicode_mult(self):
        """Test unicode multiplication sign ×."""
        ops = extract_operations("5 × 3 = 15")
        assert len(ops) == 1
        assert ops[0]['operator'] == 'mult'

    def test_multiple_operations(self):
        text = "First 5 + 3 = 8, then 8 * 2 = 16"
        ops = extract_operations(text)
        assert len(ops) == 2
        assert ops[0]['result'] == 8.0
        assert ops[1]['result'] == 16.0

    def test_incorrect_operation(self):
        """Test operation with wrong result."""
        ops = extract_operations("5 + 3 = 9")  # Should be 8
        assert len(ops) == 1
        assert ops[0]['is_correct'] is False

    def test_no_operations(self):
        ops = extract_operations("No math here")
        assert len(ops) == 0

    def test_char_positions(self):
        """Test that char_start and char_end are captured."""
        text = "The result is 5 + 3 = 8"
        ops = extract_operations(text)
        assert len(ops) == 1
        assert 'char_start' in ops[0]
        assert 'char_end' in ops[0]


class TestNormalizeToken:
    """Tests for normalize_token function."""

    def test_gpt_prefix(self):
        assert normalize_token("Ġword") == "word"

    def test_newline_prefix(self):
        assert normalize_token("Ċline") == "line"

    def test_no_prefix(self):
        assert normalize_token("word") == "word"

    def test_multiple_chars(self):
        assert normalize_token("Ġhello") == "hello"


class TestFindNumberInTokens:
    """Tests for find_number_in_tokens function."""

    def test_single_token(self):
        tokens = ['The', 'answer', 'is', '42']
        start, end = find_number_in_tokens(tokens, 42.0)
        assert start == 3
        assert end == 3

    def test_with_start_pos(self):
        tokens = ['5', 'plus', '5', 'equals', '10']
        start, end = find_number_in_tokens(tokens, 5.0, start_pos=2)
        assert start == 2
        assert end == 2

    def test_not_found(self):
        tokens = ['no', 'numbers', 'here']
        start, end = find_number_in_tokens(tokens, 42.0)
        assert start == -1
        assert end == -1

    def test_integer_vs_float(self):
        """Integer 42 should match whether searching for 42 or 42.0."""
        tokens = ['value', 'is', '42']
        start, end = find_number_in_tokens(tokens, 42.0)
        assert start == 2
        assert end == 2

    def test_embedded_in_token(self):
        """Number embedded in larger token."""
        tokens = ['value=42']
        start, end = find_number_in_tokens(tokens, 42.0)
        assert start == 0
        assert end == 0

    def test_multi_token_number(self):
        """Multi-token number should return span (start, end)."""
        tokens = ['total', 'is', '1', '6', '0', 'dollars']
        start, end = find_number_in_tokens(tokens, 160.0)
        assert start == 2
        assert end == 4


class TestFindOperatorInTokens:
    """Tests for find_operator_in_tokens function."""

    def test_find_plus(self):
        tokens = ['5', '+', '3']
        pos = find_operator_in_tokens(tokens, 'add')
        assert pos == 1

    def test_find_minus(self):
        tokens = ['5', '-', '3']
        pos = find_operator_in_tokens(tokens, 'sub')
        assert pos == 1

    def test_find_mult(self):
        tokens = ['5', '*', '3']
        pos = find_operator_in_tokens(tokens, 'mult')
        assert pos == 1

    def test_find_div(self):
        tokens = ['6', '/', '2']
        pos = find_operator_in_tokens(tokens, 'div')
        assert pos == 1

    def test_not_found(self):
        tokens = ['5', '3', '8']
        pos = find_operator_in_tokens(tokens, 'add')
        assert pos == -1

    def test_with_start_pos(self):
        tokens = ['+', '5', '+', '3']
        pos = find_operator_in_tokens(tokens, 'add', start_pos=2)
        assert pos == 2
