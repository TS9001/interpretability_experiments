"""Tests for utils/tokenization.py"""
import pytest
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tokenization import build_char_to_token_map, find_token_positions_for_span


class TestBuildCharToTokenMap:
    """Tests for build_char_to_token_map function."""

    def test_simple_tokens(self):
        tokens = ['hello', ' ', 'world']
        text = 'hello world'
        mapping = build_char_to_token_map(tokens, text)

        # 'h' at char 0 should map to token 0
        assert mapping[0] == 0
        # 'o' at char 4 should map to token 0
        assert mapping[4] == 0
        # space at char 5 should map to token 1
        assert mapping[5] == 1
        # 'w' at char 6 should map to token 2
        assert mapping[6] == 2

    def test_with_gpt_prefix(self):
        """Test handling of Ġ (GPT-style space prefix)."""
        tokens = ['hello', 'Ġworld']
        text = 'hello world'
        mapping = build_char_to_token_map(tokens, text)

        # Characters in 'hello' map to token 0
        assert mapping[0] == 0
        # Characters in ' world' map to token 1
        assert mapping[6] == 1

    def test_numbers(self):
        tokens = ['5', '+', '3', '=', '8']
        text = '5+3=8'
        mapping = build_char_to_token_map(tokens, text)

        assert mapping[0] == 0  # '5'
        assert mapping[1] == 1  # '+'
        assert mapping[2] == 2  # '3'
        assert mapping[3] == 3  # '='
        assert mapping[4] == 4  # '8'


class TestFindTokenPositionsForSpan:
    """Tests for find_token_positions_for_span function."""

    def test_single_token_span(self):
        char_to_token = {0: 0, 1: 0, 2: 1, 3: 1}
        positions = find_token_positions_for_span(0, 2, char_to_token)
        assert positions == [0]

    def test_multi_token_span(self):
        char_to_token = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
        positions = find_token_positions_for_span(0, 5, char_to_token)
        assert positions == [0, 1, 2]

    def test_not_found(self):
        char_to_token = {0: 0, 1: 0}
        positions = find_token_positions_for_span(10, 15, char_to_token)
        assert positions == [-1]

    def test_partial_overlap(self):
        char_to_token = {0: 0, 1: 0, 2: 1, 3: 1}
        positions = find_token_positions_for_span(1, 3, char_to_token)
        assert positions == [0, 1]

    def test_deduplication(self):
        """Positions should be unique even if span covers same token multiple times."""
        char_to_token = {0: 0, 1: 0, 2: 0, 3: 0}
        positions = find_token_positions_for_span(0, 4, char_to_token)
        assert positions == [0]
