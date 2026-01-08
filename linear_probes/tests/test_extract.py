"""Tests for hidden state extraction functions (POC and Full)."""
import pytest
import sys
from pathlib import Path
import torch

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from POC script (for backwards compatibility tests)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_hidden_states_poc",
    Path(__file__).parent.parent / "04_POC_extract_hidden_states.py"
)
poc_extract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc_extract_module)

get_magnitude_bin = poc_extract_module.get_magnitude_bin
get_next_op_label = poc_extract_module.get_next_op_label
get_default_batch_size = poc_extract_module.get_default_batch_size

# Import from Full extraction script
spec_full = importlib.util.spec_from_file_location(
    "extract_hidden_states_full",
    Path(__file__).parent.parent / "04_extract_hidden_states.py"
)
full_extract_module = importlib.util.module_from_spec(spec_full)
spec_full.loader.exec_module(full_extract_module)

get_prev_op_label = full_extract_module.get_prev_op_label
find_equals_positions = full_extract_module.find_equals_positions
ALL_PROBES = full_extract_module.ALL_PROBES


class TestGetMagnitudeBin:
    """Tests for get_magnitude_bin function."""

    def test_negative(self):
        assert get_magnitude_bin(-5) == 0

    def test_small(self):
        assert get_magnitude_bin(5) == 1
        assert get_magnitude_bin(0) == 1

    def test_medium(self):
        assert get_magnitude_bin(50) == 2

    def test_large(self):
        assert get_magnitude_bin(500) == 3

    def test_xlarge(self):
        assert get_magnitude_bin(5000) == 4

    def test_huge(self):
        assert get_magnitude_bin(50000) == 5


class TestGetNextOpLabel:
    """Tests for get_next_op_label function."""

    def test_add(self):
        assert get_next_op_label('add') == 0

    def test_sub(self):
        assert get_next_op_label('sub') == 1

    def test_mult(self):
        assert get_next_op_label('mult') == 2

    def test_div(self):
        assert get_next_op_label('div') == 3

    def test_none_is_end(self):
        assert get_next_op_label(None) == 4

    def test_unknown_defaults_to_end(self):
        assert get_next_op_label('unknown') == 4


class TestGetDefaultBatchSize:
    """Tests for get_default_batch_size function."""

    def test_cpu(self):
        device = torch.device('cpu')
        batch_size = get_default_batch_size(device)
        assert batch_size == 2

    def test_mps(self):
        device = torch.device('mps')
        batch_size = get_default_batch_size(device)
        assert batch_size == 4

    def test_returns_positive(self):
        """Batch size should always be positive."""
        for device_type in ['cpu', 'mps']:
            device = torch.device(device_type)
            batch_size = get_default_batch_size(device)
            assert batch_size > 0


# ============================================================
# Tests for Full Extraction Script (04_extract_hidden_states.py)
# ============================================================

class TestGetPrevOpLabel:
    """Tests for get_prev_op_label function (D6 probe)."""

    def test_add(self):
        assert get_prev_op_label('add') == 0

    def test_sub(self):
        assert get_prev_op_label('sub') == 1

    def test_mult(self):
        assert get_prev_op_label('mult') == 2

    def test_div(self):
        assert get_prev_op_label('div') == 3

    def test_none_is_first(self):
        """None means this is the first operation."""
        assert get_prev_op_label(None) == 4

    def test_unknown_defaults_to_first(self):
        assert get_prev_op_label('unknown') == 4


class TestFindEqualsPositions:
    """Tests for find_equals_positions function (C4 probe)."""

    def test_finds_equals(self):
        tokens = ['5', '+', '3', '=', '8']
        positions = find_equals_positions(tokens, 0)
        assert positions == [3]

    def test_multiple_equals(self):
        tokens = ['5', '=', '5', 'and', '3', '=', '3']
        positions = find_equals_positions(tokens, 0)
        assert positions == [1, 5]

    def test_no_equals(self):
        tokens = ['hello', 'world']
        positions = find_equals_positions(tokens, 0)
        assert positions == []

    def test_respects_response_start(self):
        """Tokens before response_start should be ignored."""
        tokens = ['prompt', '=', 'ignored', 'response', '=', 'found']
        positions = find_equals_positions(tokens, 3)
        assert positions == [4]  # Only the equals after position 3

    def test_with_tokenizer_prefix(self):
        """Test with tokenizer prefix like Ġ."""
        tokens = ['5', 'Ġ+', 'Ġ3', 'Ġ=', 'Ġ8']
        positions = find_equals_positions(tokens, 0)
        assert positions == [3]

    def test_with_underscore_prefix(self):
        """Test with tokenizer prefix like ▁."""
        tokens = ['5', '▁+', '▁3', '▁=', '▁8']
        positions = find_equals_positions(tokens, 0)
        assert positions == [3]


class TestAllProbes:
    """Tests for probe constants."""

    def test_all_probes_defined(self):
        """Check that all expected probes are in ALL_PROBES."""
        expected = [
            'A1', 'A2',
            'B1', 'B2',
            'C1', 'C3_add', 'C3_sub', 'C3_mult', 'C3_div', 'C4',
            'D1', 'D2', 'D3', 'D6',
        ]
        assert ALL_PROBES == expected

    def test_probe_count(self):
        """Should have 14 probes defined."""
        assert len(ALL_PROBES) == 14

    def test_no_duplicates(self):
        """No duplicate probes."""
        assert len(ALL_PROBES) == len(set(ALL_PROBES))
