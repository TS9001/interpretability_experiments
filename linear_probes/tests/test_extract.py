"""Tests for 04_extract_hidden_states.py functions."""
import pytest
import sys
from pathlib import Path
import torch

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_hidden_states",
    Path(__file__).parent.parent / "04_extract_hidden_states.py"
)
extract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_module)

get_magnitude_bin = extract_module.get_magnitude_bin
get_next_op_label = extract_module.get_next_op_label
get_default_batch_size = extract_module.get_default_batch_size


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
