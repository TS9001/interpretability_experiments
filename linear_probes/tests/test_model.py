"""Tests for utils/model.py"""
import pytest
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model import (
    get_model_short_name, get_device, get_device_info, clear_memory,
)


class TestGetModelShortName:
    """Tests for get_model_short_name function."""

    def test_with_org(self):
        result = get_model_short_name("Qwen/Qwen2.5-Math-1.5B")
        assert result == "Qwen2.5-Math-1.5B"

    def test_without_org(self):
        result = get_model_short_name("gpt2")
        assert result == "gpt2"

    def test_nested_path(self):
        result = get_model_short_name("org/sub/model")
        assert result == "model"


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_valid_device(self):
        import torch
        device = get_device()
        assert device.type in ('cuda', 'mps', 'cpu')
        # Verify it's a valid torch device
        assert isinstance(device, torch.device)


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_has_required_keys(self):
        info = get_device_info()
        assert 'device' in info
        assert 'type' in info
        assert info['type'] in ('cuda', 'mps', 'cpu')

    def test_cuda_info(self):
        import torch
        info = get_device_info()
        if info['type'] == 'cuda':
            assert 'name' in info
            assert 'memory_gb' in info
            assert 'compute_capability' in info
            assert 'is_h100' in info

    def test_mps_info(self):
        info = get_device_info()
        if info['type'] == 'mps':
            assert info['name'] == 'Apple Silicon'


class TestClearMemory:
    """Tests for clear_memory function."""

    def test_no_error_on_cpu(self):
        import torch
        device = torch.device('cpu')
        # Should not raise
        clear_memory(device)

    def test_no_error_on_current_device(self):
        device = get_device()
        # Should not raise regardless of device type
        clear_memory(device)
