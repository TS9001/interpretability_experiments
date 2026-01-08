"""Tests for 05_train_probes.py functions."""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_probes",
    Path(__file__).parent.parent / "05_train_probes.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

can_stratify = train_module.can_stratify
get_majority_baseline = train_module.get_majority_baseline


class TestCanStratify:
    """Tests for can_stratify function."""

    def test_balanced_can_stratify(self):
        """Balanced classes with >= 2 samples each can stratify."""
        y = np.array([0, 0, 1, 1])
        assert can_stratify(y) == True

    def test_imbalanced_cannot_stratify(self):
        """Class with only 1 sample cannot stratify."""
        y = np.array([0, 0, 0, 1])
        assert can_stratify(y) == False

    def test_single_class_cannot_stratify(self):
        """Single class cannot stratify."""
        y = np.array([0, 0, 0, 0])
        assert can_stratify(y) == False

    def test_multi_class_can_stratify(self):
        """Multiple classes with >= 2 samples each."""
        y = np.array([0, 0, 1, 1, 2, 2])
        assert can_stratify(y) == True

    def test_multi_class_one_small(self):
        """Multiple classes but one has only 1 sample."""
        y = np.array([0, 0, 1, 1, 2])
        assert can_stratify(y) == False


class TestGetMajorityBaseline:
    """Tests for get_majority_baseline function."""

    def test_balanced(self):
        y = np.array([0, 0, 1, 1])
        baseline = get_majority_baseline(y)
        assert baseline == 0.5

    def test_imbalanced_75(self):
        y = np.array([0, 0, 0, 1])
        baseline = get_majority_baseline(y)
        assert baseline == 0.75

    def test_imbalanced_80(self):
        y = np.array([0, 0, 0, 0, 1])
        baseline = get_majority_baseline(y)
        assert baseline == 0.8

    def test_single_class(self):
        y = np.array([0, 0, 0, 0])
        baseline = get_majority_baseline(y)
        assert baseline == 1.0

    def test_multi_class_majority(self):
        """Multi-class with clear majority."""
        y = np.array([0, 0, 0, 1, 2])
        baseline = get_majority_baseline(y)
        assert baseline == 0.6  # 3/5

    def test_three_balanced_classes(self):
        """Three balanced classes."""
        y = np.array([0, 0, 1, 1, 2, 2])
        baseline = get_majority_baseline(y)
        # All classes have 2/6 = 0.333, so majority is ~0.333
        assert abs(baseline - 1/3) < 0.01
