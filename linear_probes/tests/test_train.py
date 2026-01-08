"""Tests for probe training functions (POC and Full)."""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from POC training script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_probes_poc",
    Path(__file__).parent.parent / "05_POC_train_probes.py"
)
poc_train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc_train_module)

can_stratify = poc_train_module.can_stratify
get_majority_baseline = poc_train_module.get_majority_baseline
PROBE_INFO = poc_train_module.PROBE_INFO

# Import from Full training script
spec_full = importlib.util.spec_from_file_location(
    "train_probes_full",
    Path(__file__).parent.parent / "05_train_probes.py"
)
full_train_module = importlib.util.module_from_spec(spec_full)
spec_full.loader.exec_module(full_train_module)

ALL_PROBE_INFO = full_train_module.ALL_PROBE_INFO
train_multi_label_probe = full_train_module.train_multi_label_probe


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


# ============================================================
# Tests for Full Training Script (05_train_probes.py)
# ============================================================

class TestAllProbeInfo:
    """Tests for ALL_PROBE_INFO dictionary."""

    def test_contains_all_probes(self):
        """Check that all 14 probes are defined."""
        expected_probes = [
            'A1', 'A2',
            'B1', 'B2',
            'C1', 'C3_add', 'C3_sub', 'C3_mult', 'C3_div', 'C4',
            'D1', 'D2', 'D3', 'D6',
        ]
        for probe in expected_probes:
            assert probe in ALL_PROBE_INFO, f"Missing probe: {probe}"

    def test_probe_count(self):
        """Should have 14 probes defined."""
        assert len(ALL_PROBE_INFO) == 14

    def test_all_have_name(self):
        """Each probe should have a name."""
        for probe, info in ALL_PROBE_INFO.items():
            assert 'name' in info, f"{probe} missing 'name'"

    def test_all_have_random_baseline(self):
        """Each probe should have a random baseline."""
        for probe, info in ALL_PROBE_INFO.items():
            assert 'random_baseline' in info, f"{probe} missing 'random_baseline'"
            assert 0 < info['random_baseline'] <= 1, f"{probe} invalid baseline"

    def test_a1_is_multi_label(self):
        """A1 should be marked as multi-label."""
        assert ALL_PROBE_INFO['A1']['type'] == 'multi_label'
        assert ALL_PROBE_INFO['A1']['n_labels'] == 4
        assert 'label_names' in ALL_PROBE_INFO['A1']

    def test_classification_probes_have_n_classes(self):
        """Classification probes should have n_classes."""
        classification_probes = [
            'A2', 'B1', 'B2', 'C1', 'C3_add', 'C3_sub', 'C3_mult', 'C3_div',
            'C4', 'D1', 'D2', 'D3', 'D6'
        ]
        for probe in classification_probes:
            info = ALL_PROBE_INFO[probe]
            assert info.get('type', 'classification') == 'classification'
            assert 'n_classes' in info, f"{probe} missing 'n_classes'"
            assert info['n_classes'] >= 2


class TestPOCProbeInfo:
    """Tests for POC PROBE_INFO (backwards compatibility)."""

    def test_contains_poc_probes(self):
        """Check that POC probes are defined."""
        poc_probes = ['B1', 'B2', 'C1', 'D1', 'D2']
        for probe in poc_probes:
            assert probe in PROBE_INFO

    def test_poc_probe_count(self):
        """POC should have 5 probes."""
        assert len(PROBE_INFO) == 5


class TestTrainMultiLabelProbe:
    """Tests for train_multi_label_probe function."""

    def test_returns_result_dict(self):
        """Should return a result dict with expected keys."""
        np.random.seed(42)
        n_samples = 100
        n_features = 64
        n_labels = 4

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, size=(n_samples, n_labels))

        result = train_multi_label_probe(X, y)

        assert result is not None
        assert 'train_acc' in result
        assert 'test_acc' in result
        assert 'mean_f1' in result
        assert 'label_results' in result

    def test_handles_multi_labels(self):
        """Should train a classifier for each label."""
        np.random.seed(42)
        n_samples = 100
        n_features = 32
        n_labels = 4

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, size=(n_samples, n_labels))

        result = train_multi_label_probe(X, y)

        assert len(result['label_results']) == n_labels

    def test_reasonable_accuracy(self):
        """Should achieve reasonable accuracy on simple data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 32

        # Create data where label correlates with feature
        X = np.random.randn(n_samples, n_features)
        y = np.zeros((n_samples, 2))
        y[:, 0] = (X[:, 0] > 0).astype(int)  # Label 0 = feature 0 > 0
        y[:, 1] = (X[:, 1] > 0).astype(int)  # Label 1 = feature 1 > 0

        result = train_multi_label_probe(X, y)

        # Should do better than random (0.5)
        assert result['test_acc'] > 0.6
