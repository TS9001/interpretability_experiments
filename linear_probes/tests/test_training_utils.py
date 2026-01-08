"""Tests for utils/training.py functions."""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.training import (
    can_stratify,
    get_majority_baseline,
    train_probe_single_layer,
    train_multi_label_probe,
    train_probe_cv,
)


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
        assert abs(baseline - 1/3) < 0.01


class TestTrainProbeSingleLayer:
    """Tests for train_probe_single_layer function."""

    def test_returns_result_dict(self):
        """Should return a result dict with expected keys."""
        np.random.seed(42)
        n_samples = 100
        n_features = 32

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)

        result = train_probe_single_layer(X, y)

        assert result is not None
        assert 'train_acc' in result
        assert 'test_acc' in result
        assert 'n_train' in result
        assert 'n_test' in result
        assert 'y_pred' in result
        assert 'y_test' in result

    def test_reasonable_accuracy_simple_data(self):
        """Should achieve reasonable accuracy on simple linearly separable data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create linearly separable data
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)  # Class based on first feature

        result = train_probe_single_layer(X, y)

        # Should do much better than random (0.5)
        assert result['test_acc'] > 0.7

    def test_handles_multi_class(self):
        """Should handle multi-class classification."""
        np.random.seed(42)
        n_samples = 150
        n_features = 32

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, size=n_samples)

        result = train_probe_single_layer(X, y)

        assert result is not None
        assert result['n_train'] + result['n_test'] == n_samples


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


class TestTrainProbeCv:
    """Tests for train_probe_cv function."""

    def test_returns_result_dict(self):
        """Should return a result dict with expected keys."""
        np.random.seed(42)
        n_samples = 100
        n_features = 32

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)

        result = train_probe_cv(X, y, cv=3)

        assert result is not None
        assert 'mean_acc' in result
        assert 'std_acc' in result
        assert 'cv_scores' in result

    def test_cv_scores_length(self):
        """Should return cv_scores with length equal to cv folds."""
        np.random.seed(42)
        X = np.random.randn(100, 32)
        y = np.random.randint(0, 2, size=100)

        for cv in [3, 5]:
            result = train_probe_cv(X, y, cv=cv)
            assert len(result['cv_scores']) == cv

    def test_mean_within_bounds(self):
        """Mean accuracy should be between 0 and 1."""
        np.random.seed(42)
        X = np.random.randn(100, 32)
        y = np.random.randint(0, 2, size=100)

        result = train_probe_cv(X, y, cv=5)

        assert 0 <= result['mean_acc'] <= 1
        assert result['std_acc'] >= 0
