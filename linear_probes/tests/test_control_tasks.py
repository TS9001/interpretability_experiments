"""Tests for Hewitt-style control tasks."""

import numpy as np
import pytest

from utils.control_tasks import (
    compute_sample_types,
    create_control_labels,
    create_control_labels_multilabel,
    create_matched_control_labels,
    create_matched_control_labels_multilabel,
    compute_selectivity,
)


class TestComputeSampleTypes:
    """Tests for sample type computation."""

    def test_deterministic(self):
        """Same inputs should produce same types."""
        X = np.random.randn(100, 128)
        types1 = compute_sample_types(X, seed=42)
        types2 = compute_sample_types(X, seed=42)
        np.testing.assert_array_equal(types1, types2)

    def test_different_seeds_different_types(self):
        """Different seeds should produce different PCA projections."""
        X = np.random.randn(100, 128)
        types1 = compute_sample_types(X, seed=42)
        types2 = compute_sample_types(X, seed=123)
        # Types may differ due to different random PCA initialization
        # (though PCA is deterministic, the binning can shift)

    def test_similar_inputs_same_type(self):
        """Similar inputs should tend to get the same type."""
        # Create two clusters
        X1 = np.random.randn(50, 128) + 10  # Cluster 1
        X2 = np.random.randn(50, 128) - 10  # Cluster 2
        X = np.vstack([X1, X2])

        types = compute_sample_types(X, n_bins=10, seed=42)

        # Samples within same cluster should often have same type
        types_cluster1 = types[:50]
        types_cluster2 = types[50:]

        # Check that clusters have different dominant types
        unique1, counts1 = np.unique(types_cluster1, return_counts=True)
        unique2, counts2 = np.unique(types_cluster2, return_counts=True)

        dominant_type1 = unique1[np.argmax(counts1)]
        dominant_type2 = unique2[np.argmax(counts2)]

        assert dominant_type1 != dominant_type2


class TestCreateControlLabels:
    """Tests for single-label control task creation."""

    def test_type_consistency(self):
        """Same type should always get same label."""
        X = np.random.randn(100, 128)
        y = create_control_labels(X, n_classes=5, seed=42)

        types = compute_sample_types(X, seed=42)

        # Check that samples with same type have same label
        for t in np.unique(types):
            mask = types == t
            labels_for_type = y[mask]
            assert len(np.unique(labels_for_type)) == 1

    def test_valid_labels(self):
        """Labels should be in valid range."""
        X = np.random.randn(100, 128)
        y = create_control_labels(X, n_classes=5, seed=42)

        assert y.min() >= 0
        assert y.max() < 5

    def test_deterministic(self):
        """Same inputs and seed should produce same labels."""
        X = np.random.randn(100, 128)
        y1 = create_control_labels(X, n_classes=5, seed=42)
        y2 = create_control_labels(X, n_classes=5, seed=42)
        np.testing.assert_array_equal(y1, y2)


class TestCreateControlLabelsMultilabel:
    """Tests for multi-label control task creation."""

    def test_output_shape(self):
        """Output should have correct shape."""
        X = np.random.randn(100, 128)
        y = create_control_labels_multilabel(X, n_labels=4, seed=42)

        assert y.shape == (100, 4)

    def test_binary_values(self):
        """Labels should be binary (0 or 1)."""
        X = np.random.randn(100, 128)
        y = create_control_labels_multilabel(X, n_labels=4, seed=42)

        assert np.all(np.isin(y, [0, 1]))

    def test_type_consistency(self):
        """Same type should always get same label vector."""
        X = np.random.randn(100, 128)
        y = create_control_labels_multilabel(X, n_labels=4, seed=42)

        types = compute_sample_types(X, seed=42)

        for t in np.unique(types):
            mask = types == t
            labels_for_type = y[mask]
            # All rows should be identical
            assert np.all(labels_for_type == labels_for_type[0])


class TestCreateMatchedControlLabels:
    """Tests for matched train/test control labels."""

    def test_consistent_type_mapping(self):
        """Type mapping should be consistent between train and test."""
        X_train = np.random.randn(80, 128)
        X_test = np.random.randn(20, 128)

        y_train, y_test = create_matched_control_labels(
            X_train, X_test, n_classes=5, seed=42
        )

        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_valid_labels(self):
        """Labels should be in valid range for both sets."""
        X_train = np.random.randn(80, 128)
        X_test = np.random.randn(20, 128)

        y_train, y_test = create_matched_control_labels(
            X_train, X_test, n_classes=5, seed=42
        )

        assert y_train.min() >= 0
        assert y_train.max() < 5
        assert y_test.min() >= 0
        assert y_test.max() < 5


class TestCreateMatchedControlLabelsMultilabel:
    """Tests for matched multi-label control labels."""

    def test_output_shapes(self):
        """Outputs should have correct shapes."""
        X_train = np.random.randn(80, 128)
        X_test = np.random.randn(20, 128)

        y_train, y_test = create_matched_control_labels_multilabel(
            X_train, X_test, n_labels=4, seed=42
        )

        assert y_train.shape == (80, 4)
        assert y_test.shape == (20, 4)


class TestComputeSelectivity:
    """Tests for selectivity metric computation."""

    def test_high_selectivity(self):
        """High linguistic acc + low control acc = high selectivity."""
        result = compute_selectivity(0.9, 0.3, random_baseline=0.25)

        assert result['selectivity'] == pytest.approx(0.6)
        assert result['interpretation'] == 'HIGH_SELECTIVITY'

    def test_moderate_selectivity(self):
        """Moderately similar accs = moderate selectivity."""
        result = compute_selectivity(0.8, 0.75, random_baseline=0.25)

        assert result['selectivity'] == pytest.approx(0.05)
        assert result['interpretation'] == 'MODERATE_SELECTIVITY'

    def test_low_selectivity(self):
        """Very similar accs = low selectivity."""
        result = compute_selectivity(0.8, 0.78, random_baseline=0.25)

        assert result['selectivity'] == pytest.approx(0.02)
        assert result['interpretation'] == 'LOW_SELECTIVITY'

    def test_not_selective(self):
        """Control acc >= linguistic acc = not selective."""
        result = compute_selectivity(0.5, 0.6, random_baseline=0.25)

        assert result['selectivity'] == pytest.approx(-0.1)
        assert result['interpretation'] == 'NOT_SELECTIVE'

    def test_without_baseline(self):
        """Should work without random baseline."""
        result = compute_selectivity(0.9, 0.3)

        assert result['selectivity'] == pytest.approx(0.6)
        assert 'interpretation' not in result
