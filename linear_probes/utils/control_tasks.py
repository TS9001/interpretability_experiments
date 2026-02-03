"""
Hewitt-style Control Tasks for Probe Evaluation.

Implements control tasks as described in:
"Designing and Interpreting Probes with Control Tasks" (Hewitt & Liang, EMNLP 2019)

Control tasks assign random but TYPE-CONSISTENT labels to test whether probes
learn from representations or simply memorize input patterns.

Key insight: If a probe achieves high accuracy on both the linguistic task AND
the control task, the probe has too much capacity - high accuracy doesn't mean
the information is encoded in the representations.

Selectivity = linguistic_accuracy - control_accuracy
- High selectivity → information genuinely encoded in representations
- Low selectivity → probe may just be memorizing
"""

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA


def compute_sample_types(
    X: np.ndarray,
    n_bins: int = 100,
    pca_components: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute type identifiers for samples based on their hidden state features.

    Uses PCA projection + quantization to define discrete "types". All samples
    of the same type will receive the same control label.

    Args:
        X: Hidden states of shape (n_samples, hidden_dim)
        n_bins: Number of bins for quantization (more bins = finer types)
        pca_components: Number of PCA components to use for type computation
        seed: Random seed for reproducibility

    Returns:
        Array of type identifiers (integers) for each sample
    """
    n_samples, hidden_dim = X.shape

    # Use fewer components if hidden_dim is small
    n_components = min(pca_components, hidden_dim, n_samples)

    # Project to lower dimensions
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)

    # Quantize each component into bins
    # Use first component for primary type assignment
    # (captures most variance, so similar inputs get same type)
    primary_component = X_pca[:, 0]

    # Compute bin edges based on percentiles for even distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(primary_component, percentiles)

    # Assign types based on which bin each sample falls into
    types = np.digitize(primary_component, bin_edges[1:-1])

    return types


def create_control_labels(
    X: np.ndarray,
    n_classes: int,
    n_bins: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Create Hewitt-style control labels for classification tasks.

    Each unique "type" (based on hidden state features) gets a consistent
    random label. This tests whether the probe memorizes input→output mappings
    rather than extracting information from representations.

    Args:
        X: Hidden states of shape (n_samples, hidden_dim)
        n_classes: Number of classes for random label assignment
        n_bins: Number of bins for type computation (controls type granularity)
        seed: Random seed for reproducibility

    Returns:
        Control labels array of shape (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Compute types for all samples
    types = compute_sample_types(X, n_bins=n_bins, seed=seed)

    # Get unique types and assign random labels
    unique_types = np.unique(types)
    type_to_label = {t: rng.randint(0, n_classes) for t in unique_types}

    # Apply consistent mapping
    y_control = np.array([type_to_label[t] for t in types], dtype=np.int64)

    return y_control


def create_control_labels_multilabel(
    X: np.ndarray,
    n_labels: int,
    n_bins: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Create Hewitt-style control labels for multi-label classification.

    Each unique "type" gets a consistent random binary vector.

    Args:
        X: Hidden states of shape (n_samples, hidden_dim)
        n_labels: Number of labels (output dimensions)
        n_bins: Number of bins for type computation
        seed: Random seed for reproducibility

    Returns:
        Control labels array of shape (n_samples, n_labels)
    """
    rng = np.random.RandomState(seed)

    # Compute types for all samples
    types = compute_sample_types(X, n_bins=n_bins, seed=seed)

    # Get unique types and assign random label vectors
    unique_types = np.unique(types)
    type_to_labels = {
        t: rng.randint(0, 2, size=n_labels).astype(np.float32)
        for t in unique_types
    }

    # Apply consistent mapping
    y_control = np.array([type_to_labels[t] for t in types], dtype=np.float32)

    return y_control


def create_matched_control_labels(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    n_bins: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create control labels for train and test sets with consistent type mapping.

    Types are computed on train set and applied to test set. Test samples
    that don't match any train type get a random label.

    Args:
        X_train: Training hidden states (n_train, hidden_dim)
        X_test: Test hidden states (n_test, hidden_dim)
        n_classes: Number of classes
        n_bins: Number of bins for type computation
        seed: Random seed

    Returns:
        Tuple of (train_control_labels, test_control_labels)
    """
    rng = np.random.RandomState(seed)

    # Fit PCA on training data
    n_samples, hidden_dim = X_train.shape
    n_components = min(10, hidden_dim, n_samples)
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Compute bin edges from training data
    primary_train = X_train_pca[:, 0]
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(primary_train, percentiles)

    # Assign types
    train_types = np.digitize(primary_train, bin_edges[1:-1])
    test_types = np.digitize(X_test_pca[:, 0], bin_edges[1:-1])

    # Create type→label mapping from all observed types
    all_types = np.unique(np.concatenate([train_types, test_types]))
    type_to_label = {t: rng.randint(0, n_classes) for t in all_types}

    # Apply mapping
    y_train_control = np.array([type_to_label[t] for t in train_types], dtype=np.int64)
    y_test_control = np.array([type_to_label[t] for t in test_types], dtype=np.int64)

    return y_train_control, y_test_control


def create_matched_control_labels_multilabel(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_labels: int,
    n_bins: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create multi-label control labels for train and test with consistent mapping.

    Args:
        X_train: Training hidden states (n_train, hidden_dim)
        X_test: Test hidden states (n_test, hidden_dim)
        n_labels: Number of labels
        n_bins: Number of bins for type computation
        seed: Random seed

    Returns:
        Tuple of (train_control_labels, test_control_labels)
    """
    rng = np.random.RandomState(seed)

    # Fit PCA on training data
    n_samples, hidden_dim = X_train.shape
    n_components = min(10, hidden_dim, n_samples)
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Compute bin edges from training data
    primary_train = X_train_pca[:, 0]
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(primary_train, percentiles)

    # Assign types
    train_types = np.digitize(primary_train, bin_edges[1:-1])
    test_types = np.digitize(X_test_pca[:, 0], bin_edges[1:-1])

    # Create type→label mapping
    all_types = np.unique(np.concatenate([train_types, test_types]))
    type_to_labels = {
        t: rng.randint(0, 2, size=n_labels).astype(np.float32)
        for t in all_types
    }

    # Apply mapping
    y_train_control = np.array([type_to_labels[t] for t in train_types], dtype=np.float32)
    y_test_control = np.array([type_to_labels[t] for t in test_types], dtype=np.float32)

    return y_train_control, y_test_control


def compute_selectivity(
    linguistic_acc: float,
    control_acc: float,
    random_baseline: Optional[float] = None,
) -> dict:
    """
    Compute selectivity metric and related statistics.

    Selectivity = linguistic_accuracy - control_accuracy

    A selective probe achieves high accuracy on the real task but low accuracy
    on the control task, indicating it extracts genuine structure from representations
    rather than just memorizing patterns.

    Args:
        linguistic_acc: Accuracy on the real linguistic/mathematical task
        control_acc: Accuracy on the Hewitt control task
        random_baseline: Random chance baseline (optional, for context)

    Returns:
        Dict with selectivity metrics
    """
    selectivity = linguistic_acc - control_acc

    result = {
        'selectivity': selectivity,
        'linguistic_acc': linguistic_acc,
        'control_acc': control_acc,
    }

    if random_baseline is not None:
        result['random_baseline'] = random_baseline
        result['linguistic_vs_random'] = linguistic_acc - random_baseline
        result['control_vs_random'] = control_acc - random_baseline

        # Interpretation
        if selectivity > 0.1:
            result['interpretation'] = 'HIGH_SELECTIVITY'
        elif selectivity > 0.05:
            result['interpretation'] = 'MODERATE_SELECTIVITY'
        elif selectivity > 0:
            result['interpretation'] = 'LOW_SELECTIVITY'
        else:
            result['interpretation'] = 'NOT_SELECTIVE'

    return result
