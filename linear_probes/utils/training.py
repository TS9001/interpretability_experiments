"""Shared training utilities for linear probes."""
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from utils.logging import log

# Try to import cuML for GPU acceleration, fallback to sklearn
try:
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    USING_CUML = True
    log.info("Using cuML (GPU-accelerated) for logistic regression")
except ImportError:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    USING_CUML = False
    log.warning("cuML not available, falling back to sklearn (CPU)")


def can_stratify(y: np.ndarray) -> bool:
    """Check if stratified split is possible (all classes have >= 2 samples)."""
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def get_majority_baseline(y: np.ndarray) -> float:
    """Calculate majority class baseline (accuracy if always predicting most common class)."""
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / len(y)


def _create_logistic_regression(random_state: int = 42):
    """Create LogisticRegression with appropriate params for cuML or sklearn."""
    if USING_CUML:
        return LogisticRegression(
            max_iter=5000,
            solver='qn',  # Quasi-Newton, efficient on GPU
            verbose=0,
        )
    else:
        return LogisticRegression(
            max_iter=5000,
            solver='lbfgs',
            n_jobs=1,
            random_state=random_state,
        )


def train_probe_single_layer(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Optional[dict]:
    """Train a single classification probe and return metrics."""
    # Split data - use stratify only if all classes have >= 2 samples
    stratify = y if can_stratify(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Scale features (critical for convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = _create_logistic_regression(random_state)

    try:
        clf.fit(X_train_scaled, y_train)

        # Get predictions
        y_train_pred = clf.predict(X_train_scaled)
        y_pred = clf.predict(X_test_scaled)

        # Convert cuML arrays to numpy if needed
        if USING_CUML:
            y_train_pred = np.asarray(y_train_pred)
            y_pred = np.asarray(y_pred)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'y_pred': y_pred,
            'y_test': y_test,
            'model': clf,
            'scaler': scaler,
        }
    except Exception as e:
        log.warning(f"Training failed: {e}")
        return None


def train_multi_label_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Optional[dict]:
    """Train a multi-label probe (one classifier per label) and return metrics."""
    n_samples, n_labels = y.shape

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features once for all labels
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_accs = []
    test_accs = []
    f1_scores_list = []
    label_results = []
    models = []

    for label_idx in range(n_labels):
        y_train_label = y_train[:, label_idx]
        y_test_label = y_test[:, label_idx]

        clf = _create_logistic_regression(random_state)

        try:
            clf.fit(X_train_scaled, y_train_label)

            y_train_pred = clf.predict(X_train_scaled)
            y_pred = clf.predict(X_test_scaled)

            # Convert cuML arrays to numpy if needed
            if USING_CUML:
                y_train_pred = np.asarray(y_train_pred)
                y_pred = np.asarray(y_pred)

            train_acc = accuracy_score(y_train_label, y_train_pred)
            test_acc = accuracy_score(y_test_label, y_pred)
            f1 = f1_score(y_test_label, y_pred, average='binary', zero_division=0)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            f1_scores_list.append(f1)
            label_results.append({
                'train_acc': train_acc,
                'test_acc': test_acc,
                'f1': f1,
            })
            models.append(clf)
        except Exception as e:
            log.warning(f"Label {label_idx} training failed: {e}")
            label_results.append(None)
            models.append(None)

    if not test_accs:
        return None

    return {
        'train_acc': np.mean(train_accs),
        'test_acc': np.mean(test_accs),
        'mean_f1': np.mean(f1_scores_list),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'label_results': label_results,
        'y_test': y_test,
        'scaler': scaler,
        'models': models,
    }


def train_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> Optional[dict]:
    """Train probe with cross-validation."""
    try:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = _create_logistic_regression(random_state)
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            if USING_CUML:
                y_pred = np.asarray(y_pred)

            scores.append(accuracy_score(y_test, y_pred))

        scores = np.array(scores)
        return {
            'mean_acc': scores.mean(),
            'std_acc': scores.std(),
            'cv_scores': scores.tolist(),
        }
    except Exception as e:
        log.warning(f"CV training failed: {e}")
        return None
