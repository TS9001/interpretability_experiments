"""Shared training utilities for linear probes."""
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

from utils.logging import log


def can_stratify(y: np.ndarray) -> bool:
    """Check if stratified split is possible (all classes have >= 2 samples)."""
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def get_majority_baseline(y: np.ndarray) -> float:
    """Calculate majority class baseline (accuracy if always predicting most common class)."""
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / len(y)


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

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
    )

    try:
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'y_pred': y_pred,
            'y_test': y_test,
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

    train_accs = []
    test_accs = []
    f1_scores = []
    label_results = []

    for label_idx in range(n_labels):
        y_train_label = y_train[:, label_idx]
        y_test_label = y_test[:, label_idx]

        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=random_state,
        )

        try:
            clf.fit(X_train, y_train_label)
            train_acc = clf.score(X_train, y_train_label)
            test_acc = clf.score(X_test, y_test_label)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test_label, y_pred, average='binary', zero_division=0)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            f1_scores.append(f1)
            label_results.append({
                'train_acc': train_acc,
                'test_acc': test_acc,
                'f1': f1,
            })
        except Exception as e:
            log.warning(f"Label {label_idx} training failed: {e}")
            label_results.append(None)

    if not test_accs:
        return None

    return {
        'train_acc': np.mean(train_accs),
        'test_acc': np.mean(test_accs),
        'mean_f1': np.mean(f1_scores),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'label_results': label_results,
        'y_test': y_test,
    }


def train_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> Optional[dict]:
    """Train probe with cross-validation."""
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
    )

    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_acc': scores.mean(),
            'std_acc': scores.std(),
            'cv_scores': scores.tolist(),
        }
    except Exception as e:
        log.warning(f"CV training failed: {e}")
        return None
