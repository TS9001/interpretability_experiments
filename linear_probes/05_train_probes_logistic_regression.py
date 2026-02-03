#!/usr/bin/env python3
"""
Linear Probe Training with Logistic Regression.

Trains L2-regularized logistic regression probes on extracted hidden states.
Linear probes are intentionally simple - if a linear probe achieves high accuracy,
it suggests the information is linearly accessible in that layer.

Key features:
- L2 regularization with strength tuning via cross-validation
- Hewitt-style control tasks to measure probe selectivity
- Reports accuracy, AUC, F1, and selectivity scores
- Optional mean-centering vs full standardization

Control Tasks (Hewitt & Liang, EMNLP 2019):
    Uses type-consistent random labeling to test if probes memorize vs extract info.
    Selectivity = linguistic_accuracy - control_accuracy
    High selectivity → information genuinely encoded in representations

Usage:
    python 05_train_probes_logistic_regression.py
    python 05_train_probes_logistic_regression.py --probes B1,C1,D1
    python 05_train_probes_logistic_regression.py --control  # run control task

Regularization:
    By default, regularization tuning is OFF (uses fixed C=0.01).

    For rigorous regularization tuning:

    1. Enable 5-fold CV tuning (searches C in [0.001, 0.01, 0.1, 1.0, 10.0]):
       python 05_train_probes_logistic_regression.py --tune-regularization

    2. Use a specific regularization strength manually (C = 1/lambda):
       python 05_train_probes_logistic_regression.py -r 0.001  # very strong regularization
       python 05_train_probes_logistic_regression.py -r 0.01   # strong (default)
       python 05_train_probes_logistic_regression.py -r 1.0    # moderate regularization

    For publishable results, consider:
    - Running with --tune-regularization for proper hyperparameter selection
    - Using nested CV (not implemented) for unbiased performance estimates
    - Reporting results across multiple C values to show robustness
"""
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

from utils.data import load_json, save_json
from utils.logging import log, print_header, print_config
from utils.control_tasks import (
    create_matched_control_labels,
    create_matched_control_labels_multilabel,
    compute_selectivity,
)

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "probe_data"
DEFAULT_TRAIN_DIR = DEFAULT_INPUT / "train"
DEFAULT_TEST_DIR = DEFAULT_INPUT / "test"

# Regularization strengths to search (C = 1/lambda, so smaller C = stronger regularization)
REGULARIZATION_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

# All probes with metadata
ALL_PROBE_INFO = {
    # Category A - Problem Understanding
    'A1': {
        'name': 'Operation Planning',
        'type': 'multi_label',
        'n_labels': 4,
        'label_names': ['add', 'sub', 'mult', 'div'],
    },
    'A2': {'name': 'Difficulty', 'type': 'classification', 'n_classes': 5},
    # Category B - Numerical Representation
    'B1': {'name': 'Operand Magnitude', 'type': 'classification', 'n_classes': 6},
    'B2': {'name': 'Result Magnitude', 'type': 'classification', 'n_classes': 6},
    # Category C - Computation Mechanics
    'C1': {'name': 'Correctness', 'type': 'classification', 'n_classes': 2},
    'C3_add': {'name': 'Add Correctness', 'type': 'classification', 'n_classes': 2},
    'C3_sub': {'name': 'Sub Correctness', 'type': 'classification', 'n_classes': 2},
    'C3_mult': {'name': 'Mult Correctness', 'type': 'classification', 'n_classes': 2},
    'C3_div': {'name': 'Div Correctness', 'type': 'classification', 'n_classes': 2},
    'C4': {'name': 'Coarse Result', 'type': 'classification', 'n_classes': 4},
    # Category D - Sequential Reasoning
    'D1': {'name': 'Intermediate/Final', 'type': 'classification', 'n_classes': 2},
    'D2': {'name': 'Next Operation', 'type': 'classification', 'n_classes': 5},
    'D3': {'name': 'Step Position', 'type': 'classification', 'n_classes': 3},
    'D6': {'name': 'Previous Operation', 'type': 'classification', 'n_classes': 5},
}


def can_stratify(y: np.ndarray) -> bool:
    """Check if stratified split is possible (all classes have >= 2 samples)."""
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def get_majority_baseline(y: np.ndarray) -> float:
    """Calculate majority class baseline accuracy."""
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / len(y)


def preprocess_activations(
    X: np.ndarray,
    center_only: bool = False,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Preprocess activations with optional centering or full standardization.

    Args:
        X: Input activations (n_samples, hidden_dim)
        center_only: If True, only mean-center. If False, also scale to unit variance.
        fit: Whether to fit the scaler (True for train, False for test)
        scaler: Pre-fitted scaler (required if fit=False)

    Returns:
        Preprocessed activations and the scaler
    """
    if scaler is None:
        scaler = StandardScaler(with_std=not center_only)

    if fit:
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = scaler.transform(X)

    return X_processed, scaler


def find_best_regularization(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    center_only: bool = False,
) -> float:
    """
    Find optimal regularization strength via cross-validation.

    Returns the best C value (inverse of regularization strength).
    """
    # Preprocess for tuning
    X_scaled, _ = preprocess_activations(X, center_only=center_only)

    best_c = 1.0
    best_score = -np.inf

    for c in REGULARIZATION_GRID:
        clf = LogisticRegression(
            C=c,
            max_iter=5000,
            tol=1e-3,
            solver='sag',
            random_state=42,
        )

        try:
            stratify = y if can_stratify(y) else None
            if stratify is not None:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
            else:
                scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)

            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_c = c
        except Exception:
            continue

    return best_c


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regularization_c: float = 1.0,
    center_only: bool = False,
    balanced: bool = False,
) -> dict:
    """
    Train a single linear probe (logistic regression) and evaluate.

    Returns dict with metrics, predictions, and trained model.
    """
    # Preprocess
    X_train_proc, scaler = preprocess_activations(X_train, center_only=center_only)
    X_test_proc, _ = preprocess_activations(X_test, center_only=center_only, fit=False, scaler=scaler)

    # Train logistic regression with L2 regularization
    clf = LogisticRegression(
        C=regularization_c,
        max_iter=5000,
        tol=1e-3,
        solver='lbfgs',
        class_weight='balanced' if balanced else None,
        random_state=42,
    )

    clf.fit(X_train_proc, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train_proc)
    y_test_pred = clf.predict(X_test_proc)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # F1 score (macro for multiclass)
    n_classes = len(np.unique(y_train))
    average = 'binary' if n_classes == 2 else 'macro'
    test_f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)

    # AUC (only for binary or when we have probabilities)
    test_auc = None
    if n_classes == 2:
        try:
            y_prob = clf.predict_proba(X_test_proc)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            pass
    elif n_classes > 2:
        try:
            y_prob = clf.predict_proba(X_test_proc)
            test_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except Exception:
            pass

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'regularization_c': regularization_c,
        'y_pred': y_test_pred,
        'y_test': y_test,
        'model': clf,
        'scaler': scaler,
    }


def train_multi_label_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regularization_c: float = 1.0,
    center_only: bool = False,
    balanced: bool = False,
) -> dict:
    """Train a multi-label probe (one classifier per label)."""
    n_labels = y_train.shape[1]

    # Preprocess once for all labels
    X_train_proc, scaler = preprocess_activations(X_train, center_only=center_only)
    X_test_proc, _ = preprocess_activations(X_test, center_only=center_only, fit=False, scaler=scaler)

    label_results = []
    models = []
    test_accs = []
    f1_scores = []

    for label_idx in range(n_labels):
        y_train_label = y_train[:, label_idx]
        y_test_label = y_test[:, label_idx]

        clf = LogisticRegression(
            C=regularization_c,
            max_iter=5000,
            tol=1e-3,
            solver='lbfgs',
            class_weight='balanced' if balanced else None,
            random_state=42,
        )

        try:
            clf.fit(X_train_proc, y_train_label)
            y_pred = clf.predict(X_test_proc)

            acc = accuracy_score(y_test_label, y_pred)
            f1 = f1_score(y_test_label, y_pred, average='binary', zero_division=0)

            test_accs.append(acc)
            f1_scores.append(f1)
            label_results.append({'test_acc': acc, 'f1': f1})
            models.append(clf)
        except Exception as e:
            log.warning(f"Label {label_idx} training failed: {e}")
            label_results.append(None)
            models.append(None)

    if not test_accs:
        return None

    return {
        'train_acc': np.mean(test_accs),  # Approximation
        'test_acc': np.mean(test_accs),
        'mean_f1': np.mean(f1_scores),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'label_results': label_results,
        'models': models,
        'scaler': scaler,
    }


def print_results_table(
    results: dict,
    probe: str,
    layers: list[int],
    majority_baseline: float,
    is_multi_label: bool = False,
    show_selectivity: bool = False,
):
    """Print formatted results table with optional selectivity metrics."""
    info = ALL_PROBE_INFO.get(probe, {'name': probe})

    print(f"\n{'='*80}")
    print(f"Probe {probe}: {info['name']}")
    if is_multi_label:
        print(f"Type: Multi-label ({info.get('n_labels', 4)} labels)")
    print(f"Majority baseline: {majority_baseline:.1%}")
    print(f"{'='*80}")

    if is_multi_label:
        header = f"{'Layer':>6} {'Test Acc':>9} {'Mean F1':>8} {'vs Maj':>8}"
    else:
        header = f"{'Layer':>6} {'Test Acc':>9} {'Train':>7} {'F1':>7} {'AUC':>7} {'vs Maj':>8}"

    if show_selectivity:
        header += f" {'Control':>8} {'Select':>8}"
    print(header)
    print("-" * len(header))

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in results:
            continue

        r = results[layer_idx]
        test_acc = r['test_acc']
        vs_maj = test_acc - majority_baseline
        sign = '+' if vs_maj > 0 else ''

        if is_multi_label:
            mean_f1 = r.get('mean_f1', 0)
            line = f"{layer:>6} {test_acc:>9.1%} {mean_f1:>8.3f} {sign}{vs_maj:>7.1%}"
        else:
            train_acc = r.get('train_acc', test_acc)
            f1 = r.get('test_f1', 0)
            auc = r.get('test_auc')
            auc_str = f"{auc:.3f}" if auc else "N/A"
            line = f"{layer:>6} {test_acc:>9.1%} {train_acc:>7.1%} {f1:>7.3f} {auc_str:>7} {sign}{vs_maj:>7.1%}"

        if show_selectivity:
            ctrl_acc = r.get('control_acc')
            selectivity = r.get('selectivity')
            if ctrl_acc is not None and selectivity is not None:
                sel_sign = '+' if selectivity > 0 else ''
                line += f" {ctrl_acc:>8.1%} {sel_sign}{selectivity:>7.1%}"
            else:
                line += f" {'N/A':>8} {'N/A':>8}"

        print(line)


def check_class_balance(y: np.ndarray, probe: str, is_multi_label: bool = False):
    """Print class distribution."""
    if is_multi_label:
        print(f"\n  Label distribution for {probe}:")
        for i in range(y.shape[1]):
            pos = np.sum(y[:, i])
            total = len(y)
            print(f"    Label {i}: {pos}/{total} positive ({pos/total*100:.1f}%)")
    else:
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"\n  Class distribution for {probe}:")
        for cls, count in zip(unique, counts):
            pct = count / total * 100
            print(f"    Class {cls}: {count} ({pct:.1f}%)")


@app.command()
def main(
    input_dir: Optional[Path] = typer.Option(None, "--input", "-i", help="Base probe data directory (contains train/ and test/ subdirs)"),
    train_dir: Optional[Path] = typer.Option(None, "--train-dir", help="Training data directory (overrides --input)"),
    test_dir: Optional[Path] = typer.Option(None, "--test-dir", help="Test data directory (overrides --input)"),
    probes: Optional[str] = typer.Option(None, "--probes", "-p", help="Probes to train (comma-separated)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed metrics"),
    control: bool = typer.Option(False, "--control", "-c", help="Run control task only (Hewitt-style)"),
    with_selectivity: bool = typer.Option(False, "--with-selectivity", "-s", help="Run both tasks and compute selectivity (2x time)"),
    tune_regularization: bool = typer.Option(False, "--tune-regularization/--no-tune-regularization", "-t", help="Tune L2 regularization via CV (slow, adds ~25 fits per layer)"),
    regularization_c: float = typer.Option(0.01, "--regularization", "-r", help="Regularization strength (C = 1/lambda, lower = stronger)"),
    center_only: bool = typer.Option(False, "--center-only", help="Only mean-center activations (no scaling)"),
    save_probes: bool = typer.Option(True, "--save-probes/--no-save-probes", help="Save trained probes"),
    balanced: bool = typer.Option(False, "--balanced", "-b", help="Use class weighting for imbalanced classes"),
):
    """
    Train linear probes (logistic regression) and evaluate accuracy.

    Linear probes are intentionally simple. High accuracy suggests the information
    is linearly accessible in that layer. Use --control to verify with shuffled
    labels - the probe should perform at chance level on random labels.

    Expects separate train/ and test/ subdirectories with extracted hidden states.
    """
    # Determine train and test directories
    base_path = input_dir or DEFAULT_INPUT
    train_path = train_dir or (base_path / "train")
    test_path = test_dir or (base_path / "test")

    if not train_path.exists():
        log.error(f"Train directory not found: {train_path}")
        raise typer.Exit(1)

    if not test_path.exists():
        log.error(f"Test directory not found: {test_path}")
        raise typer.Exit(1)

    # Load metadata from train directory
    meta_path = train_path / 'metadata.json'
    if meta_path.exists():
        metadata = load_json(meta_path)
        available_probes = metadata.get('probes', list(ALL_PROBE_INFO.keys()))
        layers = metadata.get('layers', [0, 7, 14, 21, 27])
    else:
        available_probes = list(ALL_PROBE_INFO.keys())
        layers = [0, 7, 14, 21, 27]

    # Parse probe list
    probe_list = [p.strip() for p in probes.split(',')] if probes else available_probes

    if with_selectivity:
        mode = "Linear Probes + Selectivity (Hewitt control)"
    elif control:
        mode = "CONTROL ONLY (Hewitt-style)"
    else:
        mode = "Linear Probes"
    print_header("Logistic Regression Probe Training", mode)

    if control:
        log.warning("CONTROL MODE: Hewitt-style control task (type-consistent random labels).")
    if with_selectivity:
        log.info("SELECTIVITY MODE: Training both real + control tasks to compute selectivity.")
        log.info("Selectivity = linguistic_acc - control_acc. High selectivity = genuine signal.")

    print_config("Configuration", {
        'train_dir': str(train_path),
        'test_dir': str(test_path),
        'probes': ', '.join(probe_list),
        'layers': ', '.join(map(str, layers)),
        'regularization_c': regularization_c if not tune_regularization else "tuned via CV",
        'preprocessing': "center only" if center_only else "standardize (center + scale)",
        'class_weight': 'balanced' if balanced else 'none',
        'control_mode': control,
        'with_selectivity': with_selectivity,
    })

    all_results = {}
    control_results = {}

    for probe in probe_list:
        train_file = train_path / f"{probe}_samples.pt"
        test_file = test_path / f"{probe}_samples.pt"

        if not train_file.exists():
            log.warning(f"Train data not found: {train_file}")
            continue

        if not test_file.exists():
            log.warning(f"Test data not found: {test_file}")
            continue

        log.info(f"Training {probe}...")
        train_data = torch.load(train_file, weights_only=False)
        test_data = torch.load(test_file, weights_only=False)

        X_train_all = train_data['X'].numpy()  # (n_samples, n_layers, hidden_dim)
        y_train_all = train_data['y'].numpy()  # (n_samples,) or (n_samples, n_labels)
        X_test_all = test_data['X'].numpy()
        y_test_all = test_data['y'].numpy()

        probe_type = train_data.get('probe_type', 'classification')
        is_multi_label = probe_type == 'multi_label'

        n_train, n_layers, hidden_dim = X_train_all.shape
        n_test = X_test_all.shape[0]
        log.info(f"  Train samples: {n_train}, Test samples: {n_test}, Layers: {n_layers}, Hidden dim: {hidden_dim}")

        if verbose:
            check_class_balance(y_train_all, probe, is_multi_label)

        if n_train < 50:
            log.warning(f"  Too few train samples ({n_train}), skipping")
            continue

        # Train probes for each layer
        probe_results = {}
        probe_control_results = {}
        trained_models = {}

        for layer_idx in range(n_layers):
            layer_num = layers[layer_idx] if layer_idx < len(layers) else layer_idx
            print(f"    Layer {layer_num} ({layer_idx+1}/{n_layers})...", end=" ", flush=True)

            X_train = X_train_all[:, layer_idx, :]
            X_test = X_test_all[:, layer_idx, :]
            y_train = y_train_all.copy()
            y_test = y_test_all.copy()

            # Tune regularization if requested
            if tune_regularization:
                c = find_best_regularization(X_train, y_train, center_only=center_only)
                if layer_idx == 0:
                    log.info(f"  Tuned regularization C={c}")
            else:
                c = regularization_c

            # Generate Hewitt-style control labels if needed
            # Each unique "type" (based on hidden state features) gets a consistent
            # random label. This tests if the probe memorizes input patterns vs
            # extracts genuine information from representations.
            if control or with_selectivity:
                if is_multi_label:
                    n_labels = y_train.shape[1]
                    y_train_ctrl, y_test_ctrl = create_matched_control_labels_multilabel(
                        X_train, X_test, n_labels=n_labels,
                        n_bins=100, seed=42 + layer_idx
                    )
                else:
                    n_classes = len(np.unique(y_train))
                    y_train_ctrl, y_test_ctrl = create_matched_control_labels(
                        X_train, X_test, n_classes=n_classes,
                        n_bins=100, seed=42 + layer_idx
                    )

            try:
                # Train on real labels (unless control-only mode)
                if not control:
                    if is_multi_label:
                        result = train_multi_label_probe(
                            X_train, y_train, X_test, y_test,
                            regularization_c=c, center_only=center_only, balanced=balanced
                        )
                    else:
                        result = train_linear_probe(
                            X_train, y_train, X_test, y_test,
                            regularization_c=c, center_only=center_only, balanced=balanced
                        )

                    if result:
                        probe_results[layer_idx] = result
                        if 'model' in result:
                            trained_models[layer_idx] = result['model']

                # Train on control labels (if control mode or selectivity mode)
                if control or with_selectivity:
                    if is_multi_label:
                        ctrl_result = train_multi_label_probe(
                            X_train, y_train_ctrl, X_test, y_test_ctrl,
                            regularization_c=c, center_only=center_only, balanced=balanced
                        )
                    else:
                        ctrl_result = train_linear_probe(
                            X_train, y_train_ctrl, X_test, y_test_ctrl,
                            regularization_c=c, center_only=center_only, balanced=balanced
                        )

                    if ctrl_result:
                        probe_control_results[layer_idx] = ctrl_result

                # Print results
                if control:
                    # Control-only mode
                    if ctrl_result:
                        print(f"ctrl_acc={ctrl_result.get('test_acc', 0):.1%}", flush=True)
                    else:
                        print("no result", flush=True)
                elif with_selectivity:
                    # Selectivity mode: show both + selectivity
                    if result and ctrl_result:
                        real_acc = result.get('test_acc', 0)
                        ctrl_acc = ctrl_result.get('test_acc', 0)
                        selectivity = real_acc - ctrl_acc
                        result['control_acc'] = ctrl_acc
                        result['selectivity'] = selectivity
                        print(f"acc={real_acc:.1%} ctrl={ctrl_acc:.1%} sel={selectivity:+.1%}", flush=True)
                    elif result:
                        print(f"acc={result.get('test_acc', 0):.1%} (ctrl failed)", flush=True)
                    else:
                        print("no result", flush=True)
                else:
                    # Normal mode
                    if result:
                        print(f"acc={result.get('test_acc', 0):.1%}", flush=True)
                    else:
                        print("no result", flush=True)

                if verbose and layer_idx == n_layers - 1 and not is_multi_label and result:
                    print(f"\n  Confusion matrix (layer {layers[layer_idx]}):")
                    cm = confusion_matrix(result['y_test'], result['y_pred'])
                    print(f"  {cm}")

            except Exception as e:
                print(f"FAILED: {e}", flush=True)
                log.warning(f"  Layer {layer_idx} training failed: {e}")

        # Save trained probes
        if save_probes and trained_models and not control:
            probes_dir = base_path / 'trained_probes'
            probes_dir.mkdir(exist_ok=True)
            probe_path = probes_dir / f"{probe}_probes.joblib"
            joblib.dump({
                'models': trained_models,
                'layers': layers,
                'regularization_c': c,
                'center_only': center_only,
            }, probe_path)
            log.info(f"  Saved {len(trained_models)} probe models")

        # Calculate baselines (using train set)
        if is_multi_label:
            majority_baseline = np.mean([
                max(np.mean(y_train_all[:, i]), 1 - np.mean(y_train_all[:, i]))
                for i in range(y_train_all.shape[1])
            ])
        else:
            majority_baseline = get_majority_baseline(y_train_all)

        all_results[probe] = {
            'results': probe_results,
            'control_results': probe_control_results if (control or with_selectivity) else {},
            'majority_baseline': majority_baseline,
            'is_multi_label': is_multi_label,
        }

        print_results_table(
            probe_results, probe, layers, majority_baseline, is_multi_label,
            show_selectivity=with_selectivity
        )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    summary_data = {}
    for probe, data in all_results.items():
        results = data['results']
        baseline = data['majority_baseline']

        if not results:
            continue

        best_layer_idx = max(results.keys(), key=lambda k: results[k]['test_acc'])
        best_acc = results[best_layer_idx]['test_acc']
        best_f1 = results[best_layer_idx].get('test_f1', 0)
        best_auc = results[best_layer_idx].get('test_auc')

        # Get selectivity if available
        best_ctrl_acc = results[best_layer_idx].get('control_acc')
        best_selectivity = results[best_layer_idx].get('selectivity')

        summary_data[probe] = {
            'best_layer': layers[best_layer_idx],
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_auc': best_auc,
            'vs_baseline': best_acc - baseline,
            'majority_baseline': baseline,
            'control_acc': best_ctrl_acc,
            'selectivity': best_selectivity,
        }

        probe_info = ALL_PROBE_INFO.get(probe, {'name': probe})
        beats_baseline = best_acc > baseline + 0.1

        if control:
            # Control mode: should NOT beat baseline
            status = "PASS (at chance)" if not beats_baseline else "WARN (above chance)"
        elif with_selectivity and best_selectivity is not None:
            # Selectivity mode: judge by selectivity
            if best_selectivity > 0.1:
                status = "HIGH_SEL"
            elif best_selectivity > 0.05:
                status = "MOD_SEL"
            elif best_selectivity > 0:
                status = "LOW_SEL"
            else:
                status = "NOT_SEL"
        else:
            status = "PASS" if beats_baseline else ("MARGINAL" if best_acc > baseline else "FAIL")

        print(f"{probe} ({probe_info['name']}):")
        print(f"  Best: {best_acc:.1%} at layer {layers[best_layer_idx]} "
              f"(majority: {baseline:.1%}, +{best_acc - baseline:.1%}) [{status}]")
        if with_selectivity and best_selectivity is not None:
            print(f"  Selectivity: {best_selectivity:+.1%} (control: {best_ctrl_acc:.1%})")
        if best_auc:
            print(f"  AUC: {best_auc:.3f}, F1: {best_f1:.3f}")

    # Save results
    if with_selectivity:
        results_filename = 'probe_results_logreg_selectivity.json'
    elif control:
        results_filename = 'probe_results_logreg_control.json'
    else:
        results_filename = 'probe_results_logreg.json'
    results_path = base_path / results_filename
    save_json({
        'probes': {
            probe: {
                'results_by_layer': {
                    str(layers[k]): {
                        'test_acc': float(v['test_acc']),
                        'train_acc': float(v.get('train_acc', v['test_acc'])),
                        'f1': float(v['test_f1']) if v.get('test_f1') is not None else None,
                        'auc': float(v['test_auc']) if v.get('test_auc') is not None else None,
                        'control_acc': float(v['control_acc']) if v.get('control_acc') is not None else None,
                        'selectivity': float(v['selectivity']) if v.get('selectivity') is not None else None,
                    }
                    for k, v in data['results'].items()
                },
                'best_layer': summary_data.get(probe, {}).get('best_layer'),
                'best_acc': float(summary_data[probe]['best_acc']) if probe in summary_data else None,
                'best_selectivity': float(summary_data[probe]['selectivity']) if probe in summary_data and summary_data[probe].get('selectivity') is not None else None,
                'majority_baseline': float(data['majority_baseline']),
                'is_multi_label': data['is_multi_label'],
            }
            for probe, data in all_results.items()
        },
        'layers': layers,
        'config': {
            'regularization_c': float(regularization_c),
            'tune_regularization': tune_regularization,
            'center_only': center_only,
            'control_mode': control,
            'with_selectivity': with_selectivity,
        },
    }, results_path)

    log.success(f"Results saved to {results_path}")

    # Success criteria check
    print(f"\n{'='*70}")
    if control:
        print("CONTROL CHECK (shuffled labels - probes should fail)")
    elif with_selectivity:
        print("SELECTIVITY ANALYSIS (Hewitt et al. 2019)")
    else:
        print("SUCCESS CRITERIA")
    print(f"{'='*70}")

    success_count = 0
    control_pass_count = 0
    high_sel_count = 0

    for probe, probe_summary in summary_data.items():
        baseline = probe_summary['majority_baseline']
        beats_baseline = probe_summary['best_acc'] > baseline + 0.1
        selectivity = probe_summary.get('selectivity')

        if control:
            if not beats_baseline:
                print(f"  [PASS] {probe}: {probe_summary['best_acc']:.1%} ≈ chance ({baseline:.1%})")
                control_pass_count += 1
            else:
                print(f"  [WARN] {probe}: {probe_summary['best_acc']:.1%} > chance - may have artifacts")
        elif with_selectivity and selectivity is not None:
            if selectivity > 0.1:
                print(f"  [HIGH] {probe}: selectivity={selectivity:+.1%} (acc={probe_summary['best_acc']:.1%}, ctrl={probe_summary['control_acc']:.1%})")
                high_sel_count += 1
            elif selectivity > 0.05:
                print(f"  [MOD ] {probe}: selectivity={selectivity:+.1%} (acc={probe_summary['best_acc']:.1%}, ctrl={probe_summary['control_acc']:.1%})")
            elif selectivity > 0:
                print(f"  [LOW ] {probe}: selectivity={selectivity:+.1%} (acc={probe_summary['best_acc']:.1%}, ctrl={probe_summary['control_acc']:.1%})")
            else:
                print(f"  [NONE] {probe}: selectivity={selectivity:+.1%} - probe may just memorize")
        else:
            if beats_baseline:
                print(f"  [PASS] {probe}: {probe_summary['best_acc']:.1%} > majority+10%")
                success_count += 1
            else:
                print(f"  [    ] {probe}: {probe_summary['best_acc']:.1%} (majority: {baseline:.1%})")

    if control:
        if control_pass_count == len(summary_data):
            print(f"\nCONTROL PASSED: All probes at chance level - genuine signal verified")
        else:
            print(f"\nCONTROL WARNING: Some probes above chance - check for data artifacts")
    elif with_selectivity:
        print(f"\nSELECTIVITY SUMMARY: {high_sel_count}/{len(summary_data)} probes have HIGH selectivity (>10%)")
        print("High selectivity = information genuinely encoded in representations")
    else:
        if success_count >= 1:
            print(f"\nVALIDATED: {success_count} probe(s) meet success criteria")
        else:
            print(f"\nNOT YET VALIDATED: No probes meet success criteria")


if __name__ == '__main__':
    app()
