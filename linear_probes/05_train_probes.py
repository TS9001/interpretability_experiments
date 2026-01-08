#!/usr/bin/env python3
"""
POC Step 3: Train linear probes and evaluate.

Trains sklearn LogisticRegression probes on extracted hidden states
and reports accuracy per layer.

Usage:
    python poc_03_train_probes.py
    python poc_03_train_probes.py --probes B1,C1
    python poc_03_train_probes.py --cv 5
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from utils.data import load_json, save_json
from utils.logging import log, print_header, print_config, print_summary

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "probe_data"

# POC Probes with metadata
PROBE_INFO = {
    'B1': {'name': 'Operand Magnitude', 'n_classes': 6, 'random_baseline': 1/6},
    'B2': {'name': 'Result Magnitude', 'n_classes': 6, 'random_baseline': 1/6},
    'C1': {'name': 'Correctness', 'n_classes': 2, 'random_baseline': 0.5},
    'D1': {'name': 'Intermediate/Final', 'n_classes': 2, 'random_baseline': 0.5},
    'D2': {'name': 'Next Operation', 'n_classes': 5, 'random_baseline': 0.2},
}


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
) -> dict:
    """Train a single probe and return metrics."""
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


def train_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
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


def print_results_table(results: dict, probe: str, layers: list[int], majority_baseline: float):
    """Print formatted results table."""
    info = PROBE_INFO[probe]
    baseline = majority_baseline  # Use actual majority class baseline

    print(f"\n{'='*60}")
    print(f"Probe {probe}: {info['name']}")
    print(f"Majority baseline: {baseline:.1%}")
    print(f"{'='*60}")
    print(f"{'Layer':>8} {'Test Acc':>10} {'Train Acc':>10} {'vs Baseline':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    for layer_idx, layer in enumerate(layers):
        if layer_idx in results:
            r = results[layer_idx]
            test_acc = r['test_acc']
            train_acc = r['train_acc']
            vs_random = test_acc - baseline
            sign = '+' if vs_random > 0 else ''
            print(f"{layer:>8} {test_acc:>10.1%} {train_acc:>10.1%} {sign}{vs_random:>11.1%}")
        else:
            print(f"{layer:>8} {'N/A':>10} {'N/A':>10} {'N/A':>12}")


def check_class_balance(y: np.ndarray, probe: str):
    """Print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"\n  Class distribution for {probe}:")
    for cls, count in zip(unique, counts):
        pct = count / total * 100
        print(f"    Class {cls}: {count} ({pct:.1f}%)")


@app.command()
def main(
    input_dir: Optional[Path] = typer.Option(None, "--input", "-i", help="Input probe data directory"),
    probes: Optional[str] = typer.Option(None, "--probes", "-p", help="Probes to train (comma-separated)"),
    cv: int = typer.Option(0, "--cv", help="Cross-validation folds (0=train/test split)"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test set fraction"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed metrics"),
):
    """Train linear probes and evaluate accuracy."""
    input_path = input_dir or DEFAULT_INPUT

    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    # Load metadata
    meta_path = input_path / 'metadata.json'
    if meta_path.exists():
        metadata = load_json(meta_path)
        available_probes = metadata.get('probes', list(PROBE_INFO.keys()))
        layers = metadata.get('layers', [0, 7, 14, 21, 27])
    else:
        available_probes = list(PROBE_INFO.keys())
        layers = [0, 7, 14, 21, 27]

    # Parse probe list
    if probes:
        probe_list = [p.strip() for p in probes.split(',')]
    else:
        probe_list = available_probes

    print_header("POC Probe Training", "Step 3")
    print_config("Configuration", {
        'input': str(input_path),
        'probes': ', '.join(probe_list),
        'layers': ', '.join(map(str, layers)),
        'cv_folds': cv if cv > 0 else f"train/test ({test_size:.0%})",
    })

    all_results = {}

    for probe in probe_list:
        probe_file = input_path / f"{probe}_samples.pt"

        if not probe_file.exists():
            log.warning(f"Probe data not found: {probe_file}")
            continue

        log.info(f"Loading {probe} data...")
        data = torch.load(probe_file, weights_only=False)

        X = data['X'].numpy()  # (n_samples, n_layers, hidden_dim)
        y = data['y'].numpy()  # (n_samples,)
        n_samples, n_layers, hidden_dim = X.shape

        log.info(f"  Samples: {n_samples}, Layers: {n_layers}, Hidden dim: {hidden_dim}")

        if verbose:
            check_class_balance(y, probe)

        # Check for enough samples
        n_classes = len(np.unique(y))
        if n_samples < 50:
            log.warning(f"  Too few samples ({n_samples}), skipping")
            continue

        # Train probes for each layer
        probe_results = {}

        for layer_idx in range(n_layers):
            X_layer = X[:, layer_idx, :]  # (n_samples, hidden_dim)

            if cv > 0:
                result = train_probe_cv(X_layer, y, cv=cv)
                if result:
                    probe_results[layer_idx] = {
                        'test_acc': result['mean_acc'],
                        'train_acc': result['mean_acc'],  # Approximate
                        'std_acc': result['std_acc'],
                    }
            else:
                result = train_probe_single_layer(X_layer, y, test_size=test_size)
                if result:
                    probe_results[layer_idx] = result

                    if verbose and layer_idx == n_layers - 1:
                        # Print confusion matrix for last layer
                        print(f"\n  Confusion matrix (layer {layers[layer_idx]}):")
                        cm = confusion_matrix(result['y_test'], result['y_pred'])
                        print(f"  {cm}")

        # Calculate majority baseline for this probe
        majority_baseline = get_majority_baseline(y)

        all_results[probe] = {'results': probe_results, 'majority_baseline': majority_baseline}
        print_results_table(probe_results, probe, layers, majority_baseline)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary_data = {}
    for probe, data in all_results.items():
        results = data['results']
        baseline = data['majority_baseline']

        if not results:
            continue

        best_layer_idx = max(results.keys(), key=lambda k: results[k]['test_acc'])
        best_acc = results[best_layer_idx]['test_acc']

        summary_data[probe] = {
            'best_layer': layers[best_layer_idx],
            'best_acc': best_acc,
            'vs_baseline': best_acc - baseline,
            'majority_baseline': baseline,
        }

        status = "PASS" if best_acc > baseline + 0.1 else ("MARGINAL" if best_acc > baseline else "FAIL")
        print(f"{probe} ({PROBE_INFO[probe]['name']}):")
        print(f"  Best: {best_acc:.1%} at layer {layers[best_layer_idx]} (majority: {baseline:.1%}, +{best_acc - baseline:.1%}) [{status}]")

    # Save results
    results_path = input_path / 'probe_results.json'
    save_json({
        'probes': {
            probe: {
                'results_by_layer': {
                    str(layers[k]): {'test_acc': v['test_acc'], 'train_acc': v['train_acc']}
                    for k, v in data['results'].items()
                },
                'best_layer': summary_data.get(probe, {}).get('best_layer'),
                'best_acc': summary_data.get(probe, {}).get('best_acc'),
                'majority_baseline': data['majority_baseline'],
            }
            for probe, data in all_results.items()
        },
        'layers': layers,
        'config': {'cv': cv, 'test_size': test_size},
    }, results_path)

    log.success(f"Results saved to {results_path}")

    # POC Success check
    print(f"\n{'='*60}")
    print("POC SUCCESS CRITERIA")
    print(f"{'='*60}")

    success_count = 0
    for probe, probe_summary in summary_data.items():
        baseline = probe_summary['majority_baseline']
        if probe_summary['best_acc'] > baseline + 0.1:
            print(f"  [PASS] {probe}: {probe_summary['best_acc']:.1%} > majority+10% ({baseline:.1%})")
            success_count += 1
        else:
            print(f"  [    ] {probe}: {probe_summary['best_acc']:.1%} (majority: {baseline:.1%})")

    if success_count >= 1:
        print(f"\nPOC VALIDATED: {success_count} probe(s) meet success criteria")
    else:
        print(f"\nPOC NOT YET VALIDATED: No probes meet success criteria")


if __name__ == '__main__':
    app()
