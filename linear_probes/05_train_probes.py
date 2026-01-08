#!/usr/bin/env python3
"""
Full Probe Training: Train linear probes for all probe types.

Trains sklearn LogisticRegression probes on extracted hidden states
and reports accuracy per layer. Supports:
- Classification probes (most probes)
- Multi-label probes (A1)

Usage:
    python 05_train_probes.py
    python 05_train_probes.py --probes B1,C1,D1
    python 05_train_probes.py --cv 5
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from utils.data import load_json, save_json
from utils.logging import log, print_header, print_config, print_summary

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "probe_data"

# All probes with metadata
ALL_PROBE_INFO = {
    # Category A - Problem Understanding
    'A1': {
        'name': 'Operation Planning',
        'type': 'multi_label',
        'n_labels': 4,
        'label_names': ['add', 'sub', 'mult', 'div'],
        'random_baseline': 0.5,  # Per-label binary
    },
    'A2': {
        'name': 'Difficulty',
        'type': 'classification',
        'n_classes': 5,
        'random_baseline': 0.2,
    },
    # Category B - Numerical Representation
    'B1': {
        'name': 'Operand Magnitude',
        'type': 'classification',
        'n_classes': 6,
        'random_baseline': 1/6,
    },
    'B2': {
        'name': 'Result Magnitude',
        'type': 'classification',
        'n_classes': 6,
        'random_baseline': 1/6,
    },
    # Category C - Computation Mechanics
    'C1': {
        'name': 'Correctness',
        'type': 'classification',
        'n_classes': 2,
        'random_baseline': 0.5,
    },
    'C3_add': {
        'name': 'Add Correctness',
        'type': 'classification',
        'n_classes': 2,
        'random_baseline': 0.5,
    },
    'C3_sub': {
        'name': 'Sub Correctness',
        'type': 'classification',
        'n_classes': 2,
        'random_baseline': 0.5,
    },
    'C3_mult': {
        'name': 'Mult Correctness',
        'type': 'classification',
        'n_classes': 2,
        'random_baseline': 0.5,
    },
    'C3_div': {
        'name': 'Div Correctness',
        'type': 'classification',
        'n_classes': 2,
        'random_baseline': 0.5,
    },
    'C4': {
        'name': 'Coarse Result',
        'type': 'classification',
        'n_classes': 4,
        'random_baseline': 0.25,
    },
    # Category D - Sequential Reasoning
    'D1': {
        'name': 'Intermediate/Final',
        'type': 'classification',
        'n_classes': 2,
        'random_baseline': 0.5,
    },
    'D2': {
        'name': 'Next Operation',
        'type': 'classification',
        'n_classes': 5,
        'random_baseline': 0.2,
    },
    'D3': {
        'name': 'Step Position',
        'type': 'classification',
        'n_classes': 3,
        'random_baseline': 1/3,
    },
    'D6': {
        'name': 'Previous Operation',
        'type': 'classification',
        'n_classes': 5,
        'random_baseline': 0.2,
    },
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
) -> dict:
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


def print_results_table(results: dict, probe: str, layers: list[int], majority_baseline: float, is_multi_label: bool = False):
    """Print formatted results table."""
    info = ALL_PROBE_INFO.get(probe, {'name': probe, 'random_baseline': 0.5})
    baseline = majority_baseline

    print(f"\n{'='*60}")
    print(f"Probe {probe}: {info['name']}")
    if is_multi_label:
        print(f"Type: Multi-label ({info.get('n_labels', 4)} labels)")
    print(f"Majority baseline: {baseline:.1%}")
    print(f"{'='*60}")

    if is_multi_label:
        print(f"{'Layer':>8} {'Test Acc':>10} {'Mean F1':>10} {'vs Baseline':>12}")
    else:
        print(f"{'Layer':>8} {'Test Acc':>10} {'Train Acc':>10} {'vs Baseline':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    for layer_idx, layer in enumerate(layers):
        if layer_idx in results:
            r = results[layer_idx]
            test_acc = r['test_acc']
            vs_baseline = test_acc - baseline
            sign = '+' if vs_baseline > 0 else ''

            if is_multi_label:
                mean_f1 = r.get('mean_f1', 0)
                print(f"{layer:>8} {test_acc:>10.1%} {mean_f1:>10.3f} {sign}{vs_baseline:>11.1%}")
            else:
                train_acc = r['train_acc']
                print(f"{layer:>8} {test_acc:>10.1%} {train_acc:>10.1%} {sign}{vs_baseline:>11.1%}")
        else:
            print(f"{layer:>8} {'N/A':>10} {'N/A':>10} {'N/A':>12}")


def print_multi_label_details(results: dict, probe: str, layers: list[int]):
    """Print per-label details for multi-label probes."""
    info = ALL_PROBE_INFO.get(probe, {})
    label_names = info.get('label_names', [f'Label {i}' for i in range(4)])

    # Find best layer
    best_layer_idx = max(results.keys(), key=lambda k: results[k]['test_acc'])
    best_result = results[best_layer_idx]

    if 'label_results' not in best_result:
        return

    print(f"\n  Per-label results at best layer ({layers[best_layer_idx]}):")
    for i, (name, lr) in enumerate(zip(label_names, best_result['label_results'])):
        if lr:
            print(f"    {name}: acc={lr['test_acc']:.1%}, F1={lr['f1']:.3f}")
        else:
            print(f"    {name}: N/A")


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
        available_probes = metadata.get('probes', list(ALL_PROBE_INFO.keys()))
        layers = metadata.get('layers', [0, 7, 14, 21, 27])
    else:
        available_probes = list(ALL_PROBE_INFO.keys())
        layers = [0, 7, 14, 21, 27]

    # Parse probe list
    if probes:
        probe_list = [p.strip() for p in probes.split(',')]
    else:
        probe_list = available_probes

    print_header("Linear Probe Training", "All Probes")
    print_config("Configuration", {
        'input': str(input_path),
        'probes': ', '.join(probe_list),
        'n_probes': len(probe_list),
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
        y = data['y'].numpy()  # (n_samples,) or (n_samples, n_labels)
        probe_type = data.get('probe_type', 'classification')
        is_multi_label = probe_type == 'multi_label'

        n_samples = X.shape[0]
        n_layers = X.shape[1]
        hidden_dim = X.shape[2]

        log.info(f"  Samples: {n_samples}, Layers: {n_layers}, Hidden dim: {hidden_dim}")
        if is_multi_label:
            log.info(f"  Type: multi-label ({y.shape[1]} labels)")

        if verbose:
            check_class_balance(y, probe, is_multi_label)

        # Check for enough samples
        if n_samples < 50:
            log.warning(f"  Too few samples ({n_samples}), skipping")
            continue

        # Train probes for each layer
        probe_results = {}

        for layer_idx in range(n_layers):
            X_layer = X[:, layer_idx, :]  # (n_samples, hidden_dim)

            if is_multi_label:
                result = train_multi_label_probe(X_layer, y, test_size=test_size)
                if result:
                    probe_results[layer_idx] = result
            elif cv > 0:
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

        # Calculate majority baseline
        if is_multi_label:
            # For multi-label, baseline is average per-label majority
            majority_baseline = np.mean([max(np.mean(y[:, i]), 1-np.mean(y[:, i])) for i in range(y.shape[1])])
        else:
            majority_baseline = get_majority_baseline(y)

        all_results[probe] = {
            'results': probe_results,
            'majority_baseline': majority_baseline,
            'is_multi_label': is_multi_label,
        }
        print_results_table(probe_results, probe, layers, majority_baseline, is_multi_label)

        if verbose and is_multi_label and probe_results:
            print_multi_label_details(probe_results, probe, layers)

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

        probe_info = ALL_PROBE_INFO.get(probe, {'name': probe})
        status = "PASS" if best_acc > baseline + 0.1 else ("MARGINAL" if best_acc > baseline else "FAIL")
        print(f"{probe} ({probe_info['name']}):")
        print(f"  Best: {best_acc:.1%} at layer {layers[best_layer_idx]} (majority: {baseline:.1%}, +{best_acc - baseline:.1%}) [{status}]")

    # Save results
    results_path = input_path / 'probe_results.json'
    save_json({
        'probes': {
            probe: {
                'results_by_layer': {
                    str(layers[k]): {'test_acc': v['test_acc'], 'train_acc': v.get('train_acc', v['test_acc'])}
                    for k, v in data['results'].items()
                },
                'best_layer': summary_data.get(probe, {}).get('best_layer'),
                'best_acc': summary_data.get(probe, {}).get('best_acc'),
                'majority_baseline': data['majority_baseline'],
                'is_multi_label': data['is_multi_label'],
            }
            for probe, data in all_results.items()
        },
        'layers': layers,
        'config': {'cv': cv, 'test_size': test_size},
    }, results_path)

    log.success(f"Results saved to {results_path}")

    # Success check
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA")
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
        print(f"\nVALIDATED: {success_count} probe(s) meet success criteria")
    else:
        print(f"\nNOT YET VALIDATED: No probes meet success criteria")


if __name__ == '__main__':
    app()
