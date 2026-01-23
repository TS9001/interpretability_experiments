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

import joblib
import numpy as np
import torch
import typer
from sklearn.metrics import confusion_matrix

from utils.data import load_json, save_json
from utils.logging import log, print_header, print_config, print_summary
from utils.training import (
    can_stratify, get_majority_baseline,
    train_probe_single_layer, train_probe_cv,
)

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


def evaluate_saved_probes(
    probes_dir: Path,
    eval_data_dir: Path,
    probes_filter: Optional[str],
    verbose: bool,
):
    """Evaluate pre-trained probes on new data."""
    print_header("POC Probe Evaluation", "Eval-Only Mode")

    # Load metadata from eval data
    meta_path = eval_data_dir / 'metadata.json'
    if meta_path.exists():
        metadata = load_json(meta_path)
        available_probes = metadata.get('probes', list(PROBE_INFO.keys()))
        layers = metadata.get('layers', [0, 7, 14, 21, 27])
    else:
        available_probes = list(PROBE_INFO.keys())
        layers = [0, 7, 14, 21, 27]

    # Parse probe filter
    if probes_filter:
        probe_list = [p.strip() for p in probes_filter.split(',')]
    else:
        probe_list = available_probes

    trained_probes_dir = probes_dir / 'trained_probes'
    if not trained_probes_dir.exists():
        log.error(f"No trained probes found at {trained_probes_dir}")
        log.error("Run without --eval-only first to train probes")
        raise typer.Exit(1)

    print_config("Configuration", {
        'trained_probes': str(trained_probes_dir),
        'eval_data': str(eval_data_dir),
        'probes': ', '.join(probe_list),
    })

    all_results = {}

    for probe in probe_list:
        # Load trained probes
        probe_path = trained_probes_dir / f"{probe}_probes.joblib"
        if not probe_path.exists():
            log.warning(f"Trained probe not found: {probe_path}")
            continue

        # Load eval data
        eval_file = eval_data_dir / f"{probe}_samples.pt"
        if not eval_file.exists():
            log.warning(f"Eval data not found: {eval_file}")
            continue

        log.info(f"Evaluating {probe}...")

        saved = joblib.load(probe_path)
        trained_models = saved['models']
        trained_layers = saved['layers']

        data = torch.load(eval_file, weights_only=False)
        X = data['X'].numpy()
        y = data['y'].numpy()
        n_samples, n_layers, hidden_dim = X.shape

        log.info(f"  Eval samples: {n_samples}")

        if verbose:
            check_class_balance(y, probe)

        # Evaluate each layer
        probe_results = {}
        for layer_idx, model in trained_models.items():
            if layer_idx >= n_layers:
                continue

            X_layer = X[:, layer_idx, :]
            try:
                test_acc = model.score(X_layer, y)
                y_pred = model.predict(X_layer)
                probe_results[layer_idx] = {
                    'test_acc': test_acc,
                    'train_acc': test_acc,  # N/A for eval
                    'y_pred': y_pred,
                    'y_test': y,
                }

                if verbose and layer_idx == max(trained_models.keys()):
                    print(f"\n  Confusion matrix (layer {layers[layer_idx]}):")
                    cm = confusion_matrix(y, y_pred)
                    print(f"  {cm}")
            except Exception as e:
                log.warning(f"  Layer {layer_idx} eval failed: {e}")

        majority_baseline = get_majority_baseline(y)
        all_results[probe] = {'results': probe_results, 'majority_baseline': majority_baseline}
        print_results_table(probe_results, probe, layers, majority_baseline)

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
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

    # Save eval results
    results_path = eval_data_dir / 'eval_results.json'
    save_json({
        'probes': {
            probe: {
                'results_by_layer': {
                    str(layers[k]): {'test_acc': v['test_acc']}
                    for k, v in data['results'].items()
                },
                'best_layer': summary_data.get(probe, {}).get('best_layer'),
                'best_acc': summary_data.get(probe, {}).get('best_acc'),
                'majority_baseline': data['majority_baseline'],
            }
            for probe, data in all_results.items()
        },
        'layers': layers,
        'mode': 'eval_only',
    }, results_path)

    log.success(f"Eval results saved to {results_path}")


@app.command()
def main(
    input_dir: Optional[Path] = typer.Option(None, "--input", "-i", help="Input probe data directory"),
    probes: Optional[str] = typer.Option(None, "--probes", "-p", help="Probes to train (comma-separated)"),
    cv: int = typer.Option(0, "--cv", help="Cross-validation folds (0=train/test split)"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test set fraction"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed metrics"),
    eval_only: bool = typer.Option(False, "--eval-only", help="Evaluate pre-trained probes only"),
    eval_data: Optional[Path] = typer.Option(None, "--eval-data", help="Evaluation data directory (for --eval-only)"),
    save_probes: bool = typer.Option(True, "--save-probes/--no-save-probes", help="Save trained probes"),
):
    """Train linear probes and evaluate accuracy."""
    input_path = input_dir or DEFAULT_INPUT

    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    # Handle eval-only mode
    if eval_only:
        if not eval_data:
            log.error("--eval-data required with --eval-only")
            raise typer.Exit(1)
        if not eval_data.exists():
            log.error(f"Eval data not found: {eval_data}")
            raise typer.Exit(1)
        return evaluate_saved_probes(input_path, eval_data, probes, verbose)

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
        trained_models = {}

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
                    if 'model' in result:
                        trained_models[layer_idx] = result['model']

                    if verbose and layer_idx == n_layers - 1:
                        # Print confusion matrix for last layer
                        print(f"\n  Confusion matrix (layer {layers[layer_idx]}):")
                        cm = confusion_matrix(result['y_test'], result['y_pred'])
                        print(f"  {cm}")

        # Save trained probes
        if save_probes and trained_models:
            probes_dir = input_path / 'trained_probes'
            probes_dir.mkdir(exist_ok=True)
            probe_path = probes_dir / f"{probe}_probes.joblib"
            joblib.dump({'models': trained_models, 'layers': layers}, probe_path)
            log.info(f"  Saved {len(trained_models)} probe models to {probe_path}")

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
