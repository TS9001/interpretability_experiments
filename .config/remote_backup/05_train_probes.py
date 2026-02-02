#!/usr/bin/env python3
"""
MLP Probe Training: Train small neural network probes for all probe types.

Trains PyTorch MLP probes on extracted hidden states and reports accuracy per layer.
Uses GPU if available for fast training. Supports:
- Classification probes (most probes)
- Multi-label probes (A1)

Usage:
    python 05_train_probes.py
    python 05_train_probes.py --probes B1,C1,D1
    python 05_train_probes.py --hidden-dim 256 --epochs 50
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import typer
from sklearn.metrics import confusion_matrix, f1_score

from utils.data import load_json, save_json
from utils.logging import log, print_header, print_config, print_summary

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "probe_data"
DEFAULT_TRAIN_DIR = DEFAULT_INPUT / "train"
DEFAULT_TEST_DIR = DEFAULT_INPUT / "test"

# All probes with metadata
ALL_PROBE_INFO = {
    # Category A - Problem Understanding
    'A1': {
        'name': 'Operation Planning',
        'type': 'multi_label',
        'n_labels': 4,
        'label_names': ['add', 'sub', 'mult', 'div'],
        'random_baseline': 0.5,
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


class MLPProbe(nn.Module):
    """Single hidden layer MLP for probing hidden states."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPProbeMultiLabel(nn.Module):
    """Single hidden layer MLP for multi-label classification."""

    def __init__(self, input_dim: int, n_labels: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    hidden_dim: int = 256,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = None,
) -> dict:
    """Train an MLP probe for classification."""
    if device is None:
        device = get_device()

    input_dim = X_train.shape[1]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MLPProbe(input_dim, n_classes, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        # Train accuracy
        train_outputs = model(X_train_t.to(device))
        train_preds = train_outputs.argmax(dim=1).cpu().numpy()
        train_acc = (train_preds == y_train).mean()

        # Test accuracy
        test_outputs = model(X_test_t.to(device))
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
        test_acc = (test_preds == y_test).mean()

    return {
        'model': model,
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'y_pred': test_preds,
        'y_test': y_test,
    }


def train_mlp_probe_multi_label(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_labels: int,
    hidden_dim: int = 256,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = None,
) -> dict:
    """Train an MLP probe for multi-label classification."""
    if device is None:
        device = get_device()

    input_dim = X_train.shape[1]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MLPProbeMultiLabel(input_dim, n_labels, hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        # Test predictions
        test_outputs = model(X_test_t.to(device))
        test_preds = (test_outputs > 0.5).cpu().numpy().astype(int)

        # Per-label accuracy and F1
        label_results = []
        for i in range(n_labels):
            label_acc = (test_preds[:, i] == y_test[:, i]).mean()
            label_f1 = f1_score(y_test[:, i], test_preds[:, i], zero_division=0)
            label_results.append({
                'test_acc': float(label_acc),
                'f1': float(label_f1),
            })

        # Overall accuracy (exact match)
        test_acc = (test_preds == y_test).all(axis=1).mean()
        mean_f1 = np.mean([r['f1'] for r in label_results])

    return {
        'model': model,
        'test_acc': float(test_acc),
        'mean_f1': float(mean_f1),
        'label_results': label_results,
        'y_pred': test_preds,
        'y_test': y_test,
    }


def get_majority_baseline(y: np.ndarray) -> float:
    """Calculate majority class baseline accuracy."""
    unique, counts = np.unique(y, return_counts=True)
    return counts.max() / len(y)


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
    input_dir: Optional[Path] = typer.Option(None, "--input", "-i", help="Base probe data directory (contains train/ and test/ subdirs)"),
    train_dir: Optional[Path] = typer.Option(None, "--train-dir", help="Training data directory (overrides --input)"),
    test_dir: Optional[Path] = typer.Option(None, "--test-dir", help="Test data directory (overrides --input)"),
    probes: Optional[str] = typer.Option(None, "--probes", "-p", help="Probes to train (comma-separated)"),
    hidden_dim: int = typer.Option(256, "--hidden-dim", help="Hidden dimension for MLP"),
    epochs: int = typer.Option(30, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(256, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed metrics"),
    save_probes: bool = typer.Option(True, "--save-probes/--no-save-probes", help="Save trained probes"),
    control: bool = typer.Option(False, "--control", "-c", help="Run control task with shuffled labels"),
):
    """Train MLP probes and evaluate accuracy.

    Use --control to run a sanity check with shuffled labels.
    If the probe performs well on shuffled labels, it has too much capacity.

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

    device = get_device()
    log.info(f"Using device: {device}")

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
    if probes:
        probe_list = [p.strip() for p in probes.split(',')]
    else:
        probe_list = available_probes

    mode = "CONTROL (shuffled labels)" if control else "Neural Network Probes"
    print_header("MLP Probe Training", mode)
    if control:
        log.warning("CONTROL MODE: Labels will be shuffled. Probe should perform at chance level.")

    print_config("Configuration", {
        'train_dir': str(train_path),
        'test_dir': str(test_path),
        'probes': ', '.join(probe_list),
        'n_probes': len(probe_list),
        'layers': ', '.join(map(str, layers)),
        'hidden_dim': hidden_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'device': str(device),
        'control_mode': control,
    })

    all_results = {}

    for probe in probe_list:
        train_file = train_path / f"{probe}_samples.pt"
        test_file = test_path / f"{probe}_samples.pt"

        if not train_file.exists():
            log.warning(f"Train data not found: {train_file}")
            continue

        if not test_file.exists():
            log.warning(f"Test data not found: {test_file}")
            continue

        log.info(f"Loading {probe} data...")
        train_data = torch.load(train_file, weights_only=False)
        test_data = torch.load(test_file, weights_only=False)

        X_train_all = train_data['X'].numpy()  # (n_samples, n_layers, hidden_dim)
        y_train_all = train_data['y'].numpy()  # (n_samples,) or (n_samples, n_labels)
        X_test_all = test_data['X'].numpy()
        y_test_all = test_data['y'].numpy()

        probe_type = train_data.get('probe_type', 'classification')
        is_multi_label = probe_type == 'multi_label'

        n_train = X_train_all.shape[0]
        n_test = X_test_all.shape[0]
        n_layers = X_train_all.shape[1]
        input_dim = X_train_all.shape[2]

        log.info(f"  Train samples: {n_train}, Test samples: {n_test}, Layers: {n_layers}, Input dim: {input_dim}")
        if is_multi_label:
            log.info(f"  Type: multi-label ({y_train_all.shape[1]} labels)")

        if verbose:
            check_class_balance(y_train_all, probe, is_multi_label)

        # Check for enough samples
        if n_train < 50:
            log.warning(f"  Too few train samples ({n_train}), skipping")
            continue

        # Get number of classes
        if is_multi_label:
            n_classes = y_train_all.shape[1]
        else:
            n_classes = len(np.unique(y_train_all))

        # Train probes for each layer
        probe_results = {}
        trained_models = {}

        for layer_idx in range(n_layers):
            X_train = X_train_all[:, layer_idx, :]  # (n_samples, input_dim)
            X_test = X_test_all[:, layer_idx, :]
            y_train = y_train_all.copy()
            y_test = y_test_all.copy()

            # Control task: shuffle labels to verify probe isn't memorizing
            if control:
                rng = np.random.RandomState(42 + layer_idx)
                y_train = rng.permutation(y_train)
                y_test = rng.permutation(y_test)

            try:
                if is_multi_label:
                    result = train_mlp_probe_multi_label(
                        X_train, y_train, X_test, y_test,
                        n_labels=n_classes,
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                    )
                else:
                    result = train_mlp_probe(
                        X_train, y_train, X_test, y_test,
                        n_classes=n_classes,
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                    )

                probe_results[layer_idx] = result
                if 'model' in result:
                    trained_models[layer_idx] = result['model']

                if verbose and layer_idx == n_layers - 1 and not is_multi_label:
                    print(f"\n  Confusion matrix (layer {layers[layer_idx]}):")
                    cm = confusion_matrix(result['y_test'], result['y_pred'])
                    print(f"  {cm}")

            except Exception as e:
                log.warning(f"  Layer {layer_idx} training failed: {e}")
                continue

        # Save trained probes
        if save_probes and trained_models:
            probes_dir = base_path / 'trained_probes_mlp'
            probes_dir.mkdir(exist_ok=True)
            probe_path = probes_dir / f"{probe}_probes.pt"
            torch.save({
                'models': {k: v.state_dict() for k, v in trained_models.items()},
                'layers': layers,
                'hidden_dim': hidden_dim,
                'input_dim': input_dim,
                'n_classes': n_classes,
                'is_multi_label': is_multi_label,
            }, probe_path)
            log.info(f"  Saved {len(trained_models)} probe models to {probe_path}")

        # Calculate majority baseline (using train set)
        if is_multi_label:
            majority_baseline = np.mean([max(np.mean(y_train_all[:, i]), 1-np.mean(y_train_all[:, i])) for i in range(y_train_all.shape[1])])
        else:
            majority_baseline = get_majority_baseline(y_train_all)

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
    results_filename = 'probe_results_mlp_control.json' if control else 'probe_results_mlp.json'
    results_path = base_path / results_filename
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
        'config': {
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'control_mode': control,
        },
    }, results_path)

    log.success(f"Results saved to {results_path}")

    # Success check
    print(f"\n{'='*60}")
    if control:
        print("CONTROL CHECK (shuffled labels - probes should fail)")
    else:
        print("SUCCESS CRITERIA")
    print(f"{'='*60}")

    success_count = 0
    control_pass_count = 0
    for probe, probe_summary in summary_data.items():
        baseline = probe_summary['majority_baseline']
        beats_baseline = probe_summary['best_acc'] > baseline + 0.1

        if control:
            # In control mode, probes should NOT beat baseline (shuffled labels)
            if not beats_baseline:
                print(f"  [PASS] {probe}: {probe_summary['best_acc']:.1%} ≈ chance ({baseline:.1%}) - probe not memorizing")
                control_pass_count += 1
            else:
                print(f"  [WARN] {probe}: {probe_summary['best_acc']:.1%} > chance ({baseline:.1%}) - probe may have too much capacity")
        else:
            if beats_baseline:
                print(f"  [PASS] {probe}: {probe_summary['best_acc']:.1%} > majority+10% ({baseline:.1%})")
                success_count += 1
            else:
                print(f"  [    ] {probe}: {probe_summary['best_acc']:.1%} (majority: {baseline:.1%})")

    if control:
        if control_pass_count == len(summary_data):
            print(f"\nCONTROL PASSED: All {control_pass_count} probe(s) at chance level - no memorization detected")
        else:
            print(f"\nCONTROL WARNING: {len(summary_data) - control_pass_count} probe(s) above chance - consider reducing capacity")
    else:
        if success_count >= 1:
            print(f"\nVALIDATED: {success_count} probe(s) meet success criteria")
        else:
            print(f"\nNOT YET VALIDATED: No probes meet success criteria")


if __name__ == '__main__':
    app()
