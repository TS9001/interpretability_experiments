#!/usr/bin/env python3
"""
Step 4: PyTorch Dataset classes for loading probe training data.

Usage:
    from linear_probes.04_probe_dataset import ProbeDataset

    # Load C1 probe data, layer 14
    dataset = ProbeDataset('C1', layer=14, split='train')

    # In training loop
    for hidden_state, label in DataLoader(dataset, batch_size=32):
        ...
"""
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).parent
TRAIN_SPLIT_RATIO = 0.8


def _split_data_by_index(
    data: list,
    split: str,
    index_key: str = 'example_idx'
) -> list:
    """
    Split data into train/test based on example indices.

    Args:
        data: List of data items
        split: 'train' or 'test'
        index_key: Key to use for getting example index

    Returns:
        Filtered data for the requested split
    """
    if split not in ['train', 'test']:
        return data

    all_indices = sorted(set(d[index_key] for d in data))
    split_point = int(len(all_indices) * TRAIN_SPLIT_RATIO)

    if split == 'train':
        valid_indices = set(all_indices[:split_point])
    else:
        valid_indices = set(all_indices[split_point:])

    return [d for d in data if d[index_key] in valid_indices]


def _label_to_tensor(label: Any) -> torch.Tensor:
    """
    Convert a label to a PyTorch tensor.

    Handles:
    - Lists -> float32 tensor (for multi-label)
    - int/float -> long tensor (for classification)
    - Other -> generic tensor
    """
    if isinstance(label, list):
        return torch.tensor(label, dtype=torch.float32)
    elif isinstance(label, (int, float)):
        return torch.tensor(label, dtype=torch.long)
    else:
        return torch.tensor(label)


class ProbeDataset(Dataset):
    """
    Dataset for training linear probes on extracted hidden states.

    Args:
        probe_type: Probe identifier (A1, B1, C1, etc.)
        layer: Which layer's hidden states to use
        split: 'train' or 'test'
        data_dir: Directory containing probe_data/
    """

    def __init__(
        self,
        probe_type: str,
        layer: int,
        split: str = 'train',
        data_dir: Optional[Path] = None
    ):
        self.probe_type = probe_type
        self.layer = layer
        self.split = split

        if data_dir is None:
            data_dir = SCRIPT_DIR / "probe_data"
        else:
            data_dir = Path(data_dir)

        probe_dir = data_dir / f"{probe_type}_data"
        data_path = probe_dir / 'data.pt'

        if not data_path.exists():
            raise FileNotFoundError(f"Probe data not found: {data_path}")

        data = torch.load(data_path)
        self.data = _split_data_by_index(data, split)

        print(f"Loaded {len(self.data)} samples for {probe_type} probe, layer {layer}, {split}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        hidden_state = item['hidden_state'][self.layer].float()
        label = _label_to_tensor(item['label'])
        return hidden_state, label

    def get_num_classes(self) -> int:
        """Get number of classes for this probe."""
        first_label = self.data[0]['label']
        if isinstance(first_label, list):
            return len(first_label)
        return len(set(item['label'] for item in self.data))

    def get_hidden_dim(self) -> int:
        """Get hidden state dimension."""
        return self.data[0]['hidden_state'].shape[-1]


class MultiLayerProbeDataset(Dataset):
    """
    Dataset that returns hidden states from multiple layers.

    Useful for analyzing which layer is best for a probe.
    """

    def __init__(
        self,
        probe_type: str,
        layers: list[int],
        split: str = 'train',
        data_dir: Optional[Path] = None
    ):
        self.probe_type = probe_type
        self.layers = layers
        self.split = split

        if data_dir is None:
            data_dir = SCRIPT_DIR / "probe_data"
        else:
            data_dir = Path(data_dir)

        probe_dir = data_dir / f"{probe_type}_data"
        data_path = probe_dir / 'data.pt'

        if not data_path.exists():
            raise FileNotFoundError(f"Probe data not found: {data_path}")

        data = torch.load(data_path)
        self.data = _split_data_by_index(data, split)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        hidden_states = torch.stack([
            item['hidden_state'][l].float() for l in self.layers
        ])
        label = _label_to_tensor(item['label'])
        return hidden_states, label


class FullDataset(Dataset):
    """
    Dataset that loads the full extracted data (not probe-specific).

    Useful for custom analysis or when you need access to all positions.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        split: str = 'train'
    ):
        if data_dir is None:
            data_dir = SCRIPT_DIR / "probe_data"
        else:
            data_dir = Path(data_dir)

        data_path = data_dir / 'full_data.pt'
        if not data_path.exists():
            raise FileNotFoundError(f"Full data not found: {data_path}")

        data = torch.load(data_path)
        self.data = _split_data_by_index(data, split, index_key='index')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for DataLoader."""
    hidden_states = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return hidden_states, labels


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python probe_dataset.py <probe_type> [layer]")
        sys.exit(1)

    probe_type = sys.argv[1]
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    try:
        dataset = ProbeDataset(probe_type, layer=layer, split='train')
        print(f"\nDataset info:")
        print(f"  Samples: {len(dataset)}")
        print(f"  Hidden dim: {dataset.get_hidden_dim()}")
        print(f"  Num classes: {dataset.get_num_classes()}")

        if len(dataset) > 0:
            hidden, label = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Hidden state shape: {hidden.shape}")
            print(f"  Label: {label}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun extract_probe_hidden_states.py first to generate probe data.")
