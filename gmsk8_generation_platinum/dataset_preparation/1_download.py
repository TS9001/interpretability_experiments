#!/usr/bin/env python3
"""
Download GSM8K-Platinum dataset from HuggingFace.

Usage:
    python 1_download.py
"""
import json
from pathlib import Path
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR / "resources"


def main():
    """Download GSM8K-Platinum dataset and save to JSONL files."""
    print("=" * 60)
    print("GSM8K-Platinum Dataset Download")
    print("=" * 60)

    output_dir = RESOURCES_DIR / "gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading GSM8K-Platinum from HuggingFace...")
    dataset = load_dataset(
        "madrylab/gsm8k-platinum",
        "main",
        cache_dir=RESOURCES_DIR / "cache"
    )

    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")

    # Save train split
    if 'train' in dataset:
        train_path = output_dir / "train.jsonl"
        print(f"\nSaving train split ({len(dataset['train'])} examples)...")
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in dataset['train']:
                f.write(json.dumps(dict(example), ensure_ascii=False) + '\n')
        print(f"  Saved to: {train_path}")

    # Save test split
    if 'test' in dataset:
        test_path = output_dir / "test.jsonl"
        print(f"\nSaving test split ({len(dataset['test'])} examples)...")
        with open(test_path, 'w', encoding='utf-8') as f:
            for example in dataset['test']:
                f.write(json.dumps(dict(example), ensure_ascii=False) + '\n')
        print(f"  Saved to: {test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    total = sum(len(dataset[split]) for split in dataset.keys())
    print(f"Total examples: {total:,}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
