#!/usr/bin/env python3
"""
Download GSM8K datasets from HuggingFace.

- Train: from openai/gsm8k (GSM8K-Platinum doesn't have train split)
- Test: from madrylab/gsm8k-platinum (cleaner annotations)

Usage:
    python 1_download.py
"""
import json
from pathlib import Path
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR / "resources"


def main():
    """Download GSM8K datasets and save to JSONL files."""
    print("=" * 60)
    print("GSM8K Dataset Download")
    print("=" * 60)

    output_dir = RESOURCES_DIR / "gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download train from original GSM8K (Platinum doesn't have train)
    print("\nDownloading train split from openai/gsm8k...")
    gsm8k = load_dataset(
        "openai/gsm8k",
        "main",
        cache_dir=RESOURCES_DIR / "cache"
    )

    train_path = output_dir / "train.jsonl"
    print(f"Saving train split ({len(gsm8k['train'])} examples)...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in gsm8k['train']:
            f.write(json.dumps(dict(example), ensure_ascii=False) + '\n')
    print(f"  Saved to: {train_path}")

    # Download test from GSM8K-Platinum (cleaner annotations)
    print("\nDownloading test split from madrylab/gsm8k-platinum...")
    platinum = load_dataset(
        "madrylab/gsm8k-platinum",
        "main",
        cache_dir=RESOURCES_DIR / "cache"
    )

    test_path = output_dir / "test.jsonl"
    print(f"Saving test split ({len(platinum['test'])} examples)...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for example in platinum['test']:
            f.write(json.dumps(dict(example), ensure_ascii=False) + '\n')
    print(f"  Saved to: {test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Train: {len(gsm8k['train']):,} examples (from openai/gsm8k)")
    print(f"Test: {len(platinum['test']):,} examples (from gsm8k-platinum)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
