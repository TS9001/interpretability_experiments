#!/usr/bin/env python3
"""Download GSM8K dataset from HuggingFace."""
from datasets import load_dataset
from pathlib import Path
import json

def main():
    output_dir = Path(__file__).parent / "resources" / "gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")

    # Save train split
    train_path = output_dir / "train.jsonl"
    print(f"Saving train split ({len(dataset['train'])} examples) to {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in dataset['train']:
            f.write(json.dumps(example) + '\n')

    # Save test split
    test_path = output_dir / "test.jsonl"
    print(f"Saving test split ({len(dataset['test'])} examples) to {test_path}")
    with open(test_path, 'w', encoding='utf-8') as f:
        for example in dataset['test']:
            f.write(json.dumps(example) + '\n')

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    print(f"  Total: {len(dataset['train']) + len(dataset['test'])} examples")

if __name__ == "__main__":
    main()
