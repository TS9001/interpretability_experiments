#!/usr/bin/env python3
"""
Split GSM8K dataset into matching and non-matching examples.
Matching = operations inside <<>> tags match operations outside <<>> tags.
"""
import json
import sys
from pathlib import Path

from regexp_utils import (
    find_operations_inside_brackets,
    find_operations_outside_brackets,
    count_operations_by_type,
)


def load_dataset(filepath: Path) -> list[dict]:
    """Load JSONL dataset."""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def process_example(example: dict) -> dict:
    """Process a single example and determine if it matches."""
    answer = example.get('answer', '')

    inside_ops = find_operations_inside_brackets(answer)
    outside_ops = find_operations_outside_brackets(answer)

    inside_counts = count_operations_by_type(inside_ops)
    outside_counts = count_operations_by_type(outside_ops)

    matches = inside_counts == outside_counts

    return {
        'matches': matches,
        'inside_ops': inside_ops,
        'outside_ops': outside_ops,
        'inside_counts': inside_counts,
        'outside_counts': outside_counts,
    }


def save_jsonl(examples: list[dict], filepath: Path):
    """Save examples to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "resources" / "gsm8k"
    output_dir = base_dir / "resources" / "gsm8k_split"

    matching_dir = output_dir / "matching"
    nonmatching_dir = output_dir / "nonmatching"

    matching_dir.mkdir(parents=True, exist_ok=True)
    nonmatching_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'test']

    total_matching = 0
    total_nonmatching = 0

    for split in splits:
        input_path = input_dir / f"{split}.jsonl"

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping...")
            continue

        print(f"\nProcessing {split} split...")
        examples = load_dataset(input_path)

        matching_examples = []
        nonmatching_examples = []

        for idx, example in enumerate(examples):
            result = process_example(example)

            # Add metadata to example
            enriched = {
                'index': idx,
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
                'inside_counts': result['inside_counts'],
                'outside_counts': result['outside_counts'],
            }

            if result['matches']:
                matching_examples.append(enriched)
            else:
                # Add debug info for non-matching
                enriched['inside_ops'] = [f"{n1}{op}{n2}" for n1, op, n2, _ in result['inside_ops']]
                enriched['outside_ops'] = [f"{n1}{op}{n2}" for n1, op, n2, _ in result['outside_ops']]
                nonmatching_examples.append(enriched)

        # Save to files
        save_jsonl(matching_examples, matching_dir / f"{split}.jsonl")
        save_jsonl(nonmatching_examples, nonmatching_dir / f"{split}.jsonl")

        print(f"  {split}: {len(matching_examples)} matching, {len(nonmatching_examples)} non-matching")

        total_matching += len(matching_examples)
        total_nonmatching += len(nonmatching_examples)

    # Summary
    total = total_matching + total_nonmatching
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Matching examples:     {total_matching:,} ({total_matching/total*100:.1f}%)")
    print(f"Non-matching examples: {total_nonmatching:,} ({total_nonmatching/total*100:.1f}%)")
    print(f"Total:                 {total:,}")
    print(f"\nOutput saved to:")
    print(f"  Matching:     {matching_dir}/")
    print(f"  Non-matching: {nonmatching_dir}/")


if __name__ == "__main__":
    main()
