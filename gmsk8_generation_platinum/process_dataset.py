#!/usr/bin/env python3
"""
Process GSM8K dataset to count operations inside and outside <<>> tags.
Outputs non-matching examples and final matching percentage.
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
    """
    Process a single example to count operations inside and outside brackets.

    Returns dict with:
    - inside_ops: list of operations found inside <<>> tags
    - outside_ops: list of operations found outside <<>> tags
    - inside_counts: dict of operation counts inside brackets
    - outside_counts: dict of operation counts outside brackets
    - matches: bool indicating if counts match
    """
    answer = example.get('answer', '')

    inside_ops = find_operations_inside_brackets(answer)
    outside_ops = find_operations_outside_brackets(answer)

    inside_counts = count_operations_by_type(inside_ops)
    outside_counts = count_operations_by_type(outside_ops)

    # Check if counts match
    matches = inside_counts == outside_counts

    return {
        'inside_ops': inside_ops,
        'outside_ops': outside_ops,
        'inside_counts': inside_counts,
        'outside_counts': outside_counts,
        'matches': matches,
    }


def format_ops_list(ops: list[tuple[str, str, str, str]]) -> str:
    """Format operations list for display."""
    if not ops:
        return "(none)"
    return ", ".join(f"{n1}{op}{n2}" for n1, op, n2, _ in ops)


def main():
    # Default dataset path
    default_path = Path(__file__).parent / "annotation" / "first_100_test_examples.jsonl"

    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        dataset_path = default_path

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples\n")

    # Process all examples
    non_matching = []
    total_inside_counts = {'+': 0, '-': 0, '*': 0, '/': 0}
    total_outside_counts = {'+': 0, '-': 0, '*': 0, '/': 0}

    for idx, example in enumerate(examples):
        result = process_example(example)

        # Accumulate total counts
        for op in total_inside_counts:
            total_inside_counts[op] += result['inside_counts'][op]
            total_outside_counts[op] += result['outside_counts'][op]

        if not result['matches']:
            non_matching.append({
                'index': idx,
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
                'inside_ops': result['inside_ops'],
                'outside_ops': result['outside_ops'],
                'inside_counts': result['inside_counts'],
                'outside_counts': result['outside_counts'],
            })

    # Print non-matching examples
    print("=" * 80)
    print("NON-MATCHING EXAMPLES")
    print("=" * 80)

    for item in non_matching:
        print(f"\n--- Example {item['index']} ---")
        print(f"Question: {item['question'][:100]}...")
        print(f"\nFULL ANSWER TEXT:")
        print(item['answer'])
        print(f"\nInside <<>>: {format_ops_list(item['inside_ops'])}")
        print(f"Outside <<>>: {format_ops_list(item['outside_ops'])}")
        print(f"Inside counts: {item['inside_counts']}")
        print(f"Outside counts: {item['outside_counts']}")
        print("-" * 40)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    matching_count = len(examples) - len(non_matching)
    percentage = (matching_count / len(examples)) * 100 if examples else 0

    print(f"\nTotal operations inside <<>> tags:")
    for op, count in total_inside_counts.items():
        print(f"  {op}: {count}")

    print(f"\nTotal operations outside <<>> tags:")
    for op, count in total_outside_counts.items():
        print(f"  {op}: {count}")

    print(f"\n" + "=" * 80)
    print(f"MATCHING PERCENTAGE: {matching_count}/{len(examples)} = {percentage:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
