#!/usr/bin/env python3
"""
Tokenize enhanced JSONL files and find token positions for operation components.

For each operation, finds token positions for operand1, operator, operand2, result.
Uses -1 if a component is not found in the text (keeps all operations regardless).

Usage:
    python 4_tokenize.py

Input:  resources/gsm8k_split/matching/train_enhanced.jsonl, test_enhanced.jsonl
Output: resources/gsm8k_split/matching/train_tokenized.jsonl, test_tokenized.jsonl
"""
import json
import re
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer


# Pattern to remove <<...>> content
BRACKET_PATTERN = re.compile(r'<<[^>]+>>')


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-Math-1.5B"):
    """Load tokenizer from HuggingFace."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def clean_answer(answer: str) -> str:
    """Remove <<...>> brackets from answer text."""
    return BRACKET_PATTERN.sub('', answer)


def get_operator_text(operator: str) -> list[str]:
    """Get possible text representations of an operator."""
    if operator == 'add':
        return ['+']
    elif operator == 'sub':
        return ['-', '–', '—']
    elif operator == 'mult':
        return ['*', 'x', 'X', '×']
    elif operator == 'div':
        return ['/', '÷']
    return [operator]


def format_number_variants(num) -> list[str]:
    """Get possible string representations of a number."""
    variants = []

    if isinstance(num, float):
        if num == int(num):
            variants.append(str(int(num)))
            variants.append(f"{int(num):,}")
        else:
            s = str(num)
            variants.append(s)
            if s.startswith('0.'):
                variants.append(s[1:])  # .5 instead of 0.5
    else:
        variants.append(str(num))
        variants.append(f"{num:,}")

    return variants


def find_token_positions(
    text: str,
    token_offsets: list[tuple[int, int]],
    value,
    search_start: int = 0
) -> tuple[list[int], int]:
    """
    Find token positions for a value in the text.

    Returns (token_indices, char_end_position).
    token_indices is [-1] if not found.
    """
    # Get all possible string representations
    if isinstance(value, str):
        # It's an operator
        variants = get_operator_text(value)
    else:
        # It's a number
        variants = format_number_variants(value)

    # Find first occurrence of any variant
    best_pos = -1
    best_variant = None

    for variant in variants:
        pos = text.find(variant, search_start)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_variant = variant

    if best_pos == -1:
        return [-1], search_start

    # Find which tokens cover this character span
    char_start = best_pos
    char_end = best_pos + len(best_variant)

    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_end > char_start and tok_start < char_end:
            token_indices.append(idx)

    if not token_indices:
        return [-1], search_start

    return token_indices, char_end


def tokenize_entry(entry: dict, tokenizer) -> dict:
    """
    Tokenize an entry and find token positions for all operation components.

    Returns entry with token positions added to operations.
    """
    answer_clean = clean_answer(entry['answer'])

    # Tokenize with offset mapping
    encoding = tokenizer(
        answer_clean,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    token_offsets = encoding['offset_mapping']

    # Process each operation
    operations_with_positions = []

    for op in entry['operations']:
        # Find positions for each component
        # Note: We search sequentially but don't require strict ordering
        # Just find first occurrence of each component

        op1_positions, _ = find_token_positions(
            answer_clean, token_offsets, op['operand1'], 0
        )

        operator_positions, _ = find_token_positions(
            answer_clean, token_offsets, op['operator'], 0
        )

        op2_positions, _ = find_token_positions(
            answer_clean, token_offsets, op['operand2'], 0
        )

        result_positions, _ = find_token_positions(
            answer_clean, token_offsets, op['result'], 0
        )

        operations_with_positions.append({
            'operand1': op['operand1'],
            'operand1_positions': op1_positions,
            'operator': op['operator'],
            'operator_positions': operator_positions,
            'operand2': op['operand2'],
            'operand2_positions': op2_positions,
            'result': op['result'],
            'result_positions': result_positions,
            'is_intermediate': op.get('is_intermediate', False),
        })

    return {
        'index': entry['index'],
        'question': entry['question'],
        'answer_clean': answer_clean,
        'tokens': tokens,
        'final_result': entry['final_result'],
        'operations': operations_with_positions,
        'operation_sequence': entry['operation_sequence'],
        'operations_by_type': entry['operations_by_type'],
        'total_operations': entry['total_operations'],
    }


def process_split(input_path: Path, output_path: Path, tokenizer):
    """Process a single split."""
    entries = []

    with open(input_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            tokenized = tokenize_entry(entry, tokenizer)
            entries.append(tokenized)

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    # Count statistics
    total_ops = 0
    found_ops = 0
    for entry in entries:
        for op in entry['operations']:
            total_ops += 1
            # Count as found if all components have positions (not -1)
            all_found = (
                op['operand1_positions'] != [-1] and
                op['operator_positions'] != [-1] and
                op['operand2_positions'] != [-1] and
                op['result_positions'] != [-1]
            )
            if all_found:
                found_ops += 1

    return len(entries), total_ops, found_ops


def main(tokenizer_name: str = "Qwen/Qwen2.5-Math-1.5B"):
    """Main function to tokenize all splits."""
    print("=" * 60)
    print("Tokenizing enhanced examples")
    print("=" * 60)

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)

    base_dir = Path(__file__).parent / 'resources' / 'gsm8k_split' / 'matching'

    for split in ['train', 'test']:
        input_path = base_dir / f'{split}_enhanced.jsonl'
        output_path = base_dir / f'{split}_tokenized.jsonl'

        if not input_path.exists():
            print(f"\nWarning: {input_path} not found, skipping...")
            continue

        print(f"\nProcessing {split}...")
        num_entries, total_ops, found_ops = process_split(
            input_path, output_path, tokenizer
        )

        pct = (found_ops / total_ops * 100) if total_ops > 0 else 0
        print(f"  Entries: {num_entries}")
        print(f"  Operations: {total_ops}")
        print(f"  Fully found: {found_ops} ({pct:.1f}%)")
        print(f"  Output: {output_path.name}")

    print("\nDone!")


if __name__ == '__main__':
    main()
