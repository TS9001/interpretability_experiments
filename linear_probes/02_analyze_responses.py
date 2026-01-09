#!/usr/bin/env python3
"""
Step 2: Analyze generated responses.

Uses the proper regexp_utils.py from dataset_preparation to accurately
extract and analyze arithmetic operations in generated responses.

Loads ground truth from the original tokenized dataset files.
Merges tokens from separate tokens file and uses tokenizer to convert to strings.

Usage:
    python 02_analyze_responses.py responses/Qwen2.5-Math-1.5B/train_responses.json
    python 02_analyze_responses.py responses/Qwen2.5-Math-1.5B/train_responses.json --output analyzed/train.json
"""
import re
import sys
from pathlib import Path
from typing import Optional, Union

import typer
from transformers import AutoTokenizer

from utils.args import InputFile, OutputDir
from utils.data import load_json, load_jsonl, save_json, save_metadata, zip_files
from utils.logging import log, print_header, print_config, print_summary, create_progress
from utils.numbers import parse_number, extract_final_answer
from utils.tokenization import build_char_to_token_map

import os

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Path to original dataset (contains ground truth) - supports env var override
DATA_DIR = Path(os.environ.get(
    'PROBE_DATA_DIR',
    PROJECT_ROOT / "resources" / "gsm8k" / "matching"
))

# Default model name for tokenizer
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

# Add dataset_preparation to path for regexp_utils
sys.path.insert(0, str(PROJECT_ROOT / "gmsk8_generation_platinum" / "dataset_preparation"))
from regexp_utils import (
    normalize_operator,
    NUMBER,
    UNIT_BEFORE,
    UNIT_AFTER,
    SPACED_OPERATORS,
    ASYMMETRIC_OPS,
)

app = typer.Typer(add_completion=False)

# Correctness threshold for floating point comparison
CORRECTNESS_THRESHOLD = 0.01

# Multi-pattern approach reusing regexp_utils constants
# Each pattern captures: (operand1)(operator)(operand2) = (result)
RESULT_PATTERNS = [
    # Spaced with optional units before operator, optional currency after
    re.compile(
        rf'({NUMBER}){UNIT_BEFORE}\s+({SPACED_OPERATORS})\s+{UNIT_AFTER}({NUMBER})\s*=\s*({NUMBER})'
    ),
    # Compact (no spaces) - for expressions like "5+3=8"
    re.compile(rf'({NUMBER})([+\-*/×÷\u2013\u2014])({NUMBER})\s*=\s*({NUMBER})'),
    # Compact X multiplication - for "5x3=15" or "5X3=15"
    re.compile(rf'({NUMBER})([xX×])({NUMBER})\s*=\s*({NUMBER})'),
    # Asymmetric: space before operator only - "45 -40=5"
    re.compile(rf'({NUMBER})\s+({ASYMMETRIC_OPS})({NUMBER})\s*=\s*({NUMBER})'),
    # Asymmetric: space after operator only - "45- 40=5"
    re.compile(rf'({NUMBER})({ASYMMETRIC_OPS})\s+({NUMBER})\s*=\s*({NUMBER})'),
]


def extract_operations(text: str) -> list[dict]:
    """
    Extract arithmetic operations from response text using multi-pattern matching.

    Uses the same philosophy as regexp_utils.py:
    - Multiple patterns for different formats (spaced, compact, asymmetric)
    - Deduplication across patterns
    - Support for units, currency symbols, unicode operators

    Returns list of operation dicts with computed correctness.
    """
    operations = []
    seen = set()  # (num1, norm_op, num2, result) for deduplication

    for pattern in RESULT_PATTERNS:
        for match in pattern.finditer(text):
            num1_str, op, num2_str, result_str = match.groups()
            try:
                num1 = parse_number(num1_str)
                num2 = parse_number(num2_str)
                result = parse_number(result_str)
                norm_op = normalize_operator(op)

                # Deduplicate across patterns
                key = (num1, norm_op, num2, result)
                if key in seen:
                    continue
                seen.add(key)

                # Map operator to type
                op_type = {'+': 'add', '-': 'sub', '*': 'mult', '/': 'div'}.get(norm_op, norm_op)

                # Compute expected result
                if norm_op == '+':
                    expected = num1 + num2
                elif norm_op == '-':
                    expected = num1 - num2
                elif norm_op == '*':
                    expected = num1 * num2
                elif norm_op == '/':
                    expected = num1 / num2 if num2 != 0 else None
                else:
                    expected = None

                is_correct = expected is not None and abs(result - expected) < CORRECTNESS_THRESHOLD

                operations.append({
                    'operand1': num1,
                    'operator': op_type,
                    'operand2': num2,
                    'result': result,
                    'expected': expected,
                    'is_correct': is_correct,
                    'text_span': match.group(0),
                    'char_start': match.start(),
                    'char_end': match.end(),
                })
            except (ValueError, ZeroDivisionError):
                continue

    # Sort by position in text for consistent ordering
    operations.sort(key=lambda x: x['char_start'])
    return operations


def normalize_token(token: str) -> str:
    """Normalize token for comparison (remove Ġ prefix, Ċ newlines, etc.)."""
    # GPT-style tokenizers use Ġ for space prefix, Ċ for newline
    if token.startswith('Ġ'):
        return token[1:]
    if token.startswith('Ċ'):
        return token[1:]
    return token


def format_number_for_search(num: float) -> str:
    """Format number as string for token search."""
    if num == int(num):
        return str(int(num))
    return str(num)


def find_number_in_tokens(tokens: list[str], num: float, start_pos: int = 0) -> int:
    """
    Find a number in tokens starting from start_pos.
    Returns the token position where the number starts, or -1 if not found.

    Numbers can span multiple tokens (e.g., "48" might be ["4", "8"]).
    """
    target = format_number_for_search(num)

    # First try to find as a single token (exact match)
    for i in range(start_pos, len(tokens)):
        normalized = normalize_token(tokens[i])
        if normalized == target:
            return i

    # Check if number is embedded in a larger token (but non-empty match)
    for i in range(start_pos, len(tokens)):
        normalized = normalize_token(tokens[i])
        # Must have actual content and the target must be a significant match
        if normalized and target in normalized and len(target) > 0:
            return i

    # Try to find as consecutive tokens (for multi-digit numbers)
    for i in range(start_pos, len(tokens)):
        # Skip tokens that can't start a number
        first_normalized = normalize_token(tokens[i])
        if not first_normalized or not (first_normalized[0].isdigit() or first_normalized[0] in '.-'):
            continue

        # Build string from consecutive tokens
        combined = ""
        for j in range(i, min(i + len(target) + 2, len(tokens))):
            combined += normalize_token(tokens[j])
            if combined == target:
                return i
            # Also check if we've built the target as a prefix (before more digits)
            if combined.startswith(target) and (len(combined) == len(target) or not combined[len(target)].isdigit()):
                return i

    return -1


def find_operator_in_tokens(tokens: list[str], op_type: str, start_pos: int = 0) -> int:
    """
    Find an operator in tokens starting from start_pos.
    Returns the token position, or -1 if not found.
    """
    # Map operator type to possible token values
    operator_chars = {
        'add': ['+'],
        'sub': ['-', '−', '–', '—'],
        'mult': ['*', '×', 'x', 'X', '·'],
        'div': ['/', '÷'],
    }

    chars = operator_chars.get(op_type, [])

    for i in range(start_pos, len(tokens)):
        normalized = normalize_token(tokens[i])
        if not normalized:
            continue
        # Check if any operator char is in the token
        for char in chars:
            if char in normalized:
                return i

    return -1


def find_token_positions_sequential(
    operations: list[dict],
    tokens: list[str],
    text: str,
) -> list[dict]:
    """
    Find token positions for operations by searching left-to-right.

    For each operation, search for operand1, operator, operand2, result in order.
    Uses char_start from each operation to find the starting token position,
    avoiding false matches in echoed question text.

    If an element is not found, its position is set to -1.
    """
    if not operations or not tokens:
        return operations

    # Build char-to-token map for accurate position conversion
    char_to_token = build_char_to_token_map(tokens, text)

    for op in operations:
        operand1 = op.get('operand1')
        operator = op.get('operator')
        operand2 = op.get('operand2')
        result = op.get('result')

        # Use char_start to find the starting token position for this operation
        # This ensures we search in the actual operation, not in echoed question text
        char_start = op.get('char_start', 0)
        start_token = char_to_token.get(char_start, 0)

        # Search for operand1 starting from the operation's character position
        op1_pos = find_number_in_tokens(tokens, operand1, start_token) if operand1 is not None else -1
        op['operand1_positions'] = [op1_pos]

        # Search for operator (start from after operand1 if found)
        search_start = op1_pos + 1 if op1_pos >= 0 else start_token
        operator_pos = find_operator_in_tokens(tokens, operator, search_start) if operator else -1
        op['operator_positions'] = [operator_pos]

        # Search for operand2 (start from after operator if found)
        search_start = operator_pos + 1 if operator_pos >= 0 else search_start
        op2_pos = find_number_in_tokens(tokens, operand2, search_start) if operand2 is not None else -1
        op['operand2_positions'] = [op2_pos]

        # Search for result (start from after operand2 if found)
        search_start = op2_pos + 1 if op2_pos >= 0 else search_start
        result_pos = find_number_in_tokens(tokens, result, search_start) if result is not None else -1
        op['result_positions'] = [result_pos]

    return operations


def analyze_response(
    response: Union[dict, str],
    ground_truth_final_result: Optional[float],
    ground_truth_operations: list[dict],
) -> dict:
    """Analyze a single response."""
    # Handle both string and dict formats
    if isinstance(response, str):
        text = response
        tokens = []
        token_ids = []
    else:
        text = response['text']
        tokens = response.get('tokens', [])
        token_ids = response.get('token_ids', [])

    # Extract final answer
    final_answer = extract_final_answer(text)

    # Check final answer correctness against ground truth
    final_correct = None
    if final_answer is not None and ground_truth_final_result is not None:
        final_correct = abs(final_answer - ground_truth_final_result) < CORRECTNESS_THRESHOLD

    # Extract operations from generated text
    operations = extract_operations(text)

    # Add token positions using sequential left-to-right search
    # Pass text to use char_start for accurate position finding
    operations = find_token_positions_sequential(operations, tokens, text)

    # Count correct/incorrect
    correct_ops = sum(1 for op in operations if op.get('is_correct', False))
    incorrect_ops = len(operations) - correct_ops

    return {
        'text': text,
        'final_answer': final_answer,
        'final_correct': final_correct,
        'operations': operations,
        'num_operations': len(operations),
        'correct_operations': correct_ops,
        'incorrect_operations': incorrect_ops,
        'has_errors': incorrect_ops > 0,
    }


def analyze_example(example: dict, ground_truth: dict) -> dict:
    """Analyze all responses for an example, using ground truth from source data."""
    # Get ground truth from the source tokenized dataset
    ground_truth_final_result = ground_truth.get('final_result')
    ground_truth_operations = ground_truth.get('operations', [])
    ground_truth_answer = ground_truth.get('answer_clean', '')
    operation_sequence = ground_truth.get('operation_sequence', [])
    operations_by_type = ground_truth.get('operations_by_type', {})

    analyzed_responses = []
    for resp in example.get('responses', []):
        analyzed = analyze_response(resp, ground_truth_final_result, ground_truth_operations)
        analyzed_responses.append(analyzed)

    # Compute summary stats
    num_correct = sum(1 for r in analyzed_responses if r['final_correct'] is True)
    num_with_errors = sum(1 for r in analyzed_responses if r['has_errors'])
    total_ops_found = sum(r['num_operations'] for r in analyzed_responses)
    total_correct_ops = sum(r['correct_operations'] for r in analyzed_responses)
    total_incorrect_ops = sum(r['incorrect_operations'] for r in analyzed_responses)

    return {
        'index': example['index'],
        'question': example['question'],
        'ground_truth_answer': ground_truth_answer,
        'ground_truth_operations': ground_truth_operations,
        'ground_truth_final_result': ground_truth_final_result,
        'operation_sequence': operation_sequence,
        'operations_by_type': operations_by_type,
        'responses': analyzed_responses,
        'summary': {
            'num_responses': len(analyzed_responses),
            'num_correct_final': num_correct,
            'num_with_operation_errors': num_with_errors,
            'accuracy': num_correct / len(analyzed_responses) if analyzed_responses else 0,
            'ground_truth_ops': len(ground_truth_operations),
            'total_ops_found': total_ops_found,
            'correct_ops': total_correct_ops,
            'incorrect_ops': total_incorrect_ops,
        }
    }


def load_ground_truth(split_name: str) -> dict:
    """Load ground truth data from the original dataset (tokenized or plain)."""
    # Try different file patterns
    gt_path = None
    for pattern in [f"{split_name}_tokenized.jsonl", f"{split_name}.jsonl"]:
        candidate = DATA_DIR / pattern
        if candidate.exists():
            gt_path = candidate
            break

    if not gt_path:
        log.warning(f"Ground truth file not found for {split_name} in {DATA_DIR}")
        return {}

    ground_truth = {}
    data = load_jsonl(gt_path)
    for entry in data:
        ground_truth[entry['index']] = entry

    log.info(f"Loaded ground truth for {len(ground_truth)} examples from {gt_path.name}")
    return ground_truth


def get_split_from_filename(filename: str) -> str:
    """Extract split name (train/test) from filename."""
    name = Path(filename).stem.lower()
    if 'train' in name:
        return 'train'
    elif 'test' in name:
        return 'test'
    return 'train'  # default


def load_tokens_file(responses_path: Path) -> dict:
    """
    Load tokens from the corresponding tokens file.

    The tokens file has the same name but with _tokens instead of _responses.
    Returns a dict mapping index -> list of token_ids lists (one per response).
    """
    # Convert train_responses.json -> train_tokens.json
    tokens_filename = responses_path.stem.replace('_responses', '_tokens') + responses_path.suffix
    tokens_path = responses_path.parent / tokens_filename

    if not tokens_path.exists():
        log.warning(f"Tokens file not found: {tokens_path}")
        return {}

    token_data = load_json(tokens_path)

    # Build index -> token_ids mapping
    tokens_by_index = {}
    for entry in token_data:
        tokens_by_index[entry['index']] = entry['responses']  # list of token_id lists

    log.info(f"Loaded tokens for {len(tokens_by_index)} examples from {tokens_path.name}")
    return tokens_by_index


def merge_responses_with_tokens(
    examples: list[dict],
    tokens_by_index: dict,
    tokenizer
) -> list[dict]:
    """
    Merge response texts with their tokens.

    Converts token_ids to token strings using the tokenizer.
    Returns examples with responses as dicts containing text, tokens, token_ids.
    """
    merged = []

    for example in examples:
        idx = example['index']
        token_ids_list = tokens_by_index.get(idx, [])

        responses = []
        for i, resp in enumerate(example.get('responses', [])):
            # Handle both string and dict response formats
            if isinstance(resp, str):
                text = resp
            else:
                text = resp.get('text', '')

            # Get token_ids for this response
            if i < len(token_ids_list):
                token_ids = token_ids_list[i]
                # Convert token_ids to token strings
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
            else:
                token_ids = []
                tokens = []

            responses.append({
                'text': text,
                'tokens': tokens,
                'token_ids': token_ids,
            })

        merged.append({
            'index': idx,
            'question': example['question'],
            'responses': responses,
        })

    return merged


@app.command()
def main(
    input_file: InputFile,
    output_dir: OutputDir = None,
    zip_output: bool = typer.Option(False, "--zip", help="Create zip archive for download"),
):
    """Analyze generated responses to extract operations and check correctness."""
    input_path = Path(input_file)
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    if output_dir:
        output_path = Path(output_dir) / f"{input_path.stem}_analyzed.json"
    else:
        output_path = input_path.parent / f"{input_path.stem}_analyzed.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_header("Response Analysis", "Step 2")
    print_config("Configuration", {
        'input': str(input_path),
        'output': str(output_path),
    })

    # Load tokenizer for converting token_ids to token strings
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load ground truth from original tokenized dataset
    split_name = get_split_from_filename(input_path.name)
    ground_truth = load_ground_truth(split_name)

    log.info("Loading responses...")
    if input_path.suffix == '.json':
        examples = load_json(input_path)
    else:
        examples = load_jsonl(input_path)
    log.info(f"Loaded {len(examples)} examples")

    # Load and merge tokens
    tokens_by_index = load_tokens_file(input_path)
    if tokens_by_index:
        examples = merge_responses_with_tokens(examples, tokens_by_index, tokenizer)
        log.info("Merged tokens with responses")

    results = []
    missing_ground_truth = 0

    with create_progress() as progress:
        task = progress.add_task("Analyzing responses", total=len(examples))
        for example in examples:
            idx = example['index']
            gt = ground_truth.get(idx, {})
            if not gt:
                missing_ground_truth += 1

            analyzed = analyze_example(example, gt)
            results.append(analyzed)
            progress.advance(task)

    if missing_ground_truth > 0:
        log.warning(f"{missing_ground_truth} examples missing ground truth data")

    save_json(results, output_path)
    log.success(f"Saved results to {output_path}")

    # Compute summary stats
    total_responses = sum(len(r['responses']) for r in results)
    avg_accuracy = sum(r['summary']['accuracy'] for r in results) / len(results) if results else 0
    avg_errors = sum(r['summary']['num_with_operation_errors'] for r in results) / len(results) if results else 0

    # Count operations: ground truth vs found
    total_gt_ops = sum(len(r.get('ground_truth_operations', [])) for r in results)
    total_found_ops = sum(
        sum(len(resp['operations']) for resp in r['responses'])
        for r in results
    )
    avg_gt_ops_per_example = total_gt_ops / len(results) if results else 0
    avg_found_ops_per_response = total_found_ops / total_responses if total_responses else 0

    print_summary("Analysis Summary", {
        'examples_analyzed': len(results),
        'total_responses': total_responses,
        'average_accuracy': f"{avg_accuracy:.1%}",
        'average_responses_with_errors': f"{avg_errors:.1f}",
        'ground_truth_operations': total_gt_ops,
        'operations_found': total_found_ops,
        'avg_gt_ops_per_example': f"{avg_gt_ops_per_example:.1f}",
        'avg_found_ops_per_response': f"{avg_found_ops_per_response:.1f}",
    })

    save_metadata(
        output_path.parent,
        config={
            'input_file': str(input_path),
            'output_file': str(output_path),
        },
        stats={
            'num_examples': len(results),
            'total_responses': total_responses,
            'avg_accuracy': avg_accuracy,
            'missing_ground_truth': missing_ground_truth,
            'ground_truth_operations': total_gt_ops,
            'operations_found': total_found_ops,
        },
    )

    # Create zip archive for download
    if zip_output:
        zip_path = output_path.parent / f"{output_path.stem}.zip"
        zip_files([output_path], zip_path)
        log.success(f"Zip archive: {zip_path}")


if __name__ == '__main__':
    app()
