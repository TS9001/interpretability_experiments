#!/usr/bin/env python3
"""
Enhance matching JSONL files with detailed operation information.

Extracts operand1, operator, operand2, and result for each operation in order.

Usage:
    python 3_enhance.py

Input:  resources/gsm8k_split/matching/train.jsonl, test.jsonl
Output: resources/gsm8k_split/matching/train_enhanced.jsonl, test_enhanced.jsonl
"""
import json
import re
from pathlib import Path


# Pattern to extract content inside <<>> tags with their result
# Matches: <<expression=result>>
BRACKET_EXPR = re.compile(r'<<([^=]+)=([^>]+)>>')

# Number pattern - matches integers and decimals, with optional commas (NO leading minus)
# Also handles numbers starting with decimal point like .8, .5
UNSIGNED_NUMBER = r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)'


def normalize_operator(op: str) -> str:
    """Normalize operator to standard form: add, sub, mult, div"""
    if op == '+':
        return 'add'
    elif op == '-':
        return 'sub'
    elif op in ['*', 'x', 'X', '×']:
        return 'mult'
    elif op in ['/', '÷']:
        return 'div'
    return op


def parse_number(s: str) -> float:
    """Parse a number string, handling commas."""
    return float(s.replace(',', ''))


def tokenize_expression(expression: str) -> list[tuple[str, str]]:
    """
    Tokenize an expression into (type, value) pairs.
    Types: 'num' for numbers, 'op' for operators.
    """
    tokens = []
    pos = 0
    expr = expression.strip()

    while pos < len(expr):
        # Skip whitespace
        while pos < len(expr) and expr[pos] in ' \t':
            pos += 1
        if pos >= len(expr):
            break

        # Check for number (possibly with leading minus at start of expression)
        if pos == 0 and expr[pos] == '-':
            num_match = re.match(rf'-(?:\d[\d,]*(?:\.\d+)?|\.\d+)', expr[pos:])
            if num_match:
                tokens.append(('num', num_match.group()))
                pos += len(num_match.group())
                continue

        # Check for unsigned number
        num_match = re.match(UNSIGNED_NUMBER, expr[pos:])
        if num_match:
            tokens.append(('num', num_match.group()))
            pos += len(num_match.group())
            continue

        # Check for operator
        if expr[pos] in '+-*/xX×÷':
            tokens.append(('op', expr[pos]))
            pos += 1
            continue

        # Skip unknown characters
        pos += 1

    return tokens


def extract_operations_from_expression(expression: str, final_result_str: str) -> list[dict]:
    """
    Extract all operations from an expression like '100-50-30-15' with result '5'.
    Returns list of dicts with operand1, operand2, operator, result.
    """
    operations = []
    tokens = tokenize_expression(expression)

    # Build list of numbers and operators
    numbers = []
    operators = []
    expect_number = True

    for tok_type, tok_val in tokens:
        if tok_type == 'num' and expect_number:
            numbers.append(parse_number(tok_val))
            expect_number = False
        elif tok_type == 'op' and not expect_number:
            operators.append(tok_val)
            expect_number = True

    if len(numbers) < 2 or len(operators) < 1:
        return operations

    # Parse the actual final result from the bracket
    try:
        actual_final_result = parse_number(final_result_str.strip())
    except ValueError:
        actual_final_result = None

    def format_num(n):
        if isinstance(n, float) and n == int(n):
            return int(n)
        if isinstance(n, float):
            return round(n, 6)
        return n

    # Process left-to-right
    current_result = numbers[0]
    total_ops = min(len(operators), len(numbers) - 1)

    for i, op in enumerate(operators):
        if i + 1 >= len(numbers):
            break

        operand1 = current_result
        operand2 = numbers[i + 1]
        norm_op = normalize_operator(op)

        # Calculate result
        if norm_op == 'add':
            result = operand1 + operand2
        elif norm_op == 'sub':
            result = operand1 - operand2
        elif norm_op == 'mult':
            result = operand1 * operand2
        elif norm_op == 'div':
            result = operand1 / operand2 if operand2 != 0 else 0
        else:
            result = 0

        # For the last operation, use the actual result from the bracket
        is_last_op = (i == total_ops - 1)
        if is_last_op and actual_final_result is not None:
            result = actual_final_result

        operations.append({
            'operand1': format_num(operand1),
            'operator': norm_op,
            'operand2': format_num(operand2),
            'result': format_num(result)
        })

        current_result = result

    return operations


def extract_operations_from_answer(answer: str) -> list[dict]:
    """Extract all operations from an answer string in order."""
    all_operations = []

    for match in BRACKET_EXPR.finditer(answer):
        expression = match.group(1)
        final_result = match.group(2)
        ops = extract_operations_from_expression(expression, final_result)
        all_operations.extend(ops)

    return all_operations


def count_operations_by_type(operations: list[dict]) -> dict[str, int]:
    """Count operations by type."""
    counts = {'add': 0, 'sub': 0, 'mult': 0, 'div': 0}
    for op in operations:
        op_type = op['operator']
        if op_type in counts:
            counts[op_type] += 1
    return counts


def extract_final_result(answer: str):
    """Extract the final result from #### line."""
    match = re.search(r'####\s*(.+)', answer)
    if match:
        result_str = match.group(1).strip()
        try:
            result_str = result_str.replace(',', '')
            if '.' in result_str:
                return float(result_str)
            return int(result_str)
        except ValueError:
            return result_str
    return None


def mark_intermediate_operations(operations: list[dict]) -> None:
    """
    Mark operations as intermediate if their result feeds into the next operation.

    An operation is intermediate if its result equals the next operation's operand1.
    This indicates chained calculations where the intermediate result may not
    appear explicitly in the text.
    """
    for i, op in enumerate(operations):
        if i < len(operations) - 1:
            next_op = operations[i + 1]
            op['is_intermediate'] = (op['result'] == next_op['operand1'])
        else:
            op['is_intermediate'] = False


def enhance_jsonl(input_path: Path, output_path: Path):
    """Read a JSONL file and enhance each entry with detailed operation info."""
    enhanced_entries = []

    with open(input_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())

            # Extract detailed operations
            operations = extract_operations_from_answer(entry['answer'])
            mark_intermediate_operations(operations)
            operation_sequence = [op['operator'] for op in operations]
            final_result = extract_final_result(entry['answer'])

            # Build enhanced entry
            enhanced = {
                'index': entry['index'],
                'question': entry['question'],
                'answer': entry['answer'],
                'final_result': final_result,
                'operations': operations,
                'operation_sequence': operation_sequence,
                'operations_by_type': count_operations_by_type(operations),
                'total_operations': len(operations)
            }

            enhanced_entries.append(enhanced)

    # Write output
    with open(output_path, 'w') as f:
        for entry in enhanced_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"  Processed {len(enhanced_entries)} entries -> {output_path.name}")


def main():
    base_dir = Path(__file__).parent / 'resources' / 'gsm8k_split' / 'matching'

    print("=" * 60)
    print("Enhancing matching examples with detailed operations")
    print("=" * 60)

    # Process train and test
    for split in ['train', 'test']:
        input_path = base_dir / f'{split}.jsonl'
        output_path = base_dir / f'{split}_enhanced.jsonl'

        if input_path.exists():
            print(f"\nProcessing {split}...")
            enhance_jsonl(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping...")

    print("\nDone!")


if __name__ == '__main__':
    main()
