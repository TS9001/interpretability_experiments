"""
Regex utilities for detecting arithmetic operations in GSM8K dataset.
"""
import re

# Number pattern - matches integers and decimals, with optional commas
NUMBER = r'\d[\d,]*(?:\.\d+)?'

# Units/currency that can appear between number and operator
# Allows: $, %, currency symbols, short units (up to 15 chars), but NOT long arbitrary text
UNIT = r'(?:[$%€£¥]|[a-zA-Z]{1,15}(?:/[a-zA-Z]{1,10})?)?'

# Whitespace (including none)
WS = r'\s*'

# Operators: +, -, *, /, x, X, ×, ÷
# Note: 'x' and 'X' are used for multiplication in text like "5x5" or "5 x 5"
OPERATORS = r'[+\-*/xX×÷·]'

# Main operand pattern:
# number (optional unit) whitespace operator whitespace (optional unit) number
# This captures: "5+3", "5 + 3", "$5 + $3", "5x5", "5 X 5", "10 miles * 2"
OPERAND_PATTERN = re.compile(
    rf'({NUMBER}){WS}{UNIT}{WS}({OPERATORS}){WS}{UNIT}{WS}({NUMBER})',
    re.IGNORECASE
)

# Pattern to extract content inside <<>> tags
BRACKET_CONTENT = re.compile(r'<<([^>]+)>>')

# Pattern to find all <<...>> tags (to remove them for "outside" analysis)
BRACKET_TAGS = re.compile(r'<<[^>]+>>')


def normalize_operator(op: str) -> str:
    """Normalize operator to standard form: +, -, *, /"""
    if op in ['+']:
        return '+'
    elif op in ['-']:
        return '-'
    elif op in ['*', 'x', 'X', '×', '·']:
        return '*'
    elif op in ['/', '÷']:
        return '/'
    return op


def find_operations(text: str) -> list[tuple[str, str, str, str]]:
    """
    Find all arithmetic operations in text.

    Returns list of tuples: (num1, operator, num2, normalized_operator)
    """
    matches = OPERAND_PATTERN.findall(text)
    results = []
    for num1, op, num2 in matches:
        norm_op = normalize_operator(op)
        results.append((num1, op, num2, norm_op))
    return results


def find_operations_inside_brackets(answer: str) -> list[tuple[str, str, str, str]]:
    """Find operations inside <<>> brackets."""
    bracket_contents = BRACKET_CONTENT.findall(answer)
    all_ops = []
    for content in bracket_contents:
        ops = find_operations(content)
        all_ops.extend(ops)
    return all_ops


def find_operations_outside_brackets(answer: str) -> list[tuple[str, str, str, str]]:
    """Find operations outside <<>> brackets."""
    # Remove all <<...>> content
    text_without_brackets = BRACKET_TAGS.sub('', answer)
    return find_operations(text_without_brackets)


def count_operations_by_type(operations: list[tuple[str, str, str, str]]) -> dict[str, int]:
    """Count operations by their normalized type."""
    counts = {'+': 0, '-': 0, '*': 0, '/': 0}
    for _, _, _, norm_op in operations:
        if norm_op in counts:
            counts[norm_op] += 1
    return counts


if __name__ == "__main__":
    # Test the patterns
    test_cases = [
        "5+3",
        "5 + 3",
        "5*3",
        "5 x 3",
        "5X3",
        "10 - 5",
        "20/4",
        "20 / 4",
        "$5 + $3",
        "5 miles * 2",
        "30 years - 20 years",
        "2 trains * 80 miles",
        "He eats 3*7 = 21 eggs",
        "<<16-3-4=9>>9",
        "The result is 5+3=<<5+3=8>>8 apples.",
    ]

    for test in test_cases:
        ops = find_operations(test)
        print(f"{test!r} -> {ops}")
