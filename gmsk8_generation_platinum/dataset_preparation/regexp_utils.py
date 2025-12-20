"""
Regex utilities for detecting arithmetic operations in GSM8K dataset.
"""
import re

# Number pattern - matches integers and decimals, with optional commas
NUMBER = r'\d[\d,]*(?:\.\d+)?'

# Pattern to extract content inside <<>> tags
BRACKET_CONTENT = re.compile(r'<<([^>]+)>>')

# Pattern to find all <<...>> tags (to remove them for "outside" analysis)
BRACKET_TAGS = re.compile(r'<<[^>]+>>')

# =============================================================================
# COMPACT PATTERN (for inside <<>> brackets)
# Matches: 5+3, 100/4, 20*35, 5-2 (no spaces, numbers only)
# Also handles unicode dashes: en-dash (–) and em-dash (—)
# =============================================================================
COMPACT_PATTERN = re.compile(
    rf'({NUMBER})([+\-*/×÷\u2013\u2014])({NUMBER})'
)

# =============================================================================
# SPACED PATTERN (for outside brackets - requires spaces around operator)
# This avoids matching rates like "20/hour" or "$30/hour"
# Matches: "5 + 3", "100 / 4", "25 x 12", "30 - 20"
# Also matches with units: "100 miles / 4 gallons", "5 years + 3 years"
# =============================================================================

# Units that can appear after a number (before operator)
# Handles: miles, years, hours, dollars, gallons, etc.
# Can be multi-word like "miles per gallon"
UNIT_BEFORE = r'(?:\s+[a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})?'  # up to 4 words

# Units/currency that can appear before a number (after operator)
UNIT_AFTER = r'(?:[$€£¥]\s*)?'  # optional currency symbol

# Operators with required spaces: +, -, *, /, x, X, ×, ÷
# The 'x' and 'X' are multiplication in text like "5 x 5"
# Also includes unicode dashes: en-dash (–) U+2013, em-dash (—) U+2014
SPACED_OPERATORS = r'[+\-*/xX×÷·\u2013\u2014]'

# Spaced pattern - requires at least one space around the operator
SPACED_PATTERN = re.compile(
    rf'({NUMBER}){UNIT_BEFORE}\s+({SPACED_OPERATORS})\s+{UNIT_AFTER}({NUMBER})',
    re.IGNORECASE
)

# =============================================================================
# COMPACT WITH X (for cases like "5x3" without spaces)
# =============================================================================
COMPACT_X_PATTERN = re.compile(
    rf'({NUMBER})([xX×])({NUMBER})'
)

# =============================================================================
# ASYMMETRIC SPACED PATTERN (for cases like "45 -40" or "2 +2")
# Space on one side only - catches operations missed by both SPACED and COMPACT
# =============================================================================
# Basic operators only (not x/X which could be variable names)
ASYMMETRIC_OPS = r'[+\-*/÷\u2013\u2014]'

# Pattern 1: space before operator, no space after (e.g., "45 -40")
ASYMMETRIC_PATTERN_1 = re.compile(
    rf'({NUMBER})\s+({ASYMMETRIC_OPS})({NUMBER})'
)

# Pattern 2: no space before operator, space after (e.g., "4- 2")
ASYMMETRIC_PATTERN_2 = re.compile(
    rf'({NUMBER})({ASYMMETRIC_OPS})\s+({NUMBER})'
)


def normalize_operator(op: str) -> str:
    """Normalize operator to standard form: +, -, *, /"""
    if op in ['+']:
        return '+'
    elif op in ['-', '\u2013', '\u2014']:  # minus, en-dash, em-dash
        return '-'
    elif op in ['*', 'x', 'X', '×', '·']:
        return '*'
    elif op in ['/', '÷']:
        return '/'
    return op


def find_operations_compact(text: str) -> list[tuple[str, str, str, str]]:
    """
    Find arithmetic operations in compact form (no spaces).
    Used for inside <<>> brackets.

    Returns list of tuples: (num1, operator, num2, normalized_operator)
    """
    results = []
    # Use finditer with overlapping search to catch chained operations like 16-3-4
    pos = 0
    while pos < len(text):
        match = COMPACT_PATTERN.search(text, pos)
        if not match:
            break
        num1, op, num2 = match.groups()
        norm_op = normalize_operator(op)
        results.append((num1, op, num2, norm_op))
        # Move position to start of num2 to catch chained operations (e.g., 16-3-4)
        # Find where num2 starts in the match
        pos = match.start() + len(num1) + len(op)
    return results


def find_operations_spaced(text: str) -> list[tuple[str, str, str, str]]:
    """
    Find arithmetic operations for outside <<>> brackets.
    Uses both spaced patterns (with units) and compact patterns (number op number).

    Compact pattern safely handles rates like "20/hour" because it requires
    both operands to be numbers.

    Returns list of tuples: (num1, operator, num2, normalized_operator)
    """
    results = []
    seen = set()  # Track unique operations to avoid duplicates

    # First find spaced operations with overlapping search (handles "16 - 3 - 4")
    pos = 0
    while pos < len(text):
        match = SPACED_PATTERN.search(text, pos)
        if not match:
            break
        num1, op, num2 = match.groups()
        norm_op = normalize_operator(op)
        key = (num1, norm_op, num2)
        if key not in seen:
            seen.add(key)
            results.append((num1, op, num2, norm_op))
        # Find where num2 starts in the matched text to catch chained operations
        # Match end is after num2, so we need to find where num2 starts
        match_text = match.group(0)
        num2_start_in_match = match_text.rfind(num2)
        pos = match.start() + num2_start_in_match

    # Also find compact operations (handles "3*7", "5+2", "100/4")
    # This is safe because it requires NUMBER on both sides
    pos = 0
    while pos < len(text):
        match = COMPACT_PATTERN.search(text, pos)
        if not match:
            break
        num1, op, num2 = match.groups()
        norm_op = normalize_operator(op)
        key = (num1, norm_op, num2)
        if key not in seen:
            seen.add(key)
            results.append((num1, op, num2, norm_op))
        pos = match.start() + len(num1) + len(op)

    # Also find compact "NxN" patterns (multiplication with x/X)
    x_matches = COMPACT_X_PATTERN.findall(text)
    for num1, op, num2 in x_matches:
        norm_op = normalize_operator(op)
        key = (num1, norm_op, num2)
        if key not in seen:
            seen.add(key)
            results.append((num1, op, num2, norm_op))

    # Also find asymmetric spaced patterns (space on one side only)
    # Pattern 1: "45 -40" (space before operator)
    for pattern in [ASYMMETRIC_PATTERN_1, ASYMMETRIC_PATTERN_2]:
        pos = 0
        while pos < len(text):
            match = pattern.search(text, pos)
            if not match:
                break
            num1, op, num2 = match.groups()
            norm_op = normalize_operator(op)
            key = (num1, norm_op, num2)
            if key not in seen:
                seen.add(key)
                results.append((num1, op, num2, norm_op))
            pos = match.start() + len(num1) + 1  # Move past first number

    return results


def find_operations_inside_brackets(answer: str) -> list[tuple[str, str, str, str]]:
    """Find operations inside <<>> brackets using compact pattern."""
    bracket_contents = BRACKET_CONTENT.findall(answer)
    all_ops = []
    for content in bracket_contents:
        ops = find_operations_compact(content)
        all_ops.extend(ops)
    return all_ops


def find_operations_outside_brackets(answer: str) -> list[tuple[str, str, str, str]]:
    """Find operations outside <<>> brackets using spaced pattern."""
    # Remove all <<...>> content
    text_without_brackets = BRACKET_TAGS.sub('', answer)
    return find_operations_spaced(text_without_brackets)


def count_operations_by_type(operations: list[tuple[str, str, str, str]]) -> dict[str, int]:
    """Count operations by their normalized type."""
    counts = {'+': 0, '-': 0, '*': 0, '/': 0}
    for _, _, _, norm_op in operations:
        if norm_op in counts:
            counts[norm_op] += 1
    return counts


if __name__ == "__main__":
    # Test the patterns
    print("=== COMPACT PATTERN (inside brackets) ===")
    compact_tests = [
        "5+3",
        "100/4",
        "20*35",
        "16-3-4=9",
        "5*2=10",
        "20–12=8",  # en-dash
        "115—15=100",  # em-dash
    ]
    for test in compact_tests:
        ops = find_operations_compact(test)
        print(f"{test!r} -> {ops}")

    print("\n=== SPACED PATTERN (outside brackets) ===")
    spaced_tests = [
        "5 + 3",
        "100 / 4",
        "5 x 3",
        "5X3",  # compact X should still work
        "20x20",  # compact X
        "10 - 5",
        "$20/hour * 35 hours/week",  # should NOT match 20/hour
        "100 miles / 4 gallons",  # should match
        "25 miles per gallon x 12 gallons",  # should match
        "30 years - 20 years",
        "99 + 5 = $104",
        "16 - 3 - 4 = 9",  # chained: should find 16-3 AND 3-4
        "3 hours - 1 hour - 1 hour = 1 hour",  # chained with units
        # Asymmetric spacing tests
        "45 -40 = 5",  # space before minus only
        "2 +2 = 4",  # space before plus only
        "4*2 +2 = 10",  # mixed: compact then asymmetric
        # Unicode dash tests
        "20 – 12 = 8",  # en-dash with spaces
        "total gnomes – 12 gnomes = 8",  # en-dash in sentence
    ]
    for test in spaced_tests:
        ops = find_operations_spaced(test)
        print(f"{test!r} -> {ops}")

    print("\n=== FULL EXAMPLE ===")
    example = "First find: $20/hour * 35 hours/week = $<<20*35=700>>700/week"
    print(f"Text: {example!r}")
    print(f"Inside brackets: {find_operations_inside_brackets(example)}")
    print(f"Outside brackets: {find_operations_outside_brackets(example)}")
