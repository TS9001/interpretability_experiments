"""Number parsing and final answer extraction utilities."""

import re
from typing import Optional


# Regex pattern for numbers with optional comma separators and decimals
NUMBER_PATTERN = r'[-+]?\d[\d,]*(?:\.\d+)?'


def parse_number(s: str) -> float:
    """Parse number string, handling commas."""
    return float(s.replace(',', ''))


def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from response text.

    Tries multiple patterns in order:
    1. GSM8K format: #### followed by number
    2. Natural language: "The answer is X"
    3. Fallback: Last number in text
    """
    # Look for #### pattern (GSM8K format)
    match = re.search(r'####\s*(' + NUMBER_PATTERN + r')', text)
    if match:
        try:
            return parse_number(match.group(1))
        except ValueError:
            pass

    # Look for "The answer is X" pattern
    match = re.search(r'[Tt]he answer is[:\s]*(' + NUMBER_PATTERN + r')', text)
    if match:
        try:
            return parse_number(match.group(1))
        except ValueError:
            pass

    # Look for last number in text (fallback)
    numbers = re.findall(NUMBER_PATTERN, text)
    if numbers:
        try:
            return parse_number(numbers[-1])
        except ValueError:
            pass

    return None
