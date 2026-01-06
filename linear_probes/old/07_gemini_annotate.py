#!/usr/bin/env python3
"""
Step 7: LLM-based annotation using Google Gemini Flash.

3-step process:
1. Clean text (remove <<>> brackets, keep result)
2. Use Gemini to extract operations + find character positions
3. Mechanically map character positions to token positions

Output fields per operation:
- operand1, operator, operand2, result: Operation values
- is_correct: Whether arithmetic is correct
- op1_char_pos, operator_char_pos, op2_char_pos, equals_char_pos, result_char_pos: Character positions
- op1_token_pos, operator_token_pos, op2_token_pos, result_token_pos: Token positions (list, -1 if not found)
- pre_result_pos, is_anticipatory: C4 probe fields
- op_index, is_first, is_last, is_intermediate, next_op: Sequence metadata

Usage:
    python 07_gemini_annotate.py responses/Qwen2.5-Math-1.5B/train_responses.json
    python 07_gemini_annotate.py responses/Qwen2.5-Math-1.5B/train_responses.json --limit 100
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
import typer
from transformers import AutoTokenizer

from utils.args import InputFile, OutputDir
from utils.data import load_json, save_json, save_metadata
from utils.logging import log, print_header, print_config, print_summary, create_progress

app = typer.Typer(add_completion=False)

# Model config
GEMINI_MODEL = "gemini-2.0-flash"
TOKENIZER_NAME = "Qwen/Qwen2.5-Math-1.5B"

# Pattern to remove <<...>> content but keep result
BRACKET_PATTERN = re.compile(r'<<[^>]*=([^>]*)>>')


def clean_answer(text: str) -> str:
    """Remove <<...>> brackets from text, keeping only the result."""
    return BRACKET_PATTERN.sub(r'\1', text)


def extract_answer_only(text: str) -> str:
    """Extract only the answer portion from response text."""
    patterns = [
        r'Answer\s*:\s*',
        r'\n\n(?=\w)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    return text


# Gemini prompt for extracting operations (values only, not positions)
GEMINI_PROMPT = """Extract ALL arithmetic operations from this text, including implicit ones.

Convert implicit language to explicit operations:
- "half of 48" → 48 / 2 = 24
- "twice as many as 10" → 10 * 2 = 20
- "total of X and Y" → X + Y
- "three times 5" → 5 * 3 = 15

Text to analyze:
```
{text}
```

Output ONLY a JSON array with operations. Use +, -, *, / for operators. Verify arithmetic.
```json
[{{"operand1": 48, "operator": "/", "operand2": 2, "result": 24, "is_correct": true}}]
```"""


# Operator patterns to search for in text
OPERATOR_PATTERNS = {
    '+': ['+', ' plus ', ' and '],
    '-': ['-', '−', '–', ' minus ', ' subtract '],
    '*': ['*', '×', 'x', 'X', ' times ', ' multiplied by '],
    '/': ['/', '÷', ' divided by ', ' half of ', ' over '],
}


def format_num(n) -> Optional[str]:
    """Format number for string search."""
    if n is None:
        return None
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    return str(n)


def is_word_boundary(text: str, pos: int, pattern_len: int) -> bool:
    """Check if the match at pos is at word boundaries (not inside a larger number)."""
    if pos > 0:
        char_before = text[pos - 1]
        if char_before.isdigit():
            return False
    end_pos = pos + pattern_len
    if end_pos < len(text):
        char_after = text[end_pos]
        if char_after.isdigit():
            return False
    return True


def find_all_occurrences(text: str, pattern: str, require_word_boundary: bool = True) -> list[int]:
    """Find all occurrences of pattern in text, return list of positions."""
    positions = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            break
        if require_word_boundary and pattern and pattern[0].isdigit():
            if is_word_boundary(text, idx, len(pattern)):
                positions.append(idx)
        else:
            positions.append(idx)
        start = idx + 1
    return positions


def find_operator_in_text(text: str, op_symbol: str, start: int, end: int) -> int:
    """Find operator position between start and end positions."""
    if start >= end or start < 0:
        return -1
    search_region = text[start:end]
    patterns = OPERATOR_PATTERNS.get(op_symbol, [op_symbol])

    for pattern in patterns:
        idx = search_region.find(pattern)
        if idx >= 0:
            return start + idx
    return -1


def find_equals_in_text(text: str, start: int, end: int) -> int:
    """Find '=' position between start and end positions."""
    if start >= end or start < 0:
        return -1
    search_region = text[start:end]
    idx = search_region.find('=')
    if idx >= 0:
        return start + idx
    return -1


def find_char_positions(text: str, operations: list) -> list:
    """
    Find character positions for operations mechanically.

    Strategy:
    - Find ALL occurrences of each number in text
    - For each operation, find positions that maintain order: op1 < operator < op2 < result
    """
    result = []
    min_search_pos = 0

    for op in operations:
        op_result = dict(op)

        operand1 = op.get('operand1')
        operand2 = op.get('operand2')
        op_result_val = op.get('result')
        op_symbol = op.get('operator', '')

        op1_str = format_num(operand1)
        op2_str = format_num(operand2)
        res_str = format_num(op_result_val)

        # Find all occurrences
        op1_positions = find_all_occurrences(text, op1_str) if op1_str else []
        op2_positions = find_all_occurrences(text, op2_str) if op2_str else []
        res_positions = find_all_occurrences(text, res_str) if res_str else []

        # Filter to positions >= min_search_pos
        op1_positions = [p for p in op1_positions if p >= min_search_pos]
        op2_positions = [p for p in op2_positions if p >= min_search_pos]
        res_positions = [p for p in res_positions if p >= min_search_pos]

        # Find best matching set
        best_op1 = -1
        best_op2 = -1
        best_res = -1
        best_operator = -1
        best_equals = -1

        for op1_pos in op1_positions if op1_positions else [-1]:
            op1_end = op1_pos + len(op1_str) if op1_pos >= 0 and op1_str else min_search_pos

            valid_op2 = [p for p in op2_positions if p > op1_end] if op1_pos >= 0 else op2_positions
            op2_pos = valid_op2[0] if valid_op2 else -1
            op2_end = op2_pos + len(op2_str) if op2_pos >= 0 and op2_str else op1_end

            res_before = [p for p in res_positions if p < op1_pos and p >= min_search_pos] if op1_pos >= 0 else []
            search_after = op2_end if op2_pos >= 0 else op1_end
            res_after = [p for p in res_positions if p > search_after]

            if op2_pos < 0 and res_before:
                res_pos = res_before[0]
            elif res_after:
                res_pos = res_after[0]
            elif res_before:
                res_pos = res_before[0]
            else:
                res_pos = -1

            if op1_pos >= 0 or op2_pos >= 0 or res_pos >= 0:
                best_op1 = op1_pos
                best_op2 = op2_pos
                best_res = res_pos
                break

        # Find operator
        if best_op1 >= 0:
            before_start = max(0, min_search_pos, best_op1 - 30)
            best_operator = find_operator_in_text(text, op_symbol, before_start, best_op1)

            if best_operator < 0 and best_op2 >= 0:
                op1_end = best_op1 + len(op1_str) if op1_str else best_op1
                best_operator = find_operator_in_text(text, op_symbol, op1_end, best_op2)

        # Find equals sign
        if best_op2 >= 0 and best_res >= 0:
            op2_end = best_op2 + len(op2_str) if op2_str else best_op2
            best_equals = find_equals_in_text(text, op2_end, best_res)

        # Update min_search_pos
        rightmost = max(best_op1, best_operator, best_op2, best_equals, best_res)
        if rightmost >= 0:
            if best_res >= 0 and best_res == rightmost:
                min_search_pos = best_res + len(res_str) if res_str else best_res + 1
            elif best_op2 >= 0 and best_op2 == rightmost:
                min_search_pos = best_op2 + len(op2_str) if op2_str else best_op2 + 1
            elif best_op1 >= 0 and best_op1 == rightmost:
                min_search_pos = best_op1 + len(op1_str) if op1_str else best_op1 + 1
            else:
                min_search_pos = rightmost + 1

        # Store character positions
        op_result['op1_char_pos'] = best_op1
        op_result['operator_char_pos'] = best_operator
        op_result['op2_char_pos'] = best_op2
        op_result['equals_char_pos'] = best_equals
        op_result['result_char_pos'] = best_res

        result.append(op_result)

    return result


def init_gemini() -> genai.Client:
    """Initialize Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)
    return client


def call_gemini(client: genai.Client, prompt: str, max_retries: int = 3) -> str:
    """Call Gemini API with retries."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log.warning(f"Gemini API error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                log.error(f"Gemini API failed after {max_retries} attempts: {e}")
                raise
    return ""


def parse_json_response(text: str) -> list:
    """Extract JSON array from LLM response."""
    # Try to find JSON in code block
    match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


def char_pos_to_token_pos(
    char_pos: int,
    token_offsets: list[tuple[int, int]],
) -> list[int]:
    """
    Map a character position to token position(s).

    Args:
        char_pos: Character position in text (-1 if not found)
        token_offsets: List of (start, end) character offsets for each token

    Returns:
        List of token indices that contain this character position, or [-1] if not found
    """
    if char_pos < 0:
        return [-1]

    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_start <= char_pos < tok_end:
            token_indices.append(idx)
            break  # Usually just one token contains the position

    return token_indices if token_indices else [-1]


def char_span_to_token_pos(
    char_pos: int,
    value,
    token_offsets: list[tuple[int, int]],
) -> list[int]:
    """
    Map a character position + value to all tokens that cover the value's span.

    This handles multi-digit numbers that span multiple tokens.
    """
    if char_pos < 0:
        return [-1]

    # Get the span length
    if value is None:
        return [-1]

    value_str = str(value)
    if isinstance(value, float) and value == int(value):
        value_str = str(int(value))

    char_start = char_pos
    char_end = char_pos + len(value_str)

    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        # Token overlaps with the value span
        if tok_end > char_start and tok_start < char_end:
            token_indices.append(idx)

    return token_indices if token_indices else [-1]


def add_sequence_metadata(operations: list) -> list:
    """Add sequence metadata to operations (is_first, is_last, is_intermediate, next_op)."""
    num_ops = len(operations)

    for op_idx, op in enumerate(operations):
        op['op_index'] = op_idx
        op['is_first'] = (op_idx == 0)
        op['is_last'] = (op_idx == num_ops - 1)

        # Check if result feeds into next operation
        is_intermediate = False
        next_op = None

        if op_idx < num_ops - 1:
            next_operation = operations[op_idx + 1]
            result = op.get('result')
            next_op1 = next_operation.get('operand1')
            next_op2 = next_operation.get('operand2')

            if result is not None and (result == next_op1 or result == next_op2):
                is_intermediate = True
            next_op = next_operation.get('operator')

        op['is_intermediate'] = is_intermediate
        op['next_op'] = next_op

    return operations


def add_c4_probe_fields(operations: list) -> list:
    """Add C4 probe fields (pre_result_pos, is_anticipatory)."""
    for op in operations:
        # Get all component positions
        component_positions = [
            op.get('op1_char_pos', -1),
            op.get('operator_char_pos', -1),
            op.get('op2_char_pos', -1),
            op.get('equals_char_pos', -1),
        ]
        component_positions = [p for p in component_positions if p >= 0]

        result_pos = op.get('result_char_pos', -1)

        # pre_result_pos: position to probe before result appears
        if op.get('equals_char_pos', -1) >= 0:
            pre_result_pos = op['equals_char_pos']
        elif component_positions:
            pre_result_pos = max(component_positions)
        else:
            pre_result_pos = -1

        # is_anticipatory: True if result comes after all other components
        if result_pos >= 0 and component_positions:
            is_anticipatory = all(result_pos > p for p in component_positions)
        else:
            is_anticipatory = False

        op['pre_result_pos'] = pre_result_pos
        op['is_anticipatory'] = is_anticipatory

    return operations


def annotate_response(
    text: str,
    client: genai.Client,
    tokenizer,
) -> dict:
    """Annotate a single response with Gemini + mechanical position finding."""

    # Step 1: Clean text (remove <<>> brackets)
    answer_text = extract_answer_only(text)
    answer_clean = clean_answer(answer_text)

    # Step 2: Call Gemini to extract operations (values only)
    prompt = GEMINI_PROMPT.format(text=answer_clean)
    response = call_gemini(client, prompt)
    operations = parse_json_response(response)

    if not operations:
        return {
            "answer": answer_clean,
            "operations": [],
            "error": "Failed to extract operations"
        }

    # Step 3a: Find character positions mechanically
    operations = find_char_positions(answer_clean, operations)

    # Step 3b: Tokenize and map char positions to token positions
    encoding = tokenizer(
        answer_clean,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    token_offsets = encoding['offset_mapping']

    # Map each operation's char positions to token positions
    for op in operations:
        op['op1_token_pos'] = char_span_to_token_pos(
            op.get('op1_char_pos', -1),
            op.get('operand1'),
            token_offsets
        )
        op['operator_token_pos'] = char_pos_to_token_pos(
            op.get('operator_char_pos', -1),
            token_offsets
        )
        op['op2_token_pos'] = char_span_to_token_pos(
            op.get('op2_char_pos', -1),
            op.get('operand2'),
            token_offsets
        )
        op['result_token_pos'] = char_span_to_token_pos(
            op.get('result_char_pos', -1),
            op.get('result'),
            token_offsets
        )

    # Add sequence metadata
    operations = add_sequence_metadata(operations)

    # Add C4 probe fields
    operations = add_c4_probe_fields(operations)

    return {
        "answer": answer_clean,
        "tokens": tokens,
        "operations": operations,
    }


def process_examples(
    examples: list,
    client: genai.Client,
    tokenizer,
    responses_per_example: int = 1,
    progress=None,
    task=None,
) -> tuple[list, int, int]:
    """Process multiple examples."""
    results = []
    total_responses = 0
    total_ops = 0

    for example in examples:
        responses_annotated = []

        for resp_idx, resp in enumerate(example.get('responses', [])[:responses_per_example]):
            # Handle both string and dict response formats
            if isinstance(resp, str):
                text = resp
            else:
                text = resp.get('text', '')

            annotated = annotate_response(text, client, tokenizer)
            responses_annotated.append(annotated)

            total_responses += 1
            total_ops += len(annotated.get('operations', []))

            if progress and task:
                progress.advance(task)

        results.append({
            'index': example['index'],
            'question': example['question'],
            'responses': responses_annotated,
        })

    return results, total_responses, total_ops


@app.command()
def main(
    input_file: InputFile,
    output_dir: OutputDir = None,
    limit: int = typer.Option(0, "--limit", "-l", help="Limit examples to process (0=all)"),
    responses_per_example: int = typer.Option(1, "--responses", "-r", help="Responses to annotate per example"),
):
    """Annotate operations using Google Gemini Flash with token position mapping."""
    input_path = Path(input_file)
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    if output_dir:
        output_path = Path(output_dir) / f"{input_path.stem}_gemini_annotated.json"
    else:
        output_path = input_path.parent / f"{input_path.stem}_gemini_annotated.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_header("Gemini Annotation", "Step 7")
    print_config("Configuration", {
        'input': str(input_path),
        'output': str(output_path),
        'gemini_model': GEMINI_MODEL,
        'tokenizer': TOKENIZER_NAME,
        'limit': limit if limit > 0 else "all",
        'responses_per_example': responses_per_example,
    })

    # Initialize Gemini
    log.info(f"Initializing Gemini model: {GEMINI_MODEL}")
    client = init_gemini()

    # Load tokenizer
    log.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load input data
    log.info("Loading responses...")
    examples = load_json(input_path)

    if limit > 0:
        examples = examples[:limit]

    total_prompts = sum(min(len(ex.get('responses', [])), responses_per_example) for ex in examples)
    log.info(f"Processing {len(examples)} examples ({total_prompts} responses)")

    # Process
    with create_progress() as progress:
        task = progress.add_task("Annotating", total=total_prompts)

        results, total_responses, total_ops = process_examples(
            examples,
            client,
            tokenizer,
            responses_per_example,
            progress,
            task,
        )

    save_json(results, output_path)
    log.success(f"Saved to {output_path}")

    print_summary("Annotation Summary", {
        'examples': len(results),
        'responses_annotated': total_responses,
        'total_ops_found': total_ops,
        'avg_ops_per_response': f"{total_ops / total_responses:.1f}" if total_responses else "0",
    })

    save_metadata(
        output_path.parent,
        config={
            'script': '07_gemini_annotate.py',
            'input_file': str(input_path),
            'output_file': str(output_path),
            'gemini_model': GEMINI_MODEL,
            'tokenizer': TOKENIZER_NAME,
            'limit': limit,
            'responses_per_example': responses_per_example,
        },
        stats={
            'examples': len(results),
            'responses_annotated': total_responses,
            'total_ops_found': total_ops,
        },
    )


if __name__ == '__main__':
    app()
