#!/usr/bin/env python3
"""
Step 5: LLM-based annotation of operations in responses.

Uses MLX LLM directly (not HTTP server) for fast batched inference.

Output fields per operation:
- operand1, operator, operand2, result: Operation values
- is_correct: Whether arithmetic is correct
- op1_pos, operator_pos, op2_pos, equals_pos, result_pos: Character positions (-1 if not found)
- pre_result_pos: Position to probe for C4 (anticipatory result prediction)
- is_anticipatory: True if result appears AFTER all other components
- op_index, is_first, is_last, is_intermediate, next_op: Sequence metadata

Usage:
    python 05_llm_annotate.py responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json
    python 05_llm_annotate.py responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json --limit 100
    python 05_llm_annotate.py responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json --batch-size 16
"""
import json
import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import load, generate
import typer

from utils.args import InputFile, OutputDir
from utils.data import load_json, save_json, save_metadata
from utils.logging import log, print_header, print_config, print_summary, create_progress

app = typer.Typer(add_completion=False)

# Model config
MODEL_NAME = "mlx-community/Qwen3-30B-A3B-bf16"

# Global model and tokenizer (loaded once)
_model = None
_tokenizer = None


def load_model():
    """Load model and tokenizer (cached globally)."""
    global _model, _tokenizer
    if _model is None:
        log.info(f"Loading model {MODEL_NAME}...")
        _model, _tokenizer = load(MODEL_NAME)
        log.success("Model loaded")
    return _model, _tokenizer


def generate_single(prompt: str, max_tokens: int = 512) -> str:
    """Generate response for a single prompt."""
    model, tokenizer = load_model()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


def generate_batch(prompts: list[str], max_tokens: int = 512) -> list[str]:
    """Generate responses for a batch of prompts (sequential fallback)."""
    model, tokenizer = load_model()

    results = []
    for prompt in prompts:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        results.append(response)

    return results


def generate_batch_parallel(
    prompts: list[str],
    max_tokens: int = 512,
    batch_size: int = 8,
) -> list[str]:
    """
    Generate responses for multiple prompts in parallel using batched inference.

    Processes prompts in batches on GPU for true parallelism.
    """
    model, tokenizer = load_model()

    all_results = []

    # Process in batches
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_results = _generate_batch_parallel_inner(
            model, tokenizer, batch_prompts, max_tokens
        )
        all_results.extend(batch_results)
        mx.metal.clear_cache()

    return all_results


def _generate_batch_parallel_inner(
    model,
    tokenizer,
    prompts: list[str],
    max_tokens: int,
) -> list[str]:
    """Inner function for parallel batch generation using greedy decoding."""

    # Tokenize all prompts
    batch_tokens = [tokenizer.encode(p) for p in prompts]
    batch_size = len(batch_tokens)

    # Pad to same length
    max_len = max(len(t) for t in batch_tokens)
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_tokens = []
    prompt_lengths = []
    for tokens in batch_tokens:
        prompt_lengths.append(len(tokens))
        padding = [pad_token] * (max_len - len(tokens))
        padded_tokens.append(padding + tokens)  # Left padding for generation

    # Convert to MLX array
    input_ids = mx.array(padded_tokens)

    # Track which sequences are done
    done = [False] * batch_size
    eos_token = tokenizer.eos_token_id

    # Generated tokens for each sequence
    generated = [[] for _ in range(batch_size)]

    # Generate tokens (greedy decoding)
    for _ in range(max_tokens):
        # Forward pass
        logits = model(input_ids)
        # Only use last token's logits
        logits = logits[:, -1, :]

        # Greedy: take argmax
        next_tokens = mx.argmax(logits, axis=-1)
        mx.eval(next_tokens)
        next_tokens = next_tokens.tolist()

        # Update generated tokens and check for EOS
        all_done = True
        for i, tok in enumerate(next_tokens):
            if not done[i]:
                if tok == eos_token:
                    done[i] = True
                else:
                    generated[i].append(tok)
                    all_done = False
            else:
                all_done = all_done and True

        if all_done:
            break

        # Prepare next input (append new tokens)
        input_ids = mx.concatenate([input_ids, mx.array([[t] for t in next_tokens])], axis=1)

    # Decode results
    results = []
    for tokens in generated:
        text = tokenizer.decode(tokens)
        results.append(text)

    return results


# === PROMPT: Combined - extract and standardize in one call ===
PROMPT_COMBINED = """/no_think
Extract ALL arithmetic operations from this text, including implicit ones.

Convert implicit language:
- "half of 48" → 48 / 2 = 24
- "twice as many as 10" → 10 * 2 = 20
- "total of X and Y" → X + Y

Text:
{text}

Output ONLY a JSON array with standardized operations. Use +, -, *, / symbols. Verify arithmetic is correct.
```json
[{{"operand1": 48, "operator": "/", "operand2": 2, "result": 24, "is_correct": true}}]
```"""


# === Legacy prompts (kept for reference) ===
PROMPT_EXPLICIT = """/no_think
Analyze this math problem response and identify ALL arithmetic operations performed.

For each operation, convert implicit language to explicit arithmetic:
- "half of 48" → "48 / 2 = 24"
- "twice as many" → multiplication
- "three times" → multiplication
- "16 lawns * $33 per lawn" → "16 * 33 = 528"
- "total of X and Y" → addition

Response text:
{text}

List ALL operations found, even implicit ones. Output ONLY a JSON array, no explanation:
```json
[
  {{"implicit": "half of 48", "explicit": "48 / 2 = 24", "operand1": 48, "operator": "/", "operand2": 2, "result": 24}}
]
```"""


PROMPT_FORMAT = """/no_think
Given these operations extracted from a math response, verify and standardize them.

Operations found:
{operations}

Original text:
{text}

For each operation:
1. Verify the arithmetic is correct
2. Standardize operator symbols: use +, -, *, /
3. Ensure all values are numbers (not strings)

Output ONLY a JSON array, no explanation:
```json
[
  {{"operand1": 48, "operator": "/", "operand2": 2, "result": 24, "is_correct": true}}
]
```"""


def format_num(n):
    """Format number for string search."""
    if n is None:
        return None
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    return str(n)


# Operator patterns to search for
OPERATOR_PATTERNS = {
    '+': ['+', ' plus ', ' and '],
    '-': ['-', '−', '–', ' minus ', ' subtract '],
    '*': ['*', '×', 'x', 'X', ' times ', ' multiplied by '],
    '/': ['/', '÷', ' divided by ', ' half of ', ' over '],
}


def is_word_boundary(text: str, pos: int, pattern_len: int) -> bool:
    """Check if the match at pos is at word boundaries (not inside a larger number)."""
    # Check before
    if pos > 0:
        char_before = text[pos - 1]
        if char_before.isdigit():
            return False
    # Check after
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
        # Check word boundaries for numbers (avoid matching "2" inside "72")
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


def find_positions_with_code(text: str, operations: list) -> list:
    """
    Find character positions for operations.

    Strategy:
    - Find ALL occurrences of each number in text
    - For each operation, find positions that maintain order: op1 < operator < op2 < result
    - Search for operator both BEFORE op1 (for "half of X") and BETWEEN op1/op2
    """
    result = []
    num_ops = len(operations)
    min_search_pos = 0  # Don't reuse positions from previous operations

    for op_idx, op in enumerate(operations):
        op_result = dict(op)  # Copy operation

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

        # Find best matching set: op1 < op2 < result (with some flexibility)
        best_op1 = -1
        best_op2 = -1
        best_res = -1
        best_operator = -1
        best_equals = -1

        # Try each op1 position
        for op1_pos in op1_positions if op1_positions else [-1]:
            op1_end = op1_pos + len(op1_str) if op1_pos >= 0 and op1_str else min_search_pos

            # Find op2 after op1
            valid_op2 = [p for p in op2_positions if p > op1_end] if op1_pos >= 0 else op2_positions
            op2_pos = valid_op2[0] if valid_op2 else -1
            op2_end = op2_pos + len(op2_str) if op2_pos >= 0 and op2_str else op1_end

            # Find result - for implicit operators like "half of", result often appears BEFORE op1
            # Check both before and after, prefer the one that makes a complete operation
            res_before = [p for p in res_positions if p < op1_pos and p >= min_search_pos] if op1_pos >= 0 else []
            search_after = op2_end if op2_pos >= 0 else op1_end
            res_after = [p for p in res_positions if p > search_after]

            # Prefer result BEFORE op1 if op2 is not found (implicit operation like "24 is half of 48")
            if op2_pos < 0 and res_before:
                res_pos = res_before[0]
            elif res_after:
                res_pos = res_after[0]
            elif res_before:
                res_pos = res_before[0]
            else:
                res_pos = -1

            # Check if this is a valid sequence (at least one element found)
            if op1_pos >= 0 or op2_pos >= 0 or res_pos >= 0:
                best_op1 = op1_pos
                best_op2 = op2_pos
                best_res = res_pos
                break  # Take first valid sequence

        # Find operator - check BEFORE op1 first (for "half of X"), then between op1 and op2
        if best_op1 >= 0:
            # Check before op1 (within reasonable window, e.g., 30 chars)
            before_start = max(0, min_search_pos, best_op1 - 30)
            best_operator = find_operator_in_text(text, op_symbol, before_start, best_op1)

            # If not found before, check between op1 and op2
            if best_operator < 0 and best_op2 >= 0:
                op1_end = best_op1 + len(op1_str) if op1_str else best_op1
                best_operator = find_operator_in_text(text, op_symbol, op1_end, best_op2)

        # Find equals sign (between op2 and result)
        if best_op2 >= 0 and best_res >= 0:
            op2_end = best_op2 + len(op2_str) if op2_str else best_op2
            best_equals = find_equals_in_text(text, op2_end, best_res)

        # Update min_search_pos for next operation
        # Use the rightmost found position to avoid skipping subsequent operations
        rightmost = max(best_op1, best_operator, best_op2, best_equals, best_res)
        if rightmost >= 0:
            # Advance past the rightmost element found
            if best_res >= 0 and best_res == rightmost:
                min_search_pos = best_res + len(res_str) if res_str else best_res + 1
            elif best_op2 >= 0 and best_op2 == rightmost:
                min_search_pos = best_op2 + len(op2_str) if op2_str else best_op2 + 1
            elif best_op1 >= 0 and best_op1 == rightmost:
                min_search_pos = best_op1 + len(op1_str) if op1_str else best_op1 + 1
            else:
                min_search_pos = rightmost + 1

        # Store positions
        op_result['op1_pos'] = best_op1
        op_result['operator_pos'] = best_operator
        op_result['op2_pos'] = best_op2
        op_result['equals_pos'] = best_equals
        op_result['result_pos'] = best_res

        # Add sequence metadata
        op_result['op_index'] = op_idx
        op_result['is_first'] = (op_idx == 0)
        op_result['is_last'] = (op_idx == num_ops - 1)

        # Check if result is used in next operation (is_intermediate)
        is_intermediate = False
        next_op = None
        if op_idx < num_ops - 1:
            next_operation = operations[op_idx + 1]
            next_op1 = next_operation.get('operand1')
            next_op2 = next_operation.get('operand2')
            # Check if current result feeds into next operation
            if op_result_val is not None and (op_result_val == next_op1 or op_result_val == next_op2):
                is_intermediate = True
            next_op = next_operation.get('operator')

        op_result['is_intermediate'] = is_intermediate
        op_result['next_op'] = next_op

        # === C4 Probe Support: Anticipatory Result Prediction ===
        # pre_result_pos: Position to probe for C4 (anticipating result before it appears)
        #   - Uses equals_pos if available (explicit: "48 + 24 = 72")
        #   - Otherwise uses the last component position before result
        # is_anticipatory: True if result appears AFTER all operands and operator
        #   - Only when is_anticipatory=True can we validly probe for result anticipation
        #   - False for implicit ops like "24 is half of 48" where result comes first

        # Find positions of all components (excluding result)
        component_positions = [p for p in [best_op1, best_op2, best_operator, best_equals] if p >= 0]

        if best_equals >= 0:
            # Explicit operation with "=" sign - use equals position
            pre_result_pos = best_equals
        elif component_positions:
            # No equals sign - use last component before result
            pre_result_pos = max(component_positions)
        else:
            pre_result_pos = -1

        # Check if result comes after all other components (anticipatory)
        if best_res >= 0 and component_positions:
            is_anticipatory = all(best_res > p for p in component_positions)
        else:
            is_anticipatory = False

        op_result['pre_result_pos'] = pre_result_pos
        op_result['is_anticipatory'] = is_anticipatory

        result.append(op_result)

    return result


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


def extract_answer_only(text: str) -> str:
    """Extract only the answer portion from response text."""
    # Common patterns: "Answer:", "Answer :", "\n\nAnswer"
    patterns = [
        r'Answer\s*:\s*',
        r'\n\n(?=\w)',  # Double newline before text
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    return text


def annotate_response(text: str, ground_truth_ops: list) -> dict:
    """Single LLM call + code-based position finding."""

    # Extract answer only
    answer_text = extract_answer_only(text)

    # Single combined prompt - extract and standardize in one call
    prompt = PROMPT_COMBINED.format(text=answer_text)
    response = generate_single(prompt)
    operations = parse_json_response(response)

    if not operations:
        return {
            "answer": answer_text,
            "operations": [],
            "error": "Failed to extract operations"
        }

    # Find positions with code (not LLM)
    operations_with_positions = find_positions_with_code(answer_text, operations)

    # Clean up output - keep essential fields for probes
    clean_ops = []
    for op in operations_with_positions:
        clean_ops.append({
            # Operation data
            "operand1": op.get('operand1'),
            "operator": op.get('operator'),
            "operand2": op.get('operand2'),
            "result": op.get('result'),
            "is_correct": op.get('is_correct'),
            # Character positions (for probes B1, B2, C1, C3)
            "op1_pos": op.get('op1_pos', -1),
            "operator_pos": op.get('operator_pos', -1),
            "op2_pos": op.get('op2_pos', -1),
            "equals_pos": op.get('equals_pos', -1),
            "result_pos": op.get('result_pos', -1),
            # C4 probe: Anticipatory result prediction
            # pre_result_pos: position to probe (equals or last component before result)
            # is_anticipatory: True if result comes after all operands/operator (C4 valid)
            "pre_result_pos": op.get('pre_result_pos', -1),
            "is_anticipatory": op.get('is_anticipatory', False),
            # Sequence metadata (for probes D1, D2, D3, D6)
            "op_index": op.get('op_index', 0),
            "is_first": op.get('is_first', False),
            "is_last": op.get('is_last', False),
            "is_intermediate": op.get('is_intermediate', False),
            "next_op": op.get('next_op'),
        })

    return {
        "operations": clean_ops,
    }


def process_batch(
    examples: list,
    responses_per_example: int,
    progress=None,
    task=None,
    parallel: bool = True,
    gpu_batch_size: int = 8,
) -> tuple[list, int, int]:
    """Process examples with batched LLM calls."""
    results = []
    total_responses = 0
    total_ops = 0

    # Collect all prompts first
    prompt_data = []  # (example_idx, resp_idx, answer_text, prompt)
    for ex_idx, example in enumerate(examples):
        for resp_idx, resp in enumerate(example.get('responses', [])[:responses_per_example]):
            text = resp.get('text', '')
            answer_text = extract_answer_only(text)
            prompt = PROMPT_COMBINED.format(text=answer_text)
            prompt_data.append((ex_idx, resp_idx, answer_text, prompt, resp))

    # Generate all responses
    prompts = [p[3] for p in prompt_data]
    log.info(f"Generating {len(prompts)} LLM responses (parallel={parallel}, gpu_batch={gpu_batch_size})...")

    if parallel:
        responses = generate_batch_parallel(prompts, batch_size=gpu_batch_size)
    else:
        responses = generate_batch(prompts)

    # Process responses and build results
    example_responses = {i: [] for i in range(len(examples))}

    for (ex_idx, resp_idx, answer_text, prompt, orig_resp), llm_response in zip(prompt_data, responses):
        operations = parse_json_response(llm_response)

        if operations:
            operations_with_positions = find_positions_with_code(answer_text, operations)
            clean_ops = []
            for op in operations_with_positions:
                clean_ops.append({
                    "operand1": op.get('operand1'),
                    "operator": op.get('operator'),
                    "operand2": op.get('operand2'),
                    "result": op.get('result'),
                    "is_correct": op.get('is_correct'),
                    "op1_pos": op.get('op1_pos', -1),
                    "operator_pos": op.get('operator_pos', -1),
                    "op2_pos": op.get('op2_pos', -1),
                    "equals_pos": op.get('equals_pos', -1),
                    "result_pos": op.get('result_pos', -1),
                    "pre_result_pos": op.get('pre_result_pos', -1),
                    "is_anticipatory": op.get('is_anticipatory', False),
                    "op_index": op.get('op_index', 0),
                    "is_first": op.get('is_first', False),
                    "is_last": op.get('is_last', False),
                    "is_intermediate": op.get('is_intermediate', False),
                    "next_op": op.get('next_op'),
                })
        else:
            clean_ops = []

        example_responses[ex_idx].append({
            'answer': answer_text,
            'regex_ops': len(orig_resp.get('operations', [])),
            'llm_ops': clean_ops,
        })

        total_responses += 1
        total_ops += len(clean_ops)

        if progress and task:
            progress.advance(task)

    # Build final results
    for ex_idx, example in enumerate(examples):
        results.append({
            'index': example['index'],
            'question': example['question'],
            'question_char_end': len(example['question']),
            'ground_truth_operations': example.get('ground_truth_operations', []),
            'ground_truth_final_result': example.get('ground_truth_final_result'),
            'responses': example_responses[ex_idx],
        })

    return results, total_responses, total_ops


@app.command()
def main(
    input_file: InputFile,
    output_dir: OutputDir = None,
    limit: int = typer.Option(0, "--limit", "-l", help="Limit examples to process (0=all)"),
    responses_per_example: int = typer.Option(1, "--responses", "-r", help="Responses to annotate per example"),
    batch_size: int = typer.Option(0, "--batch-size", "-b", help="Process in batches (0=all at once)"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Use parallel GPU batching"),
    gpu_batch: int = typer.Option(8, "--gpu-batch", "-g", help="GPU batch size for parallel mode"),
):
    """Annotate operations using MLX LLM directly with parallel batching."""
    input_path = Path(input_file)
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    if output_dir:
        output_path = Path(output_dir) / f"{input_path.stem}_llm_annotated.json"
    else:
        output_path = input_path.parent / f"{input_path.stem}_llm_annotated.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_header("LLM Annotation", "Step 5")
    print_config("Configuration", {
        'input': str(input_path),
        'output': str(output_path),
        'model': MODEL_NAME,
        'limit': limit if limit > 0 else "all",
        'responses_per_example': responses_per_example,
        'batch_size': batch_size if batch_size > 0 else "all",
        'parallel': parallel,
        'gpu_batch': gpu_batch,
    })

    # Load model
    load_model()

    # Test with a simple prompt
    log.info("Testing model...")
    test_resp = generate_single("/no_think\nSay OK")
    log.info(f"Model OK: {test_resp[:30].strip()}...")

    log.info("Loading analyzed responses...")
    examples = load_json(input_path)

    if limit > 0:
        examples = examples[:limit]

    total_prompts = sum(min(len(ex.get('responses', [])), responses_per_example) for ex in examples)
    log.info(f"Processing {len(examples)} examples ({total_prompts} prompts)")

    # Process
    with create_progress() as progress:
        task = progress.add_task("Annotating", total=total_prompts)

        if batch_size > 0:
            # Process in batches
            all_results = []
            total_responses = 0
            total_ops = 0

            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]
                batch_results, batch_resp, batch_ops = process_batch(
                    batch, responses_per_example, progress, task,
                    parallel=parallel, gpu_batch_size=gpu_batch
                )
                all_results.extend(batch_results)
                total_responses += batch_resp
                total_ops += batch_ops
                mx.metal.clear_cache()  # Clear memory between batches

            results = all_results
            total_ops_found = total_ops
        else:
            # Process all at once
            results, total_responses, total_ops_found = process_batch(
                examples, responses_per_example, progress, task,
                parallel=parallel, gpu_batch_size=gpu_batch
            )

    save_json(results, output_path)
    log.success(f"Saved to {output_path}")

    print_summary("Annotation Summary", {
        'examples': len(results),
        'responses_annotated': total_responses,
        'total_llm_ops_found': total_ops_found,
        'avg_ops_per_response': f"{total_ops_found / total_responses:.1f}" if total_responses else "0",
    })

    save_metadata(
        output_path.parent,
        config={
            'script': '05_llm_annotate.py',
            'input_file': str(input_path),
            'output_file': str(output_path),
            'model': MODEL_NAME,
            'limit': limit,
            'batch_size': batch_size,
            'parallel': parallel,
            'gpu_batch': gpu_batch,
        },
        stats={
            'examples': len(results),
            'responses_annotated': total_responses,
            'total_llm_ops_found': total_ops_found,
        },
    )


if __name__ == '__main__':
    app()
