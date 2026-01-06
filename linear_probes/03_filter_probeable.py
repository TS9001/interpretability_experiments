#!/usr/bin/env python3
"""
POC Step 1: Filter to probeable responses only.

Selects responses where ALL operations have explicit token positions found.
Picks the "best" response per example based on:
1. Final answer correct (if available)
2. All operations have valid positions
3. Number of operations matches ground truth (bonus)

Usage:
    python poc_01_filter_probeable.py
    python poc_01_filter_probeable.py --max-responses 3
    python poc_01_filter_probeable.py --require-correct
"""
from pathlib import Path
from typing import Optional
from collections import Counter

import typer

from utils.data import load_json, save_json
from utils.logging import log, print_header, print_config, print_summary

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json"


def is_operation_probeable(op: dict) -> bool:
    """Check if an operation has all required token positions found."""
    positions = [
        op.get('operand1_positions', [-1])[0],
        op.get('operator_positions', [-1])[0],
        op.get('operand2_positions', [-1])[0],
        op.get('result_positions', [-1])[0],
    ]
    return all(p >= 0 for p in positions)


def is_response_probeable(resp: dict) -> bool:
    """Check if ALL operations in a response are probeable."""
    ops = resp.get('operations', [])
    if not ops:
        return False
    return all(is_operation_probeable(op) for op in ops)


def score_response(resp: dict, ground_truth_ops: list, ground_truth_result: Optional[float]) -> tuple:
    """
    Score a response for selection.

    Returns tuple for sorting (higher is better):
    (final_correct, all_ops_correct, ops_match_gt, num_ops)
    """
    ops = resp.get('operations', [])

    # Final answer correctness
    final_correct = 1 if resp.get('final_correct') else 0

    # All operations arithmetically correct
    all_ops_correct = 1 if all(op.get('is_correct', False) for op in ops) else 0

    # Number of operations matches ground truth
    ops_match_gt = 1 if len(ops) == len(ground_truth_ops) else 0

    return (final_correct, all_ops_correct, ops_match_gt, len(ops))


def add_probe_metadata(resp: dict, resp_idx: int, example: dict) -> dict:
    """Add metadata needed for probing to a response."""
    ops = resp.get('operations', [])
    total_ops = len(ops)

    # Enrich operations with sequential metadata
    enriched_ops = []
    for idx, op in enumerate(ops):
        enriched_op = {
            # Core operation data
            'operand1': op['operand1'],
            'operand2': op['operand2'],
            'operator': op['operator'],
            'result': op['result'],
            'is_correct': op.get('is_correct', True),

            # Token positions
            'operand1_pos': op.get('operand1_positions', [-1])[0],
            'operand2_pos': op.get('operand2_positions', [-1])[0],
            'operator_pos': op.get('operator_positions', [-1])[0],
            'result_pos': op.get('result_positions', [-1])[0],

            # Sequence metadata for D-probes
            'op_index': idx,
            'is_first': idx == 0,
            'is_last': idx == total_ops - 1,
            'step_position': 0 if idx == 0 else (2 if idx == total_ops - 1 else 1),  # first/middle/last

            # Check if result feeds into next operation
            'is_intermediate': False,
            'next_op': None,
        }

        # Determine if intermediate (result used in next operation)
        if idx < total_ops - 1:
            next_operation = ops[idx + 1]
            if (op['result'] == next_operation.get('operand1') or
                op['result'] == next_operation.get('operand2')):
                enriched_op['is_intermediate'] = True
            enriched_op['next_op'] = next_operation['operator']

        enriched_ops.append(enriched_op)

    return {
        'response_idx': resp_idx,
        'text': resp['text'],
        'final_answer': resp.get('final_answer'),
        'final_correct': resp.get('final_correct'),
        'num_operations': total_ops,
        'all_ops_correct': all(op.get('is_correct', False) for op in ops),
        'operations': enriched_ops,
    }


def compute_operations_by_type(ops: list) -> dict:
    """Compute operation counts by type."""
    counts = {'add': 0, 'sub': 0, 'mult': 0, 'div': 0}
    for op in ops:
        op_type = op.get('operator', '')
        if op_type in counts:
            counts[op_type] += 1
    return counts


@app.command()
def main(
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input analyzed responses JSON"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output filtered JSON"),
    max_responses: int = typer.Option(1, "--max-responses", "-n", help="Max responses to keep per example"),
    require_correct: bool = typer.Option(False, "--require-correct", help="Only keep responses with correct final answer"),
    min_operations: int = typer.Option(1, "--min-ops", help="Minimum operations required"),
):
    """Filter to probeable responses and prepare for hidden state extraction."""
    input_path = input_file or DEFAULT_INPUT

    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    output_path = output_file or input_path.parent / f"{input_path.stem}_probeable.json"

    print_header("POC Filter", "Step 1 - Filter Probeable Responses")
    print_config("Configuration", {
        'input': str(input_path),
        'output': str(output_path),
        'max_responses': max_responses,
        'require_correct': require_correct,
        'min_operations': min_operations,
    })

    log.info("Loading analyzed responses...")
    data = load_json(input_path)
    log.info(f"Loaded {len(data)} examples")

    # Process each example
    results = []
    stats = Counter()

    for example in data:
        stats['total_examples'] += 1
        ground_truth_ops = example.get('ground_truth_operations', [])
        ground_truth_result = example.get('ground_truth_final_result')

        # Filter to probeable responses
        probeable_responses = []
        for resp_idx, resp in enumerate(example.get('responses', [])):
            stats['total_responses'] += 1

            if not is_response_probeable(resp):
                stats['not_probeable'] += 1
                continue

            ops = resp.get('operations', [])
            if len(ops) < min_operations:
                stats['too_few_ops'] += 1
                continue

            if require_correct and not resp.get('final_correct'):
                stats['not_correct'] += 1
                continue

            stats['probeable'] += 1
            score = score_response(resp, ground_truth_ops, ground_truth_result)
            probeable_responses.append((score, resp_idx, resp))

        if not probeable_responses:
            stats['examples_no_probeable'] += 1
            continue

        # Sort by score (descending) and take top N
        probeable_responses.sort(key=lambda x: x[0], reverse=True)
        selected = probeable_responses[:max_responses]

        # Build output example
        selected_responses = [
            add_probe_metadata(resp, resp_idx, example)
            for score, resp_idx, resp in selected
        ]

        # Compute A1 labels (which operation types are needed) from ground truth
        gt_ops_by_type = example.get('operations_by_type', {})
        if not gt_ops_by_type:
            gt_ops_by_type = compute_operations_by_type(ground_truth_ops)

        results.append({
            'index': example['index'],
            'question': example['question'],

            # Ground truth for A-probes
            'ground_truth': {
                'final_result': ground_truth_result,
                'operations': ground_truth_ops,
                'operations_by_type': gt_ops_by_type,
                'total_operations': len(ground_truth_ops),
                'operation_sequence': example.get('operation_sequence', []),
            },

            # A-probe labels (computed from ground truth, same for all responses)
            'A1_label': [
                1 if gt_ops_by_type.get('add', 0) > 0 else 0,
                1 if gt_ops_by_type.get('sub', 0) > 0 else 0,
                1 if gt_ops_by_type.get('mult', 0) > 0 else 0,
                1 if gt_ops_by_type.get('div', 0) > 0 else 0,
            ],
            'A2_label': min(len(ground_truth_ops), 5) - 1 if ground_truth_ops else 0,

            # Selected responses
            'responses': selected_responses,
            'num_responses': len(selected_responses),
        })

        stats['examples_with_probeable'] += 1
        stats['selected_responses'] += len(selected_responses)
        stats['total_operations'] += sum(r['num_operations'] for r in selected_responses)

    # Save results
    save_json(results, output_path)
    log.success(f"Saved {len(results)} examples to {output_path}")

    # Summary
    print_summary("Filter Summary", {
        'Input examples': stats['total_examples'],
        'Input responses': stats['total_responses'],
        'Probeable responses': stats['probeable'],
        'Not probeable (missing positions)': stats['not_probeable'],
        'Too few operations': stats['too_few_ops'],
        'Not correct (filtered)': stats['not_correct'],
        'Examples with probeable': stats['examples_with_probeable'],
        'Examples without probeable': stats['examples_no_probeable'],
        'Output examples': len(results),
        'Output responses': stats['selected_responses'],
        'Total operations for probing': stats['total_operations'],
    })

    # Estimate probe training data
    avg_ops = stats['total_operations'] / max(stats['selected_responses'], 1)
    log.info(f"\n📊 Probe Training Data Estimate:")
    log.info(f"   B1 (operands): ~{int(stats['total_operations'] * 2)} samples")
    log.info(f"   B2, C1, D1, D2 (results): ~{stats['total_operations']} samples")
    log.info(f"   D6 (operators): ~{stats['total_operations']} samples")
    log.info(f"   A1, A2 (questions): ~{len(results)} samples")


if __name__ == '__main__':
    app()
