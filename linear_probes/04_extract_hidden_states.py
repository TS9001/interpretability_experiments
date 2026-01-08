#!/usr/bin/env python3
"""
Full Hidden State Extraction for All Probes.

Extracts hidden states at various token positions for training all linear probes:
- Category A: Problem Understanding (A1, A2) - question position
- Category B: Numerical Representation (B1, B2) - operand/result positions
- Category C: Computation Mechanics (C1, C3_*, C4) - result/equals positions
- Category D: Sequential Reasoning (D1, D2, D3, D6) - result/operator positions

Automatically detects CUDA/MPS/CPU and optimizes accordingly:
- CUDA (H100): Uses bfloat16, Flash Attention 2, TF32, large batches
- MPS (Apple): Uses float32, smaller batches
- CPU: Fallback mode

Usage:
    python 04_extract_hidden_states.py
    python 04_extract_hidden_states.py --layers 0,7,14,21,27 --batch-size 4
    python 04_extract_hidden_states.py --probes A1,A2,B1,B2  # Subset of probes
"""
from pathlib import Path
from typing import Optional

import torch
import typer

from utils.data import load_json, save_json, format_prompt, parse_csv_ints, parse_csv_strings
from utils.logging import log, print_header, print_config, print_summary, create_progress
from utils.model import get_device, clear_memory, load_model_and_tokenizer
from utils.hidden_states import extract_hidden_batch
from utils.probe_positions import (
    get_magnitude_bin, get_coarse_bin, get_difficulty_bin, get_step_position,
    get_A1_labels, OPERATION_TYPES,
)

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "responses/Qwen2.5-Math-1.5B/train_responses_analyzed_probeable.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "probe_data"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

# Default layers for Qwen 1.5B (28 layers total)
DEFAULT_LAYERS = "0,7,14,21,27"

# All supported probes
ALL_PROBES = [
    # Category A - Problem Understanding
    'A1', 'A2',
    # Category B - Numerical Representation
    'B1', 'B2',
    # Category C - Computation Mechanics
    'C1', 'C3_add', 'C3_sub', 'C3_mult', 'C3_div', 'C4',
    # Category D - Sequential Reasoning
    'D1', 'D2', 'D3', 'D6',
]

DEFAULT_PROBES = ','.join(ALL_PROBES)


def get_default_batch_size(device: torch.device) -> int:
    """Get recommended batch size for device."""
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_gb >= 70:  # H100 80GB
            return 32
        elif mem_gb >= 40:  # A100 40GB
            return 16
        else:
            return 8
    elif device.type == "mps":
        return 4
    return 2  # CPU


def get_next_op_label(next_op: Optional[str]) -> int:
    """Convert next operation to label."""
    mapping = {'add': 0, 'sub': 1, 'mult': 2, 'div': 3, None: 4}
    return mapping.get(next_op, 4)  # None = END


def get_prev_op_label(prev_op: Optional[str]) -> int:
    """Convert previous operation to label (D6)."""
    mapping = {'add': 0, 'sub': 1, 'mult': 2, 'div': 3, None: 4}
    return mapping.get(prev_op, 4)  # None = FIRST


def find_equals_positions(tokens: list[str], response_start: int) -> list[int]:
    """Find positions of '=' tokens in the response."""
    positions = []
    for i, tok in enumerate(tokens):
        if i < response_start:
            continue
        clean = tok.replace('Ġ', '').replace('▁', '').strip()
        if clean == '=':
            positions.append(i)
    return positions


def collect_probe_samples(
    example: dict,
    response: dict,
    hidden_states: torch.Tensor,
    positions: list[int],
    prompt_len: int,
    question_hidden: Optional[torch.Tensor],
    probe_list: list[str],
    equals_positions: list[int],
) -> dict:
    """
    Collect probe samples from a single response.

    Args:
        example: Full example dict with ground_truth
        response: Single response dict
        hidden_states: Tensor of shape (num_layers, num_positions, hidden_dim)
        positions: List of valid positions that were extracted
        prompt_len: Length of prompt in tokens
        question_hidden: Hidden state at last question token for A1/A2
        probe_list: List of probes to collect samples for
        equals_positions: Positions of '=' tokens for C4

    Returns:
        Dict mapping probe name to list of sample dicts
    """
    samples = {probe: [] for probe in probe_list}
    operations = response.get('operations', [])
    total_ops = len(operations)

    # Build position -> index mapping
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}

    # Ground truth info for A1, A2
    ground_truth = example.get('ground_truth', {})
    ops_by_type = ground_truth.get('operations_by_type', {})
    gt_total_ops = ground_truth.get('total_operations', len(operations))

    # A1: Operation planning (multi-label) - at question position
    if 'A1' in probe_list and question_hidden is not None:
        a1_labels = get_A1_labels(ops_by_type)
        samples['A1'].append({
            'hidden': question_hidden,
            'label': a1_labels,  # [needs_add, needs_sub, needs_mult, needs_div]
            'question_idx': example.get('index', 0),
        })

    # A2: Difficulty prediction - at question position
    if 'A2' in probe_list and question_hidden is not None:
        samples['A2'].append({
            'hidden': question_hidden,
            'label': get_difficulty_bin(gt_total_ops),
            'total_ops': gt_total_ops,
            'question_idx': example.get('index', 0),
        })

    # C4: Coarse result prediction at '=' positions
    if 'C4' in probe_list:
        for eq_idx, eq_pos in enumerate(equals_positions):
            if eq_pos in pos_to_idx:
                idx = pos_to_idx[eq_pos]
                # Find the operation this '=' belongs to (approximate by finding next result)
                for op in operations:
                    result_pos = op['result_pos'] + prompt_len
                    if result_pos > eq_pos:
                        samples['C4'].append({
                            'hidden': hidden_states[:, idx, :],
                            'label': get_coarse_bin(op['result']),
                            'result_value': op['result'],
                        })
                        break

    # Process each operation
    for op_idx, op in enumerate(operations):
        # Get adjusted positions (response positions + prompt offset)
        op1_pos = op['operand1_pos'] + prompt_len
        op2_pos = op['operand2_pos'] + prompt_len
        operator_pos = op['operator_pos'] + prompt_len
        result_pos = op['result_pos'] + prompt_len

        op_type = op['operator']
        is_correct = op.get('is_correct', True)
        step_label = get_step_position(op_idx, total_ops)

        # Previous operation for D6
        prev_op = operations[op_idx - 1]['operator'] if op_idx > 0 else None

        # B1: Operand magnitude (both operands)
        if 'B1' in probe_list:
            if op1_pos in pos_to_idx:
                idx = pos_to_idx[op1_pos]
                samples['B1'].append({
                    'hidden': hidden_states[:, idx, :],
                    'label': get_magnitude_bin(op['operand1']),
                    'value': op['operand1'],
                    'op_type': op_type,
                })

            if op2_pos in pos_to_idx:
                idx = pos_to_idx[op2_pos]
                samples['B1'].append({
                    'hidden': hidden_states[:, idx, :],
                    'label': get_magnitude_bin(op['operand2']),
                    'value': op['operand2'],
                    'op_type': op_type,
                })

        # D6: Previous operation (at operator position)
        if 'D6' in probe_list and operator_pos in pos_to_idx:
            idx = pos_to_idx[operator_pos]
            samples['D6'].append({
                'hidden': hidden_states[:, idx, :],
                'label': get_prev_op_label(prev_op),
                'current_op': op_type,
                'prev_op': prev_op,
            })

        # D3: Step position (at operator position)
        if 'D3' in probe_list and operator_pos in pos_to_idx:
            idx = pos_to_idx[operator_pos]
            samples['D3'].append({
                'hidden': hidden_states[:, idx, :],
                'label': step_label,
                'op_idx': op_idx,
                'total_ops': total_ops,
                'position_type': 'operator',
            })

        # Result-based probes
        if result_pos in pos_to_idx:
            idx = pos_to_idx[result_pos]
            result_hidden = hidden_states[:, idx, :]

            # B2: Result magnitude
            if 'B2' in probe_list:
                samples['B2'].append({
                    'hidden': result_hidden,
                    'label': get_magnitude_bin(op['result']),
                    'value': op['result'],
                    'op_type': op_type,
                })

            # C1: Correctness
            if 'C1' in probe_list:
                samples['C1'].append({
                    'hidden': result_hidden,
                    'label': 1 if is_correct else 0,
                    'op_type': op_type,
                })

            # C3_*: Per-operation-type correctness
            c3_probe = f'C3_{op_type}'
            if c3_probe in probe_list:
                samples[c3_probe].append({
                    'hidden': result_hidden,
                    'label': 1 if is_correct else 0,
                })

            # D1: Intermediate vs final
            if 'D1' in probe_list:
                samples['D1'].append({
                    'hidden': result_hidden,
                    'label': 1 if op.get('is_intermediate', False) else 0,
                    'op_type': op_type,
                })

            # D2: Next operation prediction
            if 'D2' in probe_list:
                samples['D2'].append({
                    'hidden': result_hidden,
                    'label': get_next_op_label(op.get('next_op')),
                    'op_type': op_type,
                })

            # D3: Step position (at result position)
            if 'D3' in probe_list:
                samples['D3'].append({
                    'hidden': result_hidden,
                    'label': step_label,
                    'op_idx': op_idx,
                    'total_ops': total_ops,
                    'position_type': 'result',
                })

    return samples


def process_batch(
    batch_examples: list[dict],
    model,
    tokenizer,
    device: torch.device,
    layers: list[int],
    probe_list: list[str],
) -> dict:
    """Process a batch of examples and collect probe samples."""
    all_samples = {probe: [] for probe in probe_list}
    needs_question_hidden = 'A1' in probe_list or 'A2' in probe_list
    needs_equals = 'C4' in probe_list

    # Prepare batch data
    texts = []
    positions_list = []
    metadata = []  # (example, response, prompt_len, question_last_pos, equals_positions)

    for example in batch_examples:
        question = example['question']
        prompt = format_prompt(question)
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)
        question_last_pos = prompt_len - 1  # Last token of prompt

        for response in example.get('responses', []):
            # Collect all positions from this response
            positions = set()

            # Add question position for A1/A2
            if needs_question_hidden:
                positions.add(question_last_pos)

            # Add operation positions
            for op in response.get('operations', []):
                positions.add(op['operand1_pos'] + prompt_len)
                positions.add(op['operand2_pos'] + prompt_len)
                positions.add(op['operator_pos'] + prompt_len)
                positions.add(op['result_pos'] + prompt_len)

            if not positions:
                continue

            full_text = prompt + response['text']

            # Find equals positions for C4
            equals_positions = []
            if needs_equals:
                full_tokens = tokenizer(full_text, add_special_tokens=False)['input_ids']
                full_token_strs = tokenizer.convert_ids_to_tokens(full_tokens)
                for i, tok in enumerate(full_token_strs):
                    if i >= prompt_len:
                        clean = tok.replace('Ġ', '').replace('▁', '').strip()
                        if clean == '=':
                            positions.add(i)
                            equals_positions.append(i)

            texts.append(full_text)
            positions_list.append(sorted(positions))
            metadata.append((example, response, prompt_len, question_last_pos, equals_positions))

    if not texts:
        return all_samples

    # Batch extract hidden states
    results = extract_hidden_batch(
        model, tokenizer, texts, positions_list, device, layers_to_save=layers
    )

    # Collect samples from each result
    for i, result in enumerate(results):
        if result is None:
            continue

        hidden_states, valid_positions = result
        example, response, prompt_len, question_last_pos, equals_positions = metadata[i]

        # Get question hidden state for A1/A2
        question_hidden = None
        if needs_question_hidden and question_last_pos in valid_positions:
            q_idx = valid_positions.index(question_last_pos)
            question_hidden = hidden_states[:, q_idx, :]

        samples = collect_probe_samples(
            example, response, hidden_states, valid_positions, prompt_len,
            question_hidden, probe_list, equals_positions
        )

        for probe in probe_list:
            all_samples[probe].extend(samples[probe])

    return all_samples


@app.command()
def main(
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input probeable JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    layers: str = typer.Option(DEFAULT_LAYERS, "--layers", "-l", help="Layers to extract (comma-separated)"),
    probes: str = typer.Option(DEFAULT_PROBES, "--probes", "-p", help="Probes to extract (comma-separated)"),
    batch_size: int = typer.Option(0, "--batch-size", "-b", help="Batch size (0=auto-detect)"),
    max_examples: int = typer.Option(-1, "--max-examples", "-n", help="Max examples to process (-1=all)"),
    no_flash_attn: bool = typer.Option(False, "--no-flash-attn", help="Disable Flash Attention 2"),
):
    """Extract hidden states at probe positions for all probes."""
    input_path = input_file or DEFAULT_INPUT
    output_path = output_dir or DEFAULT_OUTPUT

    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    layer_indices = parse_csv_ints(layers)
    probe_list = parse_csv_strings(probes, default=ALL_PROBES)

    # Validate probes
    invalid_probes = set(probe_list) - set(ALL_PROBES)
    if invalid_probes:
        log.error(f"Invalid probes: {invalid_probes}")
        log.info(f"Valid probes: {ALL_PROBES}")
        raise typer.Exit(1)

    # Get device and auto-detect batch size
    device = get_device()
    if batch_size <= 0:
        batch_size = get_default_batch_size(device)
        log.info(f"Auto-detected batch size: {batch_size}")

    print_header("Full Hidden State Extraction", "All Probes")
    print_config("Configuration", {
        'input': str(input_path),
        'output': str(output_path),
        'model': MODEL_NAME,
        'layers': layers,
        'batch_size': batch_size,
        'probes': ', '.join(probe_list),
        'n_probes': len(probe_list),
    })

    # Load model
    log.info(f"Loading model on {device}...")
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME, device, use_flash_attn=not no_flash_attn
    )

    # IMPORTANT: Use right padding for position-accurate hidden state extraction
    tokenizer.padding_side = 'right'

    # Load data
    log.info("Loading probeable data...")
    data = load_json(input_path)

    if max_examples > 0:
        data = data[:max_examples]

    log.info(f"Processing {len(data)} examples")

    # Process in batches
    all_samples = {probe: [] for probe in probe_list}

    with create_progress() as progress:
        task = progress.add_task("Extracting", total=len(data))

        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start:batch_start + batch_size]

            try:
                batch_samples = process_batch(
                    batch, model, tokenizer, device, layer_indices, probe_list
                )

                for probe in probe_list:
                    all_samples[probe].extend(batch_samples[probe])

            except Exception as e:
                log.error(f"Batch error at {batch_start}: {e}")
                import traceback
                traceback.print_exc()

            progress.advance(task, len(batch))
            clear_memory(device)

    # Convert to tensors and save
    log.info("Saving probe data...")

    samples_summary = {}
    for probe in probe_list:
        samples = all_samples[probe]
        if not samples:
            log.warning(f"No samples for probe {probe}")
            samples_summary[probe] = 0
            continue

        # Handle A1 specially (multi-label)
        if probe == 'A1':
            X = torch.stack([s['hidden'] for s in samples], dim=0)
            y = torch.tensor([s['label'] for s in samples], dtype=torch.float)  # Multi-label
            n_classes = 4  # 4 binary labels
        else:
            X = torch.stack([s['hidden'] for s in samples], dim=0)
            y = torch.tensor([s['label'] for s in samples], dtype=torch.long)
            n_classes = len(set(y.tolist()))

        probe_data = {
            'X': X,
            'y': y,
            'layers': layer_indices,
            'n_samples': len(samples),
            'n_classes': n_classes,
            'probe_type': 'multi_label' if probe == 'A1' else 'classification',
        }

        probe_path = output_path / f"{probe}_samples.pt"
        torch.save(probe_data, probe_path)
        log.info(f"  {probe}: {len(samples)} samples, {n_classes} classes → {probe_path.name}")
        samples_summary[probe] = len(samples)

    # Save metadata
    save_json({
        'model': MODEL_NAME,
        'layers': layer_indices,
        'probes': probe_list,
        'input_file': str(input_path),
        'samples_per_probe': samples_summary,
    }, output_path / 'metadata.json')

    # Summary
    total_samples = sum(samples_summary.values())
    summary_dict = {
        'Examples processed': len(data),
        'Total samples': total_samples,
    }
    for probe in probe_list:
        summary_dict[probe] = samples_summary.get(probe, 0)
    summary_dict['Output'] = str(output_path)

    print_summary("Extraction Summary", summary_dict)


if __name__ == '__main__':
    app()
