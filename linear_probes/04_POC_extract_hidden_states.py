#!/usr/bin/env python3
"""
POC Step 2: Extract hidden states at operation positions.

Extracts hidden states at operand, operator, and result token positions
for training linear probes.

Automatically detects CUDA/MPS/CPU and optimizes accordingly:
- CUDA (H100): Uses bfloat16, Flash Attention 2, TF32, large batches
- MPS (Apple): Uses float32, smaller batches
- CPU: Fallback mode

Usage:
    python 04_extract_hidden_states.py
    python 04_extract_hidden_states.py --layers 0,7,14,21,27 --batch-size 4
    python 04_extract_hidden_states.py --batch-size 32  # H100 with 80GB VRAM
"""
from pathlib import Path
from typing import Optional

import torch
import typer

from utils.data import load_json, save_json, format_prompt, parse_csv_ints
from utils.logging import log, print_header, print_config, print_summary, create_progress
from utils.model import get_device, clear_memory, load_model_and_tokenizer, get_default_batch_size
from utils.hidden_states import extract_hidden_batch
from utils.probe_positions import get_magnitude_bin, get_next_op_label

app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "responses/Qwen2.5-Math-1.5B/train_responses_analyzed_probeable.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "probe_data"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

# Default layers for Qwen 1.5B (28 layers total)
DEFAULT_LAYERS = "0,7,14,21,27"

# POC Probes
POC_PROBES = ['B1', 'B2', 'C1', 'D1', 'D2']


def collect_probe_samples(
    example: dict,
    response: dict,
    hidden_states: torch.Tensor,
    positions: list[int],
    prompt_len: int,
) -> dict:
    """
    Collect probe samples from a single response.

    Args:
        example: Full example dict
        response: Single response dict
        hidden_states: Tensor of shape (num_layers, num_positions, hidden_dim)
        positions: List of valid positions that were extracted
        prompt_len: Length of prompt in tokens (for adjusting positions)

    Returns:
        Dict mapping probe name to list of (hidden_state, label, metadata) tuples
    """
    samples = {probe: [] for probe in POC_PROBES}

    # Build position -> index mapping
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}

    for op in response.get('operations', []):
        # Get adjusted positions (response positions + prompt offset)
        op1_pos = op['operand1_pos'] + prompt_len
        op2_pos = op['operand2_pos'] + prompt_len
        operator_pos = op['operator_pos'] + prompt_len
        result_pos = op['result_pos'] + prompt_len

        # B1: Operand magnitude (both operands)
        if op1_pos in pos_to_idx:
            idx = pos_to_idx[op1_pos]
            samples['B1'].append({
                'hidden': hidden_states[:, idx, :],  # (num_layers, hidden_dim)
                'label': get_magnitude_bin(op['operand1']),
                'value': op['operand1'],
                'op_type': op['operator'],
            })

        if op2_pos in pos_to_idx:
            idx = pos_to_idx[op2_pos]
            samples['B1'].append({
                'hidden': hidden_states[:, idx, :],
                'label': get_magnitude_bin(op['operand2']),
                'value': op['operand2'],
                'op_type': op['operator'],
            })

        # B2: Result magnitude
        if result_pos in pos_to_idx:
            idx = pos_to_idx[result_pos]
            result_hidden = hidden_states[:, idx, :]

            samples['B2'].append({
                'hidden': result_hidden,
                'label': get_magnitude_bin(op['result']),
                'value': op['result'],
                'op_type': op['operator'],
            })

            # C1: Correctness (at result position)
            samples['C1'].append({
                'hidden': result_hidden,
                'label': 1 if op.get('is_correct', True) else 0,
                'op_type': op['operator'],
            })

            # D1: Intermediate vs final
            samples['D1'].append({
                'hidden': result_hidden,
                'label': 1 if op.get('is_intermediate', False) else 0,
                'op_type': op['operator'],
            })

            # D2: Next operation prediction
            samples['D2'].append({
                'hidden': result_hidden,
                'label': get_next_op_label(op.get('next_op')),
                'op_type': op['operator'],
            })

    return samples


def process_batch(
    batch_examples: list[dict],
    model,
    tokenizer,
    device: torch.device,
    layers: list[int],
) -> dict:
    """Process a batch of examples and collect probe samples."""
    all_samples = {probe: [] for probe in POC_PROBES}

    # Prepare batch data
    texts = []
    positions_list = []
    metadata = []  # (example, response, prompt_len)

    for example in batch_examples:
        question = example['question']
        prompt = format_prompt(question)
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)

        for response in example.get('responses', []):
            # Collect all positions from this response
            positions = set()
            for op in response.get('operations', []):
                # Adjust positions for prompt
                positions.add(op['operand1_pos'] + prompt_len)
                positions.add(op['operand2_pos'] + prompt_len)
                positions.add(op['operator_pos'] + prompt_len)
                positions.add(op['result_pos'] + prompt_len)

            if not positions:
                continue

            full_text = prompt + response['text']
            texts.append(full_text)
            positions_list.append(sorted(positions))
            metadata.append((example, response, prompt_len))

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
        example, response, prompt_len = metadata[i]

        samples = collect_probe_samples(
            example, response, hidden_states, valid_positions, prompt_len
        )

        for probe in POC_PROBES:
            all_samples[probe].extend(samples[probe])

    return all_samples


@app.command()
def main(
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input probeable JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    layers: str = typer.Option(DEFAULT_LAYERS, "--layers", "-l", help="Layers to extract (comma-separated)"),
    batch_size: int = typer.Option(0, "--batch-size", "-b", help="Batch size (0=auto-detect, H100: 32-64)"),
    max_examples: int = typer.Option(-1, "--max-examples", "-n", help="Max examples to process (-1=all)"),
    no_flash_attn: bool = typer.Option(False, "--no-flash-attn", help="Disable Flash Attention 2"),
):
    """Extract hidden states at probe positions."""
    input_path = input_file or DEFAULT_INPUT
    output_path = output_dir or DEFAULT_OUTPUT

    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    layer_indices = parse_csv_ints(layers)

    # Get device and auto-detect batch size
    device = get_device()
    if batch_size <= 0:
        batch_size = get_default_batch_size(device)
        log.info(f"Auto-detected batch size: {batch_size}")

    print_header("POC Hidden State Extraction", "Step 2")
    print_config("Configuration", {
        'input': str(input_path),
        'output': str(output_path),
        'model': MODEL_NAME,
        'layers': layers,
        'batch_size': batch_size,
        'probes': ', '.join(POC_PROBES),
    })

    # Load model
    log.info(f"Loading model on {device}...")
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME, device, use_flash_attn=not no_flash_attn
    )

    # IMPORTANT: Use right padding for position-accurate hidden state extraction
    # Left padding (used for generation) shifts token positions and breaks extraction
    tokenizer.padding_side = 'right'

    # Load data
    log.info("Loading probeable data...")
    data = load_json(input_path)

    if max_examples > 0:
        data = data[:max_examples]

    log.info(f"Processing {len(data)} examples")

    # Process in batches
    all_samples = {probe: [] for probe in POC_PROBES}

    with create_progress() as progress:
        task = progress.add_task("Extracting", total=len(data))

        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start:batch_start + batch_size]

            try:
                batch_samples = process_batch(batch, model, tokenizer, device, layer_indices)

                for probe in POC_PROBES:
                    all_samples[probe].extend(batch_samples[probe])

            except Exception as e:
                log.error(f"Batch error at {batch_start}: {e}")
                import traceback
                traceback.print_exc()

            progress.advance(task, len(batch))
            clear_memory(device)

    # Convert to tensors and save
    log.info("Saving probe data...")

    for probe in POC_PROBES:
        samples = all_samples[probe]
        if not samples:
            log.warning(f"No samples for probe {probe}")
            continue

        # Stack hidden states: (n_samples, n_layers, hidden_dim)
        X = torch.stack([s['hidden'] for s in samples], dim=0)
        y = torch.tensor([s['label'] for s in samples], dtype=torch.long)

        # Save metadata separately (not as tensor)
        meta = [{k: v for k, v in s.items() if k not in ['hidden', 'label']} for s in samples]

        probe_data = {
            'X': X,  # (n_samples, n_layers, hidden_dim)
            'y': y,  # (n_samples,)
            'layers': layer_indices,
            'n_samples': len(samples),
            'n_classes': len(set(y.tolist())),
        }

        probe_path = output_path / f"{probe}_samples.pt"
        torch.save(probe_data, probe_path)
        log.info(f"  {probe}: {len(samples)} samples, {probe_data['n_classes']} classes → {probe_path.name}")

    # Save metadata
    save_json({
        'model': MODEL_NAME,
        'layers': layer_indices,
        'probes': POC_PROBES,
        'input_file': str(input_path),
        'samples_per_probe': {probe: len(all_samples[probe]) for probe in POC_PROBES},
    }, output_path / 'metadata.json')

    # Summary
    total_samples = sum(len(all_samples[p]) for p in POC_PROBES)
    print_summary("Extraction Summary", {
        'Examples processed': len(data),
        'Total samples': total_samples,
        'B1 (operand magnitude)': len(all_samples['B1']),
        'B2 (result magnitude)': len(all_samples['B2']),
        'C1 (correctness)': len(all_samples['C1']),
        'D1 (intermediate)': len(all_samples['D1']),
        'D2 (next operation)': len(all_samples['D2']),
        'Output': str(output_path),
    })


if __name__ == '__main__':
    app()
