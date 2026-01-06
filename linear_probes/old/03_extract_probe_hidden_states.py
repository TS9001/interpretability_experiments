#!/usr/bin/env python3
"""
Step 3: Extract hidden states at probe-relevant positions only.

Uses probe_positions.py to identify which positions are needed for each probe,
then extracts hidden states only at those positions for efficient storage.

Usage:
    python 03_extract_probe_hidden_states.py responses/train_analyzed.jsonl
    python 03_extract_probe_hidden_states.py responses/train_analyzed.jsonl --probes A1,C1 --layers 0,7,14
"""
from pathlib import Path
from typing import Optional

import torch
import typer

from utils.args import InputFile, OutputDir, MaxExamples, Compile, BatchSize, Layers, ProbeTypes
from utils.data import load_jsonl, save_json, format_prompt, parse_csv_ints, parse_csv_strings
from utils.logging import log, print_header, print_config, print_summary, create_progress
from utils.model import get_device, clear_memory, load_model_and_tokenizer
from utils.probe_data import adjust_probe_positions, build_pos_to_idx, save_probe_data
from utils.probe_positions import PROBE_POSITION_MAP, get_all_probe_positions, get_unique_positions
from utils.labels import generate_probe_labels
from utils.hidden_states import extract_hidden_batch


SCRIPT_DIR = Path(__file__).parent
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
ALL_PROBES = list(PROBE_POSITION_MAP.keys())

app = typer.Typer(add_completion=False)


def prepare_response_data(
    example: dict,
    resp: dict,
    resp_idx: int,
    tokenizer,
    prompt_length: int,
    probes: list[str]
) -> dict:
    """Prepare response data with positions and labels (without hidden states)."""
    probe_positions = get_all_probe_positions(
        operations=resp.get('operations', []),
        tokens=resp.get('tokens', []),
        question=example['question'],
        tokenizer=tokenizer,
        operations_by_type=example.get('ground_truth', {}).get('operations_by_type'),
        total_operations=len(resp.get('operations', [])),
    )

    return {
        'resp_idx': resp_idx,
        'text': resp['text'],
        'final_answer': resp.get('final_answer'),
        'final_correct': resp.get('final_correct'),
        'operations': resp.get('operations', []),
        'probe_positions': adjust_probe_positions(probe_positions, prompt_length),
        'labels': generate_probe_labels(probes, example, resp.get('operations', [])),
    }


def process_batch(
    examples: list[dict],
    model,
    tokenizer,
    device: torch.device,
    layers_to_save: Optional[list[int]] = None,
    probes: Optional[list[str]] = None
) -> list[dict]:
    """Process multiple examples in batch for efficient forward passes."""
    if probes is None:
        probes = ALL_PROBES

    # Prepare all response data
    batch_items = []  # (example_idx, resp_data, full_text, positions)

    for example in examples:
        prompt = format_prompt(example['question'])
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])

        for resp_idx, resp in enumerate(example.get('responses', [])):
            resp_data = prepare_response_data(example, resp, resp_idx, tokenizer, prompt_length, probes)
            all_positions = get_unique_positions(resp_data['probe_positions'])

            if all_positions:
                batch_items.append((example['index'], resp_data, prompt + resp['text'], all_positions))

    # Initialize results
    example_results = {ex['index']: {
        'index': ex['index'],
        'question': ex['question'],
        'ground_truth': {
            'answer': ex.get('ground_truth_answer', ''),
            'operations': ex.get('ground_truth_operations', []),
            'final_result': ex.get('ground_truth_final_result'),
            'operation_sequence': ex.get('operation_sequence', []),
        },
        'responses': [],
    } for ex in examples}

    if not batch_items:
        return list(example_results.values())

    # Batch forward pass
    extraction_results = extract_hidden_batch(
        model, tokenizer,
        texts=[item[2] for item in batch_items],
        positions_list=[item[3] for item in batch_items],
        device=device,
        layers_to_save=layers_to_save
    )

    # Organize results
    for i, (example_idx, resp_data, _, _) in enumerate(batch_items):
        if extraction_results[i] is None:
            continue

        hidden_states, valid_positions = extraction_results[i]
        example_results[example_idx]['responses'].append({
            'text': resp_data['text'],
            'final_answer': resp_data['final_answer'],
            'final_correct': resp_data['final_correct'],
            'operations': resp_data['operations'],
            'hidden_states': hidden_states,
            'positions': valid_positions,
            'pos_to_idx': build_pos_to_idx(valid_positions),
            'probe_positions': resp_data['probe_positions'],
            'labels': resp_data['labels'],
        })

    return list(example_results.values())


@app.command()
def main(
    input_file: InputFile,
    output_dir: OutputDir = None,
    probes: ProbeTypes = None,
    layers: Layers = "0,7,14,21,27",
    max_examples: MaxExamples = -1,
    compile: Compile = False,
    batch_size: BatchSize = 8,
):
    """Extract hidden states at probe-relevant positions."""
    input_path = Path(input_file)
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        raise typer.Exit(1)

    out_dir = Path(output_dir) if output_dir else SCRIPT_DIR / "probe_data"
    layers_to_save = parse_csv_ints(layers)
    probe_list = parse_csv_strings(probes, default=ALL_PROBES)

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device, use_compile=compile)

    log.info(f"Loading responses from: {input_path}")
    examples = load_jsonl(input_path)
    log.info(f"Loaded {len(examples)} examples")

    if max_examples > 0:
        examples = examples[:max_examples]
        log.info(f"Limited to {max_examples} examples")

    print_header("Hidden State Extraction", "Step 3")
    print_config("Configuration", {
        'device': str(device),
        'probes': ', '.join(probe_list),
        'layers': layers,
        'batch_size': batch_size,
        'output': str(out_dir),
    })

    results = []
    with create_progress() as progress:
        task = progress.add_task("Extracting", total=len(examples))

        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start:batch_start + batch_size]

            try:
                batch_results = process_batch(
                    batch_examples, model, tokenizer, device, layers_to_save, probe_list
                )
                results.extend(batch_results)
            except Exception as e:
                log.error(f"Batch error at {batch_start}: {e}")

            progress.advance(task, len(batch_examples))
            clear_memory(device)

    log.info("Saving probe data...")
    save_probe_data(results, out_dir, probe_list)

    save_json({
        'model_name': MODEL_NAME,
        'input_file': str(input_path),
        'config': {'probes': probe_list, 'layers': layers, 'batch_size': batch_size},
        'num_examples': len(results),
    }, out_dir / 'metadata.json')

    print_summary("Extraction Summary", {
        'examples_processed': len(results),
        'output_directory': str(out_dir),
    })
    log.success("Complete")


if __name__ == '__main__':
    app()
