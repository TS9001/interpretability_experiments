#!/usr/bin/env python3
"""
Step 1: Generate responses (batch processing only).

Fast batched generation to collect responses. Analysis of operations/correctness
should be done in a separate script using the proper regexp_utils.py.

Automatically detects CUDA/MPS/CPU and optimizes accordingly:
- CUDA (H100): Uses bfloat16, Flash Attention 2, TF32, large batches
- MPS (Apple): Uses float32, smaller batches
- CPU: Fallback mode

Usage:
    python 01_generate_responses.py --max-examples 100
    python 01_generate_responses.py --max-examples -1 --compile --batch-size 8
    python 01_generate_responses.py --max-examples -1 --batch-size 32  # H100 with 80GB VRAM
"""
from pathlib import Path
from typing import Annotated

import torch
import typer
from tqdm import tqdm

from utils.args import (
    MaxExamples, OutputDir, Split, Resume,
    NumResponses, MaxNewTokens, Compile, BatchSize, NoFlashAttn,
)
from utils.data import load_jsonl, format_prompt, get_existing_indices, save_split, save_metadata, zip_directory
from utils.logging import log, print_config, print_header, print_summary
from utils.model import get_model_short_name, get_device, clear_memory, load_model_and_tokenizer

SaveEvery = Annotated[int, typer.Option("--save-every", help="Save results every N examples (0 to disable)")]
ZipOutput = Annotated[bool, typer.Option("--zip", help="Create zip archive of outputs for download")]


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Support PROBE_DATA_DIR env var for pipeline flexibility
import os
DATA_DIR = Path(os.environ.get(
    'PROBE_DATA_DIR',
    PROJECT_ROOT / "resources" / "gsm8k" / "matching"
))

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def get_default_batch_size(device: torch.device) -> int:
    """Get recommended batch size for device.

    Benchmarked optimal values for Qwen2.5-Math-1.5B with 10 responses, 512 tokens:
    - H100 80GB: batch=128 → 7430 tok/s, 35.7GB VRAM (best)
    - H100 80GB: batch=96  → 4804 tok/s, 27.5GB VRAM (safe)
    """
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_gb >= 70:  # H100 80GB
            return 96  # Benchmarked: 128 is optimal but 96 is safer
        elif mem_gb >= 40:  # A100 40GB
            return 48
        else:
            return 16
    elif device.type == "mps":
        return 4
    return 2  # CPU


def process_batch(
    entries: list[dict],
    model,
    tokenizer,
    device: torch.device,
    num_responses: int,
    max_new_tokens: int
) -> list[dict]:
    """Process a batch of examples, generating multiple responses for each."""
    prompts = [format_prompt(entry['question']) for entry in entries]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    ).to(device, non_blocking=(device.type == "cuda"))  # non_blocking only safe on CUDA

    prompt_lengths = (inputs.attention_mask.sum(dim=1)).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_responses,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # outputs shape: (batch_size * num_responses, seq_len)
    # Reshape to group responses by input example
    results = []
    for i, entry in enumerate(entries):
        responses = []
        for j in range(num_responses):
            seq_idx = i * num_responses + j
            seq = outputs[seq_idx]
            prompt_len = prompt_lengths[i]
            response_ids = seq[prompt_len:]

            # Remove padding tokens from response
            response_ids = response_ids[response_ids != tokenizer.pad_token_id]

            responses.append({
                'text': tokenizer.decode(response_ids, skip_special_tokens=True),
                'tokens': tokenizer.convert_ids_to_tokens(response_ids),
                'token_ids': response_ids.cpu().tolist(),
            })

        results.append({
            'index': entry['index'],
            'question': entry['question'],
            'responses': responses,
        })

    return results


app = typer.Typer()


@app.command()
def main(
    max_examples: MaxExamples = 10,
    output_dir: OutputDir = None,
    split: Split = "both",
    resume: Resume = False,
    num_responses: NumResponses = 10,
    max_new_tokens: MaxNewTokens = 512,
    compile: Compile = False,
    batch_size: BatchSize = 0,  # 0 = auto-detect based on device
    save_every: SaveEvery = 500,
    no_flash_attn: NoFlashAttn = False,
    zip_output: ZipOutput = False,
):
    """Generate responses (no hidden states)."""
    model_short_name = get_model_short_name(MODEL_NAME)
    out_path = Path(output_dir) if output_dir else SCRIPT_DIR / "responses" / model_short_name
    out_path.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Auto-detect batch size if not specified
    if batch_size <= 0:
        batch_size = get_default_batch_size(device)
        log.info(f"Auto-detected batch size: {batch_size}")

    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME, device, use_compile=compile, use_flash_attn=not no_flash_attn
    )

    splits = ['train', 'test'] if split == 'both' else [split]
    total_stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    print_header("Response Generation", "Pass 1 - No Hidden States")
    print_config("Configuration", {
        "Device": device,
        "Max examples": max_examples if max_examples > 0 else "all",
        "Responses": num_responses,
        "Batch size": batch_size,
        "Save every": save_every if save_every > 0 else "disabled",
        "Output": out_path,
    })

    for split_name in splits:
        log.info(f"Processing {split_name}...")

        # Try different file patterns (tokenized or plain)
        data_file = None
        for pattern in [f"{split_name}_tokenized.jsonl", f"{split_name}.jsonl"]:
            candidate = DATA_DIR / pattern
            if candidate.exists():
                data_file = candidate
                break

        if not data_file:
            log.warning(f"No data file found for {split_name} in {DATA_DIR}")
            continue

        try:
            dataset = load_jsonl(data_file)
            log.info(f"Loaded {len(dataset)} examples from {data_file.name}")
        except Exception as e:
            log.warning(f"Failed to load {data_file}: {e}")
            continue

        if max_examples > 0:
            dataset = dataset[:max_examples]

        existing = get_existing_indices(out_path / f"{split_name}_responses.json") if resume else set()

        # Filter out already processed
        to_process = [e for e in dataset if e['index'] not in existing]
        total_stats['skipped'] += len(dataset) - len(to_process)

        results = []
        num_batches = (len(to_process) + batch_size - 1) // batch_size
        saved_count = 0
        has_saved = False

        pbar = tqdm(
            range(0, len(to_process), batch_size),
            desc=f"{split_name}",
            unit="batch",
            total=num_batches,
        )

        for batch_start in pbar:
            batch_entries = to_process[batch_start:batch_start + batch_size]
            pbar.set_postfix(examples=len(batch_entries))

            try:
                batch_results = process_batch(
                    entries=batch_entries,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    num_responses=num_responses,
                    max_new_tokens=max_new_tokens
                )
                results.extend(batch_results)
                total_stats['processed'] += len(batch_results)

                # Periodic save to avoid memory overflow
                if save_every > 0 and len(results) >= save_every:
                    save_split(results, out_path, split_name, resume=(resume or has_saved))
                    saved_count += len(results)
                    results = []
                    has_saved = True

            except Exception as e:
                log.error(f"Error on batch {batch_start}-{batch_start + len(batch_entries)}: {e}")
                total_stats['errors'] += len(batch_entries)

            clear_memory(device)

        # Save remaining results
        if results:
            save_split(results, out_path, split_name, resume=(resume or has_saved))
            saved_count += len(results)
            results = []

        if saved_count > 0:
            log.success(f"{split_name}: {saved_count} examples saved")

    save_metadata(
        out_path,
        model_name=MODEL_NAME,
        config={
            'max_examples': max_examples,
            'num_responses': num_responses,
            'max_new_tokens': max_new_tokens,
            'batch_size': batch_size,
            'save_every': save_every,
            'split': split,
            'compile': compile,
        },
        stats=total_stats,
    )

    print_summary("Summary", {
        "Processed": total_stats['processed'],
        "Skipped": total_stats['skipped'],
        "Errors": total_stats['errors'],
        "Output": str(out_path),
    })

    # Create zip archive for download
    if zip_output:
        zip_path = zip_directory(out_path, pattern="*.json")
        log.success(f"Zip archive: {zip_path}")


if __name__ == '__main__':
    app()
