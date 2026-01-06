#!/usr/bin/env python3
"""
[OLD] Extract hidden states for selected examples.

NOTE: This is the old approach. Use 02_extract_probe_hidden_states.py instead
which extracts only probe-relevant positions for more efficient storage.

Usage:
    python extract_hidden_states.py --input responses/test_responses.jsonl
    python extract_hidden_states.py --input responses/test_responses.jsonl --filter has_errors
"""
import argparse
import gzip
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).parent.parent
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_memory(device: torch.device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_model_and_tokenizer(device: torch.device, use_compile: bool = False):
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    model.eval()

    if use_compile and device.type == "cuda":
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_responses(path: Path) -> list:
    entries = []
    with open(path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def filter_examples(examples: list, filter_type: str) -> list:
    if filter_type == 'has_errors':
        return [ex for ex in examples if ex['summary']['num_with_operation_errors'] > 0]
    elif filter_type == 'incorrect':
        return [ex for ex in examples if ex['summary']['num_correct_final'] < ex['summary']['num_responses']]
    elif filter_type == 'correct':
        return [ex for ex in examples if ex['summary']['num_correct_final'] == ex['summary']['num_responses']]
    elif filter_type == 'mixed':
        return [ex for ex in examples
                if 0 < ex['summary']['num_correct_final'] < ex['summary']['num_responses']]
    else:
        return examples


def format_prompt(question: str) -> str:
    return f"Question: {question}\n\nAnswer:"


def extract_hidden_states(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    layers_to_save: Optional[list] = None
) -> torch.Tensor:
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **encodings,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states[1:]
    num_layers = len(hidden_states)

    if layers_to_save is not None:
        layer_indices = layers_to_save
    else:
        layer_indices = list(range(num_layers))

    selected = torch.stack([hidden_states[i][0] for i in layer_indices], dim=0)

    return selected.cpu().half()


def process_example(
    example: dict,
    model,
    tokenizer,
    device: torch.device,
    layers_to_save: Optional[list] = None,
    batch_size: int = 5
) -> dict:
    question = example['question']
    prompt = format_prompt(question)

    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    prompt_length = prompt_encoding.input_ids.shape[1]

    responses_with_hidden = []

    for resp in example['responses']:
        full_text = prompt + resp['text']

        hidden_states = extract_hidden_states(
            model, tokenizer, full_text, device, layers_to_save
        )

        response_hidden = hidden_states[:, prompt_length:, :]

        responses_with_hidden.append({
            'text': resp['text'],
            'tokens': resp['tokens'],
            'token_ids': resp['token_ids'],
            'final_answer': resp['final_answer'],
            'final_correct': resp['final_correct'],
            'operations': resp['operations'],
            'hidden_states': response_hidden,
        })

        clear_memory(device)

    return {
        'index': example['index'],
        'question': question,
        'ground_truth_answer': example['ground_truth_answer'],
        'ground_truth_operations': example['ground_truth_operations'],
        'ground_truth_final_result': example['ground_truth_final_result'],
        'operation_sequence': example['operation_sequence'],
        'responses': responses_with_hidden,
        'summary': example['summary'],
    }


def save_example(result: dict, output_path: Path, compress: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        output_path = output_path.with_suffix('.pt.gz')
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(result, f)
    else:
        torch.save(result, output_path)


def parse_layers(layers_str: str) -> Optional[list]:
    if not layers_str:
        return None
    return [int(x.strip()) for x in layers_str.split(',')]


def parse_indices(indices_str: str) -> Optional[set]:
    if not indices_str:
        return None
    return {int(x.strip()) for x in indices_str.split(',')}


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states for selected examples")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--filter', type=str, default=None,
                        choices=['has_errors', 'incorrect', 'correct', 'mixed'])
    parser.add_argument('--indices', type=str, default=None)
    parser.add_argument('--layers', type=str, default=None)
    parser.add_argument('--max-examples', type=int, default=-1)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "hidden_states"
    output_dir.mkdir(parents=True, exist_ok=True)

    layers_to_save = parse_layers(args.layers)
    selected_indices = parse_indices(args.indices)

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, use_compile=args.compile)

    print(f"\nLoading responses from: {input_path}")
    examples = load_responses(input_path)
    print(f"Loaded {len(examples)} examples")

    if args.filter:
        examples = filter_examples(examples, args.filter)
        print(f"After filter '{args.filter}': {len(examples)} examples")

    if selected_indices:
        examples = [ex for ex in examples if ex['index'] in selected_indices]
        print(f"After index filter: {len(examples)} examples")

    if args.max_examples > 0:
        examples = examples[:args.max_examples]
        print(f"Limited to {len(examples)} examples")

    existing = set()
    if args.resume:
        for f in output_dir.glob("example_*"):
            try:
                idx = int(f.stem.split('_')[1])
                existing.add(idx)
            except (ValueError, IndexError):
                pass
        print(f"Resuming: {len(existing)} already processed")

    print(f"\n{'='*60}")
    print("Hidden State Extraction [OLD VERSION]")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Examples: {len(examples)}")
    print(f"Layers: {args.layers if args.layers else 'all'}")
    print(f"Compress: {args.compress}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    for example in tqdm(examples, desc="Extracting"):
        idx = example['index']

        if idx in existing:
            stats['skipped'] += 1
            continue

        try:
            result = process_example(
                example=example,
                model=model,
                tokenizer=tokenizer,
                device=device,
                layers_to_save=layers_to_save
            )

            output_path = output_dir / f"example_{idx:04d}.pt"
            save_example(result, output_path, compress=args.compress)
            stats['processed'] += 1

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            stats['errors'] += 1
            continue

        clear_memory(device)

    metadata = {
        'model_name': MODEL_NAME,
        'input_file': str(input_path),
        'config': {
            'filter': args.filter,
            'indices': args.indices,
            'layers': args.layers,
            'compress': args.compress,
        },
        'stats': stats,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
