#!/usr/bin/env python3
"""
[OLD] Generate responses and extract hidden states from Qwen2.5-Math-1.5B for linear probe training.

NOTE: This is the old approach that extracts ALL hidden states for ALL tokens.
Use 01_generate_responses.py + 02_extract_probe_hidden_states.py instead for more efficient extraction.

Usage:
    # Local testing (M4/MPS)
    python generate_probe_qwen_math.py --max-examples 10

    # Full run on H100
    python generate_probe_qwen_math.py --max-examples -1 --compile --batch-size 10

    # With compression for storage
    python generate_probe_qwen_math.py --compress

    # Resume interrupted run
    python generate_probe_qwen_math.py --resume
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


SCRIPT_DIR = Path(__file__).parent.parent  # Go up from old/
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "gmsk8_generation_platinum" / "dataset_preparation" / "resources" / "gsm8k_split" / "matching"

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


def load_model_and_tokenizer(device: torch.device, use_compile: bool = False, use_flash_attn: bool = False):
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }

    if use_flash_attn and device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs).to(device)
    model.eval()

    if use_compile and device.type == "cuda":
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_dataset(split: str) -> list:
    file_path = DATA_DIR / f"{split}_tokenized.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def format_prompt(question: str) -> str:
    return f"Question: {question}\n\nAnswer:"


def extract_hidden_states_batch(
    model,
    tokenizer,
    texts: list,
    device: torch.device,
    layers_to_save: Optional[list] = None
) -> list:
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
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

    selected = torch.stack([hidden_states[i] for i in layer_indices], dim=0)
    attention_mask = encodings.attention_mask

    results = []
    for b in range(selected.shape[1]):
        seq_len = attention_mask[b].sum().item()
        hidden = selected[:, b, :seq_len, :].cpu().half()
        results.append(hidden)

    return results


def generate_responses(
    model,
    tokenizer,
    question: str,
    num_responses: int,
    max_new_tokens: int,
    device: torch.device,
    layers_to_save: Optional[list] = None,
    hidden_batch_size: int = 5
) -> list:
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]

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

    generated_sequences = outputs
    response_texts = []
    response_data = []

    for i in range(num_responses):
        seq = generated_sequences[i]
        response_ids = seq[prompt_length:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_tokens = tokenizer.convert_ids_to_tokens(response_ids)
        full_text = tokenizer.decode(seq, skip_special_tokens=True)
        response_texts.append(full_text)

        response_data.append({
            'text': response_text,
            'tokens': response_tokens,
            'token_ids': response_ids.cpu(),
            'prompt_length': prompt_length,
        })

    all_hidden_states = []
    for i in range(0, num_responses, hidden_batch_size):
        batch_texts = response_texts[i:i + hidden_batch_size]
        batch_hidden = extract_hidden_states_batch(
            model, tokenizer, batch_texts, device, layers_to_save
        )
        all_hidden_states.extend(batch_hidden)
        clear_memory(device)

    responses = []
    for i, data in enumerate(response_data):
        full_hidden = all_hidden_states[i]
        prompt_len = data['prompt_length']
        response_hidden = full_hidden[:, prompt_len:, :]

        responses.append({
            'text': data['text'],
            'tokens': data['tokens'],
            'token_ids': data['token_ids'],
            'hidden_states': response_hidden,
        })

    return responses


def process_example(
    entry: dict,
    model,
    tokenizer,
    device: torch.device,
    num_responses: int,
    max_new_tokens: int,
    layers_to_save: Optional[list] = None,
    hidden_batch_size: int = 5
) -> dict:
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        question=entry['question'],
        num_responses=num_responses,
        max_new_tokens=max_new_tokens,
        device=device,
        layers_to_save=layers_to_save,
        hidden_batch_size=hidden_batch_size
    )

    return {
        'index': entry['index'],
        'question': entry['question'],
        'ground_truth_answer': entry.get('answer_clean', ''),
        'ground_truth_operations': entry.get('operations', []),
        'ground_truth_final_result': entry.get('final_result'),
        'operation_sequence': entry.get('operation_sequence', []),
        'responses': responses,
    }


def save_example(result: dict, output_path: Path, compress: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        output_path = output_path.with_suffix('.pt.gz')
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(result, f)
    else:
        torch.save(result, output_path)


def load_example(path: Path) -> dict:
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return torch.load(path)


def save_metadata(output_dir: Path, config: dict, stats: dict):
    metadata = {
        'model_name': MODEL_NAME,
        'config': config,
        'stats': stats,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def get_existing_indices(output_dir: Path, split: str) -> set:
    split_dir = output_dir / split
    if not split_dir.exists():
        return set()

    existing = set()
    for f in split_dir.glob("example_*"):
        try:
            idx = int(f.stem.split('_')[1])
            existing.add(idx)
        except (ValueError, IndexError):
            pass
    return existing


def parse_layers(layers_str: str) -> Optional[list]:
    if not layers_str:
        return None
    return [int(x.strip()) for x in layers_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses and extract hidden states for linear probes"
    )
    parser.add_argument('--max-examples', type=int, default=10)
    parser.add_argument('--num-responses', type=int, default=10)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--layers', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--flash-attn', action='store_true')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--compress', action='store_true')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    layers_to_save = parse_layers(args.layers)
    device = get_device()

    model, tokenizer = load_model_and_tokenizer(
        device,
        use_compile=args.compile,
        use_flash_attn=args.flash_attn
    )

    splits = ['train', 'test'] if args.split == 'both' else [args.split]
    total_stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    config = {
        'max_examples': args.max_examples,
        'num_responses': args.num_responses,
        'max_new_tokens': args.max_new_tokens,
        'layers': args.layers,
        'splits': splits,
        'compile': args.compile,
        'flash_attn': args.flash_attn,
        'batch_size': args.batch_size,
        'compress': args.compress,
    }

    print(f"\n{'='*60}")
    print("Linear Probes Dataset Generation [OLD VERSION]")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Max examples: {args.max_examples if args.max_examples > 0 else 'all'}")
    print(f"Responses: {args.num_responses}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Layers: {args.layers if args.layers else 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Compress: {args.compress}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    for split in splits:
        print(f"\nProcessing {split}...")

        try:
            dataset = load_dataset(split)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        if args.max_examples > 0:
            dataset = dataset[:args.max_examples]

        existing = get_existing_indices(output_dir, split) if args.resume else set()
        split_dir = output_dir / split

        for entry in tqdm(dataset, desc=f"{split}"):
            idx = entry['index']

            if idx in existing:
                total_stats['skipped'] += 1
                continue

            try:
                result = process_example(
                    entry=entry,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    num_responses=args.num_responses,
                    max_new_tokens=args.max_new_tokens,
                    layers_to_save=layers_to_save,
                    hidden_batch_size=args.batch_size
                )

                output_path = split_dir / f"example_{idx:04d}.pt"
                save_example(result, output_path, compress=args.compress)
                total_stats['processed'] += 1

            except Exception as e:
                print(f"\nError on example {idx}: {e}")
                total_stats['errors'] += 1
                continue

            clear_memory(device)

    save_metadata(output_dir, config, total_stats)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Processed: {total_stats['processed']}")
    print(f"Skipped: {total_stats['skipped']}")
    print(f"Errors: {total_stats['errors']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
