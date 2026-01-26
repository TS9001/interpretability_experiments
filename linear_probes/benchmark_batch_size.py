#!/usr/bin/env python3
"""
Benchmark different batch sizes to find optimal throughput on H100.

Tests generation and hidden state extraction with various batch sizes.
"""
import time
from pathlib import Path
import torch
import typer
from rich.console import Console
from rich.table import Table

from utils.model import load_model_and_tokenizer, get_device, clear_memory
from utils.data import load_jsonl, format_prompt

console = Console()
app = typer.Typer(add_completion=False)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "gmsk8_generation_platinum" / "dataset_preparation" / "resources" / "gsm8k_split" / "matching"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def benchmark_generation(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int = 512,
    num_responses: int = 1,
    warmup: bool = True,
) -> dict:
    """Benchmark generation with given batch size."""
    # Warmup run
    if warmup:
        inputs = tokenizer(
            prompts[:2],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=32,
                num_return_sequences=1,
                do_sample=False,
            )
        clear_memory(device)

    # Actual benchmark
    num_batches = len(prompts) // batch_size
    if num_batches == 0:
        return {"error": "batch_size larger than dataset"}

    total_tokens = 0
    total_time = 0

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(device, non_blocking=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

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

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Count generated tokens
        generated_tokens = outputs.shape[0] * (outputs.shape[1] - inputs.input_ids.shape[1])
        total_tokens += generated_tokens
        total_time += elapsed

        clear_memory(device)

    return {
        "batch_size": batch_size,
        "batches": num_batches,
        "total_examples": num_batches * batch_size,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": total_tokens / total_time,
        "examples_per_sec": (num_batches * batch_size) / total_time,
        "sec_per_example": total_time / (num_batches * batch_size),
        "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def benchmark_hidden_states(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[str],
    batch_size: int,
    layers: list[int] = [0, 7, 14, 21, 27],
) -> dict:
    """Benchmark hidden state extraction with given batch size."""
    num_batches = len(prompts) // batch_size
    if num_batches == 0:
        return {"error": "batch_size larger than dataset"}

    total_time = 0
    total_tokens = 0

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(device, non_blocking=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            # Extract specific layers
            hidden_states = [outputs.hidden_states[l] for l in layers]
            # Force computation
            _ = [h.cpu() for h in hidden_states]

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens += inputs.input_ids.numel()
        total_time += elapsed

        clear_memory(device)

    return {
        "batch_size": batch_size,
        "batches": num_batches,
        "total_examples": num_batches * batch_size,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": total_tokens / total_time,
        "examples_per_sec": (num_batches * batch_size) / total_time,
        "sec_per_example": total_time / (num_batches * batch_size),
        "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


@app.command()
def main(
    max_examples: int = typer.Option(64, "--max-examples", "-n", help="Number of examples to use"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens", help="Max tokens to generate"),
    num_responses: int = typer.Option(1, "--num-responses", help="Responses per prompt"),
    test_generation: bool = typer.Option(True, "--generation/--no-generation", help="Test generation"),
    test_hidden: bool = typer.Option(True, "--hidden/--no-hidden", help="Test hidden state extraction"),
):
    """Benchmark batch sizes for optimal GPU utilization."""
    device = get_device()

    if device.type != "cuda":
        console.print("[red]This benchmark requires CUDA[/red]")
        raise typer.Exit(1)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    console.print(f"\n[bold]GPU:[/bold] {gpu_name} ({gpu_mem:.0f}GB)")

    # Load data
    console.print(f"\n[bold]Loading data...[/bold]")
    train_data = load_jsonl(DATA_DIR / "train.jsonl")[:max_examples]
    prompts = [format_prompt(entry['question']) for entry in train_data]
    console.print(f"Loaded {len(prompts)} prompts")

    # Load model
    console.print(f"\n[bold]Loading model...[/bold]")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device, use_compile=False)

    # Batch sizes to test (powers of 2 + some intermediate values)
    if gpu_mem >= 70:  # H100 80GB
        batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    elif gpu_mem >= 40:  # A100 40GB
        batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    # Filter batch sizes that fit in our data
    batch_sizes = [b for b in batch_sizes if b <= len(prompts)]

    # Benchmark generation
    if test_generation:
        console.print(f"\n[bold cyan]═══ Generation Benchmark ═══[/bold cyan]")
        console.print(f"max_new_tokens={max_new_tokens}, num_responses={num_responses}")

        gen_results = []
        for bs in batch_sizes:
            console.print(f"\nTesting batch_size={bs}...")
            torch.cuda.reset_peak_memory_stats()
            try:
                result = benchmark_generation(
                    model, tokenizer, device, prompts,
                    batch_size=bs,
                    max_new_tokens=max_new_tokens,
                    num_responses=num_responses,
                    warmup=(bs == batch_sizes[0]),
                )
                gen_results.append(result)
                console.print(f"  {result['tokens_per_sec']:.0f} tok/s, {result['gpu_mem_gb']:.1f}GB VRAM")
            except torch.cuda.OutOfMemoryError:
                console.print(f"  [red]OOM[/red]")
                clear_memory(device)
                break

        # Print results table
        table = Table(title="Generation Results")
        table.add_column("Batch", justify="right")
        table.add_column("Tok/s", justify="right")
        table.add_column("Ex/s", justify="right")
        table.add_column("s/Ex", justify="right")
        table.add_column("VRAM (GB)", justify="right")

        best_throughput = max(r['tokens_per_sec'] for r in gen_results)
        for r in gen_results:
            is_best = r['tokens_per_sec'] == best_throughput
            style = "bold green" if is_best else None
            table.add_row(
                str(r['batch_size']),
                f"{r['tokens_per_sec']:.0f}",
                f"{r['examples_per_sec']:.2f}",
                f"{r['sec_per_example']:.2f}",
                f"{r['gpu_mem_gb']:.1f}",
                style=style,
            )

        console.print(table)

        best = max(gen_results, key=lambda x: x['tokens_per_sec'])
        console.print(f"\n[bold green]Optimal generation batch_size: {best['batch_size']}[/bold green]")
        console.print(f"  Throughput: {best['tokens_per_sec']:.0f} tokens/sec")
        console.print(f"  VRAM usage: {best['gpu_mem_gb']:.1f} GB")

    # Benchmark hidden state extraction
    if test_hidden:
        console.print(f"\n[bold cyan]═══ Hidden State Extraction Benchmark ═══[/bold cyan]")

        hidden_results = []
        for bs in batch_sizes:
            console.print(f"\nTesting batch_size={bs}...")
            torch.cuda.reset_peak_memory_stats()
            try:
                result = benchmark_hidden_states(
                    model, tokenizer, device, prompts,
                    batch_size=bs,
                )
                hidden_results.append(result)
                console.print(f"  {result['tokens_per_sec']:.0f} tok/s, {result['gpu_mem_gb']:.1f}GB VRAM")
            except torch.cuda.OutOfMemoryError:
                console.print(f"  [red]OOM[/red]")
                clear_memory(device)
                break

        # Print results table
        table = Table(title="Hidden State Extraction Results")
        table.add_column("Batch", justify="right")
        table.add_column("Tok/s", justify="right")
        table.add_column("Ex/s", justify="right")
        table.add_column("s/Ex", justify="right")
        table.add_column("VRAM (GB)", justify="right")

        best_throughput = max(r['tokens_per_sec'] for r in hidden_results)
        for r in hidden_results:
            is_best = r['tokens_per_sec'] == best_throughput
            style = "bold green" if is_best else None
            table.add_row(
                str(r['batch_size']),
                f"{r['tokens_per_sec']:.0f}",
                f"{r['examples_per_sec']:.2f}",
                f"{r['sec_per_example']:.2f}",
                f"{r['gpu_mem_gb']:.1f}",
                style=style,
            )

        console.print(table)

        best = max(hidden_results, key=lambda x: x['tokens_per_sec'])
        console.print(f"\n[bold green]Optimal hidden state batch_size: {best['batch_size']}[/bold green]")
        console.print(f"  Throughput: {best['tokens_per_sec']:.0f} tokens/sec")
        console.print(f"  VRAM usage: {best['gpu_mem_gb']:.1f} GB")


if __name__ == "__main__":
    app()
