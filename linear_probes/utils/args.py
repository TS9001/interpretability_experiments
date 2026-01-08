"""Argument parsing utilities using Typer."""
from typing import Annotated, Optional

import typer

# Common option definitions - reuse these in script commands
MaxExamples = Annotated[int, typer.Option("--max-examples", help="Maximum examples to process (-1 for all)")]
OutputDir = Annotated[Optional[str], typer.Option("--output-dir", help="Output directory")]
Split = Annotated[str, typer.Option("--split", help="Dataset split to process")]
Resume = Annotated[bool, typer.Option("--resume", help="Resume from existing output file")]

# Generation-specific options
NumResponses = Annotated[int, typer.Option("--num-responses", help="Responses per question")]
MaxNewTokens = Annotated[int, typer.Option("--max-new-tokens", help="Max new tokens per response")]
Compile = Annotated[bool, typer.Option("--compile", help="Use torch.compile with max-autotune (CUDA only, adds startup time)")]
BatchSize = Annotated[int, typer.Option("--batch-size", help="Batch size (H100: 32-64 for 1.5B model, 16-32 for 7B)")]
NoFlashAttn = Annotated[bool, typer.Option("--no-flash-attn", help="Disable Flash Attention 2 (enabled by default on CUDA)")]

# Analysis/extraction options
InputFile = Annotated[str, typer.Argument(help="Input JSONL file")]
Layers = Annotated[str, typer.Option("--layers", help="Comma-separated layer indices (e.g., '0,7,14,21,27')")]
ProbeTypes = Annotated[Optional[str], typer.Option("--probes", help="Comma-separated probe names (e.g., 'A1,B1,C1')")]
