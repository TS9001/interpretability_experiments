"""GRPO training on GSM8K for Qwen3-1.7B.

Usage:
    python rl/train_grpo.py
    python rl/train_grpo.py --num-checkpoints 10
    python rl/train_grpo.py --epochs 3 --lr 2e-6 --format-reward
"""

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_OUTPUT = "rl/checkpoints"

console = Console()

# ---------------------------------------------------------------------------
# Logging (self-contained, mirrors linear_probes/utils/logging.py)
# ---------------------------------------------------------------------------


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level="INFO",
    )
    return logger


log = setup_logger()


def print_config(title: str, config: dict):
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    for key, value in config.items():
        table.add_row(str(key), str(value))
    console.print(table)


def print_summary(title: str, stats: dict):
    table = Table(title=title, border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, "
    "showing your reasoning. Put your final numeric answer after ####."
)


def prepare_gsm8k_dataset(split: str = "train"):
    """Load openai/gsm8k and format as chat messages with answer column."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    def format_example(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    return ds.map(format_example, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    output_dir: str = typer.Option(DEFAULT_OUTPUT, help="Base output directory for checkpoints"),
    model: str = typer.Option(MODEL_NAME, help="Model name or path"),
    epochs: int = typer.Option(1, help="Number of training epochs"),
    lr: float = typer.Option(1e-6, help="Learning rate"),
    batch_size: int = typer.Option(16, help="Per-device batch size"),
    num_generations: int = typer.Option(8, help="Number of generations per prompt (G in GRPO)"),
    grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
    max_new_tokens: int = typer.Option(512, help="Max tokens to generate"),
    num_checkpoints: int = typer.Option(10, help="Evenly-spaced checkpoints (0 = save every epoch)"),
    format_reward: bool = typer.Option(False, help="Include format reward"),
    temperature: float = typer.Option(0.7, help="Sampling temperature for generations"),
    kl_coef: float = typer.Option(0.05, help="KL divergence coefficient"),
):
    """Train Qwen3-1.7B with GRPO on GSM8K."""
    # Lazy imports so --help is fast
    from trl import GRPOConfig, GRPOTrainer

    from rl.callbacks import EvenCheckpointCallback
    from rl.rewards import correctness_reward, format_reward as format_reward_fn

    # Resolve output dir: output_dir / model_short_name
    model_short = model.split("/")[-1]
    run_dir = str(Path(output_dir) / model_short)

    # ---- Config table ----
    config = {
        "model": model,
        "output_dir": run_dir,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_generations": num_generations,
        "grad_accum": grad_accum,
        "max_new_tokens": max_new_tokens,
        "num_checkpoints": num_checkpoints if num_checkpoints > 0 else "every epoch",
        "format_reward": format_reward,
        "temperature": temperature,
        "kl_coef": kl_coef,
    }
    console.print(Panel("[bold]GRPO Training[/bold]", border_style="blue"))
    print_config("Training Configuration", config)

    # ---- Dataset ----
    log.info("Loading GSM8K dataset...")
    dataset = prepare_gsm8k_dataset("train")
    log.info(f"Dataset: {len(dataset)} examples")

    # ---- Reward functions ----
    reward_fns = [correctness_reward]
    reward_names = ["correctness"]
    if format_reward:
        reward_fns.append(format_reward_fn)
        reward_names.append("format")
    log.info(f"Reward functions: {reward_names}")

    # ---- Callbacks ----
    callbacks = []
    if num_checkpoints > 0:
        callbacks.append(EvenCheckpointCallback(num_checkpoints))
        log.info(f"Will save {num_checkpoints} evenly-spaced checkpoints")

    # ---- GRPOConfig ----
    save_strategy = "no" if num_checkpoints > 0 else "epoch"

    training_config = GRPOConfig(
        output_dir=run_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        gradient_accumulation_steps=grad_accum,
        max_completion_length=max_new_tokens,
        temperature=temperature,
        beta=kl_coef,
        # H100 optimizations
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        # Scheduler
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # Saving
        save_strategy=save_strategy,
        save_total_limit=None,
        # Logging
        logging_steps=10,
        report_to="tensorboard",
    )

    # ---- Trainer ----
    log.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_config,
        train_dataset=dataset,
        callbacks=callbacks,
    )

    # ---- Train ----
    log.info("Starting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ---- Save final model ----
    final_dir = str(Path(run_dir) / "final")
    log.info(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)

    # ---- Summary ----
    print_summary("Training Complete", {
        "model": model,
        "output_dir": run_dir,
        "final_model": final_dir,
        "training_time": f"{elapsed / 60:.1f} min",
        "epochs": str(epochs),
        "reward_functions": ", ".join(reward_names),
    })


if __name__ == "__main__":
    app()
