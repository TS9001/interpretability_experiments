"""Logging utilities using Loguru + Rich."""
import sys
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Rich console instance
console = Console()


def setup_logger():
    """Configure loguru with custom format."""
    logger.remove()  # Remove default handler

    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level="INFO",
    )
    return logger


def print_config(title: str, config: dict):
    """Print configuration as a nice table."""
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(str(key), str(value))

    console.print(table)


def print_header(title: str, subtitle: Optional[str] = None):
    """Print a section header."""
    text = f"[bold]{title}[/bold]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text, border_style="blue"))


def print_summary(title: str, stats: dict):
    """Print a summary table."""
    table = Table(title=title, border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


def create_progress() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# Initialize logger on import
log = setup_logger()
