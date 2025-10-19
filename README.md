# Interpretability Experiments

A collection of interpretability experiments for AI/ML models.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management and virtual environment handling.

### Prerequisites

- Python 3.12+
- uv package manager

### Installation

1. Install uv (if not already installed):
   ```bash
   pip install uv
   ```

2. Create a virtual environment and sync dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

## Project Structure

```
interpretability_experiments/
├── src/
│   └── interpretability_experiments/  # Main package
├── experiments/                        # Individual experiments
│   └── example_experiment/            # Example experiment template
├── pyproject.toml                     # Project configuration
└── README.md                          # This file
```

## Adding New Experiments

1. Create a new directory under `experiments/`
2. Add your experiment code and documentation
3. See `experiments/example_experiment/` for a template structure

## Development

To add new dependencies:
```bash
uv add <package-name>
```

To run an experiment:
```bash
uv run python experiments/your_experiment/main.py
```
