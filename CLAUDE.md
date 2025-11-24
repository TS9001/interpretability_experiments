# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project annotates the GSM8K (Grade School Math 8K) dataset with mathematical operations for interpretability experiments. The annotation system automatically detects and extracts arithmetic operations from math problem solutions.

## Environment Setup

```bash
# Create and activate virtual environment
./setup_venv.sh
source .venv-pip/bin/activate

# Or use the existing .venv environment if available
source .venv/bin/activate
```

Required Python version: 3.11+

## Workflow

### 1. Download Dataset

```bash
cd gmsk8_generation
python gmsk8_download.py
```

Downloads the GSM8K dataset from HuggingFace and saves to `resources/gmsk8_dataset/`.

### 2. Annotate with Operations

```bash
cd gmsk8_generation

# Annotate train split
python annotate_operations.py train

# Annotate test split
python annotate_operations.py test

# Annotate both splits
python annotate_operations.py both
```

This creates TWO output files per split:
- `{split}_perfect_annotations.json` - Examples with reliable detection (~60%)
- `{split}_needs_fixing.json` - Examples that need manual fixing (~40%)

The "perfect" examples are ready to use immediately. The "needs_fixing" examples should be corrected with a local LLM, then re-annotated by running the same script.

## How It Works

### Annotation System

The system automatically detects and extracts **operations** from GSM8K solutions:
- **Operations list**: Arithmetic operations in execution order (add/sub/mult/div)
- **Operations by type**: Count of each operation type

### Detection Algorithm

Handles multiple operation formats:
- Standard operators: `+`, `-`, `*`, `/`
- Text multiplication: `"20 x 20"`, `"5 X 3"`
- Unicode operators: `×`, `÷`, `·`
- Operations with units: `"30 years - 20 years"`, `"2 trains * 80 miles"`
- Implied multiplication: `"5(10)"` (partially supported)

### Reliability

The script detects whether an example's operations are **reliably extracted**:
- ✅ **Reliable (~60%)**: Operations inside `<<>>` brackets match operations in text
- ⚠️ **Needs Fixing (~40%)**: Mismatches detected (operations missing, extra operations, etc.)

Examples marked as "needs fixing" should be corrected with a local LLM, then re-run through the same script.

## Files

**Scripts:**
- `gmsk8_download.py` - Download GSM8K dataset
- `annotate_operations.py` - Annotate dataset with operations

**Data Locations:**
- **Input**: `gmsk8_generation/resources/gmsk8_dataset/` (downloaded dataset)
- **Output**: `gmsk8_generation/{split}_perfect_annotations.json` (reliable examples)
- **Output**: `gmsk8_generation/{split}_needs_fixing.json` (examples to fix)

## Output Format

Each annotation contains:
```json
{
  "index": 42,
  "question": "John has 5 apples and buys 3 more...",
  "answer": "First, 5 + 3 = <<5+3=8>>8 apples...",
  "operations": ["add"],
  "operations_by_type": {"add": 1, "sub": 0, "mult": 0, "div": 0}
}
```

Examples in `needs_fixing.json` also include:
- `detection_issues`: List of issues found
- `detection_details`: Detailed counts for debugging
