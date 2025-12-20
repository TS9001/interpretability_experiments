# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project annotates the GSM8K-Platinum dataset with mathematical operations for interpretability experiments. The annotation system automatically detects and extracts arithmetic operations from math problem solutions, splitting them into "matching" (reliable) and "non-matching" examples.

## Environment Setup

```bash
# Create and activate virtual environment
./setup_venv.sh
source .venv-pip/bin/activate

# Or use the existing .venv environment if available
source .venv/bin/activate
```

Required Python version: 3.11+

## Dataset Preparation Pipeline

All dataset preparation scripts are in `gmsk8_generation_platinum/dataset_preparation/`.

### Step 1: Download Dataset

```bash
cd gmsk8_generation_platinum/dataset_preparation
python 1_download.py
```

Downloads GSM8K-Platinum from HuggingFace (`madrylab/gsm8k-platinum`) and saves to `resources/gsm8k/`.

### Step 2: Split into Matching/Non-matching

```bash
python 2_split.py
```

Splits the dataset based on whether operations inside `<<>>` brackets match operations in the surrounding text:
- **Matching**: Operations counts match → `resources/gsm8k_split/matching/`
- **Non-matching**: Mismatches detected → `resources/gsm8k_split/nonmatching/`

### Step 3: Enhance with Detailed Operations

```bash
python 3_enhance.py
```

Adds detailed operation info (operand1, operator, operand2, result) to matching examples:
- Output: `resources/gsm8k_split/matching/train_enhanced.jsonl`
- Output: `resources/gsm8k_split/matching/test_enhanced.jsonl`

## How It Works

### Matching Detection

The system compares operations found in two places:
1. **Inside `<<>>` brackets**: e.g., `<<5+3=8>>` → one `add` operation
2. **Outside brackets**: e.g., `5 + 3 = 8` → one `add` operation

If the operation counts match, the example is considered reliable.

### Detection Algorithm

Handles multiple operation formats:
- Standard operators: `+`, `-`, `*`, `/`
- Text multiplication: `"20 x 20"`, `"5 X 3"`
- Unicode operators: `×`, `÷`, `·`
- Unicode dashes: en-dash `–`, em-dash `—`
- Operations with units: `"30 years - 20 years"`, `"2 trains * 80 miles"`

## Files

```
gmsk8_generation_platinum/dataset_preparation/
├── 1_download.py       # Download GSM8K-Platinum
├── 2_split.py          # Split into matching/non-matching
├── 3_enhance.py        # Add detailed operation info
├── regexp_utils.py     # Core regex utilities
└── resources/
    └── gsm8k_split/
        ├── matching/
        │   ├── train.jsonl
        │   ├── test.jsonl
        │   ├── train_enhanced.jsonl
        │   └── test_enhanced.jsonl
        └── nonmatching/
            ├── train.jsonl
            └── test.jsonl
```

## Output Format (Enhanced)

Each enhanced example contains:
```json
{
  "index": 42,
  "question": "John has 5 apples and buys 3 more...",
  "answer": "First, 5 + 3 = <<5+3=8>>8 apples...",
  "final_result": 8,
  "operations": [
    {"operand1": 5, "operator": "add", "operand2": 3, "result": 8}
  ],
  "operation_sequence": ["add"],
  "operations_by_type": {"add": 1, "sub": 0, "mult": 0, "div": 0},
  "total_operations": 1
}
