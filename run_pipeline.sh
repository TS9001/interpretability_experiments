#!/bin/bash
# =============================================================================
# Linear Probes Pipeline for Mathematical Reasoning Interpretability
# =============================================================================
#
# Dataset Strategy:
#   - GSM8K Train (7,473 examples) → Probe training
#   - GSM8K-Platinum (1,209 examples) → Probe evaluation (higher quality test set)
#
# Pipeline Phases:
#   1. Dataset Preparation: Download and process both datasets
#   2. Response Generation: Generate model responses on train data
#   3. Probe Training: Train linear probes on GSM8K Train
#   4. Evaluation: Test probes on GSM8K-Platinum
#
# Usage:
#   ./run_pipeline.sh              # Full pipeline
#   ./run_pipeline.sh --poc        # POC mode (100 examples, fast)
#   ./run_pipeline.sh --skip-data  # Skip dataset preparation
#   ./run_pipeline.sh --train-only # Only train, skip evaluation
#   ./run_pipeline.sh --eval-only  # Only evaluate (requires trained probes)
#   ./run_pipeline.sh --help       # Show help
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
POC_MODE=false
SKIP_SETUP=false
SKIP_DATA=false
TRAIN_ONLY=false
EVAL_ONLY=false
MAX_EXAMPLES=-1  # -1 = all
BATCH_SIZE=0     # 0 = auto-detect
PER_OPERATION=true  # Use relaxed per-operation filtering

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --poc)
            POC_MODE=true
            MAX_EXAMPLES=100
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --strict-filter)
            PER_OPERATION=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Dataset Strategy:"
            echo "  GSM8K Train (7,473 examples)     → Probe training"
            echo "  GSM8K-Platinum (1,209 examples)  → Probe evaluation"
            echo ""
            echo "Options:"
            echo "  --poc            POC mode: 100 examples only (fast iteration)"
            echo "  --skip-setup     Skip virtual environment setup"
            echo "  --skip-data      Skip dataset preparation (use existing data)"
            echo "  --train-only     Only run training phases (skip evaluation)"
            echo "  --eval-only      Only run evaluation (requires trained probes)"
            echo "  --max-examples N Maximum examples to process (-1=all)"
            echo "  --batch-size N   Batch size for generation (0=auto)"
            echo "  --strict-filter  Use strict per-response filtering (default: per-operation)"
            echo "  --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --poc                    # Quick test run"
            echo "  $0 --skip-data --eval-only  # Re-run evaluation only"
            echo "  $0 --max-examples 1000      # Process 1000 examples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
}

subheader() {
    echo ""
    echo -e "${GREEN}──────────────────────────────────────${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}──────────────────────────────────────${NC}"
}

# =============================================================================
# PHASE 0: Environment Setup
# =============================================================================
if [ "$SKIP_SETUP" = false ]; then
    header "Phase 0: Environment Setup"

    if [ ! -d ".venv-pip" ]; then
        log "Creating virtual environment..."
        ./setup_venv.sh || error "Failed to create venv"
    else
        log "Virtual environment already exists"
    fi
fi

# Activate venv
if [ -f ".venv-pip/bin/activate" ]; then
    source .venv-pip/bin/activate
    log "Activated .venv-pip ($(python --version))"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    log "Activated .venv ($(python --version))"
else
    error "No virtual environment found. Run ./setup_venv.sh first"
fi

# =============================================================================
# PHASE 1: Dataset Preparation
# =============================================================================
if [ "$SKIP_DATA" = false ] && [ "$EVAL_ONLY" = false ]; then
    header "Phase 1: Dataset Preparation"

    RESOURCES_DIR="$SCRIPT_DIR/resources"
    mkdir -p "$RESOURCES_DIR"

    # Download and process datasets using inline Python
    python3 << 'PYTHON_SCRIPT'
import json
import re
import sys
from pathlib import Path
from datasets import load_dataset

RESOURCES_DIR = Path("resources")

# Regex patterns for operation detection (simplified from regexp_utils.py)
def normalize_operator(op):
    if op in ['+']:
        return '+'
    elif op in ['-', '–', '—']:
        return '-'
    elif op in ['*', 'x', 'X', '×', '·']:
        return '*'
    elif op in ['/', '÷']:
        return '/'
    return op

COMPACT_PATTERN = re.compile(r'(\d[\d,]*(?:\.\d+)?|\.\d+)([+\-*/×÷\u2013\u2014])(\d[\d,]*(?:\.\d+)?|\.\d+)')
SPACED_PATTERN = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*([+\-*/×÷xX\u2013\u2014])\s*(\d[\d,]*(?:\.\d+)?)\s*=\s*(\d[\d,]*(?:\.\d+)?)')

def find_operations_inside_brackets(answer):
    """Find operations inside <<>> brackets."""
    bracket_pattern = re.compile(r'<<([^>]+)>>')
    operations = []
    for match in bracket_pattern.finditer(answer):
        content = match.group(1)
        for m in COMPACT_PATTERN.finditer(content):
            operations.append((m.group(1), normalize_operator(m.group(2)), m.group(3)))
    return operations

def find_operations_outside_brackets(answer):
    """Find operations outside <<>> brackets."""
    # Remove bracket contents
    clean = re.sub(r'<<[^>]+>>', '', answer)
    operations = []
    for m in SPACED_PATTERN.finditer(clean):
        operations.append((m.group(1), normalize_operator(m.group(2)), m.group(3)))
    return operations

def count_by_type(operations):
    counts = {'+': 0, '-': 0, '*': 0, '/': 0}
    for op in operations:
        if op[1] in counts:
            counts[op[1]] += 1
    return counts

def is_matching(answer):
    inside = count_by_type(find_operations_inside_brackets(answer))
    outside = count_by_type(find_operations_outside_brackets(answer))
    return inside == outside

def parse_operations(answer):
    """Extract detailed operations from <<>> brackets."""
    bracket_pattern = re.compile(r'<<([^>]+)>>')
    operations = []

    for match in bracket_pattern.finditer(answer):
        content = match.group(1)
        # Parse expression like "16-3-4=9"
        if '=' not in content:
            continue
        parts = content.split('=')
        if len(parts) != 2:
            continue

        expr, result_str = parts[0], parts[1]
        try:
            final_result = float(result_str.replace(',', ''))
            if final_result == int(final_result):
                final_result = int(final_result)
        except:
            continue

        # Find all operations in expression
        tokens = re.findall(r'(\d[\d,]*\.?\d*|[+\-*/×÷xX])', expr)
        if len(tokens) < 3:
            continue

        try:
            current = float(tokens[0].replace(',', ''))
            if current == int(current):
                current = int(current)
        except:
            continue

        i = 1
        while i < len(tokens) - 1:
            op = normalize_operator(tokens[i])
            try:
                operand2 = float(tokens[i+1].replace(',', ''))
                if operand2 == int(operand2):
                    operand2 = int(operand2)
            except:
                break

            if op == '+':
                result = current + operand2
            elif op == '-':
                result = current - operand2
            elif op == '*':
                result = current * operand2
            elif op == '/':
                result = current / operand2 if operand2 != 0 else 0
            else:
                break

            if result == int(result):
                result = int(result)

            op_type = {'+': 'add', '-': 'sub', '*': 'mult', '/': 'div'}.get(op, op)
            operations.append({
                'operand1': current,
                'operator': op_type,
                'operand2': operand2,
                'result': result,
            })
            current = result
            i += 2

    return operations

def process_dataset(dataset_name, hf_path, splits, output_subdir):
    """Download and process a dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    output_dir = RESOURCES_DIR / output_subdir
    matching_dir = output_dir / "matching"
    nonmatching_dir = output_dir / "nonmatching"
    matching_dir.mkdir(parents=True, exist_ok=True)
    nonmatching_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from HuggingFace: {hf_path}")
    dataset = load_dataset(hf_path, "main")

    total_matching = 0
    total_nonmatching = 0

    for split in splits:
        if split not in dataset:
            print(f"  Split '{split}' not found, skipping...")
            continue

        data = dataset[split]
        print(f"  {split}: {len(data)} examples")

        matching = []
        nonmatching = []

        for idx, example in enumerate(data):
            entry = {
                'index': idx,
                'question': example['question'],
                'answer': example['answer'],
            }

            if is_matching(example['answer']):
                # Parse operations for matching examples
                ops = parse_operations(example['answer'])
                entry['operations'] = ops
                entry['operation_sequence'] = [op['operator'] for op in ops]
                entry['operations_by_type'] = {
                    'add': sum(1 for op in ops if op['operator'] == 'add'),
                    'sub': sum(1 for op in ops if op['operator'] == 'sub'),
                    'mult': sum(1 for op in ops if op['operator'] == 'mult'),
                    'div': sum(1 for op in ops if op['operator'] == 'div'),
                }
                entry['total_operations'] = len(ops)

                # Get final result
                final_match = re.search(r'####\s*(\d[\d,]*\.?\d*)', example['answer'])
                if final_match:
                    try:
                        fr = float(final_match.group(1).replace(',', ''))
                        entry['final_result'] = int(fr) if fr == int(fr) else fr
                    except:
                        entry['final_result'] = None
                else:
                    entry['final_result'] = None

                matching.append(entry)
            else:
                nonmatching.append(entry)

        # Save matching
        with open(matching_dir / f"{split}.jsonl", 'w') as f:
            for e in matching:
                f.write(json.dumps(e) + '\n')

        # Save nonmatching
        with open(nonmatching_dir / f"{split}.jsonl", 'w') as f:
            for e in nonmatching:
                f.write(json.dumps(e) + '\n')

        total_matching += len(matching)
        total_nonmatching += len(nonmatching)
        print(f"    → {len(matching)} matching, {len(nonmatching)} non-matching")

    print(f"  Total: {total_matching} matching, {total_nonmatching} non-matching")
    print(f"  Output: {output_dir}")
    return total_matching, total_nonmatching

# Process GSM8K original (train for probes)
gsm8k_match, gsm8k_nonmatch = process_dataset(
    "GSM8K (Original)",
    "openai/gsm8k",
    ["train", "test"],
    "gsm8k"
)

# Process GSM8K-Platinum (test for evaluation)
platinum_match, platinum_nonmatch = process_dataset(
    "GSM8K-Platinum",
    "madrylab/gsm8k-platinum",
    ["test"],
    "gsm8k_platinum"
)

print(f"\n{'='*60}")
print("Dataset Summary")
print(f"{'='*60}")
print(f"GSM8K Train:     {gsm8k_match} matching examples (for probe training)")
print(f"GSM8K-Platinum:  {platinum_match} matching examples (for evaluation)")
print(f"{'='*60}")
PYTHON_SCRIPT

    success "Dataset preparation complete"
fi

# =============================================================================
# PHASE 2: Response Generation (GSM8K Train)
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    header "Phase 2: Response Generation (GSM8K Train)"

    cd linear_probes

    # Update DATA_DIR in generate script to use GSM8K train
    # We'll pass the data path as an environment variable or modify the script

    # Build command args
    ARGS=""
    if [ "$MAX_EXAMPLES" != "-1" ]; then
        ARGS="$ARGS --max-examples $MAX_EXAMPLES"
    fi
    if [ "$BATCH_SIZE" != "0" ]; then
        ARGS="$ARGS --batch-size $BATCH_SIZE"
    fi

    subheader "Step 2.1: Generating model responses"

    # Generate responses for GSM8K train
    export PROBE_DATA_DIR="$SCRIPT_DIR/resources/gsm8k/matching"
    python 01_generate_responses.py $ARGS --split train

    subheader "Step 2.2: Analyzing responses"
    python 02_analyze_responses.py responses/Qwen2.5-Math-1.5B/train_responses.json

    subheader "Step 2.3: Filtering probeable responses"
    FILTER_ARGS="--input responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json"
    if [ "$PER_OPERATION" = true ]; then
        FILTER_ARGS="$FILTER_ARGS --per-operation"
    fi
    python 03_filter_probeable.py $FILTER_ARGS

    cd "$SCRIPT_DIR"
    success "Response generation complete"
fi

# =============================================================================
# PHASE 3: Hidden State Extraction & Probe Training
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    header "Phase 3: Hidden State Extraction & Probe Training"

    cd linear_probes

    subheader "Step 3.1: Extracting hidden states"
    EXTRACT_ARGS="--input responses/Qwen2.5-Math-1.5B/train_responses_analyzed_probeable.json"
    if [ "$BATCH_SIZE" != "0" ]; then
        EXTRACT_ARGS="$EXTRACT_ARGS --batch-size $BATCH_SIZE"
    fi
    python 04_POC_extract_hidden_states.py $EXTRACT_ARGS

    subheader "Step 3.2: Training linear probes"
    python 05_POC_train_probes.py --verbose

    cd "$SCRIPT_DIR"
    success "Probe training complete"
fi

# =============================================================================
# PHASE 4: Evaluation on GSM8K-Platinum
# =============================================================================
if [ "$TRAIN_ONLY" = false ]; then
    header "Phase 4: Evaluation on GSM8K-Platinum"

    cd linear_probes

    # Build command args for evaluation
    EVAL_ARGS=""
    if [ "$POC_MODE" = true ]; then
        EVAL_ARGS="$EVAL_ARGS --max-examples 100"
    fi
    if [ "$BATCH_SIZE" != "0" ]; then
        EVAL_ARGS="$EVAL_ARGS --batch-size $BATCH_SIZE"
    fi

    subheader "Step 4.1: Generating responses on Platinum test"
    export PROBE_DATA_DIR="$SCRIPT_DIR/resources/gsm8k_platinum/matching"
    python 01_generate_responses.py $EVAL_ARGS --split test \
        --output-dir responses/Qwen2.5-Math-1.5B/platinum/

    subheader "Step 4.2: Analyzing Platinum responses"
    python 02_analyze_responses.py responses/Qwen2.5-Math-1.5B/platinum/test_responses.json

    subheader "Step 4.3: Filtering probeable Platinum responses"
    FILTER_ARGS="--input responses/Qwen2.5-Math-1.5B/platinum/test_responses_analyzed.json"
    if [ "$PER_OPERATION" = true ]; then
        FILTER_ARGS="$FILTER_ARGS --per-operation"
    fi
    python 03_filter_probeable.py $FILTER_ARGS

    subheader "Step 4.4: Extracting hidden states from Platinum"
    python 04_POC_extract_hidden_states.py \
        --input responses/Qwen2.5-Math-1.5B/platinum/test_responses_analyzed_probeable.json \
        --output probe_data/platinum/ \
        ${BATCH_SIZE:+--batch-size $BATCH_SIZE}

    subheader "Step 4.5: Evaluating probes on Platinum"
    python 05_POC_train_probes.py \
        --eval-only \
        --eval-data probe_data/platinum/ \
        --verbose

    cd "$SCRIPT_DIR"
    success "Evaluation complete"
fi

# =============================================================================
# Summary
# =============================================================================
header "Pipeline Complete!"

echo ""
echo -e "${CYAN}Results:${NC}"
echo "  Training Data:    resources/gsm8k/matching/"
echo "  Evaluation Data:  resources/gsm8k_platinum/matching/"
echo "  Train Responses:  linear_probes/responses/Qwen2.5-Math-1.5B/"
echo "  Eval Responses:   linear_probes/responses/Qwen2.5-Math-1.5B/platinum/"
echo "  Probe Data:       linear_probes/probe_data/"
echo "  Probe Results:    linear_probes/probe_data/probe_results.json"
echo ""

if [ "$POC_MODE" = true ]; then
    echo -e "${YELLOW}Note: This was a POC run with limited examples.${NC}"
    echo "For full results, run: ./run_pipeline.sh"
fi

echo ""
echo -e "${GREEN}Dataset Strategy:${NC}"
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  GSM8K Train (7,473 examples)                           │"
echo "  │  └─→ Response generation                                │"
echo "  │  └─→ Probe training                                     │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │  GSM8K-Platinum (1,209 examples)                        │"
echo "  │  └─→ Probe evaluation (high-quality test set)           │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
