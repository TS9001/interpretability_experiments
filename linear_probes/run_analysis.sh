#!/bin/bash
# Run analysis pipeline: analyze → filter → extract
# Usage: ./run_analysis.sh [train|test]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="${1:-train}"
INPUT="../resources/Qwen2.5-Math-1.5B/${SPLIT}_responses.json"

# Set ground truth data path
export PROBE_DATA_DIR="$SCRIPT_DIR/../resources/gsm8k/matching"

echo "=== Step 1: Analyze responses ==="
echo "Ground truth: $PROBE_DATA_DIR"
python 02_analyze_responses.py "$INPUT"

echo ""
echo "=== Step 2: Filter probeable ==="
python 03_filter_probeable.py --input "../resources/Qwen2.5-Math-1.5B/${SPLIT}_responses_analyzed.json" --per-operation

echo ""
echo "=== Step 3: Extract hidden states ==="
python 04_extract_hidden_states.py --input "../resources/Qwen2.5-Math-1.5B/${SPLIT}_responses_analyzed_probeable.json"

echo ""
echo "=== Done ==="
