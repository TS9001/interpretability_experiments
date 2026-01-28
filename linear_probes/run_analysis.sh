#!/bin/bash
# Run analysis pipeline: analyze → filter → extract
# Usage: ./run_analysis.sh [train|test]

set -e

SPLIT="${1:-train}"
INPUT="../resources/Qwen2.5-Math-1.5B/${SPLIT}_responses.json"

echo "=== Step 1: Analyze responses ==="
python 02_analyze_responses.py "$INPUT"

echo ""
echo "=== Step 2: Filter probeable ==="
python 03_filter_probeable.py --input "../resources/Qwen2.5-Math-1.5B/${SPLIT}_responses_analyzed.json" --per-operation

echo ""
echo "=== Step 3: Extract hidden states ==="
python 04_extract_hidden_states.py --input "../resources/Qwen2.5-Math-1.5B/${SPLIT}_responses_analyzed_probeable.json"

echo ""
echo "=== Done ==="
