#!/bin/bash
# =============================================================================
# Remote Server Setup Script
# =============================================================================
#
# Usage:
#   ./remote_setup.sh "ssh -p 59672 root@192.222.52.140 -i ~/.ssh/vastai"
#   ./remote_setup.sh "ssh ..." --all-layers    # Use all 28 layers instead of key 9
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_TRAJECTORIES="$SCRIPT_DIR/linear_probes/resources/Qwen2.5-Math-1.5B"
REMOTE_WORKSPACE="/workspace"
REMOTE_REPO="$REMOTE_WORKSPACE/interpretability_experiments"
REPO_URL="https://github.com/TS9001/interpretability_experiments.git"

# Layers configuration
# Qwen2.5-Math-1.5B has 28 layers (0-27)
ALL_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"
# Key layers where probes showed best results (from experiments)
# 0=baseline, 6/8=D6, 14=D2/D3, 16=C3_div, 21=A1/C1, 22=C4/D1, 27=B1/B2
DEFAULT_LAYERS="0,6,7,8,14,16,21,22,27"
LAYERS="$DEFAULT_LAYERS"

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${CYAN}== $1 ==${NC}"; }

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ssh-command> [--all-layers]"
    echo "Example: $0 \"ssh -p 59672 root@192.222.52.140 -i ~/.ssh/vastai\""
    echo "         $0 \"ssh ...\" --all-layers   # Extract all 28 layers"
    exit 1
fi

SSH_CMD="$1"

# Check for --all-layers flag
if [[ "$2" == "--all-layers" ]]; then
    LAYERS="$ALL_LAYERS"
    log "Using ALL layers (0-27)"
else
    log "Using key layers: $LAYERS"
fi
SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p\s*[0-9]+' | grep -oE '[0-9]+' || echo "22")
SSH_HOST=$(echo "$SSH_CMD" | grep -oE '[a-zA-Z0-9_]+@[a-zA-Z0-9._-]+')
SSH_KEY=$(echo "$SSH_CMD" | grep -oE '\-i\s*[^ ]+' | sed 's/-i\s*//')

[ -z "$SSH_HOST" ] && error "Could not parse host from: $SSH_CMD"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

log "Host: $SSH_HOST, Port: $SSH_PORT"

# =============================================================================
header "Step 1: Clone Repository"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    cd $REMOTE_WORKSPACE
    if [ -d interpretability_experiments/.git ]; then
        echo 'Repo exists, switching to prod_test and pulling...'
        cd interpretability_experiments && git fetch origin && git checkout prod_test && git pull origin prod_test
    else
        echo 'Cloning fresh...'
        rm -rf interpretability_experiments
        git clone $REPO_URL
        cd interpretability_experiments && git checkout prod_test
    fi
"
success "Repository ready"

# =============================================================================
header "Step 2: Install Dependencies"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    cd $REMOTE_REPO
    if [ ! -d .venv-pip ]; then
        ./setup_venv.sh
    fi
    source .venv-pip/bin/activate
    echo 'Python:' \$(python --version)
"
success "Dependencies installed"

# =============================================================================
header "Step 3: Prepare GSM8K Dataset"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    cd $REMOTE_REPO
    source .venv-pip/bin/activate

    # Run dataset preparation scripts
    cd gmsk8_generation_platinum/dataset_preparation

    echo 'Downloading GSM8K...'
    python 1_download.py

    echo 'Splitting dataset...'
    python 2_split.py

    echo 'Enhancing with operations...'
    python 3_enhance.py

    echo 'Tokenizing...'
    python 4_tokenize.py
"
success "Dataset prepared"

# =============================================================================
header "Step 4: Copy Trajectories"
# =============================================================================
if [ ! -d "$LOCAL_TRAJECTORIES" ]; then
    error "Local trajectories not found: $LOCAL_TRAJECTORIES"
fi

log "Copying from: $LOCAL_TRAJECTORIES"
ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_REPO/linear_probes/responses/Qwen2.5-Math-1.5B"

RSYNC_SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT"
[ -n "$SSH_KEY" ] && RSYNC_SSH="$RSYNC_SSH -i $SSH_KEY"

rsync -avz --progress -e "$RSYNC_SSH" \
    "$LOCAL_TRAJECTORIES/" \
    "$SSH_HOST:$REMOTE_REPO/linear_probes/responses/Qwen2.5-Math-1.5B/"

success "Trajectories copied"

# =============================================================================
header "Step 5: Run Analysis Pipeline"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    cd $REMOTE_REPO
    source .venv-pip/bin/activate

    # CRITICAL: Set ground truth path
    export PROBE_DATA_DIR='$REMOTE_REPO/gmsk8_generation_platinum/dataset_preparation/resources/gsm8k_split/matching'
    echo 'PROBE_DATA_DIR=' \$PROBE_DATA_DIR

    cd linear_probes

    for SPLIT in train test; do
        RESP=\"responses/Qwen2.5-Math-1.5B/\${SPLIT}_responses.json\"

        if [ ! -f \"\$RESP\" ]; then
            echo \"Skipping \$SPLIT (no responses file)\"
            continue
        fi

        echo ''
        echo '=== Processing '\$SPLIT' ==='

        echo '[1/3] Analyzing...'
        python 02_analyze_responses.py \"\$RESP\"

        echo '[2/3] Filtering...'
        python 03_filter_probeable.py \\
            --input \"responses/Qwen2.5-Math-1.5B/\${SPLIT}_responses_analyzed.json\" \\
            --per-operation

        echo '[3/3] Extracting hidden states (layers: $LAYERS)...'
        python 04_extract_hidden_states.py \\
            --input \"responses/Qwen2.5-Math-1.5B/\${SPLIT}_responses_analyzed_probeable.json\" \\
            --split \$SPLIT \\
            --layers $LAYERS \\
            --no-flash-attn
    done
"
success "Pipeline complete"

# =============================================================================
header "Done!"
# =============================================================================
echo ""
echo -e "${GREEN}Remote setup complete!${NC}"
echo ""
echo "To train probes with selectivity (Hewitt control tasks):"
echo "  $SSH_CMD"
echo "  cd $REMOTE_REPO/linear_probes && source ../.venv-pip/bin/activate"
echo "  python 05_train_probes_logistic_regression.py --with-selectivity"
echo "  python 05_train_probes.py --with-selectivity"
echo ""
