#!/bin/bash
# =============================================================================
# Deploy local changes to remote Linux machine
# =============================================================================
#
# Usage:
#   ./deploy_to_remote.sh "ssh -p 40230 root@135.135.24.114 -i ~/.ssh/vastai"
#   ./deploy_to_remote.sh "ssh -p 40230 root@135.135.24.114 -i ~/.ssh/vastai" --setup
#
# Options:
#   --setup     Re-run dataset preparation pipeline after sync
#   --code-only Only sync code, skip responses/probe_data directories
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

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${CYAN}== $1 ==${NC}"; }

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_WORKSPACE="/workspace"
REMOTE_REPO="$REMOTE_WORKSPACE/interpretability_experiments"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ssh-command> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 \"ssh -p 40230 root@135.135.24.114 -i ~/.ssh/vastai\""
    echo "  $0 \"ssh ...\" --setup      # Re-run dataset preparation"
    echo "  $0 \"ssh ...\" --code-only  # Only sync code files"
    echo ""
    echo "This script syncs local changes to remote WITHOUT needing git push."
    exit 1
fi

SSH_CMD="$1"
RUN_SETUP=false
CODE_ONLY=false

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup) RUN_SETUP=true ;;
        --code-only) CODE_ONLY=true ;;
        *) warn "Unknown option: $1" ;;
    esac
    shift
done

# Parse SSH command components
SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p\s*[0-9]+' | grep -oE '[0-9]+' || echo "22")
SSH_HOST=$(echo "$SSH_CMD" | grep -oE '[a-zA-Z0-9_]+@[0-9.]+')
SSH_KEY=$(echo "$SSH_CMD" | grep -oE '\-i\s*[^ ]+' | sed 's/-i\s*//')

[ -z "$SSH_HOST" ] && error "Could not parse host from: $SSH_CMD"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

RSYNC_SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT"
[ -n "$SSH_KEY" ] && RSYNC_SSH="$RSYNC_SSH -i $SSH_KEY"

log "Target: $SSH_HOST:$SSH_PORT"
log "Local dir: $LOCAL_DIR"
log "Remote dir: $REMOTE_REPO"

# =============================================================================
header "Step 1: Create remote directory"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_REPO"
success "Remote directory ready"

# =============================================================================
header "Step 2: Sync project files"
# =============================================================================

# Base excludes (always skip these)
EXCLUDES=(
    --exclude '.venv*'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.git'
    --exclude '.claude'
    --exclude 'logs/'
    --exclude '.DS_Store'
    --exclude '*.egg-info'
    --exclude 'cache/'
    --exclude '*.pt'
    --exclude '*.pth'
)

# Additional excludes for code-only mode
if [ "$CODE_ONLY" = true ]; then
    log "Code-only mode: skipping responses and probe_data"
    EXCLUDES+=(
        --exclude 'responses/'
        --exclude 'probe_data/'
        --exclude 'resources/Qwen*'
    )
fi

log "Syncing files..."
rsync -avz --progress \
    "${EXCLUDES[@]}" \
    -e "$RSYNC_SSH" \
    "$LOCAL_DIR/" \
    "$SSH_HOST:$REMOTE_REPO/"

success "Files synced"

# =============================================================================
header "Step 3: Verify sync"
# =============================================================================
log "Checking key files on remote..."
ssh $SSH_OPTS "$SSH_HOST" "
    echo 'Dataset preparation scripts:'
    ls -la $REMOTE_REPO/gmsk8_generation_platinum/dataset_preparation/*.py 2>/dev/null | head -5
    echo ''
    echo 'Linear probes scripts:'
    ls -la $REMOTE_REPO/linear_probes/*.py 2>/dev/null | head -5
"
success "Verification complete"

# =============================================================================
# Optional: Run dataset preparation
# =============================================================================
if [ "$RUN_SETUP" = true ]; then
    header "Step 4: Running dataset preparation"

    ssh $SSH_OPTS "$SSH_HOST" "
        cd $REMOTE_REPO

        # Create/activate venv if needed
        if [ ! -d .venv-pip ]; then
            echo 'Creating virtual environment...'
            python3 -m venv .venv-pip
            source .venv-pip/bin/activate
            pip install --upgrade pip
            pip install datasets transformers torch typer rich scikit-learn
        else
            source .venv-pip/bin/activate
        fi

        cd gmsk8_generation_platinum/dataset_preparation

        echo ''
        echo '=== Downloading datasets ==='
        python 1_download.py

        echo ''
        echo '=== Splitting dataset ==='
        python 2_split.py

        echo ''
        echo '=== Enhancing with operations ==='
        python 3_enhance.py

        echo ''
        echo '=== Tokenizing ==='
        python 4_tokenize.py

        echo ''
        echo '=== Verifying output ==='
        ls -la resources/gsm8k_split/matching/
    "
    success "Dataset preparation complete"
fi

# =============================================================================
header "Done!"
# =============================================================================
echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "To connect and run analysis:"
echo "  $SSH_CMD"
echo "  cd $REMOTE_REPO/linear_probes && source ../.venv-pip/bin/activate"
echo ""
if [ "$RUN_SETUP" = false ]; then
    echo "To re-run dataset preparation (if needed):"
    echo "  $0 \"$1\" --setup"
    echo ""
fi
echo "To re-run analysis pipeline:"
echo "  export PROBE_DATA_DIR='$REMOTE_REPO/gmsk8_generation_platinum/dataset_preparation/resources/gsm8k_split/matching'"
echo "  python 02_analyze_responses.py responses/Qwen2.5-Math-1.5B/train_responses.json"
echo "  python 03_filter_probeable.py --input responses/Qwen2.5-Math-1.5B/train_responses_analyzed.json --per-operation"
echo "  python 04_extract_hidden_states.py --input responses/Qwen2.5-Math-1.5B/train_responses_analyzed_probeable.json --split train"
