#!/bin/bash
# =============================================================================
# GRPO Training on Remote GPU with Tensorboard
# =============================================================================
#
# Usage:
#   ./rl/run_grpo_training.sh "ssh -p 34959 root@ssh7.vast.ai -i ~/.ssh/vastai"
#   ./rl/run_grpo_training.sh "ssh -p 34959 root@ssh7.vast.ai -i ~/.ssh/vastai" --epochs 3
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

TB_PORT="${TB_PORT:-8080}"
REMOTE_DIR="/root/interpretability_experiments"
REPO_URL="https://github.com/TS9001/interpretability_experiments.git"
REPO_BRANCH="prod_test"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ssh-command> [train_grpo options]"
    echo ""
    echo "Examples:"
    echo "  $0 \"ssh -p 34959 root@ssh7.vast.ai -i ~/.ssh/vastai\""
    echo "  $0 \"ssh ...\" --epochs 3"
    echo "  $0 \"ssh ...\" --epochs 2 --lr 2e-6"
    echo ""
    echo "Environment variables:"
    echo "  TB_PORT=8080   Tensorboard port (default: 8080)"
    exit 1
fi

SSH_CMD="$1"
shift

# Collect remaining args for train_grpo.py
TRAIN_ARGS="--format-reward $*"

# Parse SSH command components (same pattern as deploy_to_remote.sh)
SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p\s*[0-9]+' | grep -oE '[0-9]+' || echo "22")
SSH_HOST=$(echo "$SSH_CMD" | grep -oE '[a-zA-Z0-9_]+@[a-zA-Z0-9._-]+')
SSH_KEY=$(echo "$SSH_CMD" | grep -oE '\-i\s*[^ ]+' | sed 's/-i\s*//')

[ -z "$SSH_HOST" ] && error "Could not parse host from: $SSH_CMD"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

log "Host: $SSH_HOST, Port: $SSH_PORT"
log "Training args: $TRAIN_ARGS"

# =============================================================================
header "Step 1: Clone/Pull Repository"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    if [ -d $REMOTE_DIR/.git ]; then
        echo 'Repo exists, pulling latest...'
        cd $REMOTE_DIR && git fetch origin && git checkout $REPO_BRANCH && git pull origin $REPO_BRANCH
    else
        echo 'Cloning fresh...'
        git clone --branch $REPO_BRANCH $REPO_URL $REMOTE_DIR
    fi
"
success "Repository ready"

# =============================================================================
header "Step 2: Install Dependencies"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "pip install -q torch transformers datasets accelerate typer rich scikit-learn numpy tqdm loguru trl tensorboard 2>&1 | tail -1"
success "Dependencies installed"

# =============================================================================
header "Step 3: Clean Up Old Processes"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    kill \$(pgrep -f train_grpo) 2>/dev/null || true
    kill \$(lsof -t -i:$TB_PORT) 2>/dev/null || true
    sleep 1
" || true
success "Old processes cleaned"

# =============================================================================
header "Step 4: Start Tensorboard"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    nohup python3 -m tensorboard.main \
        --logdir=$REMOTE_DIR/rl/checkpoints \
        --port=$TB_PORT \
        --bind_all \
        >/root/tensorboard.log 2>&1 &
    sleep 2
    if lsof -i:$TB_PORT >/dev/null 2>&1; then
        echo 'Tensorboard running on port $TB_PORT'
    else
        echo 'WARNING: Tensorboard may not have started'
        cat /root/tensorboard.log
    fi
"
success "Tensorboard started"

# =============================================================================
header "Step 5: Start GRPO Training"
# =============================================================================
ssh $SSH_OPTS "$SSH_HOST" "
    cd $REMOTE_DIR
    PYTHONPATH=$REMOTE_DIR nohup python3 -m rl.train_grpo $TRAIN_ARGS > /root/grpo_train.log 2>&1 &
    sleep 3
    if pgrep -f train_grpo > /dev/null; then
        echo 'Training process started'
    else
        echo 'ERROR: Training failed to start'
        cat /root/grpo_train.log
        exit 1
    fi
"
success "GRPO training started (correctness + format reward)"

# =============================================================================
header "Step 6: SSH Tunnel for Tensorboard"
# =============================================================================
echo ""
echo -e "${GREEN}Training is running on remote!${NC}"
echo ""
echo -e "  ${CYAN}Tensorboard:${NC}  http://localhost:$TB_PORT"
echo -e "  ${CYAN}Training log:${NC} ssh $SSH_OPTS $SSH_HOST 'tail -f /root/grpo_train.log'"
echo ""
echo "Press Ctrl+C to close the tunnel (training continues on remote)."
echo ""

ssh $SSH_OPTS -L "$TB_PORT:localhost:$TB_PORT" -N "$SSH_HOST"
