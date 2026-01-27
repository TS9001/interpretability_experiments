#!/bin/bash
# Deploy project to remote Linux machine for running probes
# Usage: ./deploy_to_remote.sh user@remote-host [remote_dir]

set -e

# Configuration
REMOTE_HOST="${1:-}"
REMOTE_DIR="${2:-~/interpretability_experiments}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_VERSION="python3.11"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ -z "$REMOTE_HOST" ]]; then
    echo "Usage: $0 user@remote-host [remote_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 user@gpu-server.example.com"
    echo "  $0 user@192.168.1.100 /home/user/projects/probes"
    echo ""
    echo "This script will:"
    echo "  1. Generate Linux-compatible requirements.txt"
    echo "  2. Sync project files to remote"
    echo "  3. Create virtual environment on remote"
    echo "  4. Install dependencies (with CUDA PyTorch if available)"
    exit 1
fi

# Step 1: Generate Linux-compatible requirements
log_info "Generating Linux-compatible requirements..."

REQUIREMENTS_FILE="$LOCAL_DIR/requirements_linux.txt"

# Export current packages, filter out Mac-specific ones
source "$LOCAL_DIR/.venv-pip/bin/activate" 2>/dev/null || {
    log_error "Cannot activate local venv at $LOCAL_DIR/.venv-pip"
    exit 1
}

# Filter out Mac-specific packages and torch (we'll install CUDA version separately)
pip freeze | grep -v -E "^(mlx|mlx-lm|mlx-metal|torch|torchvision|torchaudio)" > "$REQUIREMENTS_FILE"

# Add torch with CUDA support
cat >> "$REQUIREMENTS_FILE" << 'EOF'
--extra-index-url https://download.pytorch.org/whl/cu121
torch
torchvision
EOF

log_info "Created $REQUIREMENTS_FILE"

# Step 2: Create remote setup script
REMOTE_SETUP_SCRIPT="$LOCAL_DIR/setup_remote.sh"
cat > "$REMOTE_SETUP_SCRIPT" << 'SETUP_EOF'
#!/bin/bash
# Remote setup script - runs on the remote machine
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_VERSION="python3.11"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Python version
if ! command -v $PYTHON_VERSION &> /dev/null; then
    log_warn "$PYTHON_VERSION not found, trying python3..."
    PYTHON_VERSION="python3"
    if ! command -v $PYTHON_VERSION &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi
fi

PY_VER=$($PYTHON_VERSION -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log_info "Using Python $PY_VER"

# Create virtual environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
    log_warn "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        log_info "Keeping existing venv. Run 'source $VENV_DIR/bin/activate' to use it."
        exit 0
    fi
fi

log_info "Creating virtual environment..."
$PYTHON_VERSION -m venv "$VENV_DIR"

log_info "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    log_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    CUDA_AVAILABLE=true
else
    log_warn "No NVIDIA GPU detected. Installing CPU-only PyTorch."
    CUDA_AVAILABLE=false
fi

# Install requirements
log_info "Installing dependencies..."
if [[ "$CUDA_AVAILABLE" == "true" ]]; then
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements_linux.txt"
else
    # Install without CUDA
    grep -v "extra-index-url" "$SCRIPT_DIR/requirements_linux.txt" | \
        "$VENV_DIR/bin/pip" install -r /dev/stdin
fi

# Verify installation
log_info "Verifying installation..."
"$VENV_DIR/bin/python" -c "
import torch
import transformers
import sklearn
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print('All dependencies verified!')
"

echo ""
log_info "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  cd $SCRIPT_DIR"
echo "  source .venv/bin/activate"
echo ""
echo "H100 optimized commands (use large batch sizes!):"
echo "  cd linear_probes"
echo ""
echo "  # Generate responses (batch-size 32-64 for 1.5B model on H100)"
echo "  python 01_generate_responses.py --batch-size 32 --compile --max-examples -1"
echo ""
echo "  # Extract hidden states"
echo "  python 04_extract_hidden_states.py --batch-size 32"
echo ""
echo "  # Train probes"
echo "  python 05_train_probes.py --help"
SETUP_EOF

chmod +x "$REMOTE_SETUP_SCRIPT"

# Step 3: Sync files to remote
log_info "Syncing files to $REMOTE_HOST:$REMOTE_DIR..."

# Create remote directory
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Sync project files (excluding venv, cache, etc.)
rsync -avz --progress \
    --exclude '.venv*' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.claude' \
    --exclude 'logs/' \
    --exclude '.DS_Store' \
    --exclude '*.egg-info' \
    "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

log_info "Files synced successfully!"

# Step 4: Run remote setup
log_info "Running setup on remote machine..."
ssh -t "$REMOTE_HOST" "cd $REMOTE_DIR && bash setup_remote.sh"

echo ""
log_info "Deployment complete!"
echo ""
echo "Connect to remote and run probes:"
echo "  ssh $REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  source .venv/bin/activate"
echo "  cd linear_probes"
echo ""
echo "H100 optimized commands:"
echo "  python 01_generate_responses.py --batch-size 32 --compile --max-examples -1"
echo "  python 04_extract_hidden_states.py --batch-size 32"
