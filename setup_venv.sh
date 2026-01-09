#!/bin/bash
# Setup script for creating a Python virtual environment
#
# Usage:
#   ./setup_venv.sh           # Create venv with all dependencies
#   ./setup_venv.sh --force   # Force recreate venv
#
# Requirements: Python 3.11+

set -e

FORCE=false
if [ "$1" = "--force" ]; then
    FORCE=true
fi

# Find a suitable Python 3.11+ interpreter
PYTHON_CMD=""

for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)

        if [ "$major" = "3" ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No Python 3.11+ found. Please install Python 3.11 or newer."
    echo "Checked: python3.11, python3.12, python3.13, python3"
    exit 1
fi

echo "Found: $("$PYTHON_CMD" --version) at $(which "$PYTHON_CMD")"

# Check if .venv-pip already exists
if [ -d ".venv-pip" ]; then
    if [ "$FORCE" = true ]; then
        echo "Removing existing .venv-pip (--force)..."
        rm -rf .venv-pip
    else
        echo ".venv-pip already exists."
        echo "  To recreate: rm -rf .venv-pip && ./setup_venv.sh"
        echo "  Or use: ./setup_venv.sh --force"
        exit 0
    fi
fi

echo ""
echo "Creating virtual environment..."
"$PYTHON_CMD" -m venv .venv-pip

echo "Upgrading pip, setuptools, and wheel..."
.venv-pip/bin/pip install --upgrade pip setuptools wheel

# Install core requirements
echo ""
echo "Installing core requirements..."
.venv-pip/bin/pip install \
    datasets>=2.14.0 \
    huggingface-hub>=0.17.0 \
    transformers>=4.35.0 \
    torch>=2.0.0 \
    accelerate>=0.24.0 \
    typer>=0.9.0 \
    rich>=13.0.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    tqdm>=4.65.0 \
    loguru

# Try to install flash-attn (optional, requires CUDA)
echo ""
echo "Attempting to install flash-attn (optional, CUDA only)..."
.venv-pip/bin/pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo ""
    echo "INFO: flash-attn not installed (requires CUDA + compatible GPU)."
    echo "      The pipeline will work without it (uses standard attention)."
}

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv-pip/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  ./run_pipeline.sh --help"
echo ""
