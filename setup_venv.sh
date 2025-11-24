#!/bin/bash
# Setup script for creating a standard Python virtual environment

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Error: python3.11 not found. Please install Python 3.11+"
    exit 1
fi

# Check if .venv-pip already exists
if [ -d ".venv-pip" ]; then
    echo ".venv-pip already exists. To recreate, delete it first: rm -rf .venv-pip"
    exit 1
fi

echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv .venv-pip

echo "Upgrading pip, setuptools, and wheel..."
.venv-pip/bin/pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
.venv-pip/bin/pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv-pip/bin/activate"
echo ""
echo "Then you can run inspect commands directly:"
echo "  cd gmsk8_generation"
echo "  inspect eval inspect_annotation_prompt.py@gsm8k_annotation_task --model anthropic/claude-3-5-sonnet-latest --limit 1"

