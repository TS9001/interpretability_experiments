# GSM8K Dataset Generation

This directory contains scripts and data for working with the GSM8K (Grade School Math 8K) dataset.

## Setup

### 1. Create Virtual Environment

This project uses `uv` for fast package management. Install and set up the virtual environment:

```bash
cd gmsk8_generation
uv venv
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Download Dataset

Run the download script to fetch the GSM8K dataset and create annotation data:

```bash
.venv/bin/python gmsk8_download.py
```

This script will:
- Download the GSM8K dataset from Hugging Face
- Store it in the `resources/` folder (ignored by git)
- Create an `annotation/` folder with the first 100 test examples

## Directory Structure

```
gmsk8_generation/
├── .venv/                  # Virtual environment (ignored by git)
├── annotation/             # First 100 test examples for annotation
│   ├── first_100_test_examples.json
│   ├── first_100_test_examples.jsonl
│   └── README.md
├── resources/              # Dataset files (ignored by git)
│   ├── cache/             # Hugging Face cache
│   └── gmsk8_dataset/     # Downloaded GSM8K dataset
├── gmsk8_download.py       # Download script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dataset Information

The GSM8K dataset contains:
- **Train split**: 7,473 examples
- **Test split**: 1,319 examples

Each example includes:
- `question`: A grade-school level math word problem
- `answer`: Step-by-step solution with the final answer

## Annotation Folder

The `annotation/` folder contains the first 100 examples from the test split in two formats:
- **JSON Lines** (`.jsonl`): One example per line, easy for streaming
- **JSON** (`.json`): Standard JSON array format

This folder is tracked by git and can be used for annotation tasks.

## Resources Folder

The `resources/` folder is **excluded from git** (see `.gitignore`) and contains:
- Downloaded dataset files
- Hugging Face cache
- The download script

To regenerate the dataset, simply run the download script again.

