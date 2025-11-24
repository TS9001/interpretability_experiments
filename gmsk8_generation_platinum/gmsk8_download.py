#!/usr/bin/env python3
"""
Script to download GMSK8 dataset and prepare annotation data.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR / "resources"
ANNOTATION_DIR = SCRIPT_DIR / "annotation"

def download_gmsk8_dataset():
    """Download GMSK8/GSM8K dataset and store it in resources folder."""
    print("Downloading GSM8K dataset...")
    
    RESOURCES_DIR.mkdir(exist_ok=True, parents=True)
    
    try:
        print(f"Loading dataset: gsm8k (main config)...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", cache_dir=RESOURCES_DIR / "cache")
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        dataset_path = RESOURCES_DIR / "gmsk8_dataset"
        dataset.save_to_disk(str(dataset_path))
        print(f"Dataset saved to: {dataset_path}")
        
        return dataset
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def create_annotation_folder(dataset):
    """Create annotation folder with first 100 test examples."""
    print("\nCreating annotation folder with first 100 test examples...")
    
    ANNOTATION_DIR.mkdir(exist_ok=True, parents=True)
    
    # Get test split
    if 'test' not in dataset:
        print("Warning: 'test' split not found. Available splits:", list(dataset.keys()))
        test_split = list(dataset.keys())[0]
        print(f"Using '{test_split}' split instead")
    else:
        test_split = 'test'
    
    test_data = dataset[test_split]
    
    # Extract first 100 examples
    num_examples = min(100, len(test_data))
    print(f"Extracting {num_examples} examples from {test_split} split...")
    
    first_100 = test_data.select(range(num_examples))
    
    # Save as JSON Lines format
    output_file = ANNOTATION_DIR / "first_100_test_examples.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in first_100:
            f.write(json.dumps(dict(example)) + '\n')
    
    print(f"Saved {num_examples} examples to: {output_file}")
    
    # Also save as a single JSON file for convenience
    output_json = ANNOTATION_DIR / "first_100_test_examples.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump([dict(example) for example in first_100], f, indent=2, ensure_ascii=False)
    
    print(f"Also saved as JSON to: {output_json}")
    
    # Save a summary
    summary_file = ANNOTATION_DIR / "README.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# GMSK8 Test Set Annotation\n\n")
        f.write(f"This folder contains the first {num_examples} examples from the GMSK8 {test_split} split.\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- `first_100_test_examples.jsonl`: Examples in JSON Lines format (one per line)\n")
        f.write(f"- `first_100_test_examples.json`: Examples in standard JSON array format\n\n")
        f.write(f"## Dataset Information\n\n")
        f.write(f"- Total examples in this file: {num_examples}\n")
        f.write(f"- Source: GMSK8 dataset ({test_split} split)\n")
        if len(first_100) > 0:
            f.write(f"- Fields: {list(first_100[0].keys())}\n")
    
    print(f"Created README at: {summary_file}")

def main():
    """Main function to download dataset and create annotation folder."""
    print("=" * 60)
    print("GMSK8 Dataset Download and Setup")
    print("=" * 60)
    
    dataset = download_gmsk8_dataset()
    
    # Create annotation folder with first 100 test examples
    create_annotation_folder(dataset)
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

