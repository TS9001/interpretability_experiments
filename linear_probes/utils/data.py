"""Data loading and saving utilities."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union


def load_json(file_path: Path) -> Union[list, dict]:
    """Load a JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def load_jsonl(file_path: Path) -> list[dict]:
    """Load a JSONL file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def save_json(data: Any, file_path: Path, indent: int = 2):
    """Save data as pretty-printed JSON."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def save_jsonl(data: list[dict], file_path: Path):
    """Save data as JSONL."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_existing_indices(file_path: Path) -> set[int]:
    """Get already processed indices for resume functionality."""
    file_path = Path(file_path)
    if not file_path.exists():
        return set()

    try:
        if file_path.suffix == '.jsonl':
            data = load_jsonl(file_path)
        else:
            data = load_json(file_path)
            if not isinstance(data, list):
                return set()
        return {item['index'] for item in data if 'index' in item}
    except (json.JSONDecodeError, KeyError):
        return set()


def save_split(
    results: list[dict],
    output_dir: Path,
    split_name: str,
    resume: bool = False
) -> tuple[Path, Path]:
    """
    Save results split into main (readable) and tokens (for analysis) files.

    Args:
        results: List with 'index', 'question', 'responses' (each with 'text', 'token_ids')
        output_dir: Directory to save files
        split_name: Name prefix (e.g., 'train', 'test')
        resume: If True, merge with existing data

    Returns:
        Tuple of (main_file_path, tokens_file_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_file = output_dir / f"{split_name}_responses.json"
    tokens_file = output_dir / f"{split_name}_tokens.json"

    # Split results
    main_data = []
    token_data = []

    for result in results:
        main_data.append({
            'index': result['index'],
            'question': result['question'],
            'responses': [r['text'] for r in result['responses']],
        })
        token_data.append({
            'index': result['index'],
            'responses': [r['token_ids'] for r in result['responses']],
        })

    # Merge with existing if resuming
    if resume:
        if main_file.exists():
            main_data = load_json(main_file) + main_data
        if tokens_file.exists():
            token_data = load_json(tokens_file) + token_data

    save_json(main_data, main_file)
    save_json(token_data, tokens_file)

    return main_file, tokens_file


def save_metadata(
    output_dir: Path,
    model_name: Optional[str] = None,
    config: Optional[dict] = None,
    stats: Optional[dict] = None,
    **extra
):
    """Save metadata.json with common fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {'timestamp': datetime.now().isoformat()}
    if model_name:
        metadata['model_name'] = model_name
    if config:
        metadata['config'] = config
    if stats:
        metadata['stats'] = stats
    metadata.update(extra)

    save_json(metadata, output_dir / 'metadata.json')


def format_prompt(question: str) -> str:
    """Format a question as a prompt."""
    return f"Question: {question}\n\nAnswer:"


def parse_csv_ints(s: str) -> Optional[List[int]]:
    """
    Parse comma-separated integers from a string.

    Args:
        s: Comma-separated string like "0,7,14,21,27"

    Returns:
        List of integers, or None if input is empty/None
    """
    if not s:
        return None
    return [int(x.strip()) for x in s.split(',')]


def parse_csv_strings(s: str, default: Optional[List[str]] = None) -> List[str]:
    """
    Parse comma-separated strings from a string.

    Args:
        s: Comma-separated string like "A1,B1,C1"
        default: Default value if input is empty

    Returns:
        List of strings, or default if input is empty
    """
    if not s:
        return default if default is not None else []
    return [x.strip() for x in s.split(',')]
