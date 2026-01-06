"""Probe data utilities for hidden state extraction."""
from pathlib import Path

import torch

from utils.logging import log
from utils.probe_positions import PROBE_POSITION_MAP


def adjust_probe_positions(positions: dict, prompt_length: int) -> dict:
    """
    Adjust probe positions to account for prompt offset.

    The model processes prompt + response together, so response token positions
    need to be offset by the prompt length. The 'last_question_token' position
    is already relative to the full sequence and doesn't need adjustment.

    Args:
        positions: Dictionary of position type -> position data
        prompt_length: Number of tokens in the prompt

    Returns:
        Dictionary with adjusted positions
    """
    adjusted = {}
    for pos_type, pos_data in positions.items():
        if pos_type == 'last_question_token':
            # Already relative to full sequence
            adjusted[pos_type] = pos_data
        elif isinstance(pos_data, list):
            if pos_data and isinstance(pos_data[0], dict):
                # List of position dicts with 'position' key
                adjusted[pos_type] = [
                    {**p, 'position': p['position'] + prompt_length}
                    for p in pos_data
                ]
            else:
                # List of plain position integers
                adjusted[pos_type] = [p + prompt_length for p in pos_data]
        else:
            # Single value (shouldn't happen but handle it)
            adjusted[pos_type] = pos_data
    return adjusted


def build_pos_to_idx(positions: list[int]) -> dict[int, int]:
    """
    Build a mapping from position to index in extracted hidden states.

    Args:
        positions: List of token positions that were extracted

    Returns:
        Dictionary mapping position -> index in the hidden states tensor
    """
    return {p: i for i, p in enumerate(positions)}


def save_probe_data(results: list[dict], output_dir: Path, probes: list[str]):
    """Save extracted data organized by probe type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    torch.save(results, output_dir / 'full_data.pt')

    # Save probe-specific views
    for probe in probes:
        probe_dir = output_dir / f"{probe}_data"
        probe_dir.mkdir(exist_ok=True)

        probe_data = []
        pos_type = PROBE_POSITION_MAP.get(probe)

        for example in results:
            for resp in example.get('responses', []):
                positions_info = resp.get('probe_positions', {}).get(pos_type, [])
                hidden_states = resp.get('hidden_states')
                pos_to_idx = resp.get('pos_to_idx', {})
                labels = resp.get('labels', {}).get(probe)

                if isinstance(positions_info, int):
                    if positions_info in pos_to_idx:
                        idx = pos_to_idx[positions_info]
                        probe_data.append({
                            'example_idx': example['index'],
                            'hidden_state': hidden_states[:, idx, :],
                            'label': labels,
                        })
                elif isinstance(positions_info, list):
                    for i, pos_info in enumerate(positions_info):
                        if isinstance(pos_info, dict):
                            pos = pos_info['position']
                            if pos in pos_to_idx:
                                idx = pos_to_idx[pos]
                                item_label = pos_info.get('label', labels[i] if isinstance(labels, list) and i < len(labels) else labels)
                                probe_data.append({
                                    'example_idx': example['index'],
                                    'hidden_state': hidden_states[:, idx, :],
                                    'label': item_label,
                                    'metadata': pos_info,
                                })
                        else:
                            if pos_info in pos_to_idx:
                                idx = pos_to_idx[pos_info]
                                probe_data.append({
                                    'example_idx': example['index'],
                                    'hidden_state': hidden_states[:, idx, :],
                                    'position': pos_info,
                                })

        if probe_data:
            torch.save(probe_data, probe_dir / 'data.pt')
            log.info(f"  {probe}: {len(probe_data)} samples")
