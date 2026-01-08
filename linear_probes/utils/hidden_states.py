"""Hidden state extraction utilities."""
from typing import Optional, Tuple, List

import torch


def extract_hidden_at_positions(
    model,
    tokenizer,
    text: str,
    positions: list[int],
    device: torch.device,
    layers_to_save: Optional[list[int]] = None,
    max_length: int = 2048
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """
    Extract hidden states at specific positions only.

    Returns (hidden_states, valid_positions) or None if no valid positions.
    Shape of hidden_states: (num_layers, num_positions, hidden_dim)
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device, non_blocking=(device.type == "cuda"))

    seq_len = encodings.input_ids.shape[1]
    valid_positions = [p for p in positions if 0 <= p < seq_len]

    if not valid_positions:
        return None

    with torch.no_grad():
        outputs = model(
            **encodings,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
    num_layers = len(hidden_states)
    layer_indices = layers_to_save if layers_to_save is not None else list(range(num_layers))

    extracted = []
    for layer_idx in layer_indices:
        layer_hidden = hidden_states[layer_idx][0]  # Remove batch dim
        pos_hidden = layer_hidden[valid_positions, :]
        extracted.append(pos_hidden)

    result = torch.stack(extracted, dim=0).cpu().float()
    return result, valid_positions


def extract_hidden_batch(
    model,
    tokenizer,
    texts: list[str],
    positions_list: list[list[int]],
    device: torch.device,
    layers_to_save: Optional[list[int]] = None,
    max_length: int = 2048
) -> list:
    """
    Extract hidden states for multiple texts in a single forward pass.

    Returns list of (hidden_states, valid_positions) tuples or None entries.
    """
    if not texts:
        return []

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device, non_blocking=(device.type == "cuda"))

    batch_size = len(texts)
    attention_mask = encodings.attention_mask

    with torch.no_grad():
        outputs = model(
            **encodings,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states[1:]
    num_layers = len(hidden_states)
    layer_indices = layers_to_save if layers_to_save is not None else list(range(num_layers))

    results = []
    for b in range(batch_size):
        seq_len = attention_mask[b].sum().item()
        positions = positions_list[b]
        valid_positions = [p for p in positions if 0 <= p < seq_len]

        if not valid_positions:
            results.append(None)
            continue

        extracted = []
        for layer_idx in layer_indices:
            layer_hidden = hidden_states[layer_idx][b]
            pos_hidden = layer_hidden[valid_positions, :]
            extracted.append(pos_hidden)

        result = torch.stack(extracted, dim=0).cpu().float()
        results.append((result, valid_positions))

    return results
