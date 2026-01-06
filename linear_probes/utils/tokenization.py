"""Token position mapping utilities."""


def build_char_to_token_map(tokens: list[str], text: str) -> dict[int, int]:
    """
    Build a mapping from character indices to token indices.

    Handles tokenizer-specific prefixes like:
    - Ġ (GPT-style space prefix)
    - ▁ (SentencePiece space prefix)
    - <0x0A> (newline encoding)

    Args:
        tokens: List of token strings from tokenizer
        text: Original text that was tokenized

    Returns:
        Dictionary mapping character index to token index
    """
    char_to_token = {}
    current_char = 0

    for tok_idx, tok in enumerate(tokens):
        # Clean token (handle tokenizer-specific prefixes)
        clean_tok = tok.replace('Ġ', ' ').replace('▁', ' ').replace('<0x0A>', '\n')
        if clean_tok.startswith(' ') and current_char > 0:
            clean_tok = clean_tok[1:]

        tok_len = len(clean_tok) if clean_tok else 1
        for c in range(current_char, min(current_char + tok_len, len(text))):
            char_to_token[c] = tok_idx
        current_char += tok_len

    return char_to_token


def find_token_positions_for_span(
    start: int,
    end: int,
    char_to_token: dict[int, int]
) -> list[int]:
    """
    Find token indices that correspond to a character span.

    Args:
        start: Start character index
        end: End character index (exclusive)
        char_to_token: Character to token mapping from build_char_to_token_map

    Returns:
        Sorted list of unique token indices for the span, or [-1] if none found
    """
    positions = sorted(set(
        char_to_token.get(c, -1) for c in range(start, end)
        if char_to_token.get(c, -1) >= 0
    ))
    return positions if positions else [-1]
