#!/usr/bin/env python3
"""
Probe position extraction and label generation utilities.

Maps each probe type to the token positions it needs and generates appropriate labels.
"""
from typing import Optional


# Probe to position type mapping
PROBE_POSITION_MAP = {
    # Category A: Problem Understanding - use last question token
    'A1': 'last_question_token',
    'A2': 'last_question_token',

    # Category B: Numerical Representation
    'B1': 'operand_tokens',
    'B2': 'result_tokens',

    # Category C: Computation Mechanics
    'C1': 'result_tokens',
    'C3': 'result_tokens',
    'C4': 'equals_tokens',

    # Category D: Sequential Reasoning
    'D1': 'result_tokens',
    'D2': 'result_tokens',
    'D3': 'operator_and_result_tokens',
    'D6': 'operator_tokens',

    # Category E: Verification & Metacognition
    'E1': 'result_tokens',
    'E2': 'result_tokens',
    'E3': 'result_tokens',
}

# Magnitude bins for B1, B2 probes
MAGNITUDE_BINS = {
    'negative': 0,
    '0-10': 1,
    '10-100': 2,
    '100-1K': 3,
    '1K-10K': 4,
    '10K+': 5,
}

# Coarse result bins for C4 probe
COARSE_BINS = {
    '<10': 0,
    '10-100': 1,
    '100-1000': 2,
    '>1000': 3,
}

# Operation type mapping
OPERATION_TYPES = {
    'add': 0,
    'sub': 1,
    'mult': 2,
    'div': 3,
}

# Operation label mapping (includes None/END/FIRST as 4)
OPERATION_LABEL_MAP = {
    'add': 0,
    'sub': 1,
    'mult': 2,
    'div': 3,
    None: 4,  # END (for next_op) or FIRST (for prev_op)
}


def get_next_op_label(next_op: Optional[str]) -> int:
    """Convert next operation to label (D2 probe). None = END."""
    return OPERATION_LABEL_MAP.get(next_op, 4)


def get_prev_op_label(prev_op: Optional[str]) -> int:
    """Convert previous operation to label (D6 probe). None = FIRST."""
    return OPERATION_LABEL_MAP.get(prev_op, 4)


def get_last_position(positions: list[int]) -> int:
    """For multi-token spans, return last token position."""
    if not positions or positions[0] == -1:
        return -1
    return positions[-1]


def get_magnitude_bin(value: float) -> int:
    """Get magnitude bin for a numerical value."""
    if value < 0:
        return MAGNITUDE_BINS['negative']
    elif value < 10:
        return MAGNITUDE_BINS['0-10']
    elif value < 100:
        return MAGNITUDE_BINS['10-100']
    elif value < 1000:
        return MAGNITUDE_BINS['100-1K']
    elif value < 10000:
        return MAGNITUDE_BINS['1K-10K']
    else:
        return MAGNITUDE_BINS['10K+']


def get_coarse_bin(value: float) -> int:
    """Get coarse result bin for C4 probe."""
    abs_val = abs(value)
    if abs_val < 10:
        return COARSE_BINS['<10']
    elif abs_val < 100:
        return COARSE_BINS['10-100']
    elif abs_val < 1000:
        return COARSE_BINS['100-1000']
    else:
        return COARSE_BINS['>1000']


def get_difficulty_bin(total_operations: int) -> int:
    """Get difficulty bin for A2 probe (1, 2, 3, 4, 5+)."""
    return min(total_operations, 5) - 1


def get_step_position(step_idx: int, total_steps: int) -> int:
    """Get step position category for D3 probe (first=0, middle=1, last=2)."""
    if step_idx == 0:
        return 0  # first
    elif step_idx == total_steps - 1:
        return 2  # last
    else:
        return 1  # middle


def get_last_question_token_position(question: str, tokenizer) -> int:
    """Get position of last question token."""
    tokens = tokenizer(question, add_special_tokens=False)
    return len(tokens['input_ids']) - 1


def get_operand_positions(operations: list[dict]) -> list[dict]:
    """
    Get operand positions for B1 probe.

    Returns list of dicts with position and value for each operand.
    """
    positions = []
    for op in operations:
        # Operand 1
        pos1 = get_last_position(op.get('operand1_positions', [-1]))
        if pos1 != -1:
            positions.append({
                'position': pos1,
                'value': op['operand1'],
                'label': get_magnitude_bin(op['operand1']),
                'operation_type': op['operator'],
                'operation_idx': operations.index(op),
            })

        # Operand 2
        pos2 = get_last_position(op.get('operand2_positions', [-1]))
        if pos2 != -1:
            positions.append({
                'position': pos2,
                'value': op['operand2'],
                'label': get_magnitude_bin(op['operand2']),
                'operation_type': op['operator'],
                'operation_idx': operations.index(op),
            })

    return positions


def get_operator_positions(operations: list[dict]) -> list[dict]:
    """
    Get operator positions for D6 probe.

    Returns list of dicts with position and metadata.
    """
    positions = []
    for idx, op in enumerate(operations):
        pos = get_last_position(op.get('operator_positions', [-1]))
        if pos != -1:
            # Previous operation type (or FIRST for first operation)
            if idx == 0:
                prev_op = 4  # FIRST
            else:
                prev_op = OPERATION_TYPES.get(operations[idx - 1]['operator'], 0)

            positions.append({
                'position': pos,
                'operation_type': op['operator'],
                'operation_idx': idx,
                'label': prev_op,  # D6: previous operation
            })

    return positions


def get_result_positions(operations: list[dict]) -> list[dict]:
    """
    Get result positions for probes B2, C1, C3, D1, D2, E1, E2, E3.

    Returns list of dicts with position and all relevant labels.
    """
    positions = []
    total_ops = len(operations)

    for idx, op in enumerate(operations):
        pos = get_last_position(op.get('result_positions', [-1]))
        if pos != -1:
            # Next operation type (or END)
            if idx == total_ops - 1:
                next_op = 4  # END
            else:
                next_op = OPERATION_TYPES.get(operations[idx + 1]['operator'], 0)

            positions.append({
                'position': pos,
                'value': op['result'],
                'operation_type': op['operator'],
                'operation_idx': idx,
                'is_intermediate': op.get('is_intermediate', False),
                # Labels for different probes
                'B2_label': get_magnitude_bin(op['result']),
                'C4_label': get_coarse_bin(op['result']),
                'D1_label': 1 if op.get('is_intermediate', False) else 0,
                'D2_label': next_op,
                'D3_label': get_step_position(idx, total_ops),
            })

    return positions


def get_equals_positions(tokens: list[str]) -> list[int]:
    """
    Find positions of "=" tokens for C4 probe.
    """
    positions = []
    for i, token in enumerate(tokens):
        clean = token.replace('Ġ', '').replace('▁', '').strip()
        if clean == '=':
            positions.append(i)
    return positions


def get_operator_and_result_positions(operations: list[dict]) -> list[dict]:
    """
    Get both operator and result positions for D3 probe.
    """
    positions = []
    total_ops = len(operations)

    for idx, op in enumerate(operations):
        step_label = get_step_position(idx, total_ops)

        # Operator position
        op_pos = get_last_position(op.get('operator_positions', [-1]))
        if op_pos != -1:
            positions.append({
                'position': op_pos,
                'type': 'operator',
                'operation_idx': idx,
                'label': step_label,
            })

        # Result position
        res_pos = get_last_position(op.get('result_positions', [-1]))
        if res_pos != -1:
            positions.append({
                'position': res_pos,
                'type': 'result',
                'operation_idx': idx,
                'label': step_label,
            })

    return positions


def get_all_probe_positions(
    operations: list[dict],
    tokens: list[str],
    question: str,
    tokenizer,
    operations_by_type: Optional[dict] = None,
    total_operations: Optional[int] = None,
) -> dict:
    """
    Get all positions needed for all probes.

    Returns dict mapping position type to list of position dicts.
    """
    return {
        'last_question_token': get_last_question_token_position(question, tokenizer),
        'operand_tokens': get_operand_positions(operations),
        'operator_tokens': get_operator_positions(operations),
        'result_tokens': get_result_positions(operations),
        'equals_tokens': get_equals_positions(tokens),
        'operator_and_result_tokens': get_operator_and_result_positions(operations),
    }


def get_A1_labels(operations_by_type: dict) -> list[int]:
    """Get A1 multi-label: which operations are needed."""
    return [
        1 if operations_by_type.get('add', 0) > 0 else 0,
        1 if operations_by_type.get('sub', 0) > 0 else 0,
        1 if operations_by_type.get('mult', 0) > 0 else 0,
        1 if operations_by_type.get('div', 0) > 0 else 0,
    ]


def get_A2_label(total_operations: int) -> int:
    """Get A2 label: difficulty bin."""
    return get_difficulty_bin(total_operations)


def check_operation_correctness(op: dict) -> bool:
    """Check if an operation result is mathematically correct."""
    op1, op2, result = op['operand1'], op['operand2'], op['result']
    op_type = op['operator']

    try:
        if op_type == 'add':
            expected = op1 + op2
        elif op_type == 'sub':
            expected = op1 - op2
        elif op_type == 'mult':
            expected = op1 * op2
        elif op_type == 'div':
            expected = op1 / op2 if op2 != 0 else None
        else:
            return True

        if expected is None:
            return False

        return abs(result - expected) < 0.01
    except Exception:
        return False


def get_unique_positions(probe_positions: dict) -> list[int]:
    """Get sorted list of all unique positions across all probe types."""
    positions = set()

    # Last question token
    lqt = probe_positions.get('last_question_token', -1)
    if lqt >= 0:
        positions.add(lqt)

    # Operand tokens
    for item in probe_positions.get('operand_tokens', []):
        if item['position'] >= 0:
            positions.add(item['position'])

    # Operator tokens
    for item in probe_positions.get('operator_tokens', []):
        if item['position'] >= 0:
            positions.add(item['position'])

    # Result tokens
    for item in probe_positions.get('result_tokens', []):
        if item['position'] >= 0:
            positions.add(item['position'])

    # Equals tokens
    for pos in probe_positions.get('equals_tokens', []):
        if pos >= 0:
            positions.add(pos)

    return sorted(positions)
