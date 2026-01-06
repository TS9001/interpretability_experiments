"""
Probe label generation utilities.

Generates labels for each probe type based on example and response data.
"""
from utils.probe_positions import get_A1_labels, get_A2_label, check_operation_correctness


def generate_probe_labels(
    probes: list[str],
    example: dict,
    resp_operations: list[dict]
) -> dict:
    """
    Generate labels for specified probes.

    Args:
        probes: List of probe names (e.g., ['A1', 'A2', 'C1', 'E1'])
        example: Example dict with ground truth data
        resp_operations: List of operation dicts from the response

    Returns:
        Dictionary mapping probe name to label(s)
    """
    labels = {}

    # A1: Operation types needed (from ground truth)
    if 'A1' in probes:
        ops_by_type = example.get('ground_truth', {}).get('operations_by_type', {})
        if not ops_by_type and example.get('ground_truth_operations'):
            # Compute from operations if not directly available
            ops_by_type = {'add': 0, 'sub': 0, 'mult': 0, 'div': 0}
            for op in example['ground_truth_operations']:
                op_type = op.get('operator', '')
                if op_type in ops_by_type:
                    ops_by_type[op_type] += 1
        labels['A1'] = get_A1_labels(ops_by_type)

    # A2: Difficulty (number of operations)
    if 'A2' in probes:
        total_ops = len(example.get('ground_truth_operations', []))
        labels['A2'] = get_A2_label(total_ops)

    # C1, E1: Correctness labels for each operation
    if 'C1' in probes or 'E1' in probes:
        correctness = [
            1 if check_operation_correctness(op) else 0
            for op in resp_operations
        ]
        if 'C1' in probes:
            labels['C1'] = correctness
        if 'E1' in probes:
            labels['E1'] = correctness

    return labels
