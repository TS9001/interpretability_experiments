"""Shared fixtures for linear_probes tests."""
import pytest
from pathlib import Path


@pytest.fixture
def sample_operation():
    """Sample operation dict with all fields."""
    return {
        'operand1': 5.0,
        'operand2': 3.0,
        'operator': 'add',
        'result': 8.0,
        'is_correct': True,
        'is_intermediate': True,
        'operand1_positions': [10],
        'operand2_positions': [14],
        'operator_positions': [12],
        'result_positions': [18],
    }


@pytest.fixture
def sample_operations():
    """List of sample operations for multi-step testing."""
    return [
        {
            'operand1': 10.0,
            'operand2': 5.0,
            'operator': 'add',
            'result': 15.0,
            'is_correct': True,
            'is_intermediate': True,
            'operand1_positions': [5],
            'operand2_positions': [9],
            'operator_positions': [7],
            'result_positions': [13],
        },
        {
            'operand1': 15.0,
            'operand2': 3.0,
            'operator': 'mult',
            'result': 45.0,
            'is_correct': True,
            'is_intermediate': False,
            'operand1_positions': [20],
            'operand2_positions': [24],
            'operator_positions': [22],
            'result_positions': [28],
        },
    ]


@pytest.fixture
def sample_response():
    """Sample response dict."""
    return {
        'text': 'First, 10 + 5 = 15. Then 15 * 3 = 45.',
        'final_answer': 45.0,
        'final_correct': True,
        'operations': [
            {
                'operand1': 10.0,
                'operand2': 5.0,
                'operator': 'add',
                'result': 15.0,
                'is_correct': True,
                'operand1_positions': [5],
                'operand2_positions': [9],
                'operator_positions': [7],
                'result_positions': [13],
            }
        ],
        'num_operations': 1,
    }


@pytest.fixture
def sample_tokens():
    """Sample token list."""
    return ['First', ',', ' ', '10', ' ', '+', ' ', '5', ' ', '=', ' ', '15']


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file."""
    import json
    file_path = tmp_path / "test.json"
    data = [{'index': 0, 'value': 'test'}, {'index': 1, 'value': 'test2'}]
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file."""
    import json
    file_path = tmp_path / "test.jsonl"
    data = [{'index': 0, 'value': 'test'}, {'index': 1, 'value': 'test2'}]
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return file_path
