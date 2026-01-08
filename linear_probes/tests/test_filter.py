"""Tests for 03_filter_probeable.py functions."""
import pytest
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "filter_probeable",
    Path(__file__).parent.parent / "03_filter_probeable.py"
)
filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_module)

is_operation_probeable = filter_module.is_operation_probeable
is_response_probeable = filter_module.is_response_probeable
score_response = filter_module.score_response
compute_operations_by_type = filter_module.compute_operations_by_type


class TestIsOperationProbeable:
    """Tests for is_operation_probeable function."""

    def test_all_positions_valid(self, sample_operation):
        assert is_operation_probeable(sample_operation) is True

    def test_missing_operand1(self, sample_operation):
        sample_operation['operand1_positions'] = [-1]
        assert is_operation_probeable(sample_operation) is False

    def test_missing_operator(self, sample_operation):
        sample_operation['operator_positions'] = [-1]
        assert is_operation_probeable(sample_operation) is False

    def test_missing_result(self, sample_operation):
        sample_operation['result_positions'] = [-1]
        assert is_operation_probeable(sample_operation) is False

    def test_missing_key(self):
        """Operation missing position keys should not be probeable."""
        op = {}  # No position keys at all - defaults to [-1]
        assert is_operation_probeable(op) is False


class TestIsResponseProbeable:
    """Tests for is_response_probeable function."""

    def test_all_ops_probeable(self, sample_operations):
        resp = {'operations': sample_operations}
        assert is_response_probeable(resp) is True

    def test_one_op_not_probeable(self, sample_operations):
        sample_operations[0]['operand1_positions'] = [-1]
        resp = {'operations': sample_operations}
        assert is_response_probeable(resp) is False

    def test_no_operations(self):
        resp = {'operations': []}
        assert is_response_probeable(resp) is False

    def test_missing_operations_key(self):
        resp = {}
        assert is_response_probeable(resp) is False


class TestScoreResponse:
    """Tests for score_response function."""

    def test_all_correct(self):
        resp = {
            'final_correct': True,
            'operations': [
                {'is_correct': True, 'operator': 'add'},
                {'is_correct': True, 'operator': 'mult'},
            ]
        }
        gt_ops = [{'operator': 'add'}, {'operator': 'mult'}]
        gt_result = 100.0

        score = score_response(resp, gt_ops, gt_result)
        # (final_correct, all_ops_correct, ops_match_gt, num_ops)
        assert score == (1, 1, 1, 2)

    def test_incorrect_final(self):
        resp = {
            'final_correct': False,
            'operations': [{'is_correct': True, 'operator': 'add'}]
        }
        gt_ops = [{'operator': 'add'}]

        score = score_response(resp, gt_ops, 100.0)
        assert score[0] == 0  # final_correct

    def test_ops_mismatch(self):
        resp = {
            'final_correct': True,
            'operations': [{'is_correct': True, 'operator': 'add'}]
        }
        gt_ops = [{'operator': 'add'}, {'operator': 'mult'}]  # 2 ops

        score = score_response(resp, gt_ops, 100.0)
        assert score[2] == 0  # ops_match_gt is False


class TestComputeOperationsByType:
    """Tests for compute_operations_by_type function."""

    def test_mixed_operations(self):
        ops = [
            {'operator': 'add'},
            {'operator': 'add'},
            {'operator': 'mult'},
        ]
        result = compute_operations_by_type(ops)
        assert result == {'add': 2, 'sub': 0, 'mult': 1, 'div': 0}

    def test_empty(self):
        result = compute_operations_by_type([])
        assert result == {'add': 0, 'sub': 0, 'mult': 0, 'div': 0}

    def test_all_types(self):
        ops = [
            {'operator': 'add'},
            {'operator': 'sub'},
            {'operator': 'mult'},
            {'operator': 'div'},
        ]
        result = compute_operations_by_type(ops)
        assert result == {'add': 1, 'sub': 1, 'mult': 1, 'div': 1}

    def test_unknown_operator(self):
        """Unknown operators should be ignored."""
        ops = [{'operator': 'add'}, {'operator': 'unknown'}]
        result = compute_operations_by_type(ops)
        assert result == {'add': 1, 'sub': 0, 'mult': 0, 'div': 0}
