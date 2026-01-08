"""Tests for utils/data.py"""
import pytest
import json
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import (
    load_json, load_jsonl, save_json, save_jsonl,
    get_existing_indices, format_prompt,
    parse_csv_ints, parse_csv_strings,
)


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_valid_json(self, temp_json_file):
        data = load_json(temp_json_file)
        assert len(data) == 2
        assert data[0]['index'] == 0

    def test_load_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_load_valid_jsonl(self, temp_jsonl_file):
        data = load_jsonl(temp_jsonl_file)
        assert len(data) == 2
        assert data[1]['index'] == 1

    def test_load_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nonexistent.jsonl")


class TestSaveJson:
    """Tests for save_json function."""

    def test_roundtrip(self, tmp_path):
        data = {'key': 'value', 'number': 42}
        file_path = tmp_path / "output.json"
        save_json(data, file_path)
        loaded = load_json(file_path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        data = {'test': True}
        file_path = tmp_path / "nested" / "dir" / "file.json"
        save_json(data, file_path)
        assert file_path.exists()

    def test_list_data(self, tmp_path):
        data = [1, 2, 3]
        file_path = tmp_path / "list.json"
        save_json(data, file_path)
        loaded = load_json(file_path)
        assert loaded == data


class TestSaveJsonl:
    """Tests for save_jsonl function."""

    def test_roundtrip(self, tmp_path):
        data = [{'a': 1}, {'b': 2}]
        file_path = tmp_path / "output.jsonl"
        save_jsonl(data, file_path)
        loaded = load_jsonl(file_path)
        assert loaded == data


class TestGetExistingIndices:
    """Tests for get_existing_indices function."""

    def test_from_json(self, temp_json_file):
        indices = get_existing_indices(temp_json_file)
        assert indices == {0, 1}

    def test_from_jsonl(self, temp_jsonl_file):
        indices = get_existing_indices(temp_jsonl_file)
        assert indices == {0, 1}

    def test_nonexistent_file(self, tmp_path):
        indices = get_existing_indices(tmp_path / "nonexistent.json")
        assert indices == set()

    def test_empty_file(self, tmp_path):
        file_path = tmp_path / "empty.json"
        file_path.write_text("[]")
        indices = get_existing_indices(file_path)
        assert indices == set()


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_basic(self):
        result = format_prompt("What is 2 + 2?")
        assert "What is 2 + 2?" in result
        assert "Question:" in result
        assert "Answer:" in result

    def test_format_structure(self):
        result = format_prompt("Test question")
        expected = "Question: Test question\n\nAnswer:"
        assert result == expected


class TestParseCsvInts:
    """Tests for parse_csv_ints function."""

    def test_basic(self):
        result = parse_csv_ints("1,2,3")
        assert result == [1, 2, 3]

    def test_with_spaces(self):
        result = parse_csv_ints("1, 2, 3")
        assert result == [1, 2, 3]

    def test_single_value(self):
        result = parse_csv_ints("42")
        assert result == [42]

    def test_empty_string(self):
        result = parse_csv_ints("")
        assert result is None

    def test_none(self):
        result = parse_csv_ints(None)
        assert result is None


class TestParseCsvStrings:
    """Tests for parse_csv_strings function."""

    def test_basic(self):
        result = parse_csv_strings("a,b,c")
        assert result == ['a', 'b', 'c']

    def test_with_spaces(self):
        result = parse_csv_strings("a, b, c")
        assert result == ['a', 'b', 'c']

    def test_empty_with_default(self):
        result = parse_csv_strings("", default=['x'])
        assert result == ['x']

    def test_empty_no_default(self):
        result = parse_csv_strings("")
        assert result == []
