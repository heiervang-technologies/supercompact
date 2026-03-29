"""Tests for lib/parser.py — Turn, extract_text, _is_user_message."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn, _is_user_message, extract_text, parse_jsonl


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

class TestTurn:
    def test_default_fields(self):
        t = Turn(kind="user")
        assert t.kind == "user"
        assert t.lines == []
        assert t.index == 0

    def test_custom_fields(self):
        t = Turn(kind="system", lines=[{"foo": "bar"}], index=3)
        assert t.kind == "system"
        assert t.lines == [{"foo": "bar"}]
        assert t.index == 3

    def test_append_adds_line(self):
        t = Turn(kind="user")
        t.append({"type": "user", "message": {"content": "hello"}})
        assert len(t.lines) == 1
        assert t.lines[0]["type"] == "user"

    def test_append_multiple(self):
        t = Turn(kind="system")
        t.append({"a": 1})
        t.append({"b": 2})
        assert len(t.lines) == 2


# ---------------------------------------------------------------------------
# _is_user_message
# ---------------------------------------------------------------------------

class TestIsUserMessage:
    def test_string_content_is_user(self):
        record = {"type": "user", "message": {"content": "hello"}}
        assert _is_user_message(record) is True

    def test_text_block_is_user(self):
        record = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hi"}]},
        }
        assert _is_user_message(record) is True

    def test_tool_result_block_is_not_user(self):
        record = {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": "ok"}]},
        }
        assert _is_user_message(record) is False

    def test_non_user_type_is_false(self):
        record = {"type": "assistant", "message": {"content": "reply"}}
        assert _is_user_message(record) is False

    def test_tool_injection_with_source_uuid_is_false(self):
        record = {
            "type": "user",
            "sourceToolAssistantUUID": "abc-123",
            "message": {"content": "injected"},
        }
        assert _is_user_message(record) is False

    def test_empty_content_list(self):
        record = {"type": "user", "message": {"content": []}}
        # Empty list: no tool_result blocks → considered user
        assert _is_user_message(record) is True


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractTextStringContent:
    def test_simple_string_content(self):
        t = Turn(kind="user")
        t.append({"message": {"content": "Hello world"}})
        assert extract_text(t) == "Hello world"

    def test_multiple_records_joined(self):
        t = Turn(kind="user")
        t.append({"message": {"content": "line one"}})
        t.append({"message": {"content": "line two"}})
        result = extract_text(t)
        assert "line one" in result
        assert "line two" in result

    def test_empty_turn_returns_empty(self):
        t = Turn(kind="user")
        assert extract_text(t) == ""


class TestExtractTextListContent:
    def test_text_block(self):
        t = Turn(kind="system")
        t.append({"message": {"content": [{"type": "text", "text": "hello"}]}})
        assert extract_text(t) == "hello"

    def test_thinking_block(self):
        t = Turn(kind="system")
        t.append({"message": {"content": [
            {"type": "thinking", "thinking": "let me think"}
        ]}})
        assert "let me think" in extract_text(t)

    def test_tool_use_block_includes_name(self):
        t = Turn(kind="system")
        t.append({"message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/foo.py"}}
        ]}})
        result = extract_text(t)
        assert "Read" in result
        assert "file_path" in result
        assert "/tmp/foo.py" in result

    def test_tool_use_input_truncated_at_500(self):
        long_input = "x" * 600
        t = Turn(kind="system")
        t.append({"message": {"content": [
            {"type": "tool_use", "name": "Write", "input": {"content": long_input}}
        ]}})
        result = extract_text(t)
        assert "..." in result
        assert len(result) < 700  # shouldn't be excessively long

    def test_tool_result_string_content(self):
        t = Turn(kind="user")
        t.append({"message": {"content": [
            {"type": "tool_result", "content": "file contents here"}
        ]}})
        assert "file contents here" in extract_text(t)

    def test_tool_result_list_content(self):
        t = Turn(kind="user")
        t.append({"message": {"content": [
            {
                "type": "tool_result",
                "content": [{"type": "text", "text": "nested text"}],
            }
        ]}})
        assert "nested text" in extract_text(t)

    def test_non_dict_block_skipped(self):
        t = Turn(kind="system")
        t.append({"message": {"content": ["a string block"]}})
        # Should not crash, returns empty or whatever
        result = extract_text(t)
        assert isinstance(result, str)

    def test_multiple_blocks_concatenated(self):
        t = Turn(kind="system")
        t.append({"message": {"content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]}})
        result = extract_text(t)
        assert "first" in result
        assert "second" in result


# ---------------------------------------------------------------------------
# parse_jsonl
# ---------------------------------------------------------------------------

class TestParseJsonl:
    def _write_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        p = tmp_path / "conv.jsonl"
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return p

    def test_empty_file_returns_empty(self, tmp_path):
        p = self._write_jsonl(tmp_path, [])
        turns = parse_jsonl(p)
        assert turns == []

    def test_single_user_message(self, tmp_path):
        records = [
            {"type": "user", "message": {"content": "hello"}},
        ]
        p = self._write_jsonl(tmp_path, records)
        turns = parse_jsonl(p)
        assert any(t.kind == "user" for t in turns)

    def test_user_then_assistant(self, tmp_path):
        records = [
            {"type": "user", "message": {"content": "question"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}},
        ]
        p = self._write_jsonl(tmp_path, records)
        turns = parse_jsonl(p)
        kinds = [t.kind for t in turns]
        assert "user" in kinds
        assert "system" in kinds

    def test_turns_indexed_sequentially(self, tmp_path):
        records = [
            {"type": "user", "message": {"content": "msg1"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp1"}]}},
            {"type": "user", "message": {"content": "msg2"}},
        ]
        p = self._write_jsonl(tmp_path, records)
        turns = parse_jsonl(p)
        indices = [t.index for t in turns]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # all unique
