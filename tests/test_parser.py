"""Tests for lib.parser — Turn, _is_user_message, parse_jsonl, extract_text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn, _is_user_message, parse_jsonl, extract_text, SKIP_TYPES


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

class TestTurn:
    def test_defaults(self):
        t = Turn(kind="user")
        assert t.kind == "user"
        assert t.lines == []
        assert t.index == 0

    def test_append_adds_record(self):
        t = Turn(kind="system")
        t.append({"type": "assistant"})
        assert len(t.lines) == 1
        assert t.lines[0]["type"] == "assistant"

    def test_append_multiple(self):
        t = Turn(kind="system")
        t.append({"a": 1})
        t.append({"b": 2})
        assert len(t.lines) == 2

    def test_index_set_on_creation(self):
        t = Turn(kind="user", index=5)
        assert t.index == 5


# ---------------------------------------------------------------------------
# _is_user_message
# ---------------------------------------------------------------------------

class TestIsUserMessage:
    def test_string_content_user(self):
        record = {"type": "user", "message": {"content": "hello"}}
        assert _is_user_message(record) is True

    def test_non_user_type_false(self):
        record = {"type": "assistant", "message": {"content": "hello"}}
        assert _is_user_message(record) is False

    def test_tool_result_uuid_excluded(self):
        record = {
            "type": "user",
            "sourceToolAssistantUUID": "some-uuid",
            "message": {"content": "hello"},
        }
        assert _is_user_message(record) is False

    def test_list_content_with_text_block_is_user(self):
        record = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hi"}]},
        }
        assert _is_user_message(record) is True

    def test_list_content_with_tool_result_is_not_user(self):
        record = {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": "..."}]},
        }
        assert _is_user_message(record) is False

    def test_missing_content_is_not_user(self):
        record = {"type": "user", "message": {}}
        assert _is_user_message(record) is False

    def test_none_content_is_not_user(self):
        record = {"type": "user", "message": {"content": None}}
        assert _is_user_message(record) is False

    def test_skip_types_not_user(self):
        for skip in SKIP_TYPES:
            record = {"type": skip, "message": {"content": "x"}}
            assert _is_user_message(record) is False


# ---------------------------------------------------------------------------
# parse_jsonl
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestParseJsonl:
    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        p.write_text("")
        result = parse_jsonl(p)
        assert result == []

    def test_single_user_message(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        _write_jsonl(p, [{"type": "user", "message": {"content": "hello"}}])
        turns = parse_jsonl(p)
        assert len(turns) == 1
        assert turns[0].kind == "user"

    def test_user_then_assistant(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        _write_jsonl(p, [
            {"type": "user", "message": {"content": "hello"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}},
        ])
        turns = parse_jsonl(p)
        # user turn + system turn
        assert len(turns) == 2
        assert turns[0].kind == "user"
        assert turns[1].kind == "system"

    def test_skip_types_excluded(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        records = [{"type": t} for t in SKIP_TYPES]
        records.append({"type": "user", "message": {"content": "real msg"}})
        _write_jsonl(p, records)
        turns = parse_jsonl(p)
        assert len(turns) == 1
        assert turns[0].kind == "user"

    def test_indices_are_sequential(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        _write_jsonl(p, [
            {"type": "user", "message": {"content": "q1"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a1"}]}},
            {"type": "user", "message": {"content": "q2"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a2"}]}},
        ])
        turns = parse_jsonl(p)
        for i, t in enumerate(turns):
            assert t.index == i

    def test_invalid_json_lines_skipped(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        with open(p, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
        turns = parse_jsonl(p)
        assert len(turns) == 1

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        with open(p, "w") as f:
            f.write("\n\n")
            f.write(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
        turns = parse_jsonl(p)
        assert len(turns) == 1

    def test_tool_result_grouped_with_system(self, tmp_path):
        """Tool results (type=user with tool_result block) should land in system turn."""
        p = tmp_path / "conv.jsonl"
        _write_jsonl(p, [
            {"type": "user", "message": {"content": "do it"}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "bash", "id": "x", "input": {}}]}},
            {"type": "user", "sourceToolAssistantUUID": "some-uuid", "message": {"content": [{"type": "tool_result", "content": "output"}]}},
        ])
        turns = parse_jsonl(p)
        # Should be: user turn, system turn (assistant + tool result)
        assert turns[0].kind == "user"
        assert turns[1].kind == "system"
        assert len(turns[1].lines) == 2  # assistant + tool result

    def test_trailing_system_turn_kept(self, tmp_path):
        """System lines after the last user message are not dropped."""
        p = tmp_path / "conv.jsonl"
        _write_jsonl(p, [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}},
        ])
        turns = parse_jsonl(p)
        assert len(turns) == 1
        assert turns[0].kind == "system"


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def _turn_with(self, content) -> Turn:
        t = Turn(kind="user")
        t.lines = [{"message": {"content": content}}]
        return t

    def test_string_content_extracted(self):
        t = self._turn_with("hello world")
        assert extract_text(t) == "hello world"

    def test_text_block_extracted(self):
        t = self._turn_with([{"type": "text", "text": "block text"}])
        assert "block text" in extract_text(t)

    def test_thinking_block_extracted(self):
        t = self._turn_with([{"type": "thinking", "thinking": "reasoning here"}])
        assert "reasoning here" in extract_text(t)

    def test_tool_use_name_included(self):
        t = self._turn_with([{"type": "tool_use", "name": "bash", "id": "x", "input": {"command": "ls"}}])
        text = extract_text(t)
        assert "bash" in text
        assert "ls" in text

    def test_tool_result_content_extracted(self):
        t = self._turn_with([{"type": "tool_result", "content": "tool output"}])
        assert "tool output" in extract_text(t)

    def test_tool_result_list_content_extracted(self):
        t = self._turn_with([{
            "type": "tool_result",
            "content": [{"type": "text", "text": "nested output"}],
        }])
        assert "nested output" in extract_text(t)

    def test_empty_turn_returns_empty(self):
        t = Turn(kind="user")
        assert extract_text(t) == ""

    def test_multiple_lines_joined(self):
        t = Turn(kind="system")
        t.lines = [
            {"message": {"content": "first"}},
            {"message": {"content": "second"}},
        ]
        text = extract_text(t)
        assert "first" in text
        assert "second" in text

    def test_long_tool_input_truncated(self):
        long_value = "x" * 2000
        t = self._turn_with([{"type": "tool_use", "name": "write", "id": "y", "input": {"content": long_value}}])
        text = extract_text(t)
        # Truncated at 500 chars + "..."
        assert len(text) < 2000
        assert "..." in text
