"""Tests for lib/parser.py — JSONL parsing, turn grouping, text extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn, _is_user_message, extract_text, parse_jsonl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _user_record(text: str, *, uuid: str | None = None) -> dict:
    record = {"type": "user", "message": {"content": text}}
    if uuid:
        record["sourceToolAssistantUUID"] = uuid
    return record


def _user_record_list(blocks: list[dict], *, uuid: str | None = None) -> dict:
    record = {"type": "user", "message": {"content": blocks}}
    if uuid:
        record["sourceToolAssistantUUID"] = uuid
    return record


def _assistant_record(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _tool_use_record(name: str, inp: dict) -> dict:
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": name, "input": inp}]
        },
    }


def _tool_result_record(content: str) -> dict:
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "content": content}]
        },
    }


def _write_jsonl(records: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# _is_user_message
# ---------------------------------------------------------------------------

class TestIsUserMessage:
    def test_string_content_is_user(self):
        assert _is_user_message(_user_record("hello"))

    def test_list_with_text_block_is_user(self):
        record = _user_record_list([{"type": "text", "text": "hi"}])
        assert _is_user_message(record)

    def test_wrong_type_not_user(self):
        assert not _is_user_message({"type": "assistant", "message": {"content": "hi"}})

    def test_progress_type_not_user(self):
        assert not _is_user_message({"type": "progress"})

    def test_tool_result_block_not_user(self):
        record = _user_record_list([{"type": "tool_result", "content": "output"}])
        assert not _is_user_message(record)

    def test_source_tool_uuid_not_user(self):
        record = _user_record("hello", uuid="some-uuid")
        assert not _is_user_message(record)

    def test_empty_content_list_not_user(self):
        record = _user_record_list([])
        # Empty list has no tool_result, so it is a user message
        assert _is_user_message(record)

    def test_mixed_blocks_with_tool_result_not_user(self):
        record = _user_record_list([
            {"type": "text", "text": "result"},
            {"type": "tool_result", "content": "output"},
        ])
        assert not _is_user_message(record)


# ---------------------------------------------------------------------------
# parse_jsonl
# ---------------------------------------------------------------------------

class TestParseJsonl:
    def test_empty_file_returns_empty(self):
        path = _write_jsonl([])
        assert parse_jsonl(path) == []

    def test_single_user_message(self):
        path = _write_jsonl([_user_record("hello")])
        turns = parse_jsonl(path)
        assert len(turns) == 1
        assert turns[0].kind == "user"

    def test_user_then_assistant(self):
        path = _write_jsonl([
            _user_record("question"),
            _assistant_record("answer"),
        ])
        turns = parse_jsonl(path)
        assert len(turns) == 2
        assert turns[0].kind == "user"
        assert turns[1].kind == "system"

    def test_multiple_exchanges(self):
        path = _write_jsonl([
            _user_record("q1"),
            _assistant_record("a1"),
            _user_record("q2"),
            _assistant_record("a2"),
        ])
        turns = parse_jsonl(path)
        assert len(turns) == 4
        kinds = [t.kind for t in turns]
        assert kinds == ["user", "system", "user", "system"]

    def test_turns_are_reindexed_sequentially(self):
        path = _write_jsonl([
            _user_record("q1"),
            _assistant_record("a1"),
            _user_record("q2"),
        ])
        turns = parse_jsonl(path)
        assert [t.index for t in turns] == [0, 1, 2]

    def test_skipped_types_are_dropped(self):
        path = _write_jsonl([
            {"type": "progress"},
            {"type": "system"},
            {"type": "summary"},
            _user_record("hello"),
        ])
        turns = parse_jsonl(path)
        assert len(turns) == 1
        assert turns[0].kind == "user"

    def test_tool_result_grouped_into_system_turn(self):
        path = _write_jsonl([
            _user_record("do something"),
            _tool_use_record("Read", {"path": "/foo.py"}),
            _tool_result_record("file contents"),
        ])
        turns = parse_jsonl(path)
        assert len(turns) == 2
        assert turns[1].kind == "system"
        assert len(turns[1].lines) == 2

    def test_trailing_system_turn_included(self):
        """System turn at the end of the file (no following user message) is kept."""
        path = _write_jsonl([
            _user_record("start"),
            _assistant_record("working..."),
            _assistant_record("done"),
        ])
        turns = parse_jsonl(path)
        assert turns[-1].kind == "system"
        assert len(turns[-1].lines) == 2

    def test_malformed_json_lines_skipped(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write(json.dumps(_user_record("hello")) + "\n")
        tmp.write("not json at all\n")
        tmp.write(json.dumps(_assistant_record("reply")) + "\n")
        tmp.flush()
        turns = parse_jsonl(Path(tmp.name))
        assert len(turns) == 2

    def test_blank_lines_skipped(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write("\n")
        tmp.write(json.dumps(_user_record("hello")) + "\n")
        tmp.write("\n")
        tmp.flush()
        turns = parse_jsonl(Path(tmp.name))
        assert len(turns) == 1


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def _turn_from(self, record: dict) -> Turn:
        t = Turn(kind="user")
        t.append(record)
        return t

    def test_string_content_extracted(self):
        turn = self._turn_from({"type": "user", "message": {"content": "hello world"}})
        assert extract_text(turn) == "hello world"

    def test_text_block_extracted(self):
        record = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "response text"}
        ]}}
        turn = self._turn_from(record)
        assert "response text" in extract_text(turn)

    def test_thinking_block_extracted(self):
        record = {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "internal thought"}
        ]}}
        turn = self._turn_from(record)
        assert "internal thought" in extract_text(turn)

    def test_tool_use_name_and_input_extracted(self):
        record = _tool_use_record("Read", {"path": "/src/main.py"})
        turn = self._turn_from(record)
        text = extract_text(turn)
        assert "Read" in text
        assert "/src/main.py" in text

    def test_tool_result_string_content_extracted(self):
        record = {
            "type": "user",
            "message": {"content": [
                {"type": "tool_result", "content": "file output here"}
            ]}
        }
        turn = self._turn_from(record)
        assert "file output here" in extract_text(turn)

    def test_tool_result_list_content_extracted(self):
        record = {
            "type": "user",
            "message": {"content": [
                {"type": "tool_result", "content": [
                    {"type": "text", "text": "nested text"},
                ]}
            ]}
        }
        turn = self._turn_from(record)
        assert "nested text" in extract_text(turn)

    def test_multiple_records_concatenated(self):
        t = Turn(kind="system")
        t.append({"type": "assistant", "message": {"content": "part one"}})
        t.append({"type": "assistant", "message": {"content": "part two"}})
        text = extract_text(t)
        assert "part one" in text
        assert "part two" in text

    def test_long_tool_input_truncated(self):
        long_val = "x" * 2000
        record = _tool_use_record("Write", {"content": long_val})
        turn = self._turn_from(record)
        text = extract_text(turn)
        # Should be truncated, not pass full 2000 chars through
        assert len(text) < 2000

    def test_empty_turn_returns_empty_string(self):
        turn = Turn(kind="user")
        assert extract_text(turn) == ""
