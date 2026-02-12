#!/usr/bin/env python3
"""Tests for the Codex JSONL parser.

Validates that codex_parser correctly handles the real Codex rollout format
(payload-wrapped records) and produces Turn objects compatible with
supercompact's scoring pipeline.

Test fixtures match the actual format from codex-cli-rs 0.89+.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sys
_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from codex_parser import parse_codex_jsonl, extract_codex_text


def _write_jsonl(lines: list[dict]) -> Path:
    """Write a list of dicts as JSONL to a temp file, return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for line in lines:
        tmp.write(json.dumps(line) + "\n")
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Sample Codex rollout entries (real payload-wrapped format)
# ---------------------------------------------------------------------------

SAMPLE_SESSION_META = {
    "timestamp": "2025-07-01T10:00:00Z",
    "type": "session_meta",
    "payload": {
        "id": "test-session-001",
        "timestamp": "2025-07-01T10:00:00Z",
        "cwd": "/home/user/project",
        "originator": "codex_cli_rs",
        "cli_version": "0.89.0",
        "source": "cli",
        "model_provider": "openai",
    },
}

SAMPLE_TURN_CONTEXT = {
    "timestamp": "2025-07-01T10:00:01Z",
    "type": "turn_context",
    "payload": {
        "cwd": "/home/user/project",
        "approval_policy": "never",
        "sandbox_policy": {"type": "danger-full-access"},
        "model": "o3-mini",
        "user_instructions": "Fix the login bug in auth.py",
        "summary": "auto",
    },
}

SAMPLE_USER_MESSAGE = {
    "timestamp": "2025-07-01T10:00:02Z",
    "type": "response_item",
    "payload": {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Fix the login bug in auth.py"}],
    },
}

SAMPLE_DEVELOPER_MESSAGE = {
    "timestamp": "2025-07-01T10:00:02Z",
    "type": "response_item",
    "payload": {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are a coding agent..."}],
    },
}

SAMPLE_ASSISTANT_MESSAGE = {
    "timestamp": "2025-07-01T10:00:03Z",
    "type": "response_item",
    "payload": {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I'll look at auth.py to find the login bug."}],
    },
}

SAMPLE_FUNCTION_CALL = {
    "timestamp": "2025-07-01T10:00:04Z",
    "type": "response_item",
    "payload": {
        "type": "function_call",
        "name": "exec_command",
        "arguments": '{"cmd": "cat auth.py"}',
        "call_id": "toolcall-abc123",
    },
}

SAMPLE_FUNCTION_OUTPUT = {
    "timestamp": "2025-07-01T10:00:05Z",
    "type": "response_item",
    "payload": {
        "type": "function_call_output",
        "call_id": "toolcall-abc123",
        "output": "def login(user, password):\n    if user == 'admin':\n        return True  # BUG: doesn't check password\n    return False",
    },
}

SAMPLE_REASONING = {
    "timestamp": "2025-07-01T10:00:05Z",
    "type": "response_item",
    "payload": {
        "type": "reasoning",
        "summary": [],
        "content": [
            {
                "type": "reasoning_text",
                "text": "The login function has a bug where admin bypasses password check.",
            }
        ],
    },
}

SAMPLE_ASSISTANT_RESPONSE = {
    "timestamp": "2025-07-01T10:00:06Z",
    "type": "response_item",
    "payload": {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Found the bug! The login function doesn't check the password for admin users.",
            }
        ],
    },
}

SAMPLE_COMPACTED = {
    "timestamp": "2025-07-01T10:05:00Z",
    "type": "compacted",
    "payload": {
        "message": "The user asked to fix a login bug in auth.py. The admin login was bypassing password checks.",
    },
}

SAMPLE_EVENT_MSG = {
    "timestamp": "2025-07-01T10:00:07Z",
    "type": "event_msg",
    "payload": {
        "type": "user_message",
        "message": "Fix the login bug",
        "images": [],
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_parsing():
    """Test that a basic Codex session is parsed into user/system turns."""
    path = _write_jsonl([
        SAMPLE_SESSION_META,
        SAMPLE_TURN_CONTEXT,
        SAMPLE_ASSISTANT_MESSAGE,
        SAMPLE_FUNCTION_CALL,
        SAMPLE_FUNCTION_OUTPUT,
        SAMPLE_ASSISTANT_RESPONSE,
    ])

    turns = parse_codex_jsonl(path)
    assert len(turns) >= 2, f"Expected at least 2 turns, got {len(turns)}"

    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]

    assert len(user_turns) >= 1, f"Expected at least 1 user turn, got {len(user_turns)}"
    assert len(system_turns) >= 1, f"Expected at least 1 system turn, got {len(system_turns)}"

    print(f"  Basic parsing: {len(turns)} turns ({len(user_turns)} user, {len(system_turns)} system)")


def test_skipped_types():
    """Test that session_meta and event_msg records are skipped."""
    path = _write_jsonl([
        SAMPLE_SESSION_META,
        SAMPLE_EVENT_MSG,
        SAMPLE_TURN_CONTEXT,
        SAMPLE_EVENT_MSG,
        SAMPLE_ASSISTANT_MESSAGE,
    ])

    turns = parse_codex_jsonl(path)

    for turn in turns:
        for record in turn.lines:
            assert record.get("type") not in ("session_meta", "event_msg"), \
                f"Unexpected record type in turn: {record.get('type')}"

    print("  Skipped types: session_meta and event_msg correctly filtered")


def test_developer_messages_not_user():
    """Test that developer messages are not treated as user messages."""
    path = _write_jsonl([
        SAMPLE_SESSION_META,
        SAMPLE_DEVELOPER_MESSAGE,
        SAMPLE_TURN_CONTEXT,
        SAMPLE_ASSISTANT_MESSAGE,
    ])

    turns = parse_codex_jsonl(path)
    user_turns = [t for t in turns if t.kind == "user"]

    # Developer messages should NOT create user turns
    for ut in user_turns:
        for record in ut.lines:
            if record.get("type") == "response_item":
                payload = record.get("payload", record)
                assert payload.get("role") != "developer", \
                    "Developer message incorrectly classified as user turn"

    print("  Developer messages: correctly excluded from user turns")


def test_text_extraction_payload_format():
    """Test text extraction from real payload-wrapped Codex records."""
    Turn = type(parse_codex_jsonl(_write_jsonl([SAMPLE_TURN_CONTEXT]))[0])

    # Assistant message (payload.content[].text)
    turn = Turn(kind="system")
    turn.append(SAMPLE_ASSISTANT_MESSAGE)
    text = extract_codex_text(turn)
    assert "login bug" in text.lower(), f"Expected 'login bug' in: {text}"

    # Function call (payload.name + payload.arguments)
    turn = Turn(kind="system")
    turn.append(SAMPLE_FUNCTION_CALL)
    text = extract_codex_text(turn)
    assert "exec_command" in text, f"Expected 'exec_command' in: {text}"
    assert "cat auth.py" in text, f"Expected 'cat auth.py' in: {text}"

    # Function output (payload.output)
    turn = Turn(kind="system")
    turn.append(SAMPLE_FUNCTION_OUTPUT)
    text = extract_codex_text(turn)
    assert "login" in text.lower(), f"Expected 'login' in function output: {text}"
    assert "admin" in text.lower(), f"Expected 'admin' in function output: {text}"

    # Reasoning (payload.content[].reasoning_text)
    turn = Turn(kind="system")
    turn.append(SAMPLE_REASONING)
    text = extract_codex_text(turn)
    assert "bypass" in text.lower() or "bug" in text.lower(), \
        f"Expected reasoning text in: {text}"

    # Compacted (payload.message)
    turn = Turn(kind="system")
    turn.append(SAMPLE_COMPACTED)
    text = extract_codex_text(turn)
    assert "login bug" in text.lower() or "auth.py" in text.lower(), \
        f"Expected compaction summary in: {text}"

    # Turn context (payload.user_instructions)
    turn = Turn(kind="user")
    turn.append(SAMPLE_TURN_CONTEXT)
    text = extract_codex_text(turn)
    assert "login bug" in text.lower() or "auth.py" in text.lower(), \
        f"Expected user instructions in: {text}"

    # User message (payload.content[].input_text)
    turn = Turn(kind="user")
    turn.append(SAMPLE_USER_MESSAGE)
    text = extract_codex_text(turn)
    assert "login bug" in text.lower() or "auth.py" in text.lower(), \
        f"Expected user message text in: {text}"

    print("  Text extraction: all payload-wrapped record types handled correctly")


def test_compacted_handling():
    """Test that compacted entries are handled as system turns."""
    path = _write_jsonl([
        SAMPLE_SESSION_META,
        SAMPLE_COMPACTED,
        SAMPLE_TURN_CONTEXT,
        SAMPLE_ASSISTANT_MESSAGE,
    ])

    turns = parse_codex_jsonl(path)

    compacted_turns = [
        t for t in turns
        if t.kind == "system" and any(r.get("type") == "compacted" for r in t.lines)
    ]
    assert len(compacted_turns) == 1, f"Expected 1 compacted turn, got {len(compacted_turns)}"

    # Verify text extraction from compacted
    text = extract_codex_text(compacted_turns[0])
    assert "login bug" in text.lower() or "auth.py" in text.lower(), \
        f"Expected compacted text, got: {text}"

    print("  Compacted handling: compacted entry correctly parsed as system turn")


def test_multi_turn_conversation():
    """Test parsing a multi-turn conversation."""
    path = _write_jsonl([
        SAMPLE_SESSION_META,
        # Turn 1
        SAMPLE_TURN_CONTEXT,
        SAMPLE_ASSISTANT_MESSAGE,
        SAMPLE_FUNCTION_CALL,
        SAMPLE_FUNCTION_OUTPUT,
        SAMPLE_ASSISTANT_RESPONSE,
        # Turn 2
        {
            "timestamp": "2025-07-01T10:01:00Z",
            "type": "turn_context",
            "payload": {
                "cwd": "/home/user/project",
                "model": "o3-mini",
                "user_instructions": "Now add unit tests",
                "summary": "auto",
            },
        },
        {
            "timestamp": "2025-07-01T10:01:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I'll create unit tests for the login function."}],
            },
        },
    ])

    turns = parse_codex_jsonl(path)
    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]

    assert len(user_turns) >= 2, f"Expected at least 2 user turns, got {len(user_turns)}"
    assert len(system_turns) >= 2, f"Expected at least 2 system turns, got {len(system_turns)}"

    # Verify user turn after system turn
    for i in range(1, len(turns)):
        if turns[i-1].kind == "user":
            assert turns[i].kind == "system", \
                f"Turn {i}: expected system after user, got {turns[i].kind}"

    print(f"  Multi-turn: {len(turns)} turns parsed correctly")


def test_sequential_indices():
    """Test that turns are sequentially indexed."""
    path = _write_jsonl([
        SAMPLE_SESSION_META,
        SAMPLE_TURN_CONTEXT,
        SAMPLE_ASSISTANT_MESSAGE,
        SAMPLE_FUNCTION_CALL,
        SAMPLE_FUNCTION_OUTPUT,
    ])

    turns = parse_codex_jsonl(path)
    for i, turn in enumerate(turns):
        assert turn.index == i, f"Turn {i} has index {turn.index}"

    print("  Sequential indices: all turns correctly indexed")


def test_empty_file():
    """Test handling of empty/minimal JSONL files."""
    path = _write_jsonl([])
    turns = parse_codex_jsonl(path)
    assert len(turns) == 0, f"Expected 0 turns for empty file, got {len(turns)}"

    path = _write_jsonl([SAMPLE_SESSION_META])
    turns = parse_codex_jsonl(path)
    assert len(turns) == 0, f"Expected 0 turns for meta-only file, got {len(turns)}"

    print("  Empty file: handled correctly")


def test_malformed_json():
    """Test handling of malformed JSON lines."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.write(json.dumps(SAMPLE_SESSION_META) + "\n")
    tmp.write("not valid json\n")
    tmp.write(json.dumps(SAMPLE_TURN_CONTEXT) + "\n")
    tmp.write("{incomplete\n")
    tmp.write(json.dumps(SAMPLE_ASSISTANT_MESSAGE) + "\n")
    tmp.flush()

    turns = parse_codex_jsonl(Path(tmp.name))
    assert len(turns) >= 1, f"Expected at least 1 turn, got {len(turns)}"

    print("  Malformed JSON: invalid lines skipped gracefully")


def test_real_session_file():
    """Test against real Codex session files if available."""
    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.exists():
        print("  Real session: SKIPPED (no ~/.codex/sessions/)")
        return

    # Find the largest session file
    jsonl_files = sorted(
        sessions_dir.rglob("rollout-*.jsonl"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not jsonl_files:
        print("  Real session: SKIPPED (no rollout files)")
        return

    path = jsonl_files[0]
    turns = parse_codex_jsonl(path)

    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]

    assert len(turns) > 0, f"Expected turns from real session, got 0"

    # Verify text extraction produces non-empty strings
    non_empty = 0
    total = 0
    for turn in turns:
        text = extract_codex_text(turn)
        total += 1
        if text.strip():
            non_empty += 1

    extraction_rate = non_empty / total if total > 0 else 0
    assert extraction_rate > 0.5, \
        f"Text extraction rate too low: {non_empty}/{total} ({extraction_rate:.0%})"

    print(f"  Real session: {path.name}")
    print(f"    {len(turns)} turns ({len(user_turns)} user, {len(system_turns)} system)")
    print(f"    Text extraction: {non_empty}/{total} non-empty ({extraction_rate:.0%})")


if __name__ == "__main__":
    print("Testing Codex JSONL parser...\n")

    tests = [
        test_basic_parsing,
        test_skipped_types,
        test_developer_messages_not_user,
        test_text_extraction_payload_format,
        test_compacted_handling,
        test_multi_turn_conversation,
        test_sequential_indices,
        test_empty_file,
        test_malformed_json,
        test_real_session_file,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")

    sys.exit(1 if failed > 0 else 0)
