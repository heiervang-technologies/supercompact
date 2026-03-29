"""Tests for lib/llm_compact.py — LLM summarization baseline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.llm_compact import llm_compact, make_synthetic_turn, COMPACT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(index: int, text: str = "") -> Turn:
    t = Turn(kind="system", index=index)
    if text:
        t.lines = [{"type": "assistant", "message": {"content": text}}]
    return t


def _user(index: int, text: str = "") -> Turn:
    t = Turn(kind="user", index=index)
    if text:
        t.lines = [{"type": "user", "message": {"content": text}}]
    return t


def _fake_response(content: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    resp.text = json.dumps({"error": "test error"})
    return resp


# ---------------------------------------------------------------------------
# make_synthetic_turn
# ---------------------------------------------------------------------------

class TestMakeSyntheticTurn:
    def test_returns_turn(self):
        t = make_synthetic_turn("some summary")
        assert isinstance(t, Turn)

    def test_kind_is_system(self):
        t = make_synthetic_turn("summary text")
        assert t.kind == "system"

    def test_default_index_zero(self):
        t = make_synthetic_turn("text")
        assert t.index == 0

    def test_custom_index(self):
        t = make_synthetic_turn("text", index=42)
        assert t.index == 42

    def test_text_in_content(self):
        t = make_synthetic_turn("ValueError at /src/app.py")
        content = t.lines[0]["message"]["content"]
        assert "ValueError at /src/app.py" in content

    def test_has_one_line(self):
        t = make_synthetic_turn("text")
        assert len(t.lines) == 1

    def test_line_is_assistant_type(self):
        t = make_synthetic_turn("text")
        assert t.lines[0]["type"] == "assistant"

    def test_empty_summary(self):
        t = make_synthetic_turn("")
        assert isinstance(t, Turn)
        assert t.lines[0]["message"]["content"] == ""

    def test_role_is_assistant(self):
        t = make_synthetic_turn("text")
        assert t.lines[0]["message"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# llm_compact
# ---------------------------------------------------------------------------

class TestLlmCompact:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                llm_compact([_system(1, "hello")], budget=1000)

    def test_calls_openrouter_endpoint(self):
        fake_resp = _fake_response("summary text")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp) as mock_post:
                llm_compact([_system(1, "hello")], budget=1000)
        assert mock_post.called
        url = mock_post.call_args[0][0]
        assert "openrouter.ai" in url

    def test_returns_summary_string(self):
        fake_resp = _fake_response("This is the summary.")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp):
                result = llm_compact([_system(1, "hello")], budget=1000)
        assert result == "This is the summary."

    def test_system_prompt_sent(self):
        fake_resp = _fake_response("summary")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp) as mock_post:
                llm_compact([_system(1, "test")], budget=1000)
        payload = mock_post.call_args.kwargs["json"]
        messages = payload["messages"]
        assert any(m["role"] == "system" for m in messages)
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert "file path" in system_content.lower() or "preserve" in system_content.lower()

    def test_budget_in_user_message(self):
        fake_resp = _fake_response("summary")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp) as mock_post:
                llm_compact([_system(1, "test")], budget=2048)
        payload = mock_post.call_args.kwargs["json"]
        user_msg = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        assert "2,048" in user_msg or "2048" in user_msg

    def test_api_error_raises_runtime_error(self):
        bad_resp = _fake_response("error body", status_code=401)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=bad_resp):
                with pytest.raises(RuntimeError, match="401"):
                    llm_compact([_system(1, "test")], budget=1000)

    def test_temperature_zero(self):
        fake_resp = _fake_response("summary")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp) as mock_post:
                llm_compact([_system(1, "test")], budget=1000)
        payload = mock_post.call_args.kwargs["json"]
        assert payload.get("temperature") == 0.0

    def test_long_conversation_truncated(self):
        """Conversations > 600k chars should be truncated before the API call."""
        # Create a turn with 700k characters
        long_text = "x" * 700_000
        turns = [_system(1, long_text)]
        fake_resp = _fake_response("summary")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp) as mock_post:
                llm_compact(turns, budget=1000)
        payload = mock_post.call_args.kwargs["json"]
        user_msg = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        # The conversation part should be truncated to ≤ 600k + overhead
        assert len(user_msg) < 700_000 + 5000

    def test_multiple_turns_included(self):
        """All prefix turns should appear in the user message."""
        turns = [_user(0, "question"), _system(1, "answer"), _user(2, "followup")]
        fake_resp = _fake_response("summary")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp) as mock_post:
                llm_compact(turns, budget=1000)
        payload = mock_post.call_args.kwargs["json"]
        user_msg = next(m["content"] for m in payload["messages"] if m["role"] == "user")
        assert "question" in user_msg
        assert "answer" in user_msg
        assert "followup" in user_msg

    def test_empty_turns_list(self):
        """Empty turns list should still call the API without crashing."""
        fake_resp = _fake_response("empty summary")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("httpx.post", return_value=fake_resp):
                result = llm_compact([], budget=1000)
        assert result == "empty summary"


# ---------------------------------------------------------------------------
# COMPACT_SYSTEM_PROMPT content
# ---------------------------------------------------------------------------

class TestCompactSystemPrompt:
    def test_mentions_file_paths(self):
        assert "file path" in COMPACT_SYSTEM_PROMPT.lower()

    def test_mentions_error_messages(self):
        assert "error" in COMPACT_SYSTEM_PROMPT.lower()

    def test_mentions_function_names(self):
        assert "function" in COMPACT_SYSTEM_PROMPT.lower()

    def test_not_empty(self):
        assert len(COMPACT_SYSTEM_PROMPT) > 100
