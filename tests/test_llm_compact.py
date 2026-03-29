"""Tests for lib/llm_compact.py — make_synthetic_turn and llm_compact guards."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.llm_compact import (
    COMPACT_SYSTEM_PROMPT,
    MODEL,
    make_synthetic_turn,
)
from lib.parser import Turn, extract_text


# ---------------------------------------------------------------------------
# make_synthetic_turn
# ---------------------------------------------------------------------------

class TestMakeSyntheticTurn:
    def test_returns_turn_instance(self):
        t = make_synthetic_turn("hello")
        assert isinstance(t, Turn)

    def test_kind_is_system(self):
        t = make_synthetic_turn("some summary")
        assert t.kind == "system"

    def test_default_index_zero(self):
        t = make_synthetic_turn("summary text")
        assert t.index == 0

    def test_custom_index_stored(self):
        t = make_synthetic_turn("summary", index=42)
        assert t.index == 42

    def test_has_one_line(self):
        t = make_synthetic_turn("content")
        assert len(t.lines) == 1

    def test_line_type_is_assistant(self):
        t = make_synthetic_turn("content")
        assert t.lines[0]["type"] == "assistant"

    def test_message_role_is_assistant(self):
        t = make_synthetic_turn("content")
        assert t.lines[0]["message"]["role"] == "assistant"

    def test_summary_text_in_message_content(self):
        text = "Error: FileNotFoundError at /home/user/project/main.py"
        t = make_synthetic_turn(text)
        assert t.lines[0]["message"]["content"] == text

    def test_extract_text_returns_summary(self):
        text = "ValueError in /src/module.py on line 42"
        t = make_synthetic_turn(text)
        assert extract_text(t) == text

    def test_empty_summary_produces_valid_turn(self):
        t = make_synthetic_turn("")
        assert t.kind == "system"
        assert extract_text(t) == ""

    def test_multiline_summary_preserved(self):
        text = "line1\nline2\nline3"
        t = make_synthetic_turn(text)
        assert extract_text(t) == text

    def test_summary_with_special_chars(self):
        text = "Token: abc_123-456 / path: /usr/local/lib.so"
        t = make_synthetic_turn(text)
        assert extract_text(t) == text

    def test_different_indices_independent(self):
        t1 = make_synthetic_turn("a", index=0)
        t2 = make_synthetic_turn("b", index=99)
        assert t1.index == 0
        assert t2.index == 99
        assert t1.lines[0]["message"]["content"] == "a"
        assert t2.lines[0]["message"]["content"] == "b"


# ---------------------------------------------------------------------------
# llm_compact — missing API key guard
# ---------------------------------------------------------------------------

class TestLlmCompactApiKeyGuard:
    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from lib.llm_compact import llm_compact
        from lib.parser import Turn as T

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            llm_compact([], budget=1000)

    def test_missing_api_key_message_contains_var_name(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from lib.llm_compact import llm_compact

        with pytest.raises(RuntimeError) as exc_info:
            llm_compact([], budget=500)
        assert "OPENROUTER_API_KEY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_compact_system_prompt_is_str(self):
        assert isinstance(COMPACT_SYSTEM_PROMPT, str)

    def test_compact_system_prompt_mentions_file_paths(self):
        assert "file path" in COMPACT_SYSTEM_PROMPT.lower() or "path" in COMPACT_SYSTEM_PROMPT

    def test_compact_system_prompt_nonempty(self):
        assert len(COMPACT_SYSTEM_PROMPT) > 0

    def test_model_is_str(self):
        assert isinstance(MODEL, str)

    def test_model_nonempty(self):
        assert len(MODEL) > 0
