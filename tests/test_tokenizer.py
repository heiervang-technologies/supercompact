"""Tests for lib/tokenizer.py — estimate_tokens and turn_tokens.

These tests load the real Qwen3 tokenizer on first use (cached globally).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.tokenizer import estimate_tokens, turn_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(kind: str, index: int, text: str) -> Turn:
    t = Turn(kind=kind, index=index)
    t.append({"message": {"content": text}})
    return t


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_nonempty_string_returns_positive(self):
        count = estimate_tokens("hello world")
        assert count > 0

    def test_returns_int(self):
        result = estimate_tokens("some text")
        assert isinstance(result, int)

    def test_longer_text_more_tokens(self):
        short = "hello"
        long = "hello " * 100
        assert estimate_tokens(long) > estimate_tokens(short)

    def test_repeated_text_scales(self):
        base = "ValueError at /home/user/project/main.py"
        single = estimate_tokens(base)
        double = estimate_tokens(base * 2)
        # Rough proportionality — at least 1.5x tokens
        assert double >= single * 1.5

    def test_whitespace_only_may_have_tokens(self):
        # Don't assume whitespace-only has exactly 0 tokens
        result = estimate_tokens("   ")
        assert isinstance(result, int)
        assert result >= 0

    def test_add_special_tokens_false(self):
        # encode(..., add_special_tokens=False) — no BOS/EOS padding
        # A single word should encode to a small number of tokens
        result = estimate_tokens("hello")
        assert result < 10  # Should be 1-3 tokens at most


# ---------------------------------------------------------------------------
# turn_tokens
# ---------------------------------------------------------------------------

class TestTurnTokens:
    def test_returns_int(self):
        t = _turn("system", 0, "some content")
        assert isinstance(turn_tokens(t), int)

    def test_returns_positive_for_nonempty_turn(self):
        t = _turn("system", 0, "ValueError at /home/user/main.py")
        assert turn_tokens(t) > 0

    def test_empty_turn_returns_zero(self):
        t = Turn(kind="system", index=0)  # no lines
        assert turn_tokens(t) == 0

    def test_matches_estimate_tokens_of_text(self):
        text = "Error in /src/module.py at port :8080"
        t = _turn("system", 0, text)
        # turn_tokens should match estimate_tokens(extract_text(t))
        from lib.parser import extract_text
        assert turn_tokens(t) == estimate_tokens(extract_text(t))

    def test_longer_turn_more_tokens(self):
        short_t = _turn("system", 0, "hi")
        long_t = _turn("system", 1, "ValueError: connection refused at /home/user/project/src/main.py port :8080 " * 10)
        assert turn_tokens(long_t) > turn_tokens(short_t)
