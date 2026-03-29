"""Tests for lib/tokenizer.py — token counting helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tokenizer(n_tokens: int):
    """Return a mock tokenizer whose encode() always returns n_tokens tokens."""
    tok = MagicMock()
    tok.encode.return_value = list(range(n_tokens))
    return tok


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_returns_integer(self):
        import lib.tokenizer as tok_mod
        with patch.object(tok_mod, "_get_tokenizer", return_value=_mock_tokenizer(5)):
            from lib.tokenizer import estimate_tokens
            result = estimate_tokens("hello world")
        assert isinstance(result, int)

    def test_returns_tokenizer_count(self):
        import lib.tokenizer as tok_mod
        with patch.object(tok_mod, "_get_tokenizer", return_value=_mock_tokenizer(7)):
            from lib.tokenizer import estimate_tokens
            result = estimate_tokens("any text")
        assert result == 7

    def test_zero_tokens_for_empty_text(self):
        import lib.tokenizer as tok_mod
        with patch.object(tok_mod, "_get_tokenizer", return_value=_mock_tokenizer(0)):
            from lib.tokenizer import estimate_tokens
            result = estimate_tokens("")
        assert result == 0

    def test_passes_add_special_tokens_false(self):
        """estimate_tokens must call encode with add_special_tokens=False."""
        import lib.tokenizer as tok_mod
        mock_tok = _mock_tokenizer(3)
        with patch.object(tok_mod, "_get_tokenizer", return_value=mock_tok):
            from lib.tokenizer import estimate_tokens
            estimate_tokens("hello")
        mock_tok.encode.assert_called_once_with("hello", add_special_tokens=False)


# ---------------------------------------------------------------------------
# turn_tokens
# ---------------------------------------------------------------------------

class TestTurnTokens:
    def _make_turn(self, text: str) -> Turn:
        t = Turn(kind="system", index=0)
        t.content = [{"type": "text", "text": text}]
        return t

    def test_counts_turn_text(self):
        import lib.tokenizer as tok_mod
        with patch.object(tok_mod, "_get_tokenizer", return_value=_mock_tokenizer(4)):
            from lib.tokenizer import turn_tokens
            turn = self._make_turn("some content here")
            result = turn_tokens(turn)
        assert result == 4

    def test_returns_integer(self):
        import lib.tokenizer as tok_mod
        with patch.object(tok_mod, "_get_tokenizer", return_value=_mock_tokenizer(2)):
            from lib.tokenizer import turn_tokens
            turn = self._make_turn("hi")
            result = turn_tokens(turn)
        assert isinstance(result, int)
