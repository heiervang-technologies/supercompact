"""Tests for lib.fitness — _extract_vocab, _idf, FitnessResult."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.fitness import _extract_vocab, _idf, FitnessResult


# ---------------------------------------------------------------------------
# _extract_vocab
# ---------------------------------------------------------------------------

class TestExtractVocab:
    def test_returns_counter(self):
        result = _extract_vocab("hello world")
        assert isinstance(result, Counter)

    def test_long_words_included(self):
        result = _extract_vocab("python is great")
        assert "python" in result
        assert "great" in result

    def test_short_words_excluded(self):
        """Words shorter than MIN_WORD_LEN=4 are excluded."""
        result = _extract_vocab("to be or not to be")
        assert "to" not in result
        assert "be" not in result
        assert "or" not in result

    def test_word_lowercased(self):
        result = _extract_vocab("Python PYTHON python")
        # All three map to 'python'
        assert result.get("python", 0) >= 1

    def test_file_path_extracted(self):
        result = _extract_vocab("editing /home/user/code/main.py")
        # File paths like /home/user/code/main.py should appear
        assert any("/" in k for k in result)

    def test_empty_string_returns_empty(self):
        result = _extract_vocab("")
        assert len(result) == 0

    def test_frequency_counted(self):
        result = _extract_vocab("hello hello hello world world")
        assert result["hello"] >= 3
        assert result["world"] >= 2

    def test_words_with_underscores_included(self):
        result = _extract_vocab("function_name is important")
        assert "function_name" in result

    def test_numbers_in_word_allowed(self):
        """Words like 'model123' should be included if long enough."""
        result = _extract_vocab("model123 is fine")
        assert "model123" in result


# ---------------------------------------------------------------------------
# _idf
# ---------------------------------------------------------------------------

class TestIdf:
    def _vocabs(self, docs: list[str]) -> list[Counter]:
        return [_extract_vocab(d) for d in docs]

    def test_zero_when_term_absent(self):
        vocabs = self._vocabs(["hello world", "foo bar"])
        assert _idf("missing", vocabs, 2) == 0.0

    def test_positive_when_term_present(self):
        vocabs = self._vocabs(["hello world", "foo bar"])
        assert _idf("hello", vocabs, 2) > 0.0

    def test_lower_when_term_in_more_docs(self):
        vocabs = self._vocabs(["hello world", "hello world", "hello world"])
        idf_common = _idf("hello", vocabs, 3)
        vocabs2 = self._vocabs(["hello world", "foo bar", "baz qux"])
        idf_rare = _idf("hello", vocabs2, 3)
        assert idf_rare >= idf_common

    def test_uses_log_formula(self):
        """IDF should use log(1 + total/df)."""
        vocabs = [Counter({"hello": 1})]
        expected = math.log(1 + 1 / 1)
        assert abs(_idf("hello", vocabs, 1) - expected) < 1e-10

    def test_zero_docs_does_not_crash(self):
        result = _idf("word", [], 0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# FitnessResult.f1
# ---------------------------------------------------------------------------

class TestFitnessResultF1:
    def _result(self, recall: float, compression: float) -> FitnessResult:
        return FitnessResult(
            method="test",
            recall=recall,
            speed_s=0.0,
            compression=compression,
            budget=1000,
            total_tokens=1000,
            kept_tokens=int(1000 * compression),
            prefix_turns=10,
            suffix_turns=5,
            suffix_vocab_size=50,
            scored_count=5,
            kept_scored=3,
            dropped_scored=2,
        )

    def test_perfect_recall_perfect_compression_is_zero(self):
        """Keeping everything: recall=1, compression=1, compression_eff=0 → f1=0."""
        r = self._result(recall=1.0, compression=1.0)
        assert r.f1 == 0.0

    def test_zero_recall_is_zero(self):
        r = self._result(recall=0.0, compression=0.0)
        assert r.f1 == 0.0

    def test_balanced_recall_and_compression(self):
        """recall=0.5, compression=0.5 → compression_eff=0.5 → f1=0.5."""
        r = self._result(recall=0.5, compression=0.5)
        assert abs(r.f1 - 0.5) < 1e-10

    def test_high_recall_low_compression_gives_moderate_f1(self):
        """recall=0.9, compression=0.8 → compression_eff=0.2 → f1 = 2*0.9*0.2/(0.9+0.2)."""
        r = self._result(recall=0.9, compression=0.8)
        expected = 2 * 0.9 * 0.2 / (0.9 + 0.2)
        assert abs(r.f1 - expected) < 1e-10

    def test_f1_between_zero_and_one(self):
        for recall in [0.0, 0.3, 0.7, 1.0]:
            for comp in [0.0, 0.3, 0.7, 1.0]:
                r = self._result(recall=recall, compression=comp)
                assert 0.0 <= r.f1 <= 1.0
