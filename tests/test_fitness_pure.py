"""Tests for lib/fitness.py — pure helper functions.

Covers: _extract_vocab, _idf, FitnessResult.f1 property.
No tokenizer, scorer, or network access needed.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.fitness import FitnessResult, _extract_vocab, _idf


# ---------------------------------------------------------------------------
# _extract_vocab
# ---------------------------------------------------------------------------

class TestExtractVocab:
    def test_empty_string_returns_empty_counter(self):
        result = _extract_vocab("")
        assert isinstance(result, Counter)
        assert len(result) == 0

    def test_short_words_excluded(self):
        # Words shorter than MIN_WORD_LEN (4) should not be included
        result = _extract_vocab("go do it now")
        # All words are 3 chars or less — should be empty
        assert len(result) == 0

    def test_long_word_included(self):
        result = _extract_vocab("ValueError occurred here")
        assert "valueerror" in result

    def test_lowercased(self):
        result = _extract_vocab("ErrorMessage")
        assert "errormessage" in result
        assert "ErrorMessage" not in result

    def test_returns_counter(self):
        result = _extract_vocab("hello world hello")
        assert isinstance(result, Counter)

    def test_repeated_word_counted(self):
        result = _extract_vocab("error error error exception")
        assert result["error"] >= 2

    def test_file_path_extracted_as_token(self):
        result = _extract_vocab("/home/user/project/main.py error")
        # File path should appear as single token
        found_path = any("home" in k or "main.py" in k or "/" in k for k in result)
        assert found_path

    def test_multiple_words(self):
        result = _extract_vocab("ConnectionError SocketTimeout NetworkFailure")
        assert len(result) >= 3

    def test_numbers_in_word_included(self):
        result = _extract_vocab("Python3 package123")
        # alphanum words with 4+ chars should be included
        assert len(result) > 0

    def test_whitespace_only_returns_empty(self):
        result = _extract_vocab("   \t\n  ")
        assert len(result) == 0

    def test_typical_error_text(self):
        text = "ValueError at /home/user/project/src/main.py line 42 module import"
        result = _extract_vocab(text)
        assert isinstance(result, Counter)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _idf
# ---------------------------------------------------------------------------

class TestIdf:
    def test_returns_float(self):
        vocab = Counter({"hello": 1})
        result = _idf("hello", [vocab], total_docs=1)
        assert isinstance(result, float)

    def test_term_absent_from_all_docs_returns_zero(self):
        vocabs = [Counter({"hello": 1}), Counter({"world": 1})]
        result = _idf("missing", vocabs, total_docs=2)
        assert result == 0.0

    def test_term_in_all_docs_low_idf(self):
        # When df == total_docs, idf = log(1 + 1) = log(2) ≈ 0.693
        vocabs = [Counter({"common": 1}), Counter({"common": 1})]
        result = _idf("common", vocabs, total_docs=2)
        assert result == pytest.approx(math.log(1 + 2 / 2))

    def test_term_in_one_of_many_docs_high_idf(self):
        vocabs = [Counter({"rare": 1})] + [Counter({"other": 1})] * 9
        result = _idf("rare", vocabs, total_docs=10)
        # idf = log(1 + 10/1) = log(11)
        assert result == pytest.approx(math.log(11))

    def test_idf_decreases_as_df_increases(self):
        vocabs_1_doc = [Counter({"word": 1})] + [Counter()] * 9
        vocabs_5_docs = [Counter({"word": 1})] * 5 + [Counter()] * 5
        idf_1 = _idf("word", vocabs_1_doc, total_docs=10)
        idf_5 = _idf("word", vocabs_5_docs, total_docs=10)
        assert idf_1 > idf_5

    def test_empty_vocab_list_returns_zero(self):
        result = _idf("term", [], total_docs=0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# FitnessResult.f1
# ---------------------------------------------------------------------------

class TestFitnessResultF1:
    def _make(self, recall: float, compression: float) -> FitnessResult:
        return FitnessResult(
            method="dedup",
            recall=recall,
            speed_s=1.0,
            compression=compression,
            budget=80_000,
            total_tokens=10_000,
            kept_tokens=int(compression * 10_000),
            prefix_turns=10,
            suffix_turns=5,
            suffix_vocab_size=100,
            scored_count=8,
            kept_scored=5,
            dropped_scored=3,
        )

    def test_perfect_recall_no_compression_gives_zero_f1(self):
        # recall=1, compression=1 → compression_eff=0 → f1=0
        r = self._make(recall=1.0, compression=1.0)
        assert r.f1 == pytest.approx(0.0)

    def test_zero_recall_gives_zero_f1(self):
        # recall=0, compression_eff=0.5 → f1=0
        r = self._make(recall=0.0, compression=0.5)
        assert r.f1 == pytest.approx(0.0)

    def test_both_zero_gives_zero_f1(self):
        r = self._make(recall=0.0, compression=1.0)
        assert r.f1 == pytest.approx(0.0)

    def test_balanced_gives_nonzero_f1(self):
        # recall=0.8, compression=0.2 → compression_eff=0.8
        # f1 = 2 * 0.8 * 0.8 / (0.8 + 0.8) = 0.8
        r = self._make(recall=0.8, compression=0.2)
        assert r.f1 == pytest.approx(0.8)

    def test_f1_between_zero_and_one(self):
        r = self._make(recall=0.7, compression=0.4)
        assert 0.0 <= r.f1 <= 1.0

    def test_f1_is_harmonic_mean(self):
        recall = 0.6
        compression = 0.3
        eff = 1.0 - compression  # 0.7
        expected = 2 * recall * eff / (recall + eff)
        r = self._make(recall=recall, compression=compression)
        assert r.f1 == pytest.approx(expected)

    def test_high_compression_low_recall_gives_low_f1(self):
        # High compression (kept little) but also low recall
        r = self._make(recall=0.1, compression=0.05)
        # compression_eff = 0.95, recall = 0.1
        # f1 = 2 * 0.1 * 0.95 / (0.1 + 0.95) ≈ 0.181
        assert r.f1 < 0.25

    def test_f1_returns_float(self):
        r = self._make(recall=0.5, compression=0.5)
        assert isinstance(r.f1, float)
