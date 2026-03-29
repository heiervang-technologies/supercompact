"""Tests for lib/fitness.py — _extract_vocab, _idf, and FitnessResult.f1."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# transformers is not installed in CI; stub it out before importing fitness
if "transformers" not in sys.modules:
    _mock_transformers = MagicMock()
    sys.modules["transformers"] = _mock_transformers

from lib.fitness import _extract_vocab, _idf, FitnessResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(recall: float = 0.8, compression: float = 0.5, **kwargs) -> FitnessResult:
    defaults = dict(
        method="dedup", recall=recall, speed_s=1.0, compression=compression,
        budget=1000, total_tokens=1000, kept_tokens=500,
        prefix_turns=10, suffix_turns=5, suffix_vocab_size=50,
        scored_count=8, kept_scored=5, dropped_scored=3,
    )
    defaults.update(kwargs)
    return FitnessResult(**defaults)


# ---------------------------------------------------------------------------
# _extract_vocab
# ---------------------------------------------------------------------------

class TestExtractVocab:
    def test_empty_string_returns_empty(self):
        assert dict(_extract_vocab("")) == {}

    def test_short_words_excluded(self):
        # Words shorter than MIN_WORD_LEN (4) should not appear
        vocab = _extract_vocab("a an the or is it")
        assert len(vocab) == 0

    def test_word_at_min_length_included(self):
        # Words of exactly 4 characters are included if they match the pattern
        vocab = _extract_vocab("file code data")
        # "file", "code", "data" are 4 chars — all should be present
        assert "file" in vocab or "code" in vocab or "data" in vocab

    def test_words_lowercased(self):
        vocab = _extract_vocab("ParseError ValueError TypeError")
        # All should be lowercased
        assert "parseerror" in vocab
        assert "valueerror" in vocab
        assert "typeerror" in vocab

    def test_repeated_word_counts(self):
        vocab = _extract_vocab("error error error")
        assert vocab["error"] >= 2

    def test_file_path_extracted(self):
        vocab = _extract_vocab("edited /home/user/project/src/main.py")
        # Path should appear as a token
        path_found = any("/home" in k or "/project" in k or "main.py" in k for k in vocab)
        assert path_found

    def test_multiple_words(self):
        vocab = _extract_vocab("function method class attribute")
        # Should have several entries
        assert len(vocab) >= 2

    def test_returns_counter(self):
        result = _extract_vocab("hello world testing")
        assert isinstance(result, Counter)

    def test_word_with_underscore_included(self):
        vocab = _extract_vocab("parse_jsonl extract_text build_query")
        # These identifiers are long enough and have underscores
        found = any("parse" in k or "extract" in k or "build" in k for k in vocab)
        assert found

    def test_numeric_suffix_in_identifier(self):
        vocab = _extract_vocab("qwen3 llama2 model1234")
        # Identifiers with digits should be included if long enough
        found = any("qwen3" in k or "llama2" in k or "model1234" in k for k in vocab)
        assert found

    def test_whitespace_only_returns_empty(self):
        assert dict(_extract_vocab("   \t\n  ")) == {}


# ---------------------------------------------------------------------------
# _idf
# ---------------------------------------------------------------------------

class TestIdf:
    def test_term_not_in_any_doc_returns_zero(self):
        docs = [Counter({"hello": 1}), Counter({"world": 1})]
        result = _idf("missing", docs, 2)
        assert result == 0.0

    def test_term_in_all_docs(self):
        """Term in all docs → IDF = log(1 + n/n) = log(2)."""
        docs = [Counter({"word": 1}), Counter({"word": 2})]
        result = _idf("word", docs, 2)
        expected = math.log(1 + 2 / 2)
        assert abs(result - expected) < 1e-9

    def test_term_in_one_of_two_docs(self):
        docs = [Counter({"rare": 1}), Counter({"other": 1})]
        result = _idf("rare", docs, 2)
        expected = math.log(1 + 2 / 1)
        assert abs(result - expected) < 1e-9

    def test_term_in_one_of_many_docs_higher_idf(self):
        """Rare term (in 1 of 10 docs) should have higher IDF than common term (in 9 of 10)."""
        docs = [Counter({"common": 1}) for _ in range(9)] + [Counter({"rare": 1})]
        idf_rare = _idf("rare", docs, 10)
        idf_common = _idf("common", docs, 10)
        assert idf_rare > idf_common

    def test_empty_docs_term_not_found(self):
        result = _idf("word", [], 0)
        assert result == 0.0

    def test_single_doc_with_term(self):
        docs = [Counter({"hello": 1})]
        result = _idf("hello", docs, 1)
        expected = math.log(1 + 1 / 1)
        assert abs(result - expected) < 1e-9

    def test_total_docs_respected(self):
        """Using a larger total_docs increases IDF for rare terms."""
        docs = [Counter({"word": 1})]
        r1 = _idf("word", docs, 1)   # log(1 + 1/1) = log(2)
        r2 = _idf("word", docs, 10)  # log(1 + 10/1) = log(11)
        assert r2 > r1


# ---------------------------------------------------------------------------
# FitnessResult.f1
# ---------------------------------------------------------------------------

class TestFitnessResultF1:
    def test_perfect_recall_full_compression(self):
        """recall=1, compression=0 → compression_eff=1 → F1=1."""
        r = _result(recall=1.0, compression=0.0)
        assert abs(r.f1 - 1.0) < 1e-9

    def test_zero_recall_any_compression(self):
        """recall=0 → F1=0."""
        r = _result(recall=0.0, compression=0.3)
        assert abs(r.f1 - 0.0) < 1e-9

    def test_full_compression_efficiency_zero(self):
        """compression=1.0 → compression_eff=0 → F1=0."""
        r = _result(recall=0.8, compression=1.0)
        assert abs(r.f1 - 0.0) < 1e-9

    def test_balanced_half_half(self):
        """recall=0.5, compression=0.5 → compression_eff=0.5 → F1=0.5."""
        r = _result(recall=0.5, compression=0.5)
        assert abs(r.f1 - 0.5) < 1e-9

    def test_f1_in_0_1_range(self):
        for recall in [0.0, 0.3, 0.6, 1.0]:
            for comp in [0.0, 0.3, 0.6, 1.0]:
                r = _result(recall=recall, compression=comp)
                assert 0.0 <= r.f1 <= 1.0 + 1e-9

    def test_harmonic_mean_formula(self):
        recall = 0.8
        compression = 0.4
        compression_eff = 1.0 - compression  # 0.6
        expected = 2 * recall * compression_eff / (recall + compression_eff)
        r = _result(recall=recall, compression=compression)
        assert abs(r.f1 - expected) < 1e-9

    def test_both_zero_returns_zero(self):
        """recall=0, compression=1 → both terms 0 → F1=0."""
        r = _result(recall=0.0, compression=1.0)
        assert abs(r.f1 - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# FitnessResult dataclass fields
# ---------------------------------------------------------------------------

class TestFitnessResultFields:
    def test_fields_accessible(self):
        r = _result()
        assert r.method == "dedup"
        assert abs(r.recall - 0.8) < 1e-9
        assert abs(r.compression - 0.5) < 1e-9

    def test_budget_stored(self):
        r = _result(budget=4096)
        assert r.budget == 4096

    def test_token_counts_stored(self):
        r = _result()
        assert r.total_tokens == 1000
        assert r.kept_tokens == 500

    def test_scored_counts_stored(self):
        r = _result()
        assert r.scored_count == 8
        assert r.kept_scored == 5
        assert r.dropped_scored == 3
