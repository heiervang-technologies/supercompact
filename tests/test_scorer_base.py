"""Tests for lib/scorer_base.py — registry, get_scorer, Scorer protocol."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.scorer_base import (
    Scorer,
    DedupScorer,
    EitfScorer,
    SetcoverScorer,
    EmbedScorer,
    LlamaEmbedScorerWrapper,
    LlamaRerankScorerWrapper,
    SCORERS,
    LOCAL_METHODS,
    ALL_METHODS,
    get_scorer,
)


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_scorers_has_dedup(self):
        assert "dedup" in SCORERS

    def test_scorers_has_eitf(self):
        assert "eitf" in SCORERS

    def test_scorers_has_setcover(self):
        assert "setcover" in SCORERS

    def test_scorers_has_embed(self):
        assert "embed" in SCORERS

    def test_scorers_has_llama_embed(self):
        assert "llama-embed" in SCORERS

    def test_scorers_has_llama_rerank(self):
        assert "llama-rerank" in SCORERS

    def test_local_methods_are_subset_of_all(self):
        assert set(LOCAL_METHODS).issubset(set(ALL_METHODS))

    def test_local_methods_contains_expected(self):
        for m in ("dedup", "eitf", "setcover"):
            assert m in LOCAL_METHODS

    def test_all_methods_non_empty(self):
        assert len(ALL_METHODS) > 0

    def test_scorer_instances_have_name(self):
        for key, scorer in SCORERS.items():
            assert hasattr(scorer, "name"), f"{key} scorer missing name"
            assert isinstance(scorer.name, str)


# ---------------------------------------------------------------------------
# get_scorer
# ---------------------------------------------------------------------------

class TestGetScorer:
    def test_returns_dedup_scorer(self):
        s = get_scorer("dedup")
        assert isinstance(s, DedupScorer)

    def test_returns_eitf_scorer(self):
        s = get_scorer("eitf")
        assert isinstance(s, EitfScorer)

    def test_returns_setcover_scorer(self):
        s = get_scorer("setcover")
        assert isinstance(s, SetcoverScorer)

    def test_returns_embed_scorer(self):
        s = get_scorer("embed")
        assert isinstance(s, EmbedScorer)

    def test_invalid_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_scorer("does_not_exist")

    def test_error_message_lists_available_methods(self):
        with pytest.raises(ValueError, match="dedup"):
            get_scorer("nonexistent")

    def test_same_instance_returned_each_time(self):
        s1 = get_scorer("dedup")
        s2 = get_scorer("dedup")
        assert s1 is s2


# ---------------------------------------------------------------------------
# Scorer protocol compliance
# ---------------------------------------------------------------------------

class TestScorerProtocol:
    def test_dedup_scorer_is_scorer(self):
        assert isinstance(DedupScorer(), Scorer)

    def test_eitf_scorer_is_scorer(self):
        assert isinstance(EitfScorer(), Scorer)

    def test_setcover_scorer_is_scorer(self):
        assert isinstance(SetcoverScorer(), Scorer)

    def test_dedup_scorer_has_score_method(self):
        assert callable(DedupScorer().score)

    def test_eitf_scorer_has_score_method(self):
        assert callable(EitfScorer().score)

    def test_setcover_scorer_has_score_method(self):
        assert callable(SetcoverScorer().score)


# ---------------------------------------------------------------------------
# Concrete scorer delegation (mocked internals)
# ---------------------------------------------------------------------------

class TestDedupScorerDelegation:
    def test_score_delegates_to_dedup_scores(self):
        scorer = DedupScorer()
        mock_result = [MagicMock()]
        with patch("lib.dedup.dedup_scores", return_value=mock_result) as mock_fn:
            result = scorer.score([], [], {})
        mock_fn.assert_called_once()
        assert result is mock_result

    def test_score_passes_min_repeat_len_kwarg(self):
        scorer = DedupScorer()
        with patch("lib.dedup.dedup_scores", return_value=[]) as mock_fn:
            scorer.score([], [], {}, min_repeat_len=128)
        _, kwargs = mock_fn.call_args
        assert kwargs.get("min_repeat_len") == 128


class TestEitfScorerDelegation:
    def test_score_delegates_to_eitf_scores(self):
        scorer = EitfScorer()
        mock_result = [MagicMock()]
        with patch("lib.eitf.eitf_scores", return_value=mock_result) as mock_fn:
            result = scorer.score([], [], {})
        mock_fn.assert_called_once()
        assert result is mock_result


class TestSetcoverScorerDelegation:
    def test_score_delegates_to_setcover_scores(self):
        scorer = SetcoverScorer()
        mock_result = [MagicMock()]
        with patch("lib.setcover.setcover_scores", return_value=mock_result) as mock_fn:
            result = scorer.score([], [], {})
        mock_fn.assert_called_once()
        assert result is mock_result

    def test_score_passes_budget_kwarg(self):
        scorer = SetcoverScorer()
        with patch("lib.setcover.setcover_scores", return_value=[]) as mock_fn:
            scorer.score([], [], {}, budget=50_000)
        _, kwargs = mock_fn.call_args
        assert kwargs.get("budget") == 50_000
