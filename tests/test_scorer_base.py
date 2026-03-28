"""Tests for lib/scorer_base.py — scorer protocol and registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.scorer_base import (
    Scorer,
    DedupScorer,
    EitfScorer,
    SetcoverScorer,
    SCORERS,
    LOCAL_METHODS,
    ALL_METHODS,
    get_scorer,
)


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


# ---------------------------------------------------------------------------
# Registry / get_scorer
# ---------------------------------------------------------------------------

class TestGetScorer:
    def test_returns_scorer_for_dedup(self):
        scorer = get_scorer("dedup")
        assert isinstance(scorer, DedupScorer)

    def test_returns_scorer_for_eitf(self):
        scorer = get_scorer("eitf")
        assert isinstance(scorer, EitfScorer)

    def test_returns_scorer_for_setcover(self):
        scorer = get_scorer("setcover")
        assert isinstance(scorer, SetcoverScorer)

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_scorer("nonexistent")

    def test_all_registered_methods_retrievable(self):
        for method in SCORERS:
            scorer = get_scorer(method)
            assert scorer is not None

    def test_scorer_has_name_attribute(self):
        for method, scorer in SCORERS.items():
            assert hasattr(scorer, "name"), f"{method} scorer missing .name"
            assert scorer.name == method


# ---------------------------------------------------------------------------
# LOCAL_METHODS and ALL_METHODS
# ---------------------------------------------------------------------------

class TestMethodLists:
    def test_local_methods_are_subset_of_all(self):
        for m in LOCAL_METHODS:
            assert m in ALL_METHODS

    def test_local_methods_contains_expected(self):
        assert "dedup" in LOCAL_METHODS
        assert "eitf" in LOCAL_METHODS
        assert "setcover" in LOCAL_METHODS

    def test_all_methods_contains_local(self):
        for m in LOCAL_METHODS:
            assert m in ALL_METHODS

    def test_ml_methods_not_in_local(self):
        """Methods requiring external services should not be in LOCAL_METHODS."""
        assert "embed" not in LOCAL_METHODS
        assert "llama-embed" not in LOCAL_METHODS
        assert "llama-rerank" not in LOCAL_METHODS


# ---------------------------------------------------------------------------
# Concrete scorer protocol conformance (lazy imports via score())
# ---------------------------------------------------------------------------

class TestLocalScorerScore:
    """Smoke-test the .score() method on the three local scorers."""

    def _turns(self):
        return [
            _user(0, "help"),
            _system(1, "ValueError at /src/app.py:42"),
            _system(2, "plain response"),
        ]

    def test_dedup_scorer_score_returns_list(self):
        turns = self._turns()
        system_turns = [t for t in turns if t.kind == "system"]
        tc = {t.index: 100 for t in turns}
        result = DedupScorer().score(turns, system_turns, tc)
        assert isinstance(result, list)
        assert len(result) == len(system_turns)

    def test_eitf_scorer_score_returns_list(self):
        turns = self._turns()
        system_turns = [t for t in turns if t.kind == "system"]
        tc = {t.index: 100 for t in turns}
        result = EitfScorer().score(turns, system_turns, tc)
        assert isinstance(result, list)
        assert len(result) == len(system_turns)

    def test_setcover_scorer_score_returns_list(self):
        turns = self._turns()
        system_turns = [t for t in turns if t.kind == "system"]
        tc = {t.index: 100 for t in turns}
        result = SetcoverScorer().score(turns, system_turns, tc)
        assert isinstance(result, list)
        assert len(result) == len(system_turns)

    def test_scorer_protocol_satisfied(self):
        """All three local scorers satisfy the Scorer protocol."""
        for cls in [DedupScorer, EitfScorer, SetcoverScorer]:
            assert isinstance(cls(), Scorer)
