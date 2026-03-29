"""Tests for lib/dedup.py — SuffixAutomaton, _turn_unique_ratio, dedup_scores."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.dedup import (
    SuffixAutomaton,
    _build_automaton,
    _turn_unique_ratio,
    dedup_scores,
)
from lib.parser import Turn
from lib.types import ScoredTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(kind: str, index: int, text: str) -> Turn:
    t = Turn(kind=kind, index=index)
    t.append({"message": {"content": text}})
    return t


# ---------------------------------------------------------------------------
# SuffixAutomaton — basic invariants
# ---------------------------------------------------------------------------

class TestSuffixAutomaton:
    def test_initial_state_count(self):
        sa = SuffixAutomaton()
        assert len(sa.states) == 1

    def test_extend_adds_states(self):
        sa = SuffixAutomaton()
        sa.extend("a", 0)
        assert len(sa.states) >= 2

    def test_extend_single_char(self):
        sa = SuffixAutomaton()
        sa.extend("x", 0)
        sa.propagate_counts()
        # Should not crash and last state should be set
        assert sa.last >= 0

    def test_match_repeated_length_all_zeros_for_unique(self):
        sa = SuffixAutomaton()
        for i, c in enumerate("abcdef"):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length("abcdef")
        # Each char in "abcdef" appears only once → no repeated substrings
        assert all(l == 0 for l in lengths)

    def test_match_repeated_length_for_repeated_text(self):
        # Build automaton over "aaa" — substring "aa" appears twice
        sa = SuffixAutomaton()
        text = "aaaa"
        for i, c in enumerate(text):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length("aa")
        # "aa" occurs in "aaaa", so matching "aa" should find repeated lengths > 0
        assert any(l > 0 for l in lengths)

    def test_match_repeated_length_returns_one_per_char(self):
        sa = SuffixAutomaton()
        for i, c in enumerate("hello"):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length("hel")
        assert len(lengths) == 3


# ---------------------------------------------------------------------------
# _turn_unique_ratio
# ---------------------------------------------------------------------------

class TestTurnUniqueRatio:
    def _build_sa(self, corpus: str) -> SuffixAutomaton:
        sa = SuffixAutomaton()
        for i, c in enumerate(corpus):
            sa.extend(c, i)
        sa.propagate_counts()
        return sa

    def test_empty_text_returns_one(self):
        sa = self._build_sa("hello world")
        assert _turn_unique_ratio(sa, "") == pytest.approx(1.0)

    def test_unique_text_returns_high_ratio(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        sa = self._build_sa(corpus)
        # Text with no long repeated substrings should have high unique ratio
        ratio = _turn_unique_ratio(sa, "xyz123abc", min_repeat_len=4)
        assert ratio == pytest.approx(1.0)

    def test_highly_repeated_content_returns_lower_ratio(self):
        # Build over repeated content
        repeated = "ABCD" * 50
        sa = self._build_sa(repeated)
        ratio = _turn_unique_ratio(sa, "ABCD" * 20, min_repeat_len=4)
        assert ratio < 1.0

    def test_ratio_between_zero_and_one(self):
        sa = self._build_sa("hello world hello world")
        ratio = _turn_unique_ratio(sa, "hello", min_repeat_len=3)
        assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# dedup_scores
# ---------------------------------------------------------------------------

class TestDedupScores:
    def test_empty_system_turns_returns_empty(self):
        turns = [_turn("user", 0, "hello")]
        result = dedup_scores(turns, [], {0: 10})
        assert result == []

    def test_returns_scored_turns(self):
        t = _turn("system", 1, "unique content here")
        result = dedup_scores([t], [t], {1: 50})
        assert len(result) == 1
        assert isinstance(result[0], ScoredTurn)

    def test_scores_between_zero_and_one(self):
        turns = [
            _turn("user", 0, "question"),
            _turn("system", 1, "some response"),
            _turn("system", 2, "another response"),
        ]
        system = [turns[1], turns[2]]
        token_counts = {0: 10, 1: 20, 2: 20}
        result = dedup_scores(turns, system, token_counts)
        for st in result:
            assert 0.0 <= st.score <= 1.0

    def test_tokens_set_from_token_counts(self):
        t = _turn("system", 5, "content")
        result = dedup_scores([t], [t], {5: 999})
        assert result[0].tokens == 999

    def test_one_result_per_system_turn(self):
        turns = [_turn("system", i, f"text{i}") for i in range(4)]
        result = dedup_scores(turns, turns, {i: 10 for i in range(4)})
        assert len(result) == 4

    def test_highly_repeated_turn_scores_lower(self):
        # Same text repeated many times — turn has low unique ratio
        repeated = "ABCDE" * 40
        all_turns = [_turn("system", i, repeated) for i in range(3)]
        system = all_turns[:]
        token_counts = {i: 200 for i in range(3)}
        result = dedup_scores(all_turns, system, token_counts, min_repeat_len=5)
        # At least one turn should have lower score (repeated content)
        scores = [st.score for st in result]
        assert min(scores) < 1.0
