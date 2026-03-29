"""Tests for lib/types.py — ScoredTurn, build_query, random_scores."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.types import ScoredTurn, build_query, random_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(index: int, text: str = "") -> Turn:
    t = Turn(kind="user", index=index)
    if text:
        t.lines = [{"type": "user", "message": {"content": text}}]
    return t


def _system(index: int, text: str = "") -> Turn:
    t = Turn(kind="system", index=index)
    if text:
        t.lines = [{"type": "assistant", "message": {"content": text}}]
    return t


# ---------------------------------------------------------------------------
# ScoredTurn
# ---------------------------------------------------------------------------

class TestScoredTurn:
    def test_fields_accessible(self):
        t = _system(1, "hello")
        st = ScoredTurn(turn=t, score=0.75, tokens=100)
        assert st.turn is t
        assert st.score == 0.75
        assert st.tokens == 100

    def test_score_can_be_zero(self):
        t = _system(1, "hello")
        st = ScoredTurn(turn=t, score=0.0, tokens=50)
        assert st.score == 0.0

    def test_score_can_be_one(self):
        t = _system(1, "hello")
        st = ScoredTurn(turn=t, score=1.0, tokens=50)
        assert st.score == 1.0


# ---------------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def test_single_user_turn(self):
        u = _user(0, "What is the error?")
        q = build_query([u])
        assert "What is the error?" in q

    def test_uses_last_three_turns(self):
        turns = [_user(i, f"message {i}") for i in range(5)]
        q = build_query(turns)
        # Should include messages 2, 3, 4 (last 3)
        assert "message 2" in q
        assert "message 3" in q
        assert "message 4" in q
        # Should NOT include messages 0, 1
        assert "message 0" not in q
        assert "message 1" not in q

    def test_fewer_than_three_uses_all(self):
        turns = [_user(0, "first"), _user(1, "second")]
        q = build_query(turns)
        assert "first" in q
        assert "second" in q

    def test_exactly_three(self):
        turns = [_user(i, f"msg {i}") for i in range(3)]
        q = build_query(turns)
        for i in range(3):
            assert f"msg {i}" in q

    def test_separator_between_turns(self):
        u1 = _user(0, "first question")
        u2 = _user(1, "second question")
        q = build_query([u1, u2])
        assert "---" in q

    def test_truncated_at_max_chars(self):
        long_text = "x" * 5000
        u = _user(0, long_text)
        q = build_query([u], max_chars=100)
        assert len(q) <= 100

    def test_empty_returns_empty(self):
        q = build_query([])
        assert q == ""

    def test_custom_max_chars(self):
        u = _user(0, "a" * 500)
        q = build_query([u], max_chars=200)
        assert len(q) <= 200


# ---------------------------------------------------------------------------
# random_scores
# ---------------------------------------------------------------------------

class TestRandomScores:
    def test_returns_one_per_system_turn(self):
        turns = [_system(i) for i in range(5)]
        tc = {i: 100 for i in range(5)}
        results = random_scores(turns, tc)
        assert len(results) == 5

    def test_scores_in_0_1(self):
        turns = [_system(i, f"content {i}") for i in range(10)]
        tc = {i: 100 for i in range(10)}
        results = random_scores(turns, tc)
        for st in results:
            assert 0.0 <= st.score <= 1.0

    def test_tokens_from_token_counts(self):
        s = _system(1, "hello")
        tc = {1: 250}
        results = random_scores([s], tc)
        assert results[0].tokens == 250

    def test_missing_token_count_defaults_to_zero(self):
        s = _system(1, "hello")
        results = random_scores([s], {})
        assert results[0].tokens == 0

    def test_turn_reference_preserved(self):
        s = _system(1, "hello")
        results = random_scores([s], {1: 100})
        assert results[0].turn is s

    def test_empty_returns_empty(self):
        assert random_scores([], {}) == []
