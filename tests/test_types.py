"""Tests for lib.types — ScoredTurn, build_query, random_scores."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.types import ScoredTurn, build_query, random_scores


def _turn(index: int, text: str, kind: str = "user") -> Turn:
    t = Turn(kind=kind, index=index)
    t.lines = [{"message": {"content": text}}]
    return t


# ---------------------------------------------------------------------------
# ScoredTurn
# ---------------------------------------------------------------------------

class TestScoredTurn:
    def test_stores_turn_score_tokens(self):
        turn = _turn(0, "hello")
        st = ScoredTurn(turn=turn, score=0.75, tokens=10)
        assert st.turn is turn
        assert st.score == 0.75
        assert st.tokens == 10

    def test_score_can_be_zero(self):
        st = ScoredTurn(turn=_turn(0, "x"), score=0.0, tokens=5)
        assert st.score == 0.0

    def test_score_can_be_one(self):
        st = ScoredTurn(turn=_turn(0, "x"), score=1.0, tokens=5)
        assert st.score == 1.0


# ---------------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def test_single_turn_returns_its_text(self):
        turns = [_turn(0, "What is Python?")]
        result = build_query(turns)
        assert "What is Python?" in result

    def test_two_turns_joined_by_separator(self):
        turns = [_turn(0, "first"), _turn(1, "second")]
        result = build_query(turns)
        assert "---" in result
        assert "first" in result
        assert "second" in result

    def test_uses_last_three_turns(self):
        turns = [_turn(i, f"msg{i}") for i in range(6)]
        result = build_query(turns)
        # Only last 3 messages should appear
        assert "msg5" in result
        assert "msg4" in result
        assert "msg3" in result
        # Earlier messages should not appear
        assert "msg0" not in result
        assert "msg1" not in result

    def test_exactly_three_turns_all_included(self):
        turns = [_turn(i, f"turn{i}") for i in range(3)]
        result = build_query(turns)
        for i in range(3):
            assert f"turn{i}" in result

    def test_empty_turns_returns_empty_string(self):
        result = build_query([])
        assert result == ""

    def test_truncates_to_max_chars(self):
        # One turn with a very long text
        long_text = "x" * 10000
        turns = [_turn(0, long_text)]
        result = build_query(turns, max_chars=100)
        assert len(result) <= 100

    def test_default_max_chars_is_4000(self):
        """Query should be at most 4000 chars by default."""
        long_text = "a" * 8000
        turns = [_turn(0, long_text)]
        result = build_query(turns)
        assert len(result) <= 4000

    def test_short_query_not_truncated(self):
        turns = [_turn(0, "short")]
        result = build_query(turns, max_chars=4000)
        assert result == "short"


# ---------------------------------------------------------------------------
# random_scores
# ---------------------------------------------------------------------------

class TestRandomScores:
    def test_returns_list(self):
        turns = [_turn(i, f"t{i}", kind="system") for i in range(3)]
        result = random_scores(turns, {0: 5, 1: 10, 2: 15})
        assert isinstance(result, list)

    def test_length_matches_input(self):
        turns = [_turn(i, f"t{i}", kind="system") for i in range(5)]
        result = random_scores(turns, {i: i for i in range(5)})
        assert len(result) == 5

    def test_all_elements_are_scored_turns(self):
        turns = [_turn(0, "x", kind="system")]
        result = random_scores(turns, {0: 3})
        assert all(isinstance(r, ScoredTurn) for r in result)

    def test_scores_in_range_0_to_1(self):
        turns = [_turn(i, f"t{i}", kind="system") for i in range(20)]
        result = random_scores(turns, {i: 1 for i in range(20)})
        for st in result:
            assert 0.0 <= st.score <= 1.0

    def test_token_counts_assigned_from_dict(self):
        turns = [_turn(0, "x", kind="system"), _turn(1, "y", kind="system")]
        result = random_scores(turns, {0: 42, 1: 99})
        assert result[0].tokens == 42
        assert result[1].tokens == 99

    def test_missing_token_count_defaults_to_zero(self):
        turns = [_turn(0, "x", kind="system")]
        result = random_scores(turns, {})
        assert result[0].tokens == 0

    def test_empty_input_returns_empty_list(self):
        result = random_scores([], {})
        assert result == []

    def test_randomness_produces_variety(self):
        """Calling twice should produce different scores."""
        turns = [_turn(i, f"t{i}", kind="system") for i in range(10)]
        scores_a = [st.score for st in random_scores(turns, {})]
        scores_b = [st.score for st in random_scores(turns, {})]
        assert scores_a != scores_b
