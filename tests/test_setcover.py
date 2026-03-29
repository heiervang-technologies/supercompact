"""Tests for lib/setcover.py — setcover_scores."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.setcover import setcover_scores
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
# setcover_scores
# ---------------------------------------------------------------------------

class TestSetcoverScoresEmpty:
    def test_no_system_turns_returns_empty(self):
        turns = [_turn("user", 0, "hello")]
        result = setcover_scores(turns, [], {0: 10})
        assert result == []

    def test_empty_all_turns_returns_empty(self):
        result = setcover_scores([], [], {})
        assert result == []


class TestSetcoverScoresBasic:
    def test_returns_scored_turns(self):
        t = _turn("system", 1, "ValueError at /path/to/module.py")
        all_turns = [_turn("user", 0, "question"), t]
        result = setcover_scores(all_turns, [t], {0: 10, 1: 80})
        assert len(result) == 1
        assert isinstance(result[0], ScoredTurn)

    def test_scores_normalized_to_zero_one(self):
        turns = [
            _turn("user", 0, "what happened?"),
            _turn("system", 1, "ValueError in /src/main.py at port :8080"),
            _turn("system", 2, "ModuleNotFoundError for package requests"),
            _turn("system", 3, "generic content without entities"),
        ]
        system = turns[1:]
        token_counts = {i: 60 for i in range(4)}
        result = setcover_scores(turns, system, token_counts)
        for st in result:
            assert 0.0 <= st.score <= 1.0

    def test_max_score_is_one(self):
        turns = [
            _turn("user", 0, "question"),
            _turn("system", 1, "ValueError at /home/user/project/main.py"),
            _turn("system", 2, "just plain text no entities"),
        ]
        system = turns[1:]
        token_counts = {0: 10, 1: 80, 2: 40}
        result = setcover_scores(turns, system, token_counts)
        scores = [st.score for st in result]
        assert max(scores) == pytest.approx(1.0)

    def test_one_result_per_system_turn(self):
        turns = [_turn("system", i, f"content{i}") for i in range(5)]
        result = setcover_scores(turns, turns, {i: 50 for i in range(5)})
        assert len(result) == 5

    def test_tokens_set_from_token_counts(self):
        t = _turn("system", 2, "ValueError occurred")
        all_turns = [t]
        result = setcover_scores(all_turns, [t], {2: 456})
        assert result[0].tokens == 456

    def test_exclusive_entity_gets_higher_score(self):
        # Turn 1 has a unique entity (only appears once in system turns)
        # Turn 2 has an entity shared across many turns (less exclusive)
        common = "ValueError"  # appears in all turns
        unique_path = "/unique/path/to/special_module.py"  # only in turn 1

        turns = [
            _turn("user", 0, f"{common}"),
            _turn("system", 1, f"{common} at {unique_path}"),
            _turn("system", 2, f"{common} again"),
            _turn("system", 3, f"{common} once more"),
        ]
        system = turns[1:]
        token_counts = {i: 60 for i in range(4)}
        result = setcover_scores(turns, system, token_counts)
        score_map = {st.turn.index: st.score for st in result}
        # Turn 1 has the unique path, should score at least as high as others
        assert score_map[1] >= score_map[2]

    def test_all_plain_text_scores_all_zero_or_valid(self):
        # No extractable entities → all raw scores are 0 → max_score=1.0 default
        turns = [
            _turn("system", 0, "plain words here aaaa"),
            _turn("system", 1, "more plain words bbbbb"),
        ]
        result = setcover_scores(turns, turns, {0: 50, 1: 50})
        for st in result:
            assert 0.0 <= st.score <= 1.0
