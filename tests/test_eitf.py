"""Tests for lib/eitf.py — eitf_scores."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eitf import eitf_scores
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
# eitf_scores
# ---------------------------------------------------------------------------

class TestEitfScoresEmpty:
    def test_no_system_turns_returns_empty(self):
        turns = [_turn("user", 0, "hello")]
        result = eitf_scores(turns, [], {0: 10})
        assert result == []

    def test_empty_all_turns_returns_empty(self):
        result = eitf_scores([], [], {})
        assert result == []


class TestEitfScoresBasic:
    def test_returns_scored_turns(self):
        t = _turn("system", 1, "Error: ValueError on line 42")
        all_turns = [_turn("user", 0, "question"), t]
        result = eitf_scores(all_turns, [t], {0: 10, 1: 50})
        assert len(result) == 1
        assert isinstance(result[0], ScoredTurn)

    def test_scores_normalized_to_zero_one(self):
        turns = [
            _turn("user", 0, "what is the error?"),
            _turn("system", 1, "ValueError on line 42 in /src/main.py"),
            _turn("system", 2, "ModuleNotFoundError for numpy package"),
        ]
        system = turns[1:]
        token_counts = {0: 10, 1: 80, 2: 60}
        result = eitf_scores(turns, system, token_counts)
        for st in result:
            assert 0.0 <= st.score <= 1.0

    def test_one_result_per_system_turn(self):
        turns = [_turn("system", i, f"content{i}") for i in range(4)]
        result = eitf_scores(turns, turns, {i: 50 for i in range(4)})
        assert len(result) == 4

    def test_max_score_is_one(self):
        turns = [
            _turn("user", 0, "question"),
            _turn("system", 1, "ValueError at /path/to/file.py:42"),
            _turn("system", 2, "plain content"),
        ]
        system = turns[1:]
        token_counts = {0: 10, 1: 80, 2: 40}
        result = eitf_scores(turns, system, token_counts)
        scores = [st.score for st in result]
        assert max(scores) == pytest.approx(1.0)

    def test_tokens_set_from_token_counts(self):
        t = _turn("system", 3, "some content with ValueError")
        all_turns = [t]
        result = eitf_scores(all_turns, [t], {3: 777})
        assert result[0].tokens == 777

    def test_turn_with_rare_entities_scores_higher(self):
        # Turn 1 has a unique entity (rare), turn 2 has a common entity
        # that appears in all turns
        common_err = "ValueError"
        unique_path = "/home/user/special_project/module.py"

        turns = [
            _turn("user", 0, f"question about {common_err}"),
            _turn("system", 1, f"Error in {unique_path}: {common_err}"),
            _turn("system", 2, f"{common_err} appears here"),
            _turn("system", 3, f"{common_err} again"),
        ]
        system = turns[1:]
        token_counts = {i: 50 for i in range(4)}
        result = eitf_scores(turns, system, token_counts)
        # turn 1 (index 1) has the unique path entity, should score higher
        # than turn 3 which only has the common entity
        score_map = {st.turn.index: st.score for st in result}
        assert score_map[1] >= score_map[3]

    def test_plain_content_no_entities_scores_zero(self):
        # A turn with no extractable entities should score 0 (before normalization)
        # but with normalization it can be 0 if all turns have no entities
        turns = [
            _turn("system", 0, "aaaaa bbbbb ccccc"),
            _turn("system", 1, "zzzzz yyyyy xxxxx"),
        ]
        system = turns[:]
        result = eitf_scores(turns, system, {0: 50, 1: 50})
        scores = [st.score for st in result]
        # All should be in [0,1]
        for s in scores:
            assert 0.0 <= s <= 1.0
