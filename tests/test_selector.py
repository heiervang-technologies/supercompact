"""Tests for lib.selector — SelectionResult, select_turns."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.types import ScoredTurn
from lib.selector import SelectionResult, select_turns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(index: int, tokens: int = 100) -> Turn:
    t = Turn(kind="user", index=index)
    t.lines = [{"message": {"content": f"user turn {index}"}}]
    return t


def _system(index: int, tokens: int = 200) -> Turn:
    t = Turn(kind="system", index=index)
    t.lines = [{"message": {"content": f"system turn {index}"}}]
    return t


def _scored(turn: Turn, score: float, tokens: int) -> ScoredTurn:
    return ScoredTurn(turn=turn, score=score, tokens=tokens)


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------

class TestSelectionResult:
    def test_defaults(self):
        r = SelectionResult()
        assert r.kept_turns == []
        assert r.dropped_turns == []
        assert r.user_tokens == 0
        assert r.budget == 0

    def test_custom_budget(self):
        r = SelectionResult(budget=50_000)
        assert r.budget == 50_000


# ---------------------------------------------------------------------------
# select_turns — basic behaviour
# ---------------------------------------------------------------------------

class TestSelectTurns:
    def test_empty_returns_empty(self):
        result = select_turns([], [], {}, budget=1000)
        assert result.kept_turns == []

    def test_user_turns_always_kept(self):
        u0 = _user(0)
        u1 = _user(1)
        result = select_turns([u0, u1], [], {0: 100, 1: 100}, budget=10_000)
        kept_kinds = {t.kind for t in result.kept_turns}
        assert "user" in kept_kinds
        assert u0 in result.kept_turns
        assert u1 in result.kept_turns

    def test_short_system_turns_always_kept(self):
        u = _user(0)
        s = _system(1)  # will assign 50 tokens — below short_threshold=300
        result = select_turns([u, s], [], {0: 100, 1: 50}, budget=10_000, short_threshold=300)
        assert s in result.kept_turns
        assert result.short_system_tokens == 50

    def test_long_system_turn_not_short(self):
        """A system turn above short_threshold must be scored to be kept."""
        u = _user(0)
        s = _system(1)
        # No scored turns → long system turn should only be kept if it's the last
        result = select_turns([u, s], [], {0: 100, 1: 500}, budget=10_000, short_threshold=300)
        # s is the last system turn → always kept
        assert s in result.kept_turns

    def test_last_system_turn_always_kept(self):
        u = _user(0)
        s1 = _system(1)
        s2 = _system(2)
        # Only score s1; s2 is last system turn
        scored = [_scored(s1, 0.9, 500)]
        result = select_turns(
            [u, s1, s2], scored,
            {0: 100, 1: 500, 2: 500},
            budget=10_000, short_threshold=300,
        )
        assert s2 in result.kept_turns

    def test_budget_limits_kept_turns(self):
        u = _user(0)
        turns = [u] + [_system(i) for i in range(1, 6)]
        token_counts = {0: 100}
        scored = []
        for i in range(1, 6):
            token_counts[i] = 1000
            scored.append(_scored(turns[i], float(i) / 10, 1000))
        # Budget only allows user (100) + last system (1000) + ~1 more
        result = select_turns(turns, scored, token_counts, budget=2200, short_threshold=300)
        total_kept = sum(token_counts.get(t.index, 0) for t in result.kept_turns)
        assert total_kept <= 2200

    def test_kept_turns_in_original_order(self):
        turns = [_user(0), _system(1), _user(2), _system(3)]
        scored = [_scored(turns[1], 0.8, 200), _scored(turns[3], 0.9, 200)]
        token_counts = {0: 100, 1: 200, 2: 100, 3: 200}
        result = select_turns(turns, scored, token_counts, budget=10_000, short_threshold=50)
        indices = [t.index for t in result.kept_turns]
        assert indices == sorted(indices)

    def test_dropped_turns_tracked(self):
        u = _user(0)
        s1 = _system(1)
        s2 = _system(2)
        s3 = _system(3)
        scored = [
            _scored(s1, 0.9, 500),
            _scored(s2, 0.5, 500),
            _scored(s3, 0.1, 500),
        ]
        token_counts = {0: 100, 1: 500, 2: 500, 3: 500}
        # Budget: 100 (user) + 500 (last=s3 always kept) + 500 = 1100 → only fits 2 long turns
        result = select_turns(
            [u, s1, s2, s3], scored, token_counts,
            budget=1100, short_threshold=300,
        )
        # Some scored turns must be dropped
        assert len(result.dropped_turns) >= 1

    def test_token_accounting(self):
        u = _user(0)
        s = _system(1)
        result = select_turns([u, s], [], {0: 150, 1: 50}, budget=10_000, short_threshold=300)
        assert result.user_tokens == 150
        assert result.short_system_tokens == 50
        assert result.total_input_tokens == 200

    def test_all_user_no_system(self):
        turns = [_user(i) for i in range(5)]
        token_counts = {i: 100 for i in range(5)}
        result = select_turns(turns, [], token_counts, budget=10_000)
        assert len(result.kept_turns) == 5

    def test_high_score_turn_preferred_over_low(self):
        """Given budget for one scored turn, the higher-scored one should be kept."""
        u = _user(0)
        s_last = _system(1)  # always kept as last
        s_high = _system(2)
        s_low = _system(3)
        # Total turns: 0,1,2,3,4 but we'll set 4 as last
        s_end = _system(4)
        all_turns = [u, s_last, s_high, s_low, s_end]
        scored = [
            _scored(s_last, 0.3, 500),
            _scored(s_high, 0.9, 500),
            _scored(s_low, 0.1, 500),
        ]
        token_counts = {0: 100, 1: 500, 2: 500, 3: 500, 4: 50}
        # Budget: 100 (user) + 50 (s_end short, last) + 500 (one more) = 650
        result = select_turns(all_turns, scored, token_counts, budget=650, short_threshold=300)
        kept_indices = {t.index for t in result.kept_turns}
        # s_high (index=2) should be preferred over s_low (index=3)
        assert s_high.index in kept_indices or s_low.index not in kept_indices
