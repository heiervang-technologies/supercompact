"""Tests for lib/selector.py — budget-constrained turn selection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.selector import select_turns
from lib.types import ScoredTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(index: int) -> Turn:
    t = Turn(kind="user", index=index)
    return t


def _system(index: int) -> Turn:
    t = Turn(kind="system", index=index)
    return t


def _scored(turn: Turn, score: float, tokens: int) -> ScoredTurn:
    return ScoredTurn(turn=turn, score=score, tokens=tokens)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectTurns:
    def test_user_turns_always_kept(self):
        u = _user(0)
        s = _system(1)
        token_counts = {0: 100, 1: 5000}
        scored = [_scored(s, 0.1, 5000)]
        result = select_turns([u, s], scored, token_counts, budget=50)
        kept_indices = {t.index for t in result.kept_turns}
        assert u.index in kept_indices

    def test_short_system_turns_always_kept(self):
        u = _user(0)
        s_short = _system(1)
        token_counts = {0: 10, 1: 50}
        # score it as if long, but it's under short_threshold=300
        scored = [_scored(s_short, 0.0, 50)]
        result = select_turns([u, s_short], scored, token_counts, budget=100, short_threshold=300)
        kept_indices = {t.index for t in result.kept_turns}
        assert s_short.index in kept_indices

    def test_last_system_turn_always_kept(self):
        u = _user(0)
        s1 = _system(1)
        s2 = _system(2)
        token_counts = {0: 10, 1: 5000, 2: 5000}
        # Both long, low scores, budget too small for either
        scored = [_scored(s1, 0.1, 5000), _scored(s2, 0.1, 5000)]
        result = select_turns([u, s1, s2], scored, token_counts, budget=100, short_threshold=50)
        kept_indices = {t.index for t in result.kept_turns}
        assert s2.index in kept_indices

    def test_greedy_selection_by_score(self):
        u = _user(0)
        s_high = _system(1)
        s_low = _system(2)
        s_last = _system(3)
        token_counts = {0: 10, 1: 100, 2: 100, 3: 50}
        scored = [
            _scored(s_high, 0.9, 100),
            _scored(s_low, 0.1, 100),
        ]
        # Budget covers user + last system (60 tokens) + one more (100 tokens) = 160
        result = select_turns(
            [u, s_high, s_low, s_last],
            scored,
            token_counts,
            budget=170,
            short_threshold=40,
        )
        kept_indices = {t.index for t in result.kept_turns}
        # High-score turn should be preferred over low-score
        assert s_high.index in kept_indices
        assert s_low.index not in kept_indices

    def test_budget_respected(self):
        u = _user(0)
        turns = [u]
        scored = []
        token_counts = {0: 10}
        for i in range(1, 6):
            s = _system(i)
            turns.append(s)
            scored.append(_scored(s, 1.0, 200))
            token_counts[i] = 200
        result = select_turns(turns, scored, token_counts, budget=500, short_threshold=50)
        total = sum(token_counts[t.index] for t in result.kept_turns)
        assert total <= 500

    def test_kept_turns_in_original_order(self):
        turns = [_user(0), _system(1), _user(2), _system(3)]
        token_counts = {0: 10, 1: 50, 2: 10, 3: 50}
        scored = [_scored(turns[1], 0.5, 50), _scored(turns[3], 0.9, 50)]
        result = select_turns(turns, scored, token_counts, budget=10000, short_threshold=300)
        indices = [t.index for t in result.kept_turns]
        assert indices == sorted(indices)

    def test_token_accounting(self):
        u = _user(0)
        s = _system(1)
        token_counts = {0: 100, 1: 400}
        scored = [_scored(s, 0.5, 400)]
        result = select_turns([u, s], scored, token_counts, budget=10000, short_threshold=300)
        assert result.user_tokens == 100
        assert result.total_input_tokens == 500

    def test_empty_conversation(self):
        result = select_turns([], [], {}, budget=1000)
        assert result.kept_turns == []
        assert result.total_input_tokens == 0

    def test_only_user_turns(self):
        turns = [_user(0), _user(1)]
        token_counts = {0: 50, 1: 50}
        result = select_turns(turns, [], token_counts, budget=1000)
        assert len(result.kept_turns) == 2

    def test_dropped_turns_tracked(self):
        u = _user(0)
        s_heavy = _system(1)
        s_last = _system(2)
        u2 = _user(3)
        # s_last is still a system turn but not the final system turn (u2 comes after)
        # Actually the last system turn is determined by position in the turns list.
        # To guarantee s_heavy is dropped: make it score-eligible, budget too small,
        # and ensure it's not the last system turn.
        token_counts = {0: 10, 1: 9000, 2: 10, 3: 10}
        scored = [_scored(s_heavy, 0.5, 9000)]
        result = select_turns(
            [u, s_heavy, s_last, u2],
            scored,
            token_counts,
            budget=100,
            short_threshold=5,
        )
        dropped_indices = {st.turn.index for st in result.dropped_turns}
        assert s_heavy.index in dropped_indices
