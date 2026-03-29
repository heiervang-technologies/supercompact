"""Tests for lib.dedup — SuffixAutomaton, _turn_unique_ratio, dedup_scores."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.dedup import SuffixAutomaton, _build_automaton, _turn_unique_ratio, dedup_scores
from lib.types import ScoredTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(index: int, text: str) -> Turn:
    t = Turn(kind="system", index=index)
    t.lines = [{"message": {"content": text}}]
    return t


def _user(index: int, text: str = "u") -> Turn:
    t = Turn(kind="user", index=index)
    t.lines = [{"message": {"content": text}}]
    return t


# ---------------------------------------------------------------------------
# SuffixAutomaton — basic correctness
# ---------------------------------------------------------------------------

class TestSuffixAutomaton:
    def _build(self, s: str) -> SuffixAutomaton:
        sa = SuffixAutomaton()
        for i, c in enumerate(s):
            sa.extend(c, i)
        sa.propagate_counts()
        return sa

    def test_empty_string_builds_without_error(self):
        sa = SuffixAutomaton()
        sa.propagate_counts()
        assert len(sa.states) == 1  # just the initial state

    def test_single_char_builds(self):
        sa = self._build("a")
        assert len(sa.states) >= 2

    def test_match_repeated_length_no_repeats(self):
        sa = self._build("abcdef")
        lengths = sa.match_repeated_length("abcdef")
        # No substring repeats → all lengths should be 0
        assert all(l == 0 for l in lengths)

    def test_match_repeated_length_with_repeat(self):
        """Building on 'aaaa' — every suffix occurs many times."""
        sa = self._build("aaaa")
        lengths = sa.match_repeated_length("aa")
        # At least one position should report a non-zero repeated length
        assert any(l > 0 for l in lengths)

    def test_match_repeated_length_returns_one_per_char(self):
        sa = self._build("hello world hello")
        lengths = sa.match_repeated_length("hello")
        assert len(lengths) == len("hello")

    def test_propagate_counts_increases_root_count(self):
        sa = self._build("abab")
        # Root state should have the total count propagated
        assert sa.states[0].cnt >= 1


# ---------------------------------------------------------------------------
# _turn_unique_ratio
# ---------------------------------------------------------------------------

class TestTurnUniqueRatio:
    def test_empty_text_returns_one(self):
        sa = SuffixAutomaton()
        sa.propagate_counts()
        assert _turn_unique_ratio(sa, "") == 1.0

    def test_unique_text_returns_one(self):
        """Text that does not repeat in the automaton should score 1.0."""
        sa = SuffixAutomaton()
        for i, c in enumerate("abcdef"):
            sa.extend(c, i)
        sa.propagate_counts()
        ratio = _turn_unique_ratio(sa, "xyz")
        assert ratio == 1.0

    def test_ratio_between_zero_and_one(self):
        turns = [_system(0, "hello world"), _system(1, "hello world hello")]
        sa, _ = _build_automaton(turns)
        text = "hello world"
        ratio = _turn_unique_ratio(sa, text, min_repeat_len=5)
        assert 0.0 <= ratio <= 1.0

    def test_completely_unique_text_high_ratio(self):
        """A unique turn should have a high unique ratio."""
        turns = [_system(0, "unique content xyz123")]
        sa, _ = _build_automaton(turns)
        ratio = _turn_unique_ratio(sa, "completely different", min_repeat_len=5)
        assert ratio == 1.0

    def test_highly_repeated_text_low_ratio(self):
        """A turn that just duplicates a very long string should score low."""
        repeated = "x" * 200
        turns = [_system(0, repeated), _system(1, repeated)]
        sa, _ = _build_automaton(turns)
        ratio = _turn_unique_ratio(sa, repeated, min_repeat_len=64)
        assert ratio < 0.5


# ---------------------------------------------------------------------------
# _build_automaton
# ---------------------------------------------------------------------------

class TestBuildAutomaton:
    def test_returns_automaton_and_spans(self):
        turns = [_system(0, "hello"), _system(1, "world")]
        sa, spans = _build_automaton(turns)
        assert isinstance(sa, SuffixAutomaton)
        assert 0 in spans
        assert 1 in spans

    def test_spans_cover_text_length(self):
        turns = [_system(0, "abc"), _system(1, "defgh")]
        _, spans = _build_automaton(turns)
        start0, end0 = spans[0]
        start1, end1 = spans[1]
        assert end0 - start0 == 3
        assert end1 - start1 == 5

    def test_empty_turns_list(self):
        sa, spans = _build_automaton([])
        assert spans == {}

    def test_single_turn(self):
        turns = [_system(0, "test")]
        sa, spans = _build_automaton(turns)
        assert 0 in spans


# ---------------------------------------------------------------------------
# dedup_scores
# ---------------------------------------------------------------------------

class TestDedupScores:
    def test_returns_list_of_scored_turns(self):
        turns = [_user(0, "q"), _system(1, "answer")]
        system_turns = [turns[1]]
        token_counts = {0: 5, 1: 10}
        result = dedup_scores(turns, system_turns, token_counts)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ScoredTurn)

    def test_length_matches_system_turns(self):
        turns = [_user(0, "u")] + [_system(i, f"s{i}") for i in range(1, 4)]
        system_turns = turns[1:]
        token_counts = {i: 50 for i in range(4)}
        result = dedup_scores(turns, system_turns, token_counts)
        assert len(result) == 3

    def test_scores_in_range_0_to_1(self):
        turns = [_user(0, "q"), _system(1, "unique answer"), _system(2, "unique answer")]
        system_turns = turns[1:]
        token_counts = {i: 20 for i in range(3)}
        result = dedup_scores(turns, system_turns, token_counts)
        for st in result:
            assert 0.0 <= st.score <= 1.0

    def test_token_counts_assigned(self):
        turns = [_user(0, "q"), _system(1, "answer")]
        system_turns = [turns[1]]
        result = dedup_scores(turns, system_turns, {0: 5, 1: 42})
        assert result[0].tokens == 42

    def test_missing_token_count_defaults_to_zero(self):
        turns = [_user(0, "q"), _system(1, "answer")]
        system_turns = [turns[1]]
        result = dedup_scores(turns, system_turns, {})
        assert result[0].tokens == 0

    def test_empty_system_turns_returns_empty(self):
        turns = [_user(0, "q")]
        result = dedup_scores(turns, [], {})
        assert result == []

    def test_duplicate_turn_scores_lower_than_unique(self):
        """A turn that repeats a long block should score lower than a unique one."""
        repeated = "x" * 200
        unique = "".join(chr(ord("a") + i % 26) for i in range(200))
        turns = [
            _system(0, repeated),
            _system(1, repeated),
            _system(2, unique),
        ]
        token_counts = {i: 50 for i in range(3)}
        result = dedup_scores(turns, turns, token_counts, min_repeat_len=64)
        score_map = {st.turn.index: st.score for st in result}
        # The unique turn should score higher than the repeated turns
        assert score_map[2] >= score_map[0]
