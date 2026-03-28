"""Tests for lib/dedup.py — suffix automaton dedup scorer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.dedup import SuffixAutomaton, dedup_scores


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


def _token_counts(turns: list[Turn], default: int = 100) -> dict[int, int]:
    return {t.index: default for t in turns}


# Long string above min_repeat_len threshold (default 64)
_LONG_REPEATED = "A" * 100


# ---------------------------------------------------------------------------
# SuffixAutomaton unit tests
# ---------------------------------------------------------------------------

class TestSuffixAutomaton:
    def test_empty_automaton_has_initial_state(self):
        sa = SuffixAutomaton()
        assert len(sa.states) == 1

    def test_extend_adds_states(self):
        sa = SuffixAutomaton()
        sa.extend("a", 0)
        assert len(sa.states) == 2

    def test_propagate_counts_no_crash(self):
        sa = SuffixAutomaton()
        for i, c in enumerate("hello world"):
            sa.extend(c, i)
        sa.propagate_counts()

    def test_unique_text_match_length_all_zero(self):
        """Text where every character is distinct → no repeated substrings → all lengths zero."""
        sa = SuffixAutomaton()
        # All distinct chars → every cnt is exactly 1
        text = "abcdefghij"
        for i, c in enumerate(text):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length(text)
        assert all(length == 0 for length in lengths)

    def test_repeated_text_yields_positive_lengths(self):
        """Same chunk appended twice (with separator) → second copy shows repetition."""
        chunk = "x" * 80
        sa = SuffixAutomaton()
        full = chunk + "\x00" + chunk
        for i, c in enumerate(full):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length(chunk)
        assert any(l > 0 for l in lengths)

    def test_separator_prevents_cross_turn_span(self):
        """Null-byte separator: no substring should span across the boundary."""
        sa = SuffixAutomaton()
        t1 = "hello"
        t2 = "world"
        full = t1 + "\x00" + t2
        for i, c in enumerate(full):
            sa.extend(c, i)
        sa.propagate_counts()
        # "helloworld" would span the separator — no 10-char match should appear
        lengths = sa.match_repeated_length("helloworld")
        assert all(l < 10 for l in lengths)

    def test_match_repeated_length_returns_one_per_char(self):
        sa = SuffixAutomaton()
        text = "abcabc"
        for i, c in enumerate(text):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length(text)
        assert len(lengths) == len(text)


# ---------------------------------------------------------------------------
# dedup_scores tests
# ---------------------------------------------------------------------------

class TestDedupScores:
    def test_empty_returns_empty(self):
        result = dedup_scores([], [], {})
        assert result == []

    def test_returns_one_scored_turn_per_system_turn(self):
        u = _user(0, "question")
        s = _system(1, "unique answer here")
        turns = [u, s]
        result = dedup_scores(turns, [s], _token_counts(turns))
        assert len(result) == 1

    def test_scores_in_range_0_to_1(self):
        u = _user(0, "question")
        s1 = _system(1, "first answer with some distinct content")
        s2 = _system(2, _LONG_REPEATED)
        s3 = _system(3, _LONG_REPEATED)
        turns = [u, s1, s2, s3]
        results = dedup_scores(turns, [s1, s2, s3], _token_counts(turns))
        for st in results:
            assert 0.0 <= st.score <= 1.0, f"Score {st.score} out of [0,1]"

    def test_unique_turn_scores_higher_than_duplicate(self):
        """A turn with unique content should score higher than one with repeated content."""
        u = _user(0, "help")
        unique_text = "completely_unique_content_that_appears_nowhere_else_in_this_conversation"
        s_unique = _system(1, unique_text)
        s_dup1 = _system(2, _LONG_REPEATED)
        s_dup2 = _system(3, _LONG_REPEATED)
        turns = [u, s_unique, s_dup1, s_dup2]
        results = dedup_scores(turns, [s_unique, s_dup1, s_dup2], _token_counts(turns))
        score_map = {st.turn.index: st.score for st in results}
        assert score_map[s_unique.index] > score_map[s_dup1.index]

    def test_empty_turn_text_scores_1(self):
        """Turn with no extractable text → unique ratio defaults to 1.0."""
        u = _user(0, "question")
        s = _system(1)  # no text
        turns = [u, s]
        results = dedup_scores(turns, [s], _token_counts(turns))
        assert abs(results[0].score - 1.0) < 1e-9

    def test_tokens_stored_in_result(self):
        u = _user(0, "question")
        s = _system(1, "some answer text")
        turns = [u, s]
        tc = {0: 50, 1: 300}
        results = dedup_scores(turns, [s], tc)
        assert results[0].tokens == 300

    def test_turn_reference_preserved(self):
        u = _user(0, "question")
        s = _system(1, "some content")
        turns = [u, s]
        results = dedup_scores(turns, [s], _token_counts(turns))
        assert results[0].turn is s

    def test_identical_turns_both_score_below_1(self):
        """Two turns with identical long content → both penalized (score < 1.0)."""
        u = _user(0, "question")
        s1 = _system(1, _LONG_REPEATED)
        s2 = _system(2, _LONG_REPEATED)
        turns = [u, s1, s2]
        results = dedup_scores(turns, [s1, s2], _token_counts(turns))
        score_map = {st.turn.index: st.score for st in results}
        assert score_map[s1.index] < 1.0
        assert score_map[s2.index] < 1.0

    def test_short_repeats_below_threshold_not_penalized(self):
        """Common substring shorter than min_repeat_len should not lower the score."""
        u = _user(0, "hi")
        # Both turns share a short prefix (< 64 chars) but have distinct long tails
        # Using natural-language-like sentences avoids internal self-repetitions
        shared = "Note: "
        s1 = _system(1, shared + "The compiler reported a type mismatch in module Alpha at line forty two")
        s2 = _system(2, shared + "Deployment succeeded on staging environment with zero regressions found")
        turns = [u, s1, s2]
        results = dedup_scores(turns, [s1, s2], _token_counts(turns), min_repeat_len=64)
        for st in results:
            assert st.score > 0.9, f"Short common prefix caused excessive penalty: {st.score}"

    def test_min_repeat_len_controls_penalization(self):
        """min_repeat_len=1 penalizes even single-char repeats; high threshold does not."""
        chunk = "a" * 50
        u = _user(0, "question")
        s1 = _system(1, chunk)
        s2 = _system(2, chunk)
        turns = [u, s1, s2]

        results_tight = dedup_scores(turns, [s1, s2], _token_counts(turns), min_repeat_len=1)
        results_loose = dedup_scores(turns, [s1, s2], _token_counts(turns), min_repeat_len=200)

        for st in results_tight:
            assert st.score < 1.0, "min_repeat_len=1 should penalize repeated chunk"
        for st in results_loose:
            assert abs(st.score - 1.0) < 1e-9, "min_repeat_len > chunk size should not penalize"

    def test_ordering_preserved(self):
        """Output order should match the order of system_turns input."""
        u = _user(0, "question")
        s1 = _system(1, "first")
        s2 = _system(2, "second")
        s3 = _system(3, "third")
        turns = [u, s1, s2, s3]
        results = dedup_scores(turns, [s3, s1, s2], _token_counts(turns))
        assert [st.turn.index for st in results] == [3, 1, 2]
