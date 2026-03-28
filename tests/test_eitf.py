"""Tests for lib/eitf.py — Entity-frequency Inverse Turn Frequency scorer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.eitf import eitf_scores


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


# ---------------------------------------------------------------------------
# Basic contracts
# ---------------------------------------------------------------------------

class TestEitfScores:
    def test_empty_returns_empty(self):
        result = eitf_scores([], [], {})
        assert result == []

    def test_returns_one_scored_turn_per_system_turn(self):
        u = _user(0, "hello")
        s = _system(1, "ValueError at /src/app.py")
        turns = [u, s]
        result = eitf_scores(turns, [s], _token_counts(turns))
        assert len(result) == 1

    def test_scores_normalized_to_0_1(self):
        u = _user(0, "fix the bug")
        s1 = _system(1, "ValueError at /src/foo.py:42")
        s2 = _system(2, "ModuleNotFoundError at /src/bar.py:10")
        s3 = _system(3, "plain text no entities")
        turns = [u, s1, s2, s3]
        results = eitf_scores(turns, [s1, s2, s3], _token_counts(turns))
        for st in results:
            assert 0.0 <= st.score <= 1.0, f"Score {st.score} out of [0,1]"

    def test_max_score_is_1(self):
        u = _user(0, "question")
        s1 = _system(1, "ValueError in /src/main.py — run pip install requests")
        s2 = _system(2, "plain response")
        turns = [u, s1, s2]
        results = eitf_scores(turns, [s1, s2], _token_counts(turns))
        max_score = max(st.score for st in results)
        assert abs(max_score - 1.0) < 1e-9

    def test_single_system_turn_scores_1(self):
        """With only one system turn it should normalize to exactly 1.0."""
        u = _user(0, "question")
        s = _system(1, "ValueError at /src/app.py")
        turns = [u, s]
        results = eitf_scores(turns, [s], _token_counts(turns))
        assert abs(results[0].score - 1.0) < 1e-9

    def test_entity_rich_turn_scores_higher_than_empty(self):
        u = _user(0, "debug this")
        s_rich = _system(1,
            "ValueError at /src/app.py:42 — run pip install requests "
            "See https://docs.python.org for details"
        )
        s_empty = _system(2, "I see. Let me think about that.")
        turns = [u, s_rich, s_empty]
        results = eitf_scores(turns, [s_rich, s_empty], _token_counts(turns))
        score_map = {st.turn.index: st.score for st in results}
        assert score_map[s_rich.index] > score_map[s_empty.index]

    def test_rare_entity_scores_higher_than_common(self):
        """ITF: entity appearing in fewer turns should yield higher score."""
        u = _user(0, "help")
        # s1 has a unique exception (only in s1) — high ITF
        s1 = _system(1, "AttributeError at /project/unique_module.py line 99")
        # s2 and s3 both have the same entity — low ITF
        s2 = _system(2, "ValueError at /project/common.py — happens often")
        s3 = _system(3, "ValueError at /project/common.py again")
        turns = [u, s1, s2, s3]
        results = eitf_scores(turns, [s1, s2, s3], _token_counts(turns))
        score_map = {st.turn.index: st.score for st in results}
        # s1's unique entity should have higher ITF → higher score
        assert score_map[s1.index] >= score_map[s2.index]

    def test_all_no_entity_turns_do_not_crash(self):
        """All turns with no extractable entities → scores all 0 after normalization."""
        u = _user(0, "hi")
        s1 = _system(1, "hello")
        s2 = _system(2, "world")
        turns = [u, s1, s2]
        results = eitf_scores(turns, [s1, s2], _token_counts(turns))
        assert len(results) == 2
        for st in results:
            assert 0.0 <= st.score <= 1.0

    def test_tokens_stored_in_result(self):
        u = _user(0, "question")
        s = _system(1, "ValueError at /src/app.py")
        turns = [u, s]
        tc = {0: 50, 1: 300}
        results = eitf_scores(turns, [s], tc)
        assert results[0].tokens == 300

    def test_turn_reference_preserved(self):
        u = _user(0, "question")
        s = _system(1, "some content")
        turns = [u, s]
        results = eitf_scores(turns, [s], _token_counts(turns))
        assert results[0].turn is s

    def test_score_uses_length_normalization(self):
        """Longer turns (more tokens) should be penalized relative to shorter ones
        with the same entities (BM25-style length normalization)."""
        u = _user(0, "question")
        # Both turns have the same entity but different token counts
        s_short = _system(1, "ValueError at /src/app.py")
        s_long = _system(2, "ValueError at /src/app.py")
        turns = [u, s_short, s_long]
        # Give s_long 4x more tokens
        tc = {0: 50, 1: 100, 2: 400}
        results = eitf_scores(turns, [s_short, s_long], tc)
        score_map = {st.turn.index: st.score for st in results}
        # s_short (fewer tokens) should score >= s_long (same entity, more tokens)
        assert score_map[s_short.index] >= score_map[s_long.index]
