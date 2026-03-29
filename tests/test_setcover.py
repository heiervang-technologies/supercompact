"""Tests for lib/setcover.py — enhanced EITF scorer with exclusivity bonus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.setcover import setcover_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(index: int, text: str = "") -> Turn:
    """Build a system turn with optional text content."""
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

class TestSetcoverScores:
    def test_empty_returns_empty(self):
        result = setcover_scores([], [], {})
        assert result == []

    def test_returns_one_scored_turn_per_system_turn(self):
        u = _user(0, "hello")
        s = _system(1, "ValueError at /src/app.py")
        turns = [u, s]
        token_counts = _token_counts(turns)
        result = setcover_scores(turns, [s], token_counts)
        assert len(result) == 1

    def test_scores_normalized_to_0_1(self):
        u = _user(0, "fix the bug")
        s1 = _system(1, "ValueError at /src/foo.py:42")
        s2 = _system(2, "ModuleNotFoundError at /src/bar.py:10")
        s3 = _system(3, "the answer is 42")
        turns = [u, s1, s2, s3]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s1, s2, s3], tc)
        for st in results:
            assert 0.0 <= st.score <= 1.0, f"Score {st.score} out of [0,1]"

    def test_max_score_is_1(self):
        u = _user(0, "question")
        s1 = _system(1, "ValueError in /src/main.py running pip install requests")
        s2 = _system(2, "plain response")
        turns = [u, s1, s2]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s1, s2], tc)
        max_score = max(st.score for st in results)
        assert abs(max_score - 1.0) < 1e-9

    def test_entity_rich_turn_scores_higher(self):
        u = _user(0, "debug this")
        # s1 has many entities: file path, exception, package, URL
        s1 = _system(1,
            "ValueError at /src/app.py:42 — run pip install requests "
            "See https://docs.python.org for details"
        )
        # s2 has no extractable entities
        s2 = _system(2, "I see. Let me think about that.")
        turns = [u, s1, s2]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s1, s2], tc)
        score_map = {st.turn.index: st.score for st in results}
        assert score_map[s1.index] > score_map[s2.index]

    def test_exclusive_entity_turn_scores_higher_than_shared(self):
        """A turn with an entity unique to it should score >= one with shared entities."""
        u = _user(0, "help me debug")
        # s1 mentions a unique exception (only in s1)
        s1 = _system(1, "AttributeError at /project/unique_module.py line 99")
        # s2 mentions the same exception as s1 AND s3 — shared
        s2 = _system(2, "ValueError at /project/shared.py — also in s3")
        s3 = _system(3, "ValueError at /project/shared.py again here too")
        turns = [u, s1, s2, s3]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s1, s2, s3], tc)
        score_map = {st.turn.index: st.score for st in results}
        # s1's unique AttributeError should earn exclusivity bonus
        # so it should score at least as well as s2 (which shares its entities with s3)
        assert score_map[s1.index] >= score_map[s2.index]

    def test_tokens_stored_in_result(self):
        u = _user(0, "question")
        s = _system(1, "ValueError at /src/app.py")
        turns = [u, s]
        tc = {0: 50, 1: 300}
        results = setcover_scores(turns, [s], tc)
        assert results[0].tokens == 300

    def test_turn_reference_preserved(self):
        u = _user(0, "question")
        s = _system(1, "some content")
        turns = [u, s]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s], tc)
        assert results[0].turn is s

    def test_single_system_turn_scores_1(self):
        """With only one system turn it should normalize to exactly 1.0."""
        u = _user(0, "question")
        s = _system(1, "ValueError at /src/app.py")
        turns = [u, s]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s], tc)
        assert abs(results[0].score - 1.0) < 1e-9

    def test_all_zero_score_still_normalizes(self):
        """Turns with no entities get score 0 — should not crash."""
        u = _user(0, "hi")
        s1 = _system(1, "hello")
        s2 = _system(2, "world")
        turns = [u, s1, s2]
        tc = _token_counts(turns)
        results = setcover_scores(turns, [s1, s2], tc)
        assert len(results) == 2
        for st in results:
            assert 0.0 <= st.score <= 1.0
