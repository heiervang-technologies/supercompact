"""Tests for lib/types.py (ScoredTurn, build_query, random_scores) and
lib/eval/probes.py (Probe, ProbeSet, _format_turns_for_prompt)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.types import ScoredTurn, build_query, random_scores
from lib.eval.probes import Probe, ProbeSet, _format_turns_for_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(text: str, kind: str = "user", index: int = 0) -> Turn:
    t = Turn(kind=kind, index=index)
    t.append({"message": {"content": text}})
    return t


# ---------------------------------------------------------------------------
# ScoredTurn dataclass
# ---------------------------------------------------------------------------

class TestScoredTurn:
    def test_fields_stored(self):
        t = _turn("hello")
        st = ScoredTurn(turn=t, score=0.75, tokens=120)
        assert st.turn is t
        assert st.score == pytest.approx(0.75)
        assert st.tokens == 120


# ---------------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def test_single_user_turn(self):
        turns = [_turn("what is the bug?")]
        result = build_query(turns)
        assert "what is the bug?" in result

    def test_uses_last_three_turns(self):
        turns = [_turn(f"msg{i}") for i in range(5)]
        result = build_query(turns)
        # Only last 3 should appear
        assert "msg4" in result
        assert "msg3" in result
        assert "msg2" in result
        # Earlier messages excluded
        assert "msg0" not in result
        assert "msg1" not in result

    def test_fewer_than_three_returns_all(self):
        turns = [_turn("first"), _turn("second")]
        result = build_query(turns)
        assert "first" in result
        assert "second" in result

    def test_truncated_at_max_chars(self):
        big_text = "x" * 5000
        turns = [_turn(big_text)]
        result = build_query(turns, max_chars=500)
        assert len(result) <= 500

    def test_empty_turns_returns_empty(self):
        result = build_query([])
        assert result == ""

    def test_parts_separated_by_divider(self):
        turns = [_turn("alpha"), _turn("beta"), _turn("gamma")]
        result = build_query(turns)
        assert "---" in result


# ---------------------------------------------------------------------------
# random_scores
# ---------------------------------------------------------------------------

class TestRandomScores:
    def test_returns_one_per_turn(self):
        turns = [_turn("a", kind="system", index=i) for i in range(4)]
        token_counts = {0: 100, 1: 200, 2: 150, 3: 50}
        scored = random_scores(turns, token_counts)
        assert len(scored) == 4

    def test_returns_scored_turns(self):
        turns = [_turn("hello", kind="system", index=0)]
        scored = random_scores(turns, {0: 100})
        assert isinstance(scored[0], ScoredTurn)

    def test_scores_between_zero_and_one(self):
        turns = [_turn("x", kind="system", index=i) for i in range(10)]
        token_counts = {i: 50 for i in range(10)}
        scored = random_scores(turns, token_counts)
        for st in scored:
            assert 0.0 <= st.score <= 1.0

    def test_tokens_from_token_counts(self):
        t = _turn("hello", kind="system", index=7)
        scored = random_scores([t], {7: 333})
        assert scored[0].tokens == 333

    def test_missing_token_count_defaults_to_zero(self):
        t = _turn("hello", kind="system", index=99)
        scored = random_scores([t], {})
        assert scored[0].tokens == 0


# ---------------------------------------------------------------------------
# Probe dataclass
# ---------------------------------------------------------------------------

class TestProbeDataclass:
    def test_fields_stored(self):
        p = Probe(
            id="esr_001",
            dimension="error_solution",
            tier="factual",
            question="What caused the error?",
            gold_answer="Missing import",
        )
        assert p.id == "esr_001"
        assert p.dimension == "error_solution"
        assert p.tier == "factual"
        assert p.question == "What caused the error?"
        assert p.gold_answer == "Missing import"

    def test_default_evidence_turns_empty(self):
        p = Probe(id="p1", dimension="progress", tier="factual",
                  question="Q?", gold_answer="A")
        assert p.evidence_turns == []

    def test_default_difficulty_medium(self):
        p = Probe(id="p1", dimension="progress", tier="factual",
                  question="Q?", gold_answer="A")
        assert p.difficulty == "medium"


# ---------------------------------------------------------------------------
# ProbeSet dataclass
# ---------------------------------------------------------------------------

class TestProbeSet:
    def test_defaults(self):
        ps = ProbeSet()
        assert ps.probes == []
        assert ps.conv_hash == ""
        assert ps.split_ratio == pytest.approx(0.70)
        assert ps.version == "1"

    def test_to_dict_structure(self):
        p = Probe(id="p1", dimension="progress", tier="factual",
                  question="Q?", gold_answer="A")
        ps = ProbeSet(probes=[p], conv_hash="abc123", split_ratio=0.7, version="1")
        d = ps.to_dict()
        assert d["conv_hash"] == "abc123"
        assert d["split_ratio"] == pytest.approx(0.7)
        assert d["version"] == "1"
        assert len(d["probes"]) == 1
        assert d["probes"][0]["id"] == "p1"

    def test_from_dict_roundtrip(self):
        p = Probe(id="p1", dimension="instruction", tier="comprehension",
                  question="Why?", gold_answer="Because",
                  evidence_turns=[0, 2], difficulty="hard")
        ps = ProbeSet(probes=[p], conv_hash="xyz", split_ratio=0.8, version="2")
        d = ps.to_dict()
        restored = ProbeSet.from_dict(d)
        assert restored.conv_hash == "xyz"
        assert restored.split_ratio == pytest.approx(0.8)
        assert len(restored.probes) == 1
        rp = restored.probes[0]
        assert rp.id == "p1"
        assert rp.dimension == "instruction"
        assert rp.difficulty == "hard"
        assert rp.evidence_turns == [0, 2]

    def test_from_dict_missing_optional_fields(self):
        data = {"probes": []}
        ps = ProbeSet.from_dict(data)
        assert ps.probes == []
        assert ps.conv_hash == ""
        assert ps.split_ratio == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# _format_turns_for_prompt
# ---------------------------------------------------------------------------

class TestFormatTurnsForPrompt:
    def test_single_turn_included(self):
        t = _turn("hello world", kind="user", index=0)
        result = _format_turns_for_prompt([t])
        assert "hello world" in result

    def test_turn_header_included(self):
        t = _turn("content", kind="user", index=5)
        result = _format_turns_for_prompt([t])
        assert "Turn 5" in result
        assert "user" in result

    def test_multiple_turns_ordered(self):
        turns = [_turn(f"msg{i}", index=i) for i in range(3)]
        result = _format_turns_for_prompt(turns)
        # All messages should appear
        for i in range(3):
            assert f"msg{i}" in result

    def test_truncated_at_max_chars(self):
        big_text = "y" * 200_000
        t = _turn(big_text, index=0)
        result = _format_turns_for_prompt([t], max_chars=1000)
        assert "truncated" in result

    def test_empty_list_returns_empty(self):
        result = _format_turns_for_prompt([])
        assert result == ""
