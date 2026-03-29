"""Tests for lib/eval/probes.py — Probe, ProbeSet, _format_turns_for_prompt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.probes import Probe, ProbeSet, _format_turns_for_prompt
from lib.parser import Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(index: int, text: str, kind: str = "system") -> Turn:
    t = Turn(kind=kind, index=index)
    t.lines = [{"message": {"content": text}}]
    return t


def _probe(**kwargs) -> Probe:
    defaults = dict(
        id="esr_001",
        dimension="error_solution",
        tier="factual",
        question="What error occurred?",
        gold_answer="ValueError on line 42",
        evidence_turns=[1, 2],
        difficulty="medium",
    )
    defaults.update(kwargs)
    return Probe(**defaults)


# ---------------------------------------------------------------------------
# Probe dataclass
# ---------------------------------------------------------------------------

class TestProbe:
    def test_defaults(self):
        p = Probe(
            id="x",
            dimension="progress",
            tier="factual",
            question="Q?",
            gold_answer="A",
        )
        assert p.evidence_turns == []
        assert p.difficulty == "medium"

    def test_stores_fields(self):
        p = _probe(id="env_001", difficulty="hard")
        assert p.id == "env_001"
        assert p.difficulty == "hard"


# ---------------------------------------------------------------------------
# ProbeSet.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestProbeSetRoundtrip:
    def test_empty_probeset_roundtrip(self):
        ps = ProbeSet(probes=[], conv_hash="abc123", split_ratio=0.70, version="2")
        d = ps.to_dict()
        ps2 = ProbeSet.from_dict(d)
        assert ps2.conv_hash == "abc123"
        assert ps2.split_ratio == 0.70
        assert ps2.version == "2"
        assert ps2.probes == []

    def test_probeset_with_probes_roundtrip(self):
        p = _probe()
        ps = ProbeSet(probes=[p], conv_hash="xyz", split_ratio=0.8, version="1")
        d = ps.to_dict()
        ps2 = ProbeSet.from_dict(d)
        assert len(ps2.probes) == 1
        assert ps2.probes[0].id == "esr_001"
        assert ps2.probes[0].dimension == "error_solution"

    def test_evidence_turns_preserved(self):
        p = _probe(evidence_turns=[3, 5, 7])
        ps = ProbeSet(probes=[p])
        ps2 = ProbeSet.from_dict(ps.to_dict())
        assert ps2.probes[0].evidence_turns == [3, 5, 7]

    def test_difficulty_preserved(self):
        p = _probe(difficulty="hard")
        ps = ProbeSet(probes=[p])
        ps2 = ProbeSet.from_dict(ps.to_dict())
        assert ps2.probes[0].difficulty == "hard"

    def test_to_dict_has_required_keys(self):
        ps = ProbeSet()
        d = ps.to_dict()
        assert "conv_hash" in d
        assert "split_ratio" in d
        assert "version" in d
        assert "probes" in d

    def test_multiple_probes_preserved(self):
        probes = [_probe(id=f"p{i}") for i in range(5)]
        ps = ProbeSet(probes=probes)
        ps2 = ProbeSet.from_dict(ps.to_dict())
        assert len(ps2.probes) == 5
        ids = {p.id for p in ps2.probes}
        assert ids == {"p0", "p1", "p2", "p3", "p4"}

    def test_missing_optional_fields_use_defaults(self):
        """from_dict with minimal data fills defaults."""
        data = {
            "probes": [],
        }
        ps = ProbeSet.from_dict(data)
        assert ps.conv_hash == ""
        assert ps.split_ratio == 0.70
        assert ps.version == "1"


# ---------------------------------------------------------------------------
# _format_turns_for_prompt
# ---------------------------------------------------------------------------

class TestFormatTurnsForPrompt:
    def test_empty_turns_returns_empty(self):
        result = _format_turns_for_prompt([])
        assert result == ""

    def test_single_turn_contains_text(self):
        t = _turn(0, "hello world")
        result = _format_turns_for_prompt([t])
        assert "hello world" in result

    def test_includes_turn_index(self):
        t = _turn(5, "some content")
        result = _format_turns_for_prompt([t])
        assert "5" in result

    def test_includes_turn_kind(self):
        t = _turn(0, "text", kind="user")
        result = _format_turns_for_prompt([t])
        assert "user" in result

    def test_truncates_at_max_chars(self):
        turns = [_turn(i, "x" * 1000) for i in range(10)]
        result = _format_turns_for_prompt(turns, max_chars=500)
        assert "truncated" in result
        assert len(result) < 2000

    def test_short_content_not_truncated(self):
        t = _turn(0, "short text")
        result = _format_turns_for_prompt([t], max_chars=10000)
        assert "truncated" not in result
        assert "short text" in result

    def test_multiple_turns_all_appear_if_under_limit(self):
        turns = [_turn(i, f"msg{i}") for i in range(3)]
        result = _format_turns_for_prompt(turns, max_chars=10000)
        for i in range(3):
            assert f"msg{i}" in result
