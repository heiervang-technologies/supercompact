"""Tests for lib/eval/probes.py — Probe and ProbeSet serialization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.probes import Probe, ProbeSet, DIMENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(pid: str = "p1", dimension: str = "error_solution") -> Probe:
    return Probe(
        id=pid,
        dimension=dimension,
        tier="factual",
        question="What was the error?",
        gold_answer="ValueError at /src/app.py",
        evidence_turns=[1, 3],
        difficulty="medium",
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class TestProbe:
    def test_fields_accessible(self):
        p = _probe("abc", "instruction")
        assert p.id == "abc"
        assert p.dimension == "instruction"
        assert p.tier == "factual"

    def test_default_difficulty(self):
        p = Probe(id="x", dimension="noise", tier="factual", question="Q", gold_answer="A")
        assert p.difficulty == "medium"

    def test_default_evidence_turns_empty(self):
        p = Probe(id="x", dimension="noise", tier="factual", question="Q", gold_answer="A")
        assert p.evidence_turns == []

    def test_evidence_turns_set(self):
        p = _probe()
        assert p.evidence_turns == [1, 3]


# ---------------------------------------------------------------------------
# DIMENSIONS constant
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_expected_keys(self):
        expected = {"error_solution", "instruction", "progress", "environment", "noise"}
        assert set(DIMENSIONS.keys()) == expected

    def test_weights_sum_to_1(self):
        assert abs(sum(DIMENSIONS.values()) - 1.0) < 1e-9

    def test_all_weights_positive(self):
        assert all(w > 0 for w in DIMENSIONS.values())


# ---------------------------------------------------------------------------
# ProbeSet.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestProbeSetRoundTrip:
    def test_to_dict_structure(self):
        ps = ProbeSet(probes=[_probe()], conv_hash="abc123", split_ratio=0.7, version="2")
        d = ps.to_dict()
        assert "conv_hash" in d
        assert "split_ratio" in d
        assert "version" in d
        assert "probes" in d

    def test_probes_serialized(self):
        ps = ProbeSet(probes=[_probe("p1"), _probe("p2")])
        d = ps.to_dict()
        assert len(d["probes"]) == 2

    def test_from_dict_restores_probes(self):
        original = ProbeSet(probes=[_probe("p1"), _probe("p2")])
        d = original.to_dict()
        restored = ProbeSet.from_dict(d)
        assert len(restored.probes) == 2

    def test_from_dict_preserves_probe_fields(self):
        probe = _probe("myid", "instruction")
        ps = ProbeSet(probes=[probe])
        d = ps.to_dict()
        restored = ProbeSet.from_dict(d)
        rp = restored.probes[0]
        assert rp.id == "myid"
        assert rp.dimension == "instruction"
        assert rp.tier == probe.tier
        assert rp.question == probe.question
        assert rp.gold_answer == probe.gold_answer

    def test_from_dict_preserves_conv_hash(self):
        ps = ProbeSet(conv_hash="deadbeef")
        d = ps.to_dict()
        restored = ProbeSet.from_dict(d)
        assert restored.conv_hash == "deadbeef"

    def test_from_dict_preserves_split_ratio(self):
        ps = ProbeSet(split_ratio=0.8)
        d = ps.to_dict()
        restored = ProbeSet.from_dict(d)
        assert abs(restored.split_ratio - 0.8) < 1e-9

    def test_from_dict_preserves_version(self):
        ps = ProbeSet(version="3")
        d = ps.to_dict()
        restored = ProbeSet.from_dict(d)
        assert restored.version == "3"

    def test_from_dict_missing_conv_hash_defaults(self):
        d = {"probes": [], "split_ratio": 0.7, "version": "1"}
        ps = ProbeSet.from_dict(d)
        assert ps.conv_hash == ""

    def test_from_dict_missing_split_ratio_defaults(self):
        d = {"probes": [], "conv_hash": "abc", "version": "1"}
        ps = ProbeSet.from_dict(d)
        assert abs(ps.split_ratio - 0.70) < 1e-9

    def test_empty_probes_round_trip(self):
        ps = ProbeSet(probes=[])
        restored = ProbeSet.from_dict(ps.to_dict())
        assert restored.probes == []

    def test_evidence_turns_preserved(self):
        probe = Probe(
            id="p1", dimension="error_solution", tier="factual",
            question="Q", gold_answer="A", evidence_turns=[2, 5, 7],
        )
        ps = ProbeSet(probes=[probe])
        restored = ProbeSet.from_dict(ps.to_dict())
        assert restored.probes[0].evidence_turns == [2, 5, 7]

    def test_difficulty_preserved(self):
        probe = Probe(
            id="p1", dimension="noise", tier="comprehension",
            question="Q", gold_answer="A", difficulty="hard",
        )
        ps = ProbeSet(probes=[probe])
        restored = ProbeSet.from_dict(ps.to_dict())
        assert restored.probes[0].difficulty == "hard"
