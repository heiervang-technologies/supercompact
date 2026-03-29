"""Tests for lib/eval/evidence_coverage.py — dataclasses, _dcg, compute_evidence_coverage."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.evidence_coverage import (
    DIFFICULTY_WEIGHTS,
    DimensionCoverage,
    EvidenceCoverageResult,
    ProbeCoverage,
    _dcg,
    compute_evidence_coverage,
)
from lib.eval.probes import Probe, ProbeSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(id="p1", dimension="progress", difficulty="medium",
           evidence_turns=None) -> Probe:
    return Probe(
        id=id,
        dimension=dimension,
        tier="factual",
        question="Q?",
        gold_answer="A",
        difficulty=difficulty,
        evidence_turns=evidence_turns or [],
    )


def _probe_set(*probes) -> ProbeSet:
    return ProbeSet(probes=list(probes))


# ---------------------------------------------------------------------------
# DIFFICULTY_WEIGHTS constant
# ---------------------------------------------------------------------------

class TestDifficultyWeights:
    def test_easy_is_one(self):
        assert DIFFICULTY_WEIGHTS["easy"] == 1.0

    def test_medium_is_two(self):
        assert DIFFICULTY_WEIGHTS["medium"] == 2.0

    def test_hard_is_three(self):
        assert DIFFICULTY_WEIGHTS["hard"] == 3.0


# ---------------------------------------------------------------------------
# ProbeCoverage dataclass
# ---------------------------------------------------------------------------

class TestProbeCoverage:
    def test_fields_stored(self):
        pc = ProbeCoverage(
            probe_id="p1",
            dimension="progress",
            difficulty="medium",
            evidence_turns=[0, 1, 2],
            kept_evidence=[0, 1],
            dropped_evidence=[2],
            coverage=2/3,
        )
        assert pc.probe_id == "p1"
        assert pc.dimension == "progress"
        assert pc.evidence_turns == [0, 1, 2]
        assert pc.kept_evidence == [0, 1]
        assert pc.dropped_evidence == [2]
        assert pc.coverage == pytest.approx(2/3)


# ---------------------------------------------------------------------------
# DimensionCoverage dataclass
# ---------------------------------------------------------------------------

class TestDimensionCoverage:
    def test_fields_stored(self):
        dc = DimensionCoverage(
            dimension="instruction",
            weight=0.25,
            mean_coverage=0.8,
            probe_count=4,
            coverages=[0.5, 0.75, 1.0, 0.9],
        )
        assert dc.dimension == "instruction"
        assert dc.mean_coverage == pytest.approx(0.8)
        assert dc.probe_count == 4

    def test_default_coverages_empty(self):
        dc = DimensionCoverage(dimension="noise", weight=0.05, mean_coverage=0.0, probe_count=0)
        assert dc.coverages == []


# ---------------------------------------------------------------------------
# EvidenceCoverageResult dataclass
# ---------------------------------------------------------------------------

class TestEvidenceCoverageResult:
    def test_defaults(self):
        ecr = EvidenceCoverageResult(method="dedup", budget=80_000)
        assert ecr.composite == 0.0
        assert ecr.ndcg == 0.0
        assert ecr.dimensions == []
        assert ecr.probe_details == []

    def test_dimension_map_property(self):
        dc = DimensionCoverage(dimension="progress", weight=0.25, mean_coverage=0.7, probe_count=2)
        ecr = EvidenceCoverageResult(method="dedup", budget=80_000, dimensions=[dc])
        assert "progress" in ecr.dimension_map
        assert ecr.dimension_map["progress"] is dc

    def test_to_dict_has_required_keys(self):
        ecr = EvidenceCoverageResult(method="dedup", budget=80_000)
        d = ecr.to_dict()
        for key in ("method", "budget", "composite", "ndcg", "dimensions", "probe_details"):
            assert key in d

    def test_to_dict_method_budget(self):
        ecr = EvidenceCoverageResult(method="eitf", budget=40_000)
        d = ecr.to_dict()
        assert d["method"] == "eitf"
        assert d["budget"] == 40_000


# ---------------------------------------------------------------------------
# _dcg
# ---------------------------------------------------------------------------

class TestDcg:
    def test_empty_returns_zero(self):
        assert _dcg([]) == pytest.approx(0.0)

    def test_single_item(self):
        result = _dcg([(1.0, 2.0)])
        assert result == pytest.approx(2.0 / math.log2(2))

    def test_zero_score_returns_zero(self):
        assert _dcg([(0.0, 3.0), (0.0, 1.0)]) == pytest.approx(0.0)

    def test_sorted_by_weight_desc(self):
        # Higher weight item should come first (lower denominator)
        hard = (0.5, 3.0)
        easy = (0.5, 1.0)
        result = _dcg([easy, hard])
        expected = (0.5 * 3.0) / math.log2(2) + (0.5 * 1.0) / math.log2(3)
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_evidence_coverage — core paths
# ---------------------------------------------------------------------------

class TestComputeEvidenceCoverageEmpty:
    def test_no_probes_returns_zero_composite(self):
        ps = _probe_set()
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 80_000)
        assert result.composite == pytest.approx(0.0)

    def test_no_probes_returns_ndcg_zero(self):
        ps = _probe_set()
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 80_000)
        assert result.ndcg == pytest.approx(0.0)

    def test_no_probes_returns_no_probe_details(self):
        ps = _probe_set()
        result = compute_evidence_coverage(ps, set(), "dedup", 80_000)
        assert result.probe_details == []

    def test_probe_without_evidence_turns_is_skipped(self):
        ps = _probe_set(_probe("p1", evidence_turns=[]))
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 80_000)
        assert result.probe_details == []

    def test_method_and_budget_stored(self):
        ps = _probe_set()
        result = compute_evidence_coverage(ps, set(), "setcover", 60_000)
        assert result.method == "setcover"
        assert result.budget == 60_000


class TestComputeEvidenceCoverageFullCoverage:
    def test_all_evidence_kept_gives_coverage_one(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[0, 1, 2]))
        result = compute_evidence_coverage(ps, {0, 1, 2}, "dedup", 80_000)
        assert result.probe_details[0].coverage == pytest.approx(1.0)

    def test_all_evidence_kept_gives_ndcg_one(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[0, 1]))
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 80_000)
        assert result.ndcg == pytest.approx(1.0)

    def test_all_evidence_kept_gives_non_zero_composite(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[5]))
        result = compute_evidence_coverage(ps, {5}, "dedup", 80_000)
        assert result.composite > 0.0


class TestComputeEvidenceCoverageZeroCoverage:
    def test_no_evidence_kept_gives_coverage_zero(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[3, 4]))
        result = compute_evidence_coverage(ps, set(), "dedup", 80_000)
        assert result.probe_details[0].coverage == pytest.approx(0.0)

    def test_no_evidence_kept_ndcg_zero(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[3]))
        result = compute_evidence_coverage(ps, {0, 1, 2}, "dedup", 80_000)
        assert result.ndcg == pytest.approx(0.0)


class TestComputeEvidenceCoveragePartial:
    def test_partial_coverage_value(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[0, 1, 2, 3]))
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 80_000)
        assert result.probe_details[0].coverage == pytest.approx(0.5)

    def test_kept_and_dropped_lists(self):
        ps = _probe_set(_probe("p1", "progress", evidence_turns=[0, 1, 2]))
        result = compute_evidence_coverage(ps, {0, 2}, "dedup", 80_000)
        pc = result.probe_details[0]
        assert set(pc.kept_evidence) == {0, 2}
        assert set(pc.dropped_evidence) == {1}


class TestComputeEvidenceCoverageDimension:
    def test_multiple_probes_mean_coverage(self):
        ps = _probe_set(
            _probe("p1", "progress", evidence_turns=[0, 1]),
            _probe("p2", "progress", evidence_turns=[2, 3]),
        )
        # Keep all for p1 (coverage=1.0), none for p2 (coverage=0.0)
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 80_000)
        progress = result.dimension_map["progress"]
        assert progress.mean_coverage == pytest.approx(0.5)
        assert progress.probe_count == 2

    def test_ndcg_between_zero_and_one_for_partial(self):
        ps = _probe_set(
            _probe("p1", "progress", difficulty="medium", evidence_turns=[0]),
            _probe("p2", "instruction", difficulty="hard", evidence_turns=[1]),
        )
        result = compute_evidence_coverage(ps, {0}, "dedup", 80_000)
        assert 0.0 <= result.ndcg <= 1.0
