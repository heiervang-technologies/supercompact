"""Tests for lib/eval/evidence_coverage.py — probe evidence turn coverage."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.evidence_coverage import (
    _dcg,
    compute_evidence_coverage,
    EvidenceCoverageResult,
    DimensionCoverage,
    ProbeCoverage,
    DIFFICULTY_WEIGHTS,
)
from lib.eval.probes import Probe, ProbeSet, DIMENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(
    pid: str,
    dimension: str = "error_solution",
    evidence_turns: list[int] | None = None,
    difficulty: str = "medium",
) -> Probe:
    return Probe(
        id=pid,
        dimension=dimension,
        tier="factual",
        question="Q?",
        gold_answer="A",
        evidence_turns=evidence_turns or [],
        difficulty=difficulty,
    )


def _probe_set(probes: list[Probe]) -> ProbeSet:
    return ProbeSet(probes=probes)


# ---------------------------------------------------------------------------
# _dcg
# ---------------------------------------------------------------------------

class TestDcg:
    def test_empty_returns_zero(self):
        assert _dcg([]) == 0.0

    def test_single_item(self):
        result = _dcg([(1.0, 1.0)])
        expected = (1.0 * 1.0) / math.log2(2)
        assert abs(result - expected) < 1e-9

    def test_sorts_by_weight_descending(self):
        # _dcg sorts internally, so order of input shouldn't matter
        r1 = _dcg([(1.0, 3.0), (1.0, 1.0)])
        r2 = _dcg([(1.0, 1.0), (1.0, 3.0)])
        assert abs(r1 - r2) < 1e-9

    def test_zero_score_gives_zero(self):
        result = _dcg([(0.0, 2.0), (0.0, 1.0)])
        assert result == 0.0

    def test_two_items_formula(self):
        # i=0 → log2(2)=1, i=1 → log2(3)
        # sorted by weight desc: (1.0, 3.0) then (1.0, 1.0)
        expected = (1.0 * 3.0) / math.log2(2) + (1.0 * 1.0) / math.log2(3)
        result = _dcg([(1.0, 3.0), (1.0, 1.0)])
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# compute_evidence_coverage — basic contracts
# ---------------------------------------------------------------------------

class TestComputeEvidenceCoverageBasic:
    def test_returns_result_object(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0, 1])])
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 1000)
        assert isinstance(result, EvidenceCoverageResult)

    def test_method_propagated(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, {0}, "eitf", 1000)
        assert result.method == "eitf"

    def test_budget_propagated(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 4096)
        assert result.budget == 4096

    def test_dimensions_count_matches_known(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        assert len(result.dimensions) == len(DIMENSIONS)

    def test_empty_probe_set_returns_result(self):
        ps = _probe_set([])
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 1000)
        assert isinstance(result, EvidenceCoverageResult)
        assert result.composite == 0.0
        assert result.ndcg == 0.0

    def test_probe_without_evidence_turns_skipped(self):
        # Probe with no evidence_turns should not appear in probe_details
        ps = _probe_set([_probe("p1", evidence_turns=[])])
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 1000)
        assert len(result.probe_details) == 0


# ---------------------------------------------------------------------------
# Coverage calculations
# ---------------------------------------------------------------------------

class TestCoverageCalculations:
    def test_all_evidence_kept_coverage_1(self):
        ps = _probe_set([_probe("p1", evidence_turns=[1, 2, 3])])
        result = compute_evidence_coverage(ps, {1, 2, 3}, "dedup", 1000)
        assert abs(result.probe_details[0].coverage - 1.0) < 1e-9

    def test_no_evidence_kept_coverage_0(self):
        ps = _probe_set([_probe("p1", evidence_turns=[1, 2, 3])])
        result = compute_evidence_coverage(ps, set(), "dedup", 1000)
        assert abs(result.probe_details[0].coverage - 0.0) < 1e-9

    def test_partial_coverage(self):
        ps = _probe_set([_probe("p1", evidence_turns=[1, 2, 3, 4])])
        result = compute_evidence_coverage(ps, {1, 2}, "dedup", 1000)
        assert abs(result.probe_details[0].coverage - 0.5) < 1e-9

    def test_kept_evidence_correct(self):
        ps = _probe_set([_probe("p1", evidence_turns=[1, 2, 3])])
        result = compute_evidence_coverage(ps, {1, 3}, "dedup", 1000)
        pc = result.probe_details[0]
        assert sorted(pc.kept_evidence) == [1, 3]

    def test_dropped_evidence_correct(self):
        ps = _probe_set([_probe("p1", evidence_turns=[1, 2, 3])])
        result = compute_evidence_coverage(ps, {1, 3}, "dedup", 1000)
        pc = result.probe_details[0]
        assert pc.dropped_evidence == [2]

    def test_single_evidence_turn_kept(self):
        ps = _probe_set([_probe("p1", evidence_turns=[5])])
        result = compute_evidence_coverage(ps, {5}, "dedup", 1000)
        assert abs(result.probe_details[0].coverage - 1.0) < 1e-9

    def test_single_evidence_turn_dropped(self):
        ps = _probe_set([_probe("p1", evidence_turns=[5])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        assert abs(result.probe_details[0].coverage - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Dimension aggregation
# ---------------------------------------------------------------------------

class TestDimensionAggregation:
    def test_mean_coverage_one_probe(self):
        ps = _probe_set([_probe("p1", "error_solution", evidence_turns=[0, 1])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        dim = result.dimension_map["error_solution"]
        assert abs(dim.mean_coverage - 0.5) < 1e-9

    def test_mean_coverage_two_probes_same_dim(self):
        ps = _probe_set([
            _probe("p1", "error_solution", evidence_turns=[0, 1]),  # 0.5
            _probe("p2", "error_solution", evidence_turns=[2, 3]),  # 1.0
        ])
        result = compute_evidence_coverage(ps, {0, 2, 3}, "dedup", 1000)
        dim = result.dimension_map["error_solution"]
        assert abs(dim.mean_coverage - 0.75) < 1e-9

    def test_probe_count_correct(self):
        ps = _probe_set([
            _probe("p1", "error_solution", evidence_turns=[0]),
            _probe("p2", "error_solution", evidence_turns=[1]),
            _probe("p3", "error_solution", evidence_turns=[2]),
        ])
        result = compute_evidence_coverage(ps, {0, 1, 2}, "dedup", 1000)
        dim = result.dimension_map["error_solution"]
        assert dim.probe_count == 3

    def test_empty_dimension_probe_count_zero(self):
        ps = _probe_set([_probe("p1", "error_solution", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        # "instruction" dim has no probes
        dim = result.dimension_map["instruction"]
        assert dim.probe_count == 0
        assert abs(dim.mean_coverage - 0.0) < 1e-9

    def test_coverages_list_stored(self):
        ps = _probe_set([
            _probe("p1", "error_solution", evidence_turns=[0]),
            _probe("p2", "error_solution", evidence_turns=[1]),
        ])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        dim = result.dimension_map["error_solution"]
        assert sorted(dim.coverages) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestComposite:
    def test_composite_in_0_1_range(self):
        ps = _probe_set([
            _probe(f"p{i}", list(DIMENSIONS.keys())[i % len(DIMENSIONS)],
                   evidence_turns=[i])
            for i in range(5)
        ])
        result = compute_evidence_coverage(ps, {0, 1, 2}, "dedup", 1000)
        assert 0.0 <= result.composite <= 1.0 + 1e-9

    def test_full_coverage_composite(self):
        """Perfect coverage: composite = sum of all dimension weights = 1.0."""
        probes = [
            _probe(f"p{i}", dim, evidence_turns=[i])
            for i, dim in enumerate(DIMENSIONS)
        ]
        ps = _probe_set(probes)
        kept = set(range(len(probes)))
        result = compute_evidence_coverage(ps, kept, "dedup", 1000)
        assert abs(result.composite - 1.0) < 1e-9

    def test_zero_coverage_composite_zero(self):
        ps = _probe_set([_probe("p1", "error_solution", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, set(), "dedup", 1000)
        assert abs(result.composite - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# NDCG
# ---------------------------------------------------------------------------

class TestNdcg:
    def test_ndcg_in_0_1_range(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0, 1], difficulty="hard")])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        assert 0.0 <= result.ndcg <= 1.0 + 1e-9

    def test_perfect_coverage_ndcg_1(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0, 1], difficulty="hard")])
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 1000)
        assert abs(result.ndcg - 1.0) < 1e-9

    def test_zero_coverage_ndcg_0(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0], difficulty="medium")])
        result = compute_evidence_coverage(ps, set(), "dedup", 1000)
        assert abs(result.ndcg - 0.0) < 1e-9

    def test_no_evidence_probes_ndcg_0(self):
        """All probes have no evidence turns → no scored items → ndcg = 0."""
        ps = _probe_set([_probe("p1", evidence_turns=[])])
        result = compute_evidence_coverage(ps, {0, 1}, "dedup", 1000)
        assert abs(result.ndcg - 0.0) < 1e-9

    def test_difficulty_weights_constant_exists(self):
        assert "easy" in DIFFICULTY_WEIGHTS
        assert "medium" in DIFFICULTY_WEIGHTS
        assert "hard" in DIFFICULTY_WEIGHTS
        assert DIFFICULTY_WEIGHTS["hard"] > DIFFICULTY_WEIGHTS["easy"]


# ---------------------------------------------------------------------------
# dimension_map property
# ---------------------------------------------------------------------------

class TestDimensionMap:
    def test_returns_dict(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        assert isinstance(result.dimension_map, dict)

    def test_keys_match_dimensions(self):
        ps = _probe_set([_probe("p1", evidence_turns=[0])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        assert set(result.dimension_map.keys()) == set(DIMENSIONS.keys())


# ---------------------------------------------------------------------------
# EvidenceCoverageResult.to_dict
# ---------------------------------------------------------------------------

class TestEvidenceCoverageResultToDict:
    def _result(self) -> EvidenceCoverageResult:
        ps = _probe_set([_probe("p1", "error_solution", evidence_turns=[0, 1])])
        return compute_evidence_coverage(ps, {0}, "setcover", 2048)

    def test_top_level_keys(self):
        d = self._result().to_dict()
        for key in ("method", "budget", "composite", "ndcg", "speed_s",
                    "kept_tokens", "total_tokens", "dimensions", "probe_details"):
            assert key in d, f"Missing key: {key}"

    def test_method_in_dict(self):
        d = self._result().to_dict()
        assert d["method"] == "setcover"

    def test_budget_in_dict(self):
        d = self._result().to_dict()
        assert d["budget"] == 2048

    def test_dimensions_is_dict(self):
        d = self._result().to_dict()
        assert isinstance(d["dimensions"], dict)

    def test_dimensions_keyed_by_name(self):
        d = self._result().to_dict()
        assert "error_solution" in d["dimensions"]

    def test_dimension_entry_has_required_keys(self):
        d = self._result().to_dict()
        dim = d["dimensions"]["error_solution"]
        for key in ("coverage", "weight", "probe_count", "coverages"):
            assert key in dim, f"Missing key: {key}"

    def test_probe_details_is_list(self):
        d = self._result().to_dict()
        assert isinstance(d["probe_details"], list)

    def test_probe_detail_has_required_keys(self):
        d = self._result().to_dict()
        entry = d["probe_details"][0]
        for key in ("probe_id", "dimension", "difficulty", "coverage",
                    "evidence_turns", "kept", "dropped"):
            assert key in entry, f"Missing key: {key}"

    def test_probe_detail_coverage_value(self):
        d = self._result().to_dict()
        entry = d["probe_details"][0]
        assert abs(entry["coverage"] - 0.5) < 1e-9

    def test_empty_probe_details_when_no_evidence(self):
        ps = _probe_set([_probe("p1", evidence_turns=[])])
        result = compute_evidence_coverage(ps, {0}, "dedup", 1000)
        d = result.to_dict()
        assert d["probe_details"] == []
