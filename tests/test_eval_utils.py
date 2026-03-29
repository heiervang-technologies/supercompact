"""Tests for lib/eval/aggregate._dcg, evidence_coverage._dcg, and dataclass helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.aggregate import _dcg as agg_dcg, AggregateResult, DimensionScore
from lib.eval.evidence_coverage import (
    _dcg as ev_dcg,
    EvidenceCoverageResult,
    DimensionCoverage,
    ProbeCoverage,
)


# ---------------------------------------------------------------------------
# aggregate._dcg
# ---------------------------------------------------------------------------

class TestAggDcg:
    def test_empty_returns_zero(self):
        assert agg_dcg([]) == 0.0

    def test_single_item(self):
        # position 0 → log2(0+2) = log2(2) = 1
        result = agg_dcg([(3, 1.0)])
        expected = 3 * 1.0 / math.log2(2)
        assert abs(result - expected) < 1e-10

    def test_sorted_by_difficulty_weight(self):
        """Items should be sorted by weight descending before DCG is computed."""
        # High weight item first should give higher DCG than low weight first
        items = [(1, 1.0), (1, 3.0)]
        result = agg_dcg(items)
        # Sorted: (1, 3.0) at pos 0, (1, 1.0) at pos 1
        expected = (1 * 3.0) / math.log2(2) + (1 * 1.0) / math.log2(3)
        assert abs(result - expected) < 1e-10

    def test_zero_scores_give_zero(self):
        assert agg_dcg([(0, 1.0), (0, 2.0)]) == 0.0

    def test_returns_float(self):
        assert isinstance(agg_dcg([(1, 1.0)]), float)

    def test_higher_difficulty_weight_increases_dcg(self):
        low = agg_dcg([(1, 1.0)])
        high = agg_dcg([(1, 3.0)])
        assert high > low


# ---------------------------------------------------------------------------
# evidence_coverage._dcg
# ---------------------------------------------------------------------------

class TestEvDcg:
    def test_empty_returns_zero(self):
        assert ev_dcg([]) == 0.0

    def test_single_item(self):
        result = ev_dcg([(1.0, 1.0)])
        expected = 1.0 * 1.0 / math.log2(2)
        assert abs(result - expected) < 1e-10

    def test_float_scores(self):
        result = ev_dcg([(0.5, 2.0), (0.8, 1.0)])
        # Sorted by weight desc: (0.5, 2.0) at pos 0, (0.8, 1.0) at pos 1
        expected = (0.5 * 2.0) / math.log2(2) + (0.8 * 1.0) / math.log2(3)
        assert abs(result - expected) < 1e-10

    def test_returns_float(self):
        assert isinstance(ev_dcg([(1.0, 1.0)]), float)


# ---------------------------------------------------------------------------
# AggregateResult dataclass
# ---------------------------------------------------------------------------

class TestAggregateResult:
    def _dim(self, name: str, weight: float, score: float) -> DimensionScore:
        return DimensionScore(
            dimension=name,
            weight=weight,
            mean_score=score,
            probe_count=5,
        )

    def test_dimension_map_keyed_by_name(self):
        r = AggregateResult(
            method="test", budget=10000,
            model_key="m", model_label="M",
            dimensions=[self._dim("facts", 0.5, 0.8), self._dim("commands", 0.3, 0.6)],
        )
        dm = r.dimension_map
        assert "facts" in dm
        assert "commands" in dm

    def test_dimension_map_empty_when_no_dims(self):
        r = AggregateResult(method="x", budget=1000, model_key="k", model_label="L")
        assert r.dimension_map == {}

    def test_defaults(self):
        r = AggregateResult(method="m", budget=0, model_key="k", model_label="l")
        assert r.composite == 0.0
        assert r.ndcg == 0.0
        assert r.kept_tokens == 0


# ---------------------------------------------------------------------------
# EvidenceCoverageResult dataclass
# ---------------------------------------------------------------------------

class TestEvidenceCoverageResult:
    def _dim(self, name: str, weight: float, cov: float) -> DimensionCoverage:
        return DimensionCoverage(
            dimension=name,
            weight=weight,
            mean_coverage=cov,
            probe_count=3,
        )

    def test_dimension_map_keyed_by_name(self):
        r = EvidenceCoverageResult(
            method="dedup", budget=80000,
            dimensions=[self._dim("facts", 0.5, 0.9), self._dim("commands", 0.3, 0.7)],
        )
        dm = r.dimension_map
        assert "facts" in dm
        assert "commands" in dm

    def test_to_dict_has_expected_keys(self):
        r = EvidenceCoverageResult(method="dedup", budget=50000)
        d = r.to_dict()
        assert "method" in d
        assert "budget" in d
        assert "composite" in d
        assert "dimensions" in d

    def test_to_dict_method_preserved(self):
        r = EvidenceCoverageResult(method="llama-rerank", budget=100000)
        assert r.to_dict()["method"] == "llama-rerank"

    def test_defaults(self):
        r = EvidenceCoverageResult(method="x", budget=0)
        assert r.composite == 0.0
        assert r.ndcg == 0.0
        assert r.kept_tokens == 0
        assert r.dimensions == []


# ---------------------------------------------------------------------------
# ProbeCoverage / DimensionCoverage helpers
# ---------------------------------------------------------------------------

class TestProbeCoverage:
    def test_stores_coverage_value(self):
        pc = ProbeCoverage(
            probe_id="p1", dimension="facts", difficulty="hard",
            evidence_turns=[0, 1, 2],
            kept_evidence=[0, 1],
            dropped_evidence=[2],
            coverage=2 / 3,
        )
        assert abs(pc.coverage - 2 / 3) < 1e-10
        assert pc.difficulty == "hard"

    def test_all_kept_gives_coverage_one(self):
        pc = ProbeCoverage(
            probe_id="p2", dimension="x", difficulty="easy",
            evidence_turns=[0, 1],
            kept_evidence=[0, 1],
            dropped_evidence=[],
            coverage=1.0,
        )
        assert pc.coverage == 1.0
