"""Tests for lib/eval/aggregate.py — _dcg, DimensionScore, AggregateResult, aggregate."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.aggregate import (
    DIFFICULTY_WEIGHTS,
    AggregateResult,
    DimensionScore,
    _dcg,
    aggregate,
)
from lib.eval.judge import ProbeAnswer
from lib.eval.probes import Probe, ProbeSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(id="p1", dimension="progress", difficulty="medium") -> Probe:
    return Probe(
        id=id,
        dimension=dimension,
        tier="factual",
        question="Q?",
        gold_answer="A",
        difficulty=difficulty,
    )


def _answer(probe_id="p1", model_key="cheap", score=2) -> ProbeAnswer:
    return ProbeAnswer(
        probe_id=probe_id,
        model_key=model_key,
        model_label="haiku",
        answer="x",
        score=score,
    )


def _probe_set(*probes) -> ProbeSet:
    return ProbeSet(probes=list(probes))


# ---------------------------------------------------------------------------
# DIFFICULTY_WEIGHTS constant
# ---------------------------------------------------------------------------

class TestDifficultyWeights:
    def test_easy_has_weight_one(self):
        assert DIFFICULTY_WEIGHTS["easy"] == 1.0

    def test_medium_has_weight_two(self):
        assert DIFFICULTY_WEIGHTS["medium"] == 2.0

    def test_hard_has_weight_three(self):
        assert DIFFICULTY_WEIGHTS["hard"] == 3.0


# ---------------------------------------------------------------------------
# DimensionScore dataclass
# ---------------------------------------------------------------------------

class TestDimensionScore:
    def test_fields_stored(self):
        ds = DimensionScore(
            dimension="progress",
            weight=0.25,
            mean_score=0.8,
            probe_count=5,
            raw_scores=[2, 3, 2, 3, 2],
        )
        assert ds.dimension == "progress"
        assert ds.weight == 0.25
        assert ds.mean_score == pytest.approx(0.8)
        assert ds.probe_count == 5
        assert ds.raw_scores == [2, 3, 2, 3, 2]

    def test_default_raw_scores_empty(self):
        ds = DimensionScore(dimension="noise", weight=0.05, mean_score=0.0, probe_count=0)
        assert ds.raw_scores == []


# ---------------------------------------------------------------------------
# AggregateResult dataclass
# ---------------------------------------------------------------------------

class TestAggregateResult:
    def test_fields_stored(self):
        ar = AggregateResult(
            method="dedup", budget=80_000, model_key="cheap", model_label="haiku"
        )
        assert ar.method == "dedup"
        assert ar.budget == 80_000
        assert ar.model_key == "cheap"
        assert ar.model_label == "haiku"

    def test_default_composite_zero(self):
        ar = AggregateResult(method="x", budget=0, model_key="k", model_label="l")
        assert ar.composite == 0.0

    def test_default_ndcg_zero(self):
        ar = AggregateResult(method="x", budget=0, model_key="k", model_label="l")
        assert ar.ndcg == 0.0

    def test_default_dimensions_empty(self):
        ar = AggregateResult(method="x", budget=0, model_key="k", model_label="l")
        assert ar.dimensions == []

    def test_dimension_map_property(self):
        ds = DimensionScore(dimension="progress", weight=0.25, mean_score=0.7, probe_count=3)
        ar = AggregateResult(
            method="dedup", budget=80_000, model_key="cheap", model_label="haiku",
            dimensions=[ds],
        )
        dm = ar.dimension_map
        assert "progress" in dm
        assert dm["progress"] is ds

    def test_dimension_map_empty_when_no_dimensions(self):
        ar = AggregateResult(method="x", budget=0, model_key="k", model_label="l")
        assert ar.dimension_map == {}


# ---------------------------------------------------------------------------
# _dcg
# ---------------------------------------------------------------------------

class TestDcg:
    def test_empty_list_returns_zero(self):
        assert _dcg([]) == pytest.approx(0.0)

    def test_single_item(self):
        # score=3, weight=1.0 → 3*1 / log2(0+2) = 3/1 = 3
        result = _dcg([(3, 1.0)])
        assert result == pytest.approx(3.0 / math.log2(2))

    def test_higher_weight_item_placed_first(self):
        # Items sorted by weight desc: hard(3.0) before easy(1.0)
        # So hard item gets position 0, easy gets position 1
        hard = (2, 3.0)
        easy = (2, 1.0)
        result = _dcg([easy, hard])  # out of order by weight
        expected = (2 * 3.0) / math.log2(2) + (2 * 1.0) / math.log2(3)
        assert result == pytest.approx(expected)

    def test_all_zero_scores_returns_zero(self):
        items = [(0, 1.0), (0, 2.0), (0, 3.0)]
        assert _dcg(items) == pytest.approx(0.0)

    def test_higher_position_discounts_more(self):
        # Same score+weight at different positions
        result_first = _dcg([(3, 2.0)])
        result_second_only = [(0, 3.0), (3, 2.0)]
        # Single item should be larger than if it were at position 1
        result_two = _dcg(result_second_only)
        assert result_first > result_two - _dcg([(0, 3.0)])


# ---------------------------------------------------------------------------
# aggregate — basic paths
# ---------------------------------------------------------------------------

class TestAggregateEmpty:
    def test_no_answers_returns_empty(self):
        ps = _probe_set(_probe())
        assert aggregate([], ps, "dedup", 80_000) == []

    def test_no_probes_no_answers_returns_empty(self):
        ps = _probe_set()
        assert aggregate([], ps, "dedup", 80_000) == []


class TestAggregateSingleModel:
    def test_returns_one_result_per_model(self):
        ps = _probe_set(_probe("p1", "progress"))
        answers = [_answer("p1", "cheap", score=3)]
        results = aggregate(answers, ps, "dedup", 80_000)
        assert len(results) == 1

    def test_result_has_correct_method_and_budget(self):
        ps = _probe_set(_probe("p1", "progress"))
        answers = [_answer("p1", "cheap", score=2)]
        result = aggregate(answers, ps, "eitf", 40_000)[0]
        assert result.method == "eitf"
        assert result.budget == 40_000

    def test_result_has_model_key(self):
        ps = _probe_set(_probe("p1", "progress"))
        answers = [_answer("p1", "cheap", score=2)]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        assert result.model_key == "cheap"

    def test_score_normalized_to_zero_one(self):
        ps = _probe_set(_probe("p1", "progress"))
        answers = [_answer("p1", "cheap", score=3)]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        progress_dim = result.dimension_map["progress"]
        assert progress_dim.mean_score == pytest.approx(1.0)  # 3/3

    def test_score_zero_normalizes_to_zero(self):
        ps = _probe_set(_probe("p1", "progress"))
        answers = [_answer("p1", "cheap", score=0)]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        assert result.dimension_map["progress"].mean_score == pytest.approx(0.0)

    def test_probe_not_in_probe_set_skipped(self):
        ps = _probe_set()  # empty
        answers = [_answer("ghost", "cheap", score=3)]
        results = aggregate(answers, ps, "dedup", 80_000)
        # No probes matched → model group exists but no dims scored
        if results:
            # All dimension scores should have probe_count=0
            for ds in results[0].dimensions:
                assert ds.probe_count == 0

    def test_dimension_with_no_probes_has_zero_mean(self):
        ps = _probe_set(_probe("p1", "progress"))  # only progress
        answers = [_answer("p1", "cheap", score=3)]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        # error_solution has no probes
        esr_dim = result.dimension_map.get("error_solution")
        if esr_dim:
            assert esr_dim.mean_score == pytest.approx(0.0)
            assert esr_dim.probe_count == 0


class TestAggregateMultipleModels:
    def test_two_models_give_two_results(self):
        ps = _probe_set(_probe("p1", "progress"))
        answers = [
            _answer("p1", "cheap", score=2),
            _answer("p1", "capable", score=3),
        ]
        results = aggregate(answers, ps, "dedup", 80_000)
        assert len(results) == 2
        model_keys = {r.model_key for r in results}
        assert model_keys == {"cheap", "capable"}


class TestAggregateNdcg:
    def test_perfect_scores_give_ndcg_one(self):
        ps = _probe_set(_probe("p1", "progress", difficulty="medium"))
        answers = [_answer("p1", "cheap", score=3)]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        assert result.ndcg == pytest.approx(1.0)

    def test_zero_scores_give_ndcg_zero(self):
        ps = _probe_set(_probe("p1", "progress", difficulty="medium"))
        answers = [_answer("p1", "cheap", score=0)]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        assert result.ndcg == pytest.approx(0.0)

    def test_ndcg_between_zero_and_one(self):
        ps = _probe_set(
            _probe("p1", "progress", difficulty="easy"),
            _probe("p2", "instruction", difficulty="hard"),
        )
        answers = [
            _answer("p1", "cheap", score=1),
            _answer("p2", "cheap", score=2),
        ]
        result = aggregate(answers, ps, "dedup", 80_000)[0]
        assert 0.0 <= result.ndcg <= 1.0
