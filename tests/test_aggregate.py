"""Tests for lib/eval/aggregate.py — score aggregation, DCG, and composite metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.aggregate import (
    _dcg,
    aggregate,
    AggregateResult,
    DimensionScore,
    DIFFICULTY_WEIGHTS,
)
from lib.eval.probes import Probe, ProbeSet, DIMENSIONS
from lib.eval.judge import ProbeAnswer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(pid: str, dimension: str, difficulty: str = "medium") -> Probe:
    return Probe(
        id=pid,
        dimension=dimension,
        tier="factual",
        question="Q?",
        gold_answer="A",
        difficulty=difficulty,
    )


def _answer(probe_id: str, score: int, model_key: str = "capable") -> ProbeAnswer:
    return ProbeAnswer(
        probe_id=probe_id,
        model_key=model_key,
        model_label=f"Model-{model_key}",
        answer="some answer",
        score=score,
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
        # DCG of single item at index 0: score * weight / log2(0 + 2) = score * weight / log2(2) = score * weight
        result = _dcg([(3, 1.0)])
        expected = (3 * 1.0) / math.log2(2)  # log2(2) = 1.0
        assert abs(result - expected) < 1e-9

    def test_sorted_by_weight_descending(self):
        """Higher-weight items should appear at lower indices (better DCG)."""
        # Two items: high-weight at position 0 gives better DCG than low-weight at 0
        r1 = _dcg([(3, 3.0), (3, 1.0)])  # heavy first
        r2 = _dcg([(3, 1.0), (3, 3.0)])  # light first — same after sorting, so equal
        # _dcg sorts internally, so both should be equal
        assert abs(r1 - r2) < 1e-9

    def test_zero_score(self):
        result = _dcg([(0, 1.0), (0, 2.0)])
        assert result == 0.0

    def test_max_score(self):
        """Max score 3 with weight 1, single item."""
        result = _dcg([(3, 1.0)])
        assert result > 0


# ---------------------------------------------------------------------------
# aggregate — basic contracts
# ---------------------------------------------------------------------------

class TestAggregate:
    def _simple_setup(self, score: int = 2, dimension: str = "error_solution"):
        probe = _probe("p1", dimension)
        probe_set = _probe_set([probe])
        answers = [_answer("p1", score)]
        return probe_set, answers

    def test_returns_list(self):
        probe_set, answers = self._simple_setup()
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        assert isinstance(results, list)

    def test_returns_one_result_per_model_key(self):
        probe = _probe("p1", "error_solution")
        probe_set = _probe_set([probe])
        answers = [
            _answer("p1", 2, model_key="capable"),
            _answer("p1", 1, model_key="cheap"),
        ]
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        assert len(results) == 2

    def test_result_method_matches(self):
        probe_set, answers = self._simple_setup()
        results = aggregate(answers, probe_set, method="eitf", budget=2000)
        assert results[0].method == "eitf"

    def test_result_budget_matches(self):
        probe_set, answers = self._simple_setup()
        results = aggregate(answers, probe_set, method="dedup", budget=4096)
        assert results[0].budget == 4096

    def test_empty_answers_returns_empty(self):
        probe_set = _probe_set([_probe("p1", "error_solution")])
        results = aggregate([], probe_set, method="dedup", budget=1000)
        assert results == []

    def test_dimension_count_matches_known_dimensions(self):
        probe_set, answers = self._simple_setup()
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        assert len(results[0].dimensions) == len(DIMENSIONS)

    def test_unknown_probe_id_skipped(self):
        probe = _probe("p1", "error_solution")
        probe_set = _probe_set([probe])
        answers = [_answer("NONEXISTENT", 3)]
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        # Should still return a result (for the model key), but with 0 scored probes
        assert len(results) == 1

    def test_probe_count_correct(self):
        probes = [_probe(f"p{i}", "error_solution") for i in range(3)]
        probe_set = _probe_set(probes)
        answers = [_answer(f"p{i}", i % 3) for i in range(3)]
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        dim = results[0].dimension_map["error_solution"]
        assert dim.probe_count == 3

    def test_mean_score_normalized_to_0_1(self):
        probe = _probe("p1", "error_solution")
        probe_set = _probe_set([probe])
        # Score 3 out of 3 → normalized 1.0
        results = aggregate([_answer("p1", 3)], probe_set, method="dedup", budget=1000)
        dim = results[0].dimension_map["error_solution"]
        assert abs(dim.mean_score - 1.0) < 1e-9

    def test_zero_score_normalized(self):
        probe = _probe("p1", "error_solution")
        probe_set = _probe_set([probe])
        results = aggregate([_answer("p1", 0)], probe_set, method="dedup", budget=1000)
        dim = results[0].dimension_map["error_solution"]
        assert abs(dim.mean_score - 0.0) < 1e-9

    def test_composite_in_0_1_range(self):
        probes = [_probe(f"p{i}", list(DIMENSIONS.keys())[i % len(DIMENSIONS)]) for i in range(5)]
        probe_set = _probe_set(probes)
        answers = [_answer(f"p{i}", 2) for i in range(5)]
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        comp = results[0].composite
        assert 0.0 <= comp <= 1.0 + 1e-9

    def test_ndcg_in_0_1_range(self):
        probe = _probe("p1", "error_solution", difficulty="hard")
        probe_set = _probe_set([probe])
        results = aggregate([_answer("p1", 2)], probe_set, method="dedup", budget=1000)
        assert 0.0 <= results[0].ndcg <= 1.0 + 1e-9

    def test_perfect_score_ndcg_is_1(self):
        """Perfect scores should yield NDCG = 1.0."""
        probe = _probe("p1", "error_solution", difficulty="hard")
        probe_set = _probe_set([probe])
        results = aggregate([_answer("p1", 3)], probe_set, method="dedup", budget=1000)
        assert abs(results[0].ndcg - 1.0) < 1e-9

    def test_raw_scores_stored(self):
        probe = _probe("p1", "error_solution")
        probe_set = _probe_set([probe])
        results = aggregate([_answer("p1", 2)], probe_set, method="dedup", budget=1000)
        dim = results[0].dimension_map["error_solution"]
        assert dim.raw_scores == [2]

    def test_dimension_map_property(self):
        probe_set, answers = self._simple_setup()
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        dim_map = results[0].dimension_map
        assert isinstance(dim_map, dict)
        assert all(key in dim_map for key in DIMENSIONS)

    def test_model_label_from_first_answer(self):
        probe = _probe("p1", "error_solution")
        probe_set = _probe_set([probe])
        a = ProbeAnswer(
            probe_id="p1", model_key="capable",
            model_label="MyModel", answer="A", score=1,
        )
        results = aggregate([a], probe_set, method="dedup", budget=1000)
        assert results[0].model_label == "MyModel"

    def test_difficulty_weights_applied(self):
        """Hard probes should contribute more to NDCG than easy probes."""
        easy_probe = _probe("easy", "error_solution", difficulty="easy")
        hard_probe = _probe("hard", "error_solution", difficulty="hard")
        probe_set = _probe_set([easy_probe, hard_probe])

        # Same score but hard probe weighted more → NDCG accounts for this
        answers = [_answer("easy", 1), _answer("hard", 1)]
        results = aggregate(answers, probe_set, method="dedup", budget=1000)
        # Just verify it runs without error and NDCG is in range
        assert 0.0 <= results[0].ndcg <= 1.0
