"""Tests for lib/eval/report.py — export_json and export_trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.report import export_json, export_trace
from lib.eval.aggregate import AggregateResult, DimensionScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> AggregateResult:
    defaults = dict(
        method="dedup",
        budget=80_000,
        model_key="cheap",
        model_label="claude-haiku",
        dimensions=[],
        composite=0.75,
        ndcg=0.82,
        speed_s=1.5,
        kept_tokens=20_000,
        total_tokens=100_000,
    )
    defaults.update(overrides)
    return AggregateResult(**defaults)


def _make_dim(**overrides) -> DimensionScore:
    defaults = dict(
        dimension="progress",
        weight=1.0,
        mean_score=0.8,
        probe_count=5,
        raw_scores=[2, 3, 2, 3, 3],
    )
    defaults.update(overrides)
    return DimensionScore(**defaults)


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "results.json"
        export_json([], path)
        assert path.exists()

    def test_empty_results_writes_empty_array(self, tmp_path):
        path = tmp_path / "results.json"
        export_json([], path)
        data = json.loads(path.read_text())
        assert data == []

    def test_single_result_structure(self, tmp_path):
        result = _make_result()
        path = tmp_path / "results.json"
        export_json([result], path)
        data = json.loads(path.read_text())
        assert len(data) == 1
        entry = data[0]
        assert entry["method"] == "dedup"
        assert entry["budget"] == 80_000
        assert entry["model_key"] == "cheap"
        assert entry["composite"] == pytest.approx(0.75)
        assert entry["ndcg"] == pytest.approx(0.82)

    def test_keeps_speed_and_token_counts(self, tmp_path):
        result = _make_result(speed_s=2.5, kept_tokens=15_000, total_tokens=80_000)
        path = tmp_path / "results.json"
        export_json([result], path)
        data = json.loads(path.read_text())
        assert data[0]["speed_s"] == pytest.approx(2.5)
        assert data[0]["kept_tokens"] == 15_000
        assert data[0]["total_tokens"] == 80_000

    def test_dimension_scores_included(self, tmp_path):
        dim = _make_dim(dimension="error_solution", mean_score=0.9, probe_count=3)
        result = _make_result(dimensions=[dim])
        path = tmp_path / "results.json"
        export_json([result], path)
        data = json.loads(path.read_text())
        dims = data[0]["dimensions"]
        assert "error_solution" in dims
        assert dims["error_solution"]["score"] == pytest.approx(0.9)
        assert dims["error_solution"]["probe_count"] == 3

    def test_multiple_results(self, tmp_path):
        results = [_make_result(method="dedup"), _make_result(method="eitf")]
        path = tmp_path / "results.json"
        export_json(results, path)
        data = json.loads(path.read_text())
        assert len(data) == 2
        methods = {e["method"] for e in data}
        assert methods == {"dedup", "eitf"}

    def test_output_is_valid_json(self, tmp_path):
        path = tmp_path / "results.json"
        export_json([_make_result()], path)
        # Should parse without exception
        json.loads(path.read_text())


# ---------------------------------------------------------------------------
# export_trace
# ---------------------------------------------------------------------------

class TestExportTrace:
    def _make_probe_set(self, probes=None):
        ps = MagicMock()
        ps.probes = probes or []
        return ps

    def _make_answer(self, probe_id="p1", score=2, **overrides):
        a = MagicMock()
        a.probe_id = probe_id
        a.score = score
        a.model_key = overrides.get("model_key", "cheap")
        a.model_label = overrides.get("model_label", "haiku")
        a.answer = overrides.get("answer", "some answer")
        a.judge_reasoning = overrides.get("judge_reasoning", "looks good")
        return a

    def _make_probe(self, id="p1", dimension="progress", tier="factual",
                    difficulty="medium", question="Q?", gold_answer="A",
                    evidence_turns=None):
        p = MagicMock()
        p.id = id
        p.dimension = dimension
        p.tier = tier
        p.difficulty = difficulty
        p.question = question
        p.gold_answer = gold_answer
        p.evidence_turns = evidence_turns or []
        return p

    def test_creates_trace_file(self, tmp_path):
        probe_set = self._make_probe_set()
        result_path = export_trace("dedup", 80_000, probe_set, [], tmp_path)
        assert result_path.exists()

    def test_trace_file_in_trace_dir(self, tmp_path):
        probe_set = self._make_probe_set()
        result_path = export_trace("eitf", 40_000, probe_set, [], tmp_path)
        assert result_path.parent == tmp_path

    def test_trace_filename_contains_method_and_budget(self, tmp_path):
        probe_set = self._make_probe_set()
        result_path = export_trace("setcover", 60_000, probe_set, [], tmp_path)
        assert "setcover" in result_path.name
        assert "60000" in result_path.name

    def test_trace_has_method_and_budget(self, tmp_path):
        probe_set = self._make_probe_set()
        result_path = export_trace("dedup", 80_000, probe_set, [], tmp_path)
        data = json.loads(result_path.read_text())
        assert data["method"] == "dedup"
        assert data["budget"] == 80_000

    def test_empty_answers_produces_empty_entries(self, tmp_path):
        probe_set = self._make_probe_set()
        result_path = export_trace("dedup", 80_000, probe_set, [], tmp_path)
        data = json.loads(result_path.read_text())
        assert data["entries"] == []

    def test_answer_with_matching_probe_included(self, tmp_path):
        probe = self._make_probe(id="p1", question="What happened?", gold_answer="Error")
        probe_set = self._make_probe_set(probes=[probe])
        answer = self._make_answer(probe_id="p1", score=3, answer="Error occurred")
        result_path = export_trace("dedup", 80_000, probe_set, [answer], tmp_path)
        data = json.loads(result_path.read_text())
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["probe_id"] == "p1"
        assert entry["score"] == 3

    def test_answer_without_matching_probe_skipped(self, tmp_path):
        probe_set = self._make_probe_set(probes=[])
        answer = self._make_answer(probe_id="missing_probe")
        result_path = export_trace("dedup", 80_000, probe_set, [answer], tmp_path)
        data = json.loads(result_path.read_text())
        assert data["entries"] == []

    def test_creates_trace_dir_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "traces"
        probe_set = self._make_probe_set()
        export_trace("dedup", 80_000, probe_set, [], nested)
        assert nested.is_dir()
