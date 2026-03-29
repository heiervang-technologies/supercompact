"""Tests for lib/eval/report.py — JSON export and trace export."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.aggregate import AggregateResult, DimensionScore
from lib.eval.probes import Probe, ProbeSet, DIMENSIONS
from lib.eval.judge import ProbeAnswer
from lib.eval.report import export_json, export_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(method: str = "dedup", budget: int = 1000,
                 composite: float = 0.7, ndcg: float = 0.65) -> AggregateResult:
    dims = [
        DimensionScore(
            dimension=dim, weight=w, mean_score=0.6, probe_count=2, raw_scores=[2, 1]
        )
        for dim, w in DIMENSIONS.items()
    ]
    return AggregateResult(
        method=method,
        budget=budget,
        model_key="capable",
        model_label="TestModel",
        dimensions=dims,
        composite=composite,
        ndcg=ndcg,
        speed_s=1.5,
        kept_tokens=500,
        total_tokens=2000,
    )


def _make_probe_set() -> ProbeSet:
    probes = [
        Probe(
            id="p1", dimension="error_solution", tier="factual",
            question="Q?", gold_answer="A", difficulty="medium",
        )
    ]
    return ProbeSet(probes=probes)


def _make_answers() -> list[ProbeAnswer]:
    return [
        ProbeAnswer(
            probe_id="p1", model_key="capable", model_label="TestModel",
            answer="My answer", score=2, judge_reasoning="Good.",
        )
    ]


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result()], out)
        assert out.exists()

    def test_output_is_valid_json(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result()], out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)

    def test_one_entry_per_result(self, tmp_path):
        out = tmp_path / "results.json"
        results = [_make_result("dedup"), _make_result("eitf")]
        export_json(results, out)
        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_method_preserved(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result(method="setcover")], out)
        data = json.loads(out.read_text())
        assert data[0]["method"] == "setcover"

    def test_budget_preserved(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result(budget=4096)], out)
        data = json.loads(out.read_text())
        assert data[0]["budget"] == 4096

    def test_composite_preserved(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result(composite=0.8123)], out)
        data = json.loads(out.read_text())
        assert abs(data[0]["composite"] - 0.8123) < 1e-9

    def test_ndcg_preserved(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result(ndcg=0.75)], out)
        data = json.loads(out.read_text())
        assert abs(data[0]["ndcg"] - 0.75) < 1e-9

    def test_dimensions_exported(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result()], out)
        data = json.loads(out.read_text())
        dims = data[0]["dimensions"]
        assert isinstance(dims, dict)
        assert "error_solution" in dims

    def test_dimension_has_score_weight_count(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result()], out)
        data = json.loads(out.read_text())
        dim = data[0]["dimensions"]["error_solution"]
        assert "score" in dim
        assert "weight" in dim
        assert "probe_count" in dim

    def test_empty_results_list(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([], out)
        data = json.loads(out.read_text())
        assert data == []

    def test_speed_and_token_counts(self, tmp_path):
        out = tmp_path / "results.json"
        export_json([_make_result()], out)
        data = json.loads(out.read_text())
        entry = data[0]
        assert entry["speed_s"] == 1.5
        assert entry["kept_tokens"] == 500
        assert entry["total_tokens"] == 2000


# ---------------------------------------------------------------------------
# export_trace
# ---------------------------------------------------------------------------

class TestExportTrace:
    def test_creates_file(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "dedup", 1000, _make_probe_set(), _make_answers(), trace_dir
        )
        assert path.exists()

    def test_filename_pattern(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "eitf", 2048, _make_probe_set(), _make_answers(), trace_dir
        )
        assert "eitf" in path.name
        assert "2048" in path.name

    def test_creates_trace_dir(self, tmp_path):
        trace_dir = tmp_path / "new" / "nested" / "dir"
        export_trace("dedup", 1000, _make_probe_set(), _make_answers(), trace_dir)
        assert trace_dir.exists()

    def test_valid_json(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "dedup", 1000, _make_probe_set(), _make_answers(), trace_dir
        )
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_method_and_budget_in_json(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "setcover", 4096, _make_probe_set(), _make_answers(), trace_dir
        )
        data = json.loads(path.read_text())
        assert data["method"] == "setcover"
        assert data["budget"] == 4096

    def test_entries_present(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "dedup", 1000, _make_probe_set(), _make_answers(), trace_dir
        )
        data = json.loads(path.read_text())
        assert len(data["entries"]) == 1

    def test_entry_has_required_fields(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "dedup", 1000, _make_probe_set(), _make_answers(), trace_dir
        )
        data = json.loads(path.read_text())
        entry = data["entries"][0]
        for field in ("probe_id", "dimension", "question", "gold_answer",
                      "generated_answer", "score", "judge_reasoning"):
            assert field in entry, f"Missing field: {field}"

    def test_score_in_entry(self, tmp_path):
        trace_dir = tmp_path / "traces"
        path = export_trace(
            "dedup", 1000, _make_probe_set(), _make_answers(), trace_dir
        )
        data = json.loads(path.read_text())
        assert data["entries"][0]["score"] == 2

    def test_unknown_probe_id_skipped(self, tmp_path):
        """Answers with unknown probe IDs should not appear in entries."""
        trace_dir = tmp_path / "traces"
        answers = _make_answers() + [
            ProbeAnswer(
                probe_id="UNKNOWN", model_key="capable", model_label="M",
                answer="x", score=1,
            )
        ]
        path = export_trace("dedup", 1000, _make_probe_set(), answers, trace_dir)
        data = json.loads(path.read_text())
        # Only 1 valid probe → 1 entry
        assert len(data["entries"]) == 1
