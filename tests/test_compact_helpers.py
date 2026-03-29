"""Tests for compact.py pure helper functions.

Covers _resolve_methods, build_parser, and _export_eval_json without
executing any I/O or model inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compact import _resolve_methods, build_parser, _export_eval_json
from lib.scorer_base import ALL_METHODS


# ---------------------------------------------------------------------------
# _resolve_methods
# ---------------------------------------------------------------------------

class TestResolveMethods:
    def test_all_returns_all_methods(self):
        result = _resolve_methods("all")
        assert result == ALL_METHODS

    def test_all_result_is_list(self):
        result = _resolve_methods("all")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_single_method_returned_as_list(self):
        for method in ["eitf", "dedup", "setcover"]:
            result = _resolve_methods(method)
            assert result == [method], f"Expected [{method!r}], got {result}"

    def test_returns_list_not_string(self):
        result = _resolve_methods("eitf")
        assert isinstance(result, list)

    def test_unknown_method_still_wrapped(self):
        # _resolve_methods doesn't validate — it just wraps non-"all" values
        result = _resolve_methods("nonexistent")
        assert result == ["nonexistent"]

    def test_all_methods_are_non_empty_strings(self):
        for m in _resolve_methods("all"):
            assert isinstance(m, str) and m


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def setup_method(self):
        self.parser = build_parser()

    def test_returns_argument_parser(self):
        import argparse
        assert isinstance(self.parser, argparse.ArgumentParser)

    def test_compact_subcommand_exists(self):
        args = self.parser.parse_args(["compact", "file.jsonl"])
        assert args.command == "compact"

    def test_evaluate_subcommand_exists(self):
        args = self.parser.parse_args(["evaluate", "file.jsonl"])
        assert args.command == "evaluate"

    def test_plot_subcommand_exists(self):
        args = self.parser.parse_args(["plot", "results.json"])
        assert args.command == "plot"

    def test_compact_default_method_is_eitf(self):
        args = self.parser.parse_args(["compact", "file.jsonl"])
        assert args.method == "eitf"

    def test_compact_default_budget(self):
        args = self.parser.parse_args(["compact", "file.jsonl"])
        assert args.budget == 80_000

    def test_compact_method_flag(self):
        args = self.parser.parse_args(["compact", "file.jsonl", "--method", "dedup"])
        assert args.method == "dedup"

    def test_compact_budget_flag(self):
        args = self.parser.parse_args(["compact", "file.jsonl", "--budget", "50000"])
        assert args.budget == 50_000

    def test_compact_dry_run_default_false(self):
        args = self.parser.parse_args(["compact", "file.jsonl"])
        assert args.dry_run is False

    def test_compact_dry_run_flag(self):
        args = self.parser.parse_args(["compact", "file.jsonl", "--dry-run"])
        assert args.dry_run is True

    def test_evaluate_split_ratio_default(self):
        args = self.parser.parse_args(["evaluate", "file.jsonl"])
        assert args.split_ratio == pytest.approx(0.70)

    def test_compact_verbose_flag(self):
        args = self.parser.parse_args(["compact", "file.jsonl", "--verbose"])
        assert args.verbose is True

    def test_compact_verbose_default_false(self):
        args = self.parser.parse_args(["compact", "file.jsonl"])
        assert args.verbose is False

    def test_plot_output_default_none(self):
        args = self.parser.parse_args(["plot", "results.json"])
        assert args.output is None

    def test_compact_method_choices_include_all(self):
        # 'all' is a valid method choice
        args = self.parser.parse_args(["compact", "file.jsonl", "--method", "all"])
        assert args.method == "all"

    def test_compact_method_choices_include_claude_code(self):
        args = self.parser.parse_args(["compact", "file.jsonl", "--method", "claude-code"])
        assert args.method == "claude-code"


# ---------------------------------------------------------------------------
# _export_eval_json
# ---------------------------------------------------------------------------

def _mock_ev(method="eitf", budget=80000):
    ev = MagicMock()
    ev.to_dict.return_value = {"method": method, "budget": budget, "composite": 0.75}
    return ev


def _mock_ent(method="eitf", budget=80000):
    ent = MagicMock()
    ent.to_dict.return_value = {"method": method, "budget": budget, "coverage": 0.8}
    return ent


class TestExportEvalJson:
    def test_writes_json_file(self, tmp_path):
        out = tmp_path / "results.json"
        _export_eval_json([_mock_ev()], [], out)
        assert out.exists()

    def test_output_is_valid_json(self, tmp_path):
        out = tmp_path / "results.json"
        _export_eval_json([_mock_ev()], [], out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)

    def test_evidence_only(self, tmp_path):
        out = tmp_path / "results.json"
        ev = _mock_ev(method="eitf", budget=80000)
        _export_eval_json([ev], [], out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["method"] == "eitf"

    def test_entity_only_when_no_evidence(self, tmp_path):
        out = tmp_path / "results.json"
        ent = _mock_ent(method="dedup", budget=60000)
        _export_eval_json([], [ent], out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["method"] == "dedup"

    def test_merges_entity_into_evidence_entry(self, tmp_path):
        out = tmp_path / "results.json"
        ev = _mock_ev()
        ent = _mock_ent()
        _export_eval_json([ev], [ent], out)
        data = json.loads(out.read_text())
        assert "entity_coverage" in data[0]
        assert data[0]["entity_coverage"]["coverage"] == 0.8

    def test_multiple_evidence_entries(self, tmp_path):
        out = tmp_path / "results.json"
        evs = [_mock_ev("eitf"), _mock_ev("dedup")]
        _export_eval_json(evs, [], out)
        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_entity_not_merged_when_more_evidence_than_entity(self, tmp_path):
        out = tmp_path / "results.json"
        evs = [_mock_ev("eitf"), _mock_ev("dedup")]
        ents = [_mock_ent("eitf")]  # only one entity result
        _export_eval_json(evs, ents, out)
        data = json.loads(out.read_text())
        assert "entity_coverage" in data[0]
        assert "entity_coverage" not in data[1]

    def test_empty_inputs_writes_empty_list(self, tmp_path):
        out = tmp_path / "results.json"
        _export_eval_json([], [], out)
        data = json.loads(out.read_text())
        assert data == []
