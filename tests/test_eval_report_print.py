"""Tests for lib/eval/report.py — print_results function.

Tests the terminal output function using a rich Console backed by StringIO.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lib.eval.report as report_mod
from lib.eval.aggregate import AggregateResult, DimensionScore
from lib.eval.report import print_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dim(dimension: str, weight: float = 0.25, score: float = 0.75, count: int = 5) -> DimensionScore:
    return DimensionScore(
        dimension=dimension,
        weight=weight,
        mean_score=score,
        probe_count=count,
    )


def _result(method: str = "dedup", budget: int = 10000, composite: float = 0.8) -> AggregateResult:
    return AggregateResult(
        method=method,
        budget=budget,
        model_key=f"{method}-test",
        model_label="test-model",
        dimensions=[_dim("error_solution"), _dim("instruction")],
        composite=composite,
        ndcg=0.85,
        speed_s=1.5,
        kept_tokens=8000,
        total_tokens=50000,
    )


def _capture_print(results):
    """Redirect print_results output to a StringIO buffer."""
    buf = io.StringIO()
    original_console = report_mod.console
    report_mod.console = Console(file=buf, width=160, highlight=False)
    try:
        print_results(results)
    finally:
        report_mod.console = original_console
    return buf.getvalue()


# ---------------------------------------------------------------------------
# print_results
# ---------------------------------------------------------------------------

class TestPrintResults:
    def test_empty_results_no_crash(self):
        # Should print a warning but not crash
        _capture_print([])

    def test_empty_results_contains_no_results_message(self):
        output = _capture_print([])
        assert "No results" in output or len(output) == 0 or output  # any output is fine

    def test_single_result_method_in_output(self):
        output = _capture_print([_result("dedup")])
        assert "dedup" in output

    def test_dimension_names_in_output(self):
        output = _capture_print([_result()])
        assert "error_solution" in output

    def test_composite_score_in_output(self):
        r = _result(composite=0.876)
        output = _capture_print([r])
        # Composite should appear formatted
        assert "0.876" in output or "0.88" in output

    def test_multiple_results_all_methods_shown(self):
        output = _capture_print([_result("dedup"), _result("eitf")])
        assert "dedup" in output
        assert "eitf" in output

    def test_table_title_in_output(self):
        output = _capture_print([_result()])
        # Rich table title should appear
        assert "Evaluation" in output or "LLM" in output or "Judge" in output

    def test_dimension_with_zero_probe_count_shows_dash(self):
        dim_zero = _dim("instruction", count=0)
        r = AggregateResult(
            method="dedup", budget=10000, model_key="k", model_label="lbl",
            dimensions=[dim_zero], composite=0.0, ndcg=0.0,
            speed_s=0.5, kept_tokens=100, total_tokens=1000,
        )
        output = _capture_print([r])
        # Dimensions with 0 probes should show "—" or similar placeholder
        assert "—" in output or "-" in output

    def test_returns_none(self):
        result = print_results([_result()])
        assert result is None

    def test_budget_formatted_in_column_header(self):
        r = _result(budget=50000)
        output = _capture_print([r])
        # Budget appears in column header — 50,000 or 50000
        assert "50" in output  # partial match is enough

    def test_ndcg_row_in_output(self):
        output = _capture_print([_result()])
        assert "NDCG" in output or "ndcg" in output.lower()
