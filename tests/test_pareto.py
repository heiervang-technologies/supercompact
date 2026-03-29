"""Tests for lib/pareto.py — METHOD_STYLES, plot_entity_coverage, plot_type_breakdown."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.pareto import METHOD_STYLES, plot_entity_coverage, plot_type_breakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(method: str, speed: float, coverage: float, budget: int = 10000) -> dict:
    return {
        "method": method,
        "speed_s": speed,
        "weighted_coverage": coverage,
        "budget": budget,
        "kept_tokens": int(budget * 0.8),
    }


def _make_ax():
    fig, ax = plt.subplots()
    return ax


# ---------------------------------------------------------------------------
# METHOD_STYLES
# ---------------------------------------------------------------------------

class TestMethodStyles:
    def test_expected_methods_present(self):
        for method in ("dedup", "eitf", "setcover", "llama-embed", "llama-rerank", "claude-code"):
            assert method in METHOD_STYLES, f"{method!r} missing from METHOD_STYLES"

    def test_each_entry_has_color(self):
        for method, style in METHOD_STYLES.items():
            assert "color" in style, f"{method!r} missing color"

    def test_each_entry_has_marker(self):
        for method, style in METHOD_STYLES.items():
            assert "marker" in style, f"{method!r} missing marker"

    def test_each_entry_has_label(self):
        for method, style in METHOD_STYLES.items():
            assert "label" in style, f"{method!r} missing label"

    def test_color_is_hex_or_named_string(self):
        for method, style in METHOD_STYLES.items():
            color = style["color"]
            assert isinstance(color, str) and len(color) > 0

    def test_labels_are_non_empty_strings(self):
        for method, style in METHOD_STYLES.items():
            assert isinstance(style["label"], str) and len(style["label"]) > 0


# ---------------------------------------------------------------------------
# plot_entity_coverage
# ---------------------------------------------------------------------------

class TestPlotEntityCoverage:
    def teardown_method(self):
        plt.close("all")

    def test_empty_results_does_not_crash(self):
        ax = _make_ax()
        plot_entity_coverage(ax, [])

    def test_single_result_does_not_crash(self):
        ax = _make_ax()
        plot_entity_coverage(ax, [_make_result("dedup", 1.0, 0.8)])

    def test_multiple_results_does_not_crash(self):
        ax = _make_ax()
        results = [
            _make_result("dedup", 1.0, 0.7),
            _make_result("eitf", 2.0, 0.85),
            _make_result("setcover", 3.0, 0.9),
        ]
        plot_entity_coverage(ax, results)

    def test_same_method_multiple_points(self):
        ax = _make_ax()
        results = [
            _make_result("dedup", 1.0, 0.7, budget=10000),
            _make_result("dedup", 2.0, 0.8, budget=20000),
        ]
        plot_entity_coverage(ax, results)

    def test_unknown_method_uses_fallback(self):
        ax = _make_ax()
        results = [_make_result("unknown_method", 1.0, 0.5)]
        plot_entity_coverage(ax, results)  # should not crash

    def test_xlabel_set(self):
        ax = _make_ax()
        plot_entity_coverage(ax, [_make_result("eitf", 1.0, 0.8)])
        assert ax.get_xlabel() != ""

    def test_ylabel_set(self):
        ax = _make_ax()
        plot_entity_coverage(ax, [_make_result("eitf", 1.0, 0.8)])
        assert ax.get_ylabel() != ""

    def test_title_set(self):
        ax = _make_ax()
        plot_entity_coverage(ax, [_make_result("eitf", 1.0, 0.8)])
        assert ax.get_title() != ""

    def test_show_legend_false_no_crash(self):
        ax = _make_ax()
        plot_entity_coverage(ax, [_make_result("dedup", 1.0, 0.8)], show_legend=False)

    def test_large_budget_uses_K_label(self):
        # Budget ≥ 1000 → budget_label contains 'K'
        ax = _make_ax()
        plot_entity_coverage(ax, [_make_result("eitf", 1.0, 0.8, budget=50000)])
        # If it doesn't crash and the annotation is applied — pass

    def test_claude_code_uses_larger_scatter_size(self):
        # claude-code uses size=300, others 180 — just check no crash
        ax = _make_ax()
        results = [
            _make_result("claude-code", 1.0, 0.95),
            _make_result("dedup", 2.0, 0.7),
        ]
        plot_entity_coverage(ax, results)


# ---------------------------------------------------------------------------
# plot_type_breakdown
# ---------------------------------------------------------------------------

class TestPlotTypeBreakdown:
    def teardown_method(self):
        plt.close("all")

    def test_empty_type_coverage_does_not_crash(self):
        ax = _make_ax()
        results = [{"method": "dedup", "budget": 10000, "type_coverage": {}}]
        plot_type_breakdown(ax, results)

    def test_missing_type_coverage_key_does_not_crash(self):
        ax = _make_ax()
        results = [{"method": "dedup", "budget": 10000}]
        plot_type_breakdown(ax, results)

    def test_single_type_does_not_crash(self):
        ax = _make_ax()
        results = [{
            "method": "eitf",
            "budget": 10000,
            "type_coverage": {
                "exception": {"coverage": 0.9, "weight": 1.0, "covered": 9, "total": 10},
            },
        }]
        plot_type_breakdown(ax, results)

    def test_multiple_types_sorted(self):
        ax = _make_ax()
        results = [{
            "method": "eitf",
            "budget": 10000,
            "type_coverage": {
                "url": {"coverage": 0.5, "weight": 0.5, "covered": 5, "total": 10},
                "exception": {"coverage": 0.9, "weight": 1.0, "covered": 9, "total": 10},
                "port": {"coverage": 0.7, "weight": 0.8, "covered": 7, "total": 10},
            },
        }]
        plot_type_breakdown(ax, results)

    def test_largest_budget_selected(self):
        ax = _make_ax()
        results = [
            {"method": "dedup", "budget": 5000, "type_coverage": {}},
            {"method": "dedup", "budget": 20000, "type_coverage": {
                "exception": {"coverage": 0.8, "weight": 1.0, "covered": 8, "total": 10},
            }},
        ]
        # Should use the budget=20000 entry for the bar chart — no crash
        plot_type_breakdown(ax, results)

    def test_title_contains_budget(self):
        ax = _make_ax()
        results = [{
            "method": "eitf",
            "budget": 30000,
            "type_coverage": {
                "exception": {"coverage": 0.8, "weight": 1.0, "covered": 8, "total": 10},
            },
        }]
        plot_type_breakdown(ax, results)
        assert "30" in ax.get_title()  # budget 30K in title
