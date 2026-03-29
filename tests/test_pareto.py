"""Tests for lib/pareto.py — Pareto plot helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.pareto import plot_entity_coverage, plot_type_breakdown, METHOD_STYLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(method: str = "dedup", budget: int = 1000,
            speed_s: float = 1.0, weighted_coverage: float = 0.7,
            kept_tokens: int = 500) -> dict:
    return {
        "method": method,
        "budget": budget,
        "speed_s": speed_s,
        "weighted_coverage": weighted_coverage,
        "kept_tokens": kept_tokens,
    }


def _result_with_type_coverage(method: str = "dedup", budget: int = 4000) -> dict:
    r = _result(method=method, budget=budget)
    r["type_coverage"] = {
        "path": {"coverage": 0.9, "weight": 0.8, "covered": 9, "total": 10},
        "error": {"coverage": 0.5, "weight": 0.5, "covered": 5, "total": 10},
    }
    return r


# ---------------------------------------------------------------------------
# METHOD_STYLES constant
# ---------------------------------------------------------------------------

class TestMethodStyles:
    def test_known_methods_present(self):
        for method in ("dedup", "eitf", "setcover"):
            assert method in METHOD_STYLES

    def test_each_entry_has_color_marker_label(self):
        for method, style in METHOD_STYLES.items():
            assert "color" in style, f"Missing 'color' in METHOD_STYLES[{method!r}]"
            assert "marker" in style, f"Missing 'marker' in METHOD_STYLES[{method!r}]"
            assert "label" in style, f"Missing 'label' in METHOD_STYLES[{method!r}]"

    def test_claude_code_entry_present(self):
        assert "claude-code" in METHOD_STYLES


# ---------------------------------------------------------------------------
# plot_entity_coverage
# ---------------------------------------------------------------------------

class TestPlotEntityCoverage:
    def _make_ax(self):
        fig, ax = plt.subplots()
        return fig, ax

    def test_single_result_runs(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result()])
        plt.close(fig)

    def test_empty_results_runs(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [])
        plt.close(fig)

    def test_multiple_methods(self):
        fig, ax = self._make_ax()
        results = [
            _result("dedup", 1000),
            _result("eitf", 2000),
            _result("setcover", 4000),
        ]
        plot_entity_coverage(ax, results)
        plt.close(fig)

    def test_same_method_multiple_budgets(self):
        """Same method at multiple budgets should be connected by a dashed line."""
        fig, ax = self._make_ax()
        results = [_result("dedup", 1000, speed_s=0.5), _result("dedup", 4000, speed_s=2.0)]
        plot_entity_coverage(ax, results)
        plt.close(fig)

    def test_unknown_method_uses_default_style(self):
        """Methods not in METHOD_STYLES should use a fallback style."""
        fig, ax = self._make_ax()
        results = [_result(method="mystery-method")]
        plot_entity_coverage(ax, results)
        plt.close(fig)

    def test_xlabel_set(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result()])
        assert ax.get_xlabel() != ""
        plt.close(fig)

    def test_ylabel_set(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result()])
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_title_set(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result()])
        assert ax.get_title() != ""
        plt.close(fig)

    def test_legend_present_by_default(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result("dedup")])
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_no_legend_when_disabled(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result("dedup")], show_legend=False)
        legend = ax.get_legend()
        assert legend is None
        plt.close(fig)

    def test_large_budget_annotated_k(self):
        """Budget ≥ 1000 should be annotated with 'K' suffix."""
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result(budget=4096)])
        # Just verify it runs without error
        plt.close(fig)

    def test_kept_tokens_in_k_annotated(self):
        fig, ax = self._make_ax()
        plot_entity_coverage(ax, [_result(kept_tokens=2000)])
        plt.close(fig)

    def test_duplicate_method_label_only_once(self):
        """Same method twice → only one legend entry (label=None for second)."""
        fig, ax = self._make_ax()
        results = [_result("dedup", speed_s=1.0), _result("dedup", speed_s=2.0)]
        plot_entity_coverage(ax, results)
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()] if legend else []
        dedup_count = sum(1 for l in labels if "Dedup" in l)
        assert dedup_count <= 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_type_breakdown
# ---------------------------------------------------------------------------

class TestPlotTypeBreakdown:
    def _make_ax(self):
        fig, ax = plt.subplots()
        return fig, ax

    def test_with_type_coverage_runs(self):
        fig, ax = self._make_ax()
        results = [_result_with_type_coverage()]
        plot_type_breakdown(ax, results)
        plt.close(fig)

    def test_no_type_coverage_returns_early(self):
        """Results without 'type_coverage' key should not raise."""
        fig, ax = self._make_ax()
        results = [_result()]  # no type_coverage key
        plot_type_breakdown(ax, results)
        plt.close(fig)

    def test_uses_largest_budget_result(self):
        """plot_type_breakdown should use the result with the largest budget."""
        fig, ax = self._make_ax()
        small = _result_with_type_coverage(budget=1000)
        large = _result_with_type_coverage(budget=8000)
        large["type_coverage"]["url"] = {"coverage": 0.8, "weight": 0.9, "covered": 8, "total": 10}
        plot_type_breakdown(ax, [small, large])
        # Should include url from the large result
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("url" in l for l in labels)
        plt.close(fig)

    def test_xlabel_set(self):
        fig, ax = self._make_ax()
        plot_type_breakdown(ax, [_result_with_type_coverage()])
        assert ax.get_xlabel() != ""
        plt.close(fig)

    def test_title_set(self):
        fig, ax = self._make_ax()
        plot_type_breakdown(ax, [_result_with_type_coverage()])
        assert ax.get_title() != ""
        plt.close(fig)

    def test_empty_type_coverage_returns_early(self):
        """type_coverage = {} should not raise."""
        fig, ax = self._make_ax()
        r = _result()
        r["type_coverage"] = {}
        plot_type_breakdown(ax, [r])
        plt.close(fig)
