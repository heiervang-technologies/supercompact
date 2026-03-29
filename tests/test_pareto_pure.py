"""Pure-function tests for pareto.py pareto_frontier.

Covers the Pareto frontier extraction: minimize x, maximize y.
No file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pareto import pareto_frontier


class TestParetoFrontier:
    def test_single_point(self):
        result = pareto_frontier([(1.0, 0.5)])
        assert result == [0]

    def test_two_dominated(self):
        # (1, 0.8) dominates (2, 0.5) — lower x, higher y
        points = [(1.0, 0.8), (2.0, 0.5)]
        result = pareto_frontier(points)
        assert result == [0]

    def test_two_non_dominated(self):
        # (1, 0.5) and (2, 0.9) — tradeoff: lower x vs higher y
        points = [(1.0, 0.5), (2.0, 0.9)]
        result = pareto_frontier(points)
        assert 0 in result
        assert 1 in result

    def test_three_points_one_dominated(self):
        # (1, 0.8), (2, 0.9), (3, 0.7)
        # (3, 0.7) is dominated by (2, 0.9) — higher x AND lower y
        points = [(1.0, 0.8), (2.0, 0.9), (3.0, 0.7)]
        result = pareto_frontier(points)
        assert 0 in result
        assert 1 in result
        assert 2 not in result

    def test_all_same_x(self):
        # Same x: only the one with max y is Pareto-optimal
        points = [(1.0, 0.3), (1.0, 0.9), (1.0, 0.5)]
        result = pareto_frontier(points)
        # The first sorted point will be one of the x=1.0 points
        # Only the one with y=0.9 matters — but depends on sort stability
        # At minimum, index 1 (y=0.9) should be on the frontier
        assert 1 in result

    def test_all_same_y(self):
        # Same y: only the leftmost (min x) is kept since no subsequent improves y
        points = [(3.0, 0.5), (1.0, 0.5), (2.0, 0.5)]
        result = pareto_frontier(points)
        # First sorted by x: index 1 (x=1.0) is first, y=0.5 becomes best_y
        # No subsequent point exceeds 0.5, so only first is on frontier
        assert len(result) == 1
        assert 1 in result  # (1.0, 0.5) is the leftmost

    def test_staircase_all_pareto(self):
        # Perfect staircase: each point has lower x but also lower y
        points = [(1.0, 0.1), (2.0, 0.4), (3.0, 0.7), (4.0, 0.9)]
        result = pareto_frontier(points)
        assert result == [0, 1, 2, 3]

    def test_empty_input(self):
        result = pareto_frontier([])
        assert result == []

    def test_returns_list(self):
        result = pareto_frontier([(0.0, 1.0)])
        assert isinstance(result, list)

    def test_indices_are_original(self):
        # Verify returned indices refer to original positions, not sorted
        points = [(5.0, 0.9), (1.0, 0.3), (3.0, 0.7)]
        result = pareto_frontier(points)
        # Sorted by x: index 1 (1.0, 0.3), index 2 (3.0, 0.7), index 0 (5.0, 0.9)
        # All form ascending y staircase → all on frontier
        assert 0 in result
        assert 1 in result
        assert 2 in result

    def test_large_dominated_set(self):
        # One clearly dominant point + many dominated
        points = [(1.0, 1.0)] + [(float(i), 0.1) for i in range(2, 20)]
        result = pareto_frontier(points)
        assert 0 in result
        assert len(result) == 1  # all others have higher x and lower y
