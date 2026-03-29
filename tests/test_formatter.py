"""Tests for lib/formatter.py — write_summary_text, write_compacted_jsonl, write_scores_csv."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.formatter import write_compacted_jsonl, write_scores_csv, write_summary_text
from lib.parser import Turn
from lib.selector import SelectionResult
from lib.types import ScoredTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(kind: str, index: int, text: str = "content") -> Turn:
    t = Turn(kind=kind, index=index)
    t.append({"message": {"content": text}})
    return t


def _selection(*turns: Turn) -> SelectionResult:
    sr = SelectionResult()
    sr.kept_turns = list(turns)
    return sr


def _scored(turn: Turn, score: float, tokens: int) -> ScoredTurn:
    return ScoredTurn(turn=turn, score=score, tokens=tokens)


# ---------------------------------------------------------------------------
# write_summary_text
# ---------------------------------------------------------------------------

class TestWriteSummaryText:
    def test_creates_file(self, tmp_path):
        result = _selection(_turn("user", 0, "hello"))
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        assert path.exists()

    def test_empty_selection_creates_empty_file(self, tmp_path):
        result = _selection()
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        assert path.read_text() == ""

    def test_user_turn_labeled(self, tmp_path):
        result = _selection(_turn("user", 0, "what is the fix?"))
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        text = path.read_text()
        assert "User" in text
        assert "what is the fix?" in text

    def test_system_turn_labeled_assistant(self, tmp_path):
        result = _selection(_turn("system", 1, "here is the fix"))
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        text = path.read_text()
        assert "Assistant" in text
        assert "here is the fix" in text

    def test_turn_index_in_output(self, tmp_path):
        result = _selection(_turn("user", 7, "msg"))
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        assert "7" in path.read_text()

    def test_multiple_turns_separated(self, tmp_path):
        result = _selection(
            _turn("user", 0, "question"),
            _turn("system", 1, "answer"),
        )
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        text = path.read_text()
        assert "question" in text
        assert "answer" in text
        assert "---" in text

    def test_long_turn_truncated(self, tmp_path):
        long_text = "x" * 5000
        result = _selection(_turn("user", 0, long_text))
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        text = path.read_text()
        assert "truncated" in text

    def test_blank_turn_skipped(self, tmp_path):
        t = Turn(kind="user", index=0)  # no content lines
        result = _selection(t)
        path = tmp_path / "summary.txt"
        write_summary_text(result, path)
        # Empty turn should be skipped → empty file
        assert path.read_text() == ""


# ---------------------------------------------------------------------------
# write_compacted_jsonl
# ---------------------------------------------------------------------------

class TestWriteCompactedJsonl:
    def test_creates_file(self, tmp_path):
        result = _selection(_turn("user", 0))
        path = tmp_path / "out.jsonl"
        write_compacted_jsonl(result, path)
        assert path.exists()

    def test_empty_selection_creates_empty_file(self, tmp_path):
        result = _selection()
        path = tmp_path / "out.jsonl"
        write_compacted_jsonl(result, path)
        assert path.read_text().strip() == ""

    def test_each_line_is_valid_json(self, tmp_path):
        result = _selection(_turn("user", 0, "hello"), _turn("system", 1, "world"))
        path = tmp_path / "out.jsonl"
        write_compacted_jsonl(result, path)
        for line in path.read_text().splitlines():
            if line.strip():
                json.loads(line)  # should not raise

    def test_records_written_in_turn_order(self, tmp_path):
        t0 = _turn("user", 0, "first")
        t1 = _turn("system", 1, "second")
        result = _selection(t0, t1)
        path = tmp_path / "out.jsonl"
        write_compacted_jsonl(result, path)
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        contents = [l.get("message", {}).get("content", "") for l in lines]
        assert contents[0] == "first"
        assert contents[1] == "second"

    def test_multiple_records_per_turn(self, tmp_path):
        t = Turn(kind="user", index=0)
        t.append({"message": {"content": "rec1"}})
        t.append({"message": {"content": "rec2"}})
        result = _selection(t)
        path = tmp_path / "out.jsonl"
        write_compacted_jsonl(result, path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# write_scores_csv
# ---------------------------------------------------------------------------

class TestWriteScoresCsv:
    def test_creates_file(self, tmp_path):
        t = _turn("system", 0)
        scored = [_scored(t, 0.75, 200)]
        path = tmp_path / "scores.csv"
        write_scores_csv(scored, {0}, path)
        assert path.exists()

    def test_header_row(self, tmp_path):
        path = tmp_path / "scores.csv"
        write_scores_csv([], set(), path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "turn_index" in header
        assert "score" in header
        assert "tokens" in header
        assert "kept" in header

    def test_score_written(self, tmp_path):
        t = _turn("system", 5, "content")
        scored = [_scored(t, 0.9876, 300)]
        path = tmp_path / "scores.csv"
        write_scores_csv(scored, {5}, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["score"]) == pytest.approx(0.9876, abs=0.001)

    def test_kept_flag_true_when_in_kept_indices(self, tmp_path):
        t = _turn("system", 3)
        scored = [_scored(t, 0.5, 100)]
        path = tmp_path / "scores.csv"
        write_scores_csv(scored, {3}, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["kept"] == "True"

    def test_kept_flag_false_when_not_in_kept_indices(self, tmp_path):
        t = _turn("system", 4)
        scored = [_scored(t, 0.3, 50)]
        path = tmp_path / "scores.csv"
        write_scores_csv(scored, set(), path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["kept"] == "False"

    def test_sorted_by_turn_index(self, tmp_path):
        t5 = _turn("system", 5)
        t2 = _turn("system", 2)
        scored = [_scored(t5, 0.9, 100), _scored(t2, 0.1, 200)]
        path = tmp_path / "scores.csv"
        write_scores_csv(scored, set(), path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        indices = [int(r["turn_index"]) for r in rows]
        assert indices == sorted(indices)
