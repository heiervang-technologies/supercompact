"""Tests for lib.formatter — write_summary_text, write_compacted_jsonl, write_scores_csv."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.types import ScoredTurn
from lib.selector import SelectionResult
from lib.formatter import write_summary_text, write_compacted_jsonl, write_scores_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(index: int, text: str) -> Turn:
    t = Turn(kind="user", index=index)
    t.lines = [{"message": {"content": text}}]
    return t


def _system(index: int, text: str) -> Turn:
    t = Turn(kind="system", index=index)
    t.lines = [{"message": {"content": text}}]
    return t


def _scored(turn: Turn, score: float, tokens: int) -> ScoredTurn:
    return ScoredTurn(turn=turn, score=score, tokens=tokens)


def _result(turns: list[Turn]) -> SelectionResult:
    r = SelectionResult()
    r.kept_turns = turns
    return r


# ---------------------------------------------------------------------------
# write_summary_text
# ---------------------------------------------------------------------------

class TestWriteSummaryText:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _result([_user(0, "hello")])
        write_summary_text(r, path)
        assert path.exists()

    def test_user_turn_labeled_user(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _result([_user(0, "what is python?")])
        write_summary_text(r, path)
        text = path.read_text()
        assert "User" in text
        assert "what is python?" in text

    def test_system_turn_labeled_assistant(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _result([_system(0, "python is a language")])
        write_summary_text(r, path)
        text = path.read_text()
        assert "Assistant" in text

    def test_separator_between_turns(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _result([_user(0, "first"), _system(1, "second")])
        write_summary_text(r, path)
        text = path.read_text()
        assert "---" in text

    def test_empty_text_turn_skipped(self, tmp_path):
        path = tmp_path / "summary.txt"
        t = Turn(kind="user", index=0)
        t.lines = []  # no content → empty text
        r = _result([t, _user(1, "real content")])
        write_summary_text(r, path)
        text = path.read_text()
        assert "real content" in text

    def test_long_text_truncated(self, tmp_path):
        path = tmp_path / "summary.txt"
        long_text = "x" * 10000
        r = _result([_system(0, long_text)])
        write_summary_text(r, path)
        text = path.read_text()
        assert "[... truncated]" in text

    def test_empty_result_creates_empty_file(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _result([])
        write_summary_text(r, path)
        assert path.exists()

    def test_includes_turn_index(self, tmp_path):
        path = tmp_path / "summary.txt"
        r = _result([_user(7, "question")])
        write_summary_text(r, path)
        text = path.read_text()
        assert "7" in text


# ---------------------------------------------------------------------------
# write_compacted_jsonl
# ---------------------------------------------------------------------------

class TestWriteCompactedJsonl:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        r = _result([_user(0, "hello")])
        write_compacted_jsonl(r, path)
        assert path.exists()

    def test_each_line_is_valid_json(self, tmp_path):
        path = tmp_path / "out.jsonl"
        r = _result([_user(0, "hello"), _system(1, "world")])
        write_compacted_jsonl(r, path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_line_count_matches_records(self, tmp_path):
        path = tmp_path / "out.jsonl"
        # Two turns, each with 1 line = 2 total records
        r = _result([_user(0, "q"), _system(1, "a")])
        write_compacted_jsonl(r, path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_empty_result_creates_empty_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        r = _result([])
        write_compacted_jsonl(r, path)
        assert path.read_text().strip() == ""

    def test_records_preserved(self, tmp_path):
        path = tmp_path / "out.jsonl"
        t = Turn(kind="user", index=0)
        record = {"type": "user", "message": {"content": "preserved text"}}
        t.lines = [record]
        r = _result([t])
        write_compacted_jsonl(r, path)
        obj = json.loads(path.read_text().strip())
        assert obj["type"] == "user"


# ---------------------------------------------------------------------------
# write_scores_csv
# ---------------------------------------------------------------------------

class TestWriteScoresCsv:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "scores.csv"
        t = _system(0, "hello")
        scored = [_scored(t, 0.75, 100)]
        write_scores_csv(scored, {0}, path)
        assert path.exists()

    def test_header_present(self, tmp_path):
        path = tmp_path / "scores.csv"
        write_scores_csv([], set(), path)
        text = path.read_text()
        assert "turn_index" in text
        assert "score" in text
        assert "tokens" in text
        assert "kept" in text

    def test_kept_flag_true_when_in_set(self, tmp_path):
        path = tmp_path / "scores.csv"
        t = _system(3, "kept turn")
        scored = [_scored(t, 0.9, 200)]
        write_scores_csv(scored, {3}, path)
        with open(path) as f:
            reader = list(csv.DictReader(f))
        assert reader[0]["kept"] == "True"

    def test_kept_flag_false_when_not_in_set(self, tmp_path):
        path = tmp_path / "scores.csv"
        t = _system(5, "dropped turn")
        scored = [_scored(t, 0.1, 100)]
        write_scores_csv(scored, {99}, path)
        with open(path) as f:
            reader = list(csv.DictReader(f))
        assert reader[0]["kept"] == "False"

    def test_score_written_correctly(self, tmp_path):
        path = tmp_path / "scores.csv"
        t = _system(0, "x")
        scored = [_scored(t, 0.1234, 50)]
        write_scores_csv(scored, set(), path)
        with open(path) as f:
            reader = list(csv.DictReader(f))
        assert float(reader[0]["score"]) == pytest.approx(0.1234, abs=1e-3)

    def test_sorted_by_turn_index(self, tmp_path):
        path = tmp_path / "scores.csv"
        turns = [_system(i, f"t{i}") for i in [3, 1, 2]]
        scored = [_scored(t, 0.5, 100) for t in turns]
        write_scores_csv(scored, set(), path)
        with open(path) as f:
            reader = list(csv.DictReader(f))
        indices = [int(r["turn_index"]) for r in reader]
        assert indices == sorted(indices)

    def test_empty_scored_writes_only_header(self, tmp_path):
        path = tmp_path / "scores.csv"
        write_scores_csv([], set(), path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header only


import pytest
