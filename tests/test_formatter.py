"""Tests for lib/formatter.py — output formatting utilities."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn
from lib.selector import SelectionResult
from lib.types import ScoredTurn
from lib.formatter import write_summary_text, write_compacted_jsonl, write_scores_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system(index: int, text: str = "") -> Turn:
    t = Turn(kind="system", index=index)
    if text:
        t.lines = [{"type": "assistant", "message": {"content": text}}]
    return t


def _user(index: int, text: str = "") -> Turn:
    t = Turn(kind="user", index=index)
    if text:
        t.lines = [{"type": "user", "message": {"content": text}}]
    return t


def _scored(turn: Turn, score: float, tokens: int = 100) -> ScoredTurn:
    return ScoredTurn(turn=turn, score=score, tokens=tokens)


def _result(*turns: Turn) -> SelectionResult:
    return SelectionResult(kept_turns=list(turns))


# ---------------------------------------------------------------------------
# write_summary_text
# ---------------------------------------------------------------------------

class TestWriteSummaryText:
    def test_creates_file(self):
        s = _system(1, "Hello from assistant")
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        assert path.exists()
        path.unlink()

    def test_contains_turn_text(self):
        s = _system(1, "ValueError found in main.py")
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        content = path.read_text()
        assert "ValueError found in main.py" in content
        path.unlink()

    def test_user_and_assistant_labels(self):
        u = _user(0, "What is the error?")
        s = _system(1, "It is a ValueError")
        result = _result(u, s)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        content = path.read_text()
        assert "User" in content
        assert "Assistant" in content
        path.unlink()

    def test_turn_index_in_output(self):
        s = _system(7, "Important context")
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        content = path.read_text()
        assert "7" in content
        path.unlink()

    def test_empty_turns_skipped(self):
        s_empty = _system(1)  # no text
        s_content = _system(2, "Has content")
        result = _result(s_empty, s_content)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        content = path.read_text()
        assert "Has content" in content
        path.unlink()

    def test_long_text_truncated(self):
        long_text = "x" * 5000
        s = _system(1, long_text)
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        content = path.read_text()
        assert "[... truncated]" in content
        path.unlink()

    def test_separator_between_turns(self):
        s1 = _system(1, "First turn content")
        s2 = _system(2, "Second turn content")
        result = _result(s1, s2)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        write_summary_text(result, path)
        content = path.read_text()
        assert "---" in content
        path.unlink()


# ---------------------------------------------------------------------------
# write_compacted_jsonl
# ---------------------------------------------------------------------------

class TestWriteCompactedJsonl:
    def test_creates_file(self):
        s = _system(1, "Hello")
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        write_compacted_jsonl(result, path)
        assert path.exists()
        path.unlink()

    def test_valid_jsonl_output(self):
        s = _system(1, "Hello from system")
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        write_compacted_jsonl(result, path)
        lines = path.read_text().splitlines()
        for line in lines:
            if line.strip():
                json.loads(line)  # should not raise
        path.unlink()

    def test_all_lines_from_kept_turns(self):
        """All records from kept turns should appear in the output."""
        s = _system(1, "Content here")
        result = _result(s)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        write_compacted_jsonl(result, path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(s.lines)
        path.unlink()

    def test_multiple_turns_written(self):
        s1 = _system(1, "First")
        s2 = _system(2, "Second")
        result = _result(s1, s2)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        write_compacted_jsonl(result, path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        path.unlink()


# ---------------------------------------------------------------------------
# write_scores_csv
# ---------------------------------------------------------------------------

class TestWriteScoresCsv:
    def test_creates_file(self):
        s = _system(1, "Content")
        scored = [_scored(s, 0.75)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        write_scores_csv(scored, {1}, path)
        assert path.exists()
        path.unlink()

    def test_header_row(self):
        s = _system(1, "Content")
        scored = [_scored(s, 0.75)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        write_scores_csv(scored, {1}, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            assert "turn_index" in reader.fieldnames
            assert "score" in reader.fieldnames
            assert "tokens" in reader.fieldnames
            assert "kept" in reader.fieldnames
        path.unlink()

    def test_score_value_in_output(self):
        s = _system(1, "Content")
        scored = [_scored(s, 0.8765, tokens=200)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        write_scores_csv(scored, {1}, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["turn_index"] == "1"
        assert "0.8765" in rows[0]["score"]
        assert rows[0]["tokens"] == "200"
        path.unlink()

    def test_kept_flag(self):
        s1 = _system(1, "Kept")
        s2 = _system(2, "Dropped")
        scored = [_scored(s1, 0.9), _scored(s2, 0.1)]
        kept_indices = {1}  # only s1 is kept
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        write_scores_csv(scored, kept_indices, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = {int(r["turn_index"]): r for r in reader}
        assert rows[1]["kept"] == "True"
        assert rows[2]["kept"] == "False"
        path.unlink()

    def test_sorted_by_turn_index(self):
        s3 = _system(3, "Third")
        s1 = _system(1, "First")
        s2 = _system(2, "Second")
        scored = [_scored(s3, 0.3), _scored(s1, 0.1), _scored(s2, 0.2)]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        write_scores_csv(scored, set(), path)
        with open(path) as f:
            reader = csv.DictReader(f)
            indices = [int(r["turn_index"]) for r in reader]
        assert indices == [1, 2, 3]
        path.unlink()
