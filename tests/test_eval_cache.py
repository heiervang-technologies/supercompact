"""Tests for lib/eval/cache.py — conv_hash, load_probes, save_probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.cache import conv_hash, load_probes, save_probes, _cache_path
from lib.eval.probes import ProbeSet, Probe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_probeset(**kwargs) -> ProbeSet:
    defaults = dict(
        probes=[],
        conv_hash="abc123",
        split_ratio=0.70,
        version="1",
    )
    defaults.update(kwargs)
    return ProbeSet(**defaults)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# conv_hash
# ---------------------------------------------------------------------------

class TestConvHash:
    def test_returns_16_char_hex(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        p.write_text('{"a": 1}\n')
        result = conv_hash(p, 0.7)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_file(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n')
        h1 = conv_hash(p, 0.7)
        h2 = conv_hash(p, 0.7)
        assert h1 == h2

    def test_different_split_ratio_gives_different_hash(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        p.write_text('{"x": 1}\n')
        h1 = conv_hash(p, 0.7)
        h2 = conv_hash(p, 0.8)
        assert h1 != h2

    def test_different_content_gives_different_hash(self, tmp_path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        p1.write_text('{"a": 1}\n')
        p2.write_text('{"b": 9999}\n')
        h1 = conv_hash(p1, 0.7)
        h2 = conv_hash(p2, 0.7)
        assert h1 != h2

    def test_large_file_hashed(self, tmp_path):
        """File larger than 4096 bytes should still produce a valid hash."""
        p = tmp_path / "large.jsonl"
        p.write_text('{"line": ' + '"x" * 10}\n' * 1000)
        result = conv_hash(p, 0.7)
        assert len(result) == 16

    def test_empty_file_hashed(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = conv_hash(p, 0.7)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# _cache_path
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_returns_path_object(self, tmp_path):
        result = _cache_path(tmp_path, "abc123", "1")
        assert isinstance(result, Path)

    def test_path_contains_key_and_version(self, tmp_path):
        result = _cache_path(tmp_path, "abc123", "2")
        assert "abc123" in result.name
        assert "v2" in result.name

    def test_path_under_cache_dir(self, tmp_path):
        result = _cache_path(tmp_path / "mydir", "key", "1")
        assert result.parent == tmp_path / "mydir"

    def test_path_is_json_file(self, tmp_path):
        result = _cache_path(tmp_path, "key", "1")
        assert result.suffix == ".json"


# ---------------------------------------------------------------------------
# load_probes / save_probes roundtrip
# ---------------------------------------------------------------------------

class TestLoadSave:
    def test_load_missing_returns_none(self, tmp_path):
        result = load_probes(tmp_path, "nonexistent_key", "1")
        assert result is None

    def test_save_creates_file(self, tmp_path):
        ps = _sample_probeset()
        path = save_probes(tmp_path, ps)
        assert path.exists()

    def test_save_returns_path(self, tmp_path):
        ps = _sample_probeset()
        result = save_probes(tmp_path, ps)
        assert isinstance(result, Path)

    def test_roundtrip_empty_probeset(self, tmp_path):
        ps = _sample_probeset(conv_hash="deadbeef", version="3")
        save_probes(tmp_path, ps)
        loaded = load_probes(tmp_path, "deadbeef", "3")
        assert loaded is not None
        assert loaded.conv_hash == "deadbeef"
        assert loaded.version == "3"
        assert loaded.probes == []

    def test_roundtrip_with_probes(self, tmp_path):
        probe = Probe(
            id="p1",
            dimension="progress",
            tier="factual",
            question="Q?",
            gold_answer="A",
            evidence_turns=[1, 2],
            difficulty="hard",
        )
        ps = _sample_probeset(probes=[probe], conv_hash="feed1234", version="1")
        save_probes(tmp_path, ps)
        loaded = load_probes(tmp_path, "feed1234", "1")
        assert loaded is not None
        assert len(loaded.probes) == 1
        assert loaded.probes[0].id == "p1"
        assert loaded.probes[0].difficulty == "hard"
        assert loaded.probes[0].evidence_turns == [1, 2]

    def test_save_creates_cache_dir_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "cache"
        ps = _sample_probeset()
        save_probes(nested, ps)
        assert nested.is_dir()

    def test_version_mismatch_returns_none(self, tmp_path):
        ps = _sample_probeset(conv_hash="key1", version="1")
        save_probes(tmp_path, ps)
        loaded = load_probes(tmp_path, "key1", "2")
        assert loaded is None

    def test_file_is_valid_json(self, tmp_path):
        ps = _sample_probeset(conv_hash="jsonkey", version="1")
        path = save_probes(tmp_path, ps)
        data = json.loads(path.read_text())
        assert "probes" in data
        assert "conv_hash" in data
