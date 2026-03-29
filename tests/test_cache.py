"""Tests for lib/eval/cache.py — probe caching."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.cache import conv_hash, load_probes, save_probes, _cache_path
from lib.eval.probes import Probe, ProbeSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_set(conv_hash_val: str = "abc123", version: str = "1") -> ProbeSet:
    return ProbeSet(
        probes=[
            Probe(
                id="p1", dimension="error_solution", tier="factual",
                question="Q?", gold_answer="A", difficulty="medium",
            )
        ],
        conv_hash=conv_hash_val,
        split_ratio=0.7,
        version=version,
    )


# ---------------------------------------------------------------------------
# conv_hash
# ---------------------------------------------------------------------------

class TestConvHash:
    def test_returns_16_char_hex(self, tmp_path):
        f = tmp_path / "conv.jsonl"
        f.write_text('{"type": "user"}\n')
        result = conv_hash(f, 0.7)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_file_same_ratio_deterministic(self, tmp_path):
        f = tmp_path / "conv.jsonl"
        f.write_bytes(b"hello\n" * 100)
        h1 = conv_hash(f, 0.7)
        h2 = conv_hash(f, 0.7)
        assert h1 == h2

    def test_different_ratio_different_hash(self, tmp_path):
        f = tmp_path / "conv.jsonl"
        f.write_bytes(b"hello\n" * 100)
        h1 = conv_hash(f, 0.7)
        h2 = conv_hash(f, 0.8)
        assert h1 != h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text("content A\n" * 20)
        f2.write_text("content B\n" * 20)
        h1 = conv_hash(f1, 0.7)
        h2 = conv_hash(f2, 0.7)
        assert h1 != h2

    def test_large_file_reads_head_and_tail(self, tmp_path):
        """Large files (>4KB) should still produce a valid hash."""
        f = tmp_path / "large.jsonl"
        f.write_bytes(b"x" * 10_000)
        result = conv_hash(f, 0.7)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# _cache_path
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_contains_key(self, tmp_path):
        path = _cache_path(tmp_path, "mykey123", "1")
        assert "mykey123" in path.name

    def test_contains_version(self, tmp_path):
        path = _cache_path(tmp_path, "key", "2")
        assert "v2" in path.name

    def test_has_json_extension(self, tmp_path):
        path = _cache_path(tmp_path, "key", "1")
        assert path.suffix == ".json"

    def test_under_cache_dir(self, tmp_path):
        path = _cache_path(tmp_path / "cache", "key", "1")
        assert str(tmp_path / "cache") in str(path)


# ---------------------------------------------------------------------------
# save_probes / load_probes round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadProbes:
    def test_save_creates_file(self, tmp_path):
        ps = _probe_set()
        save_probes(tmp_path, ps)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_save_creates_dir(self, tmp_path):
        nested = tmp_path / "nested" / "cache"
        ps = _probe_set()
        save_probes(nested, ps)
        assert nested.exists()

    def test_save_returns_path(self, tmp_path):
        ps = _probe_set()
        path = save_probes(tmp_path, ps)
        assert isinstance(path, Path)
        assert path.exists()

    def test_load_missing_returns_none(self, tmp_path):
        result = load_probes(tmp_path, "nonexistent", "1")
        assert result is None

    def test_round_trip(self, tmp_path):
        ps = _probe_set(conv_hash_val="deadbeef", version="2")
        save_probes(tmp_path, ps)
        loaded = load_probes(tmp_path, "deadbeef", "2")
        assert loaded is not None
        assert loaded.conv_hash == "deadbeef"
        assert loaded.version == "2"

    def test_round_trip_preserves_probes(self, tmp_path):
        ps = _probe_set()
        save_probes(tmp_path, ps)
        loaded = load_probes(tmp_path, ps.conv_hash, ps.version)
        assert len(loaded.probes) == 1
        assert loaded.probes[0].id == "p1"

    def test_wrong_version_returns_none(self, tmp_path):
        ps = _probe_set(version="1")
        save_probes(tmp_path, ps)
        # Try loading with a different version
        result = load_probes(tmp_path, ps.conv_hash, "2")
        assert result is None

    def test_split_ratio_preserved(self, tmp_path):
        ps = ProbeSet(probes=[], conv_hash="xyz", split_ratio=0.85, version="1")
        save_probes(tmp_path, ps)
        loaded = load_probes(tmp_path, "xyz", "1")
        assert abs(loaded.split_ratio - 0.85) < 1e-9
