"""Disk caching for probe sets, keyed by (conv_hash, split_ratio, version)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .probes import ProbeSet


DEFAULT_CACHE_DIR = Path("eval_cache")


def conv_hash(jsonl_path: Path, split_ratio: float) -> str:
    """Compute a stable hash for a conversation file + split ratio.

    Uses file size + first/last 4KB + split_ratio to avoid reading
    the entire file for large conversations.
    """
    stat = jsonl_path.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(f"{split_ratio:.4f}".encode())

    with open(jsonl_path, "rb") as f:
        h.update(f.read(4096))
        if stat.st_size > 4096:
            f.seek(max(0, stat.st_size - 4096))
            h.update(f.read(4096))

    return h.hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str, version: str) -> Path:
    return cache_dir / f"probes_{key}_v{version}.json"


def load_probes(cache_dir: Path, key: str, version: str = "1") -> ProbeSet | None:
    """Load a cached ProbeSet, or return None if not found."""
    path = _cache_path(cache_dir, key, version)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ProbeSet.from_dict(data)


def save_probes(cache_dir: Path, probe_set: ProbeSet) -> Path:
    """Save a ProbeSet to disk. Returns the cache file path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, probe_set.conv_hash, probe_set.version)
    path.write_text(json.dumps(probe_set.to_dict(), indent=2))
    return path
