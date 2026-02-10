"""Entity/fact preservation metric for compaction evaluation.

Inspired by Łajewska et al. (EMNLP 2025 Findings) — "Understanding and
Improving Information Preservation in Prompt Compression for LLMs" — which
found entity preservation to be the most discriminating metric for
compression quality.

Instead of bag-of-words overlap (TF-IDF recall in fitness.py), this metric
extracts structured entities from text — file paths, port numbers, error
codes, function names, URLs, package names, shell commands — and measures
what fraction of suffix-referenced entities survive in the kept prefix turns.

Entity types are weighted by importance for coding agent continuation:
  - error messages and file paths matter most (agent needs these to debug)
  - ports, URLs, commands matter for environment continuity
  - generic identifiers matter less
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

# File paths: /foo/bar, ./foo/bar, ~/foo/bar (but not URLs)
_PATH_RE = re.compile(r"(?<!:/)(?<!//)(?:[./~])?(?:/[\w.\-]+){2,}")

# Port numbers: :8080, port 8080, PORT=8080
_PORT_RE = re.compile(
    r"(?:(?:[Pp]ort|PORT)[= ]+(\d{2,5}))"
    r"|(?::(\d{2,5})(?:[/\s,\)]|$))"
)

# HTTP status codes: 401, 404, 500 etc. in error context
_HTTP_STATUS_RE = re.compile(
    r"\b((?:1|2|3|4|5)\d{2})\b"
    r"(?:\s+(?:Unauthorized|Forbidden|Not Found|Internal Server Error"
    r"|Bad Request|OK|Created|Accepted|No Content|Bad Gateway"
    r"|Service Unavailable|Gateway Timeout|error|Error|ERROR))"
)

# Python/JS exceptions: ValueError, ModuleNotFoundError, TypeError, etc.
_EXCEPTION_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:Error|Exception|Warning|Fault))\b"
)

# Function/method names: snake_case with parens, or dotted.method()
_FUNC_RE = re.compile(
    r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)\s*\("
)

# Class names: CamelCase identifiers (2+ caps transitions)
_CLASS_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"
)

# URLs: http(s)://...
_URL_RE = re.compile(
    r"https?://[^\s<>\"'`\])]+",
)

# Package names: pip install X, npm install X, pacman -S X, yay -S X
# Must start with a letter (not a flag like --noconfirm or -e)
_PACKAGE_RE = re.compile(
    r"(?:pip install|pip3 install|npm install|yarn add"
    r"|pacman -S|yay -S|cargo install"
    r"|gem install|go install)\s+([a-zA-Z][a-zA-Z0-9_\-]{1,})",
)

# Shell commands: common CLI tools at start of line or after $
_COMMAND_RE = re.compile(
    r"(?:^|\$\s+)((?:git|docker|npm|pip|python|node|cargo|make|curl|wget"
    r"|ssh|scp|rsync|kubectl|uv|hyprctl|systemctl"
    r")\s+[a-z][a-z0-9_\- ]{2,40})",
    re.MULTILINE,
)

# Environment variables: FOO_BAR=value or $FOO_BAR
_ENV_VAR_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,})(?:=|\b)",
)


# ---------------------------------------------------------------------------
# Entity types with importance weights
# ---------------------------------------------------------------------------

ENTITY_TYPES: dict[str, float] = {
    "file_path": 1.0,
    "error": 1.0,
    "exception": 0.9,
    "url": 0.8,
    "port": 0.8,
    "command": 0.7,
    "package": 0.7,
    "function": 0.5,
    "class_name": 0.4,
    "env_var": 0.4,
    "http_status": 0.6,
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@dataclass
class EntitySet:
    """Typed entity occurrences extracted from text."""
    entities: dict[str, set[str]] = field(default_factory=dict)
    # entity_type -> set of normalized entity strings

    @property
    def total_count(self) -> int:
        return sum(len(v) for v in self.entities.values())

    def all_entities(self) -> set[tuple[str, str]]:
        """Return set of (type, value) pairs."""
        result: set[tuple[str, str]] = set()
        for etype, values in self.entities.items():
            for v in values:
                result.add((etype, v))
        return result


def extract_entities(text: str) -> EntitySet:
    """Extract structured entities from text.

    Returns an EntitySet with entities grouped by type.
    """
    result = EntitySet()

    def _add(etype: str, value: str) -> None:
        normalized = value.strip().lower()
        if not normalized or len(normalized) < 2:
            return
        result.entities.setdefault(etype, set()).add(normalized)

    # URLs first (so we can exclude URL-derived paths below)
    url_spans: list[tuple[int, int]] = []
    for m in _URL_RE.finditer(text):
        url = m.group().rstrip(".,;:)")
        _add("url", url)
        url_spans.append((m.start(), m.end()))

    # File paths (skip any that overlap with a URL match)
    for m in _PATH_RE.finditer(text):
        if any(s <= m.start() < e for s, e in url_spans):
            continue
        path = m.group().rstrip(".,;:)")
        _add("file_path", path)

    # Ports
    for m in _PORT_RE.finditer(text):
        port = m.group(1) or m.group(2)
        if port:
            port_int = int(port)
            # Filter out unlikely port numbers — keep well-known and common ranges
            # Skip 100-999 (too many false positives), allow 80-99 and 1000+
            if (80 <= port_int <= 99) or (1024 <= port_int <= 65535):
                _add("port", port)

    # HTTP status codes
    for m in _HTTP_STATUS_RE.finditer(text):
        _add("http_status", m.group(1))

    # Exceptions
    for m in _EXCEPTION_RE.finditer(text):
        _add("exception", m.group(1))

    # Functions — filter out very common/short ones
    _SKIP_FUNCS = {"print", "len", "str", "int", "list", "dict", "set",
                   "type", "range", "open", "super", "self", "init",
                   "main", "test", "run", "get", "put", "post"}
    for m in _FUNC_RE.finditer(text):
        fname = m.group(1)
        if fname not in _SKIP_FUNCS and len(fname) >= 4:
            _add("function", fname)

    # Class names
    for m in _CLASS_RE.finditer(text):
        _add("class_name", m.group(1))

    # (URLs already extracted above, before file paths)

    # Packages
    for m in _PACKAGE_RE.finditer(text):
        _add("package", m.group(1))

    # Commands
    for m in _COMMAND_RE.finditer(text):
        _add("command", m.group(1))

    # Environment variables — must look like ENV_VAR (has underscore or is known pattern)
    _SKIP_ENVS = {"HOME", "PATH", "USER", "SHELL", "PWD", "TRUE", "FALSE",
                  "NULL", "NONE", "TODO", "NOTE", "WARN", "INFO", "DEBUG",
                  "ERROR", "PASS", "FAIL", "TYPE", "NAME", "FILE", "DATA",
                  "TEST", "SELF", "ARGS", "KWARGS", "ALSO", "WITH", "FROM",
                  "THEN", "WHEN", "THAT", "THIS", "WILL", "HAVE", "BEEN",
                  "DOES", "WHAT", "EACH", "SOME", "ONLY", "JUST", "MORE",
                  "MOST", "VERY", "INTO", "OVER", "SUCH", "THAN", "THEM",
                  "THESE", "THOSE", "AFTER", "BEFORE", "BETWEEN", "SHOULD"}
    for m in _ENV_VAR_RE.finditer(text):
        var = m.group(1)
        # Must contain an underscore OR be a known env var pattern
        if (var not in _SKIP_ENVS
                and len(var) >= 4
                and ("_" in var or var.startswith(("CUDA", "OPENROUTER",
                     "SIGNAL", "DOCKER", "KUBECONFIG", "PYTHONPATH",
                     "NODE", "RUST", "CARGO", "GOPATH")))):
            _add("env_var", var)

    return result


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

@dataclass
class EntityCoverageResult:
    """Result of entity coverage evaluation."""

    method: str
    budget: int
    speed_s: float

    # Overall scores
    coverage: float         # unweighted: |suffix ∩ kept| / |suffix|
    weighted_coverage: float  # weighted by entity type importance

    # Per-type breakdown
    type_coverage: dict[str, dict] = field(default_factory=dict)
    # entity_type -> {covered, total, coverage, weight}

    # Compression stats
    total_tokens: int = 0
    kept_tokens: int = 0
    compression: float = 0.0

    # Entity counts
    suffix_entity_count: int = 0
    prefix_entity_count: int = 0
    covered_count: int = 0

    @property
    def f1(self) -> float:
        """F1 of coverage and compression efficiency."""
        compression_eff = 1.0 - self.compression
        if self.weighted_coverage + compression_eff == 0:
            return 0.0
        return (2 * self.weighted_coverage * compression_eff
                / (self.weighted_coverage + compression_eff))

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "budget": self.budget,
            "speed_s": self.speed_s,
            "coverage": self.coverage,
            "weighted_coverage": self.weighted_coverage,
            "f1": self.f1,
            "total_tokens": self.total_tokens,
            "kept_tokens": self.kept_tokens,
            "compression": self.compression,
            "suffix_entity_count": self.suffix_entity_count,
            "prefix_entity_count": self.prefix_entity_count,
            "covered_count": self.covered_count,
            "type_coverage": self.type_coverage,
        }


def compute_coverage(
    suffix_entities: EntitySet,
    kept_prefix_entities: EntitySet,
) -> tuple[float, float, dict[str, dict]]:
    """Compute entity coverage of suffix entities by kept prefix entities.

    Returns:
        (unweighted_coverage, weighted_coverage, per_type_breakdown)
    """
    suffix_all = suffix_entities.all_entities()
    kept_all = kept_prefix_entities.all_entities()

    if not suffix_all:
        return 1.0, 1.0, {}

    # For coverage: a suffix entity is "covered" if the same (type, value)
    # exists in the kept prefix
    covered = suffix_all & kept_all

    unweighted = len(covered) / len(suffix_all)

    # Weighted coverage: weight each entity by its type importance
    total_weight = 0.0
    covered_weight = 0.0
    type_breakdown: dict[str, dict] = {}

    for etype in ENTITY_TYPES:
        suffix_of_type = {v for t, v in suffix_all if t == etype}
        kept_of_type = {v for t, v in kept_all if t == etype}

        if not suffix_of_type:
            continue

        covered_of_type = suffix_of_type & kept_of_type
        type_weight = ENTITY_TYPES[etype]
        type_cov = len(covered_of_type) / len(suffix_of_type)

        total_weight += type_weight * len(suffix_of_type)
        covered_weight += type_weight * len(covered_of_type)

        type_breakdown[etype] = {
            "covered": len(covered_of_type),
            "total": len(suffix_of_type),
            "coverage": type_cov,
            "weight": type_weight,
        }

    weighted = covered_weight / total_weight if total_weight > 0 else 1.0

    return unweighted, weighted, type_breakdown


# ---------------------------------------------------------------------------
# Top-level evaluation function (mirrors fitness.evaluate interface)
# ---------------------------------------------------------------------------

def evaluate(
    turns: list,
    method: str,
    budget: int = 80_000,
    split_ratio: float = 0.70,
    short_threshold: int = 300,
    min_repeat_len: int = 64,
    device: str = "cpu",
    batch_size: int = 16,
    embed_url: str = "http://localhost:8080",
    rerank_url: str = "http://localhost:8181",
) -> EntityCoverageResult:
    """Run a compaction method and evaluate entity preservation.

    Same interface as fitness.evaluate — splits conversation, compacts the
    prefix, then measures what fraction of suffix-referenced entities survive
    in the kept prefix turns.
    """
    import time
    from ..parser import Turn, extract_text
    from ..tokenizer import turn_tokens
    from ..types import ScoredTurn, build_query
    from ..selector import select_turns

    # --- 1. Split into prefix and suffix ---
    split_idx = int(len(turns) * split_ratio)
    while split_idx < len(turns) and turns[split_idx].kind != "user":
        split_idx += 1

    prefix_turns = turns[:split_idx]
    suffix_turns = turns[split_idx:]

    if not prefix_turns or not suffix_turns:
        raise ValueError(
            f"Split at {split_ratio:.0%} ({split_idx}/{len(turns)}) "
            f"produced empty prefix or suffix"
        )

    # Re-index prefix turns
    for i, t in enumerate(prefix_turns):
        t.index = i

    # --- 2. Extract entities from suffix ---
    suffix_texts = [extract_text(t) for t in suffix_turns if t.kind == "system"]
    suffix_combined = "\n".join(suffix_texts)
    suffix_entities = extract_entities(suffix_combined)

    if suffix_entities.total_count == 0:
        raise ValueError("No entities extracted from suffix")

    # --- 3. Token counts ---
    token_counts: dict[int, int] = {}
    for t in prefix_turns:
        token_counts[t.index] = turn_tokens(t)

    total_prefix_tokens = sum(token_counts.values())

    # --- 4. Run compaction ---
    prefix_system = [t for t in prefix_turns if t.kind == "system"]
    prefix_long = [
        t for t in prefix_system
        if token_counts.get(t.index, 0) > short_threshold
    ]

    t_start = time.monotonic()

    scored: list[ScoredTurn]

    if method == "dedup":
        from ..dedup import dedup_scores
        scored = dedup_scores(
            prefix_turns, prefix_long, token_counts,
            min_repeat_len=min_repeat_len,
        )
    elif method == "embed":
        from ..scorer import Scorer
        user_turns = [t for t in prefix_turns if t.kind == "user"]
        scorer = Scorer(device=device)
        query = build_query(user_turns)
        scored = scorer.score_turns(
            prefix_long, query, token_counts, batch_size=batch_size,
        )
    elif method == "llama-embed":
        from ..llama_embed import LlamaEmbedScorer
        user_turns = [t for t in prefix_turns if t.kind == "user"]
        scorer = LlamaEmbedScorer(base_url=embed_url)
        query = build_query(user_turns)
        scored = scorer.score_turns(
            prefix_long, query, token_counts, batch_size=batch_size,
        )
    elif method == "llama-rerank":
        from ..llama_rerank import LlamaRerankScorer
        user_turns = [t for t in prefix_turns if t.kind == "user"]
        scorer = LlamaRerankScorer(base_url=rerank_url)
        query = build_query(user_turns)
        scored = scorer.score_turns(prefix_long, query, token_counts)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = select_turns(
        turns=prefix_turns,
        scored=scored,
        token_counts=token_counts,
        budget=budget,
        short_threshold=short_threshold,
    )

    t_elapsed = time.monotonic() - t_start

    # --- 5. Extract entities from kept prefix turns ---
    kept_texts = [extract_text(t) for t in result.kept_turns]
    kept_combined = "\n".join(kept_texts)
    kept_entities = extract_entities(kept_combined)

    # --- 6. Compute coverage ---
    coverage, weighted_coverage, type_breakdown = compute_coverage(
        suffix_entities, kept_entities,
    )

    kept_tokens = sum(token_counts.get(t.index, 0) for t in result.kept_turns)

    return EntityCoverageResult(
        method=method,
        budget=budget,
        speed_s=t_elapsed,
        coverage=coverage,
        weighted_coverage=weighted_coverage,
        type_coverage=type_breakdown,
        total_tokens=total_prefix_tokens,
        kept_tokens=kept_tokens,
        compression=kept_tokens / total_prefix_tokens if total_prefix_tokens > 0 else 0,
        suffix_entity_count=suffix_entities.total_count,
        prefix_entity_count=kept_entities.total_count,
        covered_count=len(suffix_entities.all_entities() & kept_entities.all_entities()),
    )
