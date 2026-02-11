# supercompact

Harder better faster stronger compacting for your AI agent.

**supercompact** is a conversation compaction tool for AI coding agents. It takes Claude Code session transcripts (JSONL) and compacts them to fit within a token budget while preserving the entities and context the agent needs to keep working — file paths, error messages, function names, commands, URLs, and more.

Unlike Claude Code's built-in `/compact` (which summarizes via LLM), supercompact uses **score-and-select**: it scores every assistant turn by relevance, then greedily selects the most important turns to keep within budget. The original turns are preserved verbatim — nothing is paraphrased or lost.

## Why?

When a Claude Code session gets long, context compaction becomes critical. The built-in `/compact` command uses LLM summarization, which:

- Is slow (~30s+ per compaction)
- Loses exact technical details (file paths get paraphrased, error messages get summarized)
- Costs API tokens for the summarization call itself

supercompact's EITF method runs in **<1 second** on any hardware (no GPU, no API calls) and preserves **~2x more entities** than LLM summarization at the same token budget.

## Installation

Requires Python 3.11+. Install with [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url>
cd supercompact
uv sync
```

For the `embed` method (local PyTorch scorer):

```bash
uv sync --extra torch
```

The `llama-embed` and `llama-rerank` methods require a running [llama.cpp](https://github.com/ggerganov/llama.cpp) server (see [Embedding & Reranking Methods](#embedding--reranking-methods)).

## Quick Start

```bash
# Compact a conversation to 80k tokens using EITF (default)
uv run python compact.py session.jsonl --method eitf --budget 80000 --output compacted.jsonl

# Try different methods
uv run python compact.py session.jsonl --method setcover --budget 60000 --output compacted.jsonl
uv run python compact.py session.jsonl --method dedup --output compacted.jsonl

# Evaluate methods against each other
uv run python compact.py evaluate session.jsonl --method all --budget 100000 --eval-output results.json

# Generate Pareto plots from evaluation results
uv run python compact.py plot results.json -o pareto.png
```

## Compaction Methods

All methods follow the same pipeline: **score** each assistant turn, then **select** turns greedily by score until the token budget is filled. User turns and short system turns (<=300 tokens) are always kept.

### Local Methods (no model, instant)

| Method | How it scores | Best for |
|--------|--------------|----------|
| **eitf** | Entity-frequency Inverse Turn Frequency. Extracts structured entities (file paths, errors, functions, etc.) and scores turns by weighted entity importance × rarity. BM25-style length normalization. | General use. Fast, good entity preservation. |
| **setcover** | EITF + exclusivity bonus. Entities that appear in only 1-2 system turns get a 20% score boost, since dropping that turn loses them entirely. | Slightly better coverage than EITF at tight budgets. |
| **dedup** | Suffix automaton deduplication. Builds an O(n) suffix automaton over the full conversation, scores turns by unique content ratio. Turns with mostly-repeated content score low. | Removing redundant tool output, repeated errors. |

### Embedding & Reranking Methods (need llama.cpp server)

| Method | How it scores | Requirements |
|--------|--------------|-------------|
| **llama-embed** | Qwen3-Embedding-0.6B cosine similarity. Embeds a query (recent user messages) and all system turns, ranks by similarity. | `llama-server` on port 8080 with Qwen3-Embedding-0.6B GGUF |
| **llama-rerank** | Qwen3-Reranker-0.6B cross-encoder. Sends query + document pairs to the reranker server for direct relevance scoring. | `llama-server` on port 8181 with `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` |
| **embed** | Same as llama-embed but runs Qwen3-Embedding-0.6B locally via PyTorch. | `torch` extra installed, GPU recommended |

### LLM Baseline

| Method | How it works | Requirements |
|--------|-------------|-------------|
| **claude-code** | Sends the conversation to Claude Sonnet via OpenRouter for LLM summarization. Simulates `/compact` behavior. Used as a baseline in evaluations. | `OPENROUTER_API_KEY` env var |

## CLI Reference

supercompact has three subcommands: `compact` (default), `evaluate`, and `plot`.

### compact

Score and select turns to fit a token budget.

```
uv run python compact.py [compact] FILE.jsonl [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `eitf` | Scoring method: `dedup`, `eitf`, `setcover`, `embed`, `llama-embed`, `llama-rerank` |
| `--budget` | `80000` | Target token budget |
| `--output` | — | Write compacted JSONL to this path |
| `--format` | `jsonl` | Output format: `jsonl` or `summary` (text for Claude context) |
| `--short-threshold` | `300` | System turns <= this token count are always kept |
| `--min-repeat-len` | `64` | Minimum repeated substring length for dedup scoring |
| `--scores-file` | — | Write per-turn scores to CSV |
| `--dry-run` | — | Use random scores (for testing the pipeline) |
| `--verbose` | — | Show detailed breakdown |
| `--device` | `cpu` | PyTorch device for `embed` method |
| `--batch-size` | `16` | Embedding batch size |
| `--embed-url` | `http://localhost:8080` | llama.cpp embedding server URL |
| `--rerank-url` | `http://localhost:8181` | llama.cpp reranker server URL |

### evaluate

Run entity preservation evaluation across methods and budgets.

```
uv run python compact.py evaluate FILE.jsonl [options]
```

Splits the conversation into prefix (70%) and suffix (30%). Compacts the prefix, then measures what fraction of suffix-referenced entities survive in the kept turns.

Additional options beyond `compact`:

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `eitf` | Use `all` to evaluate every method |
| `--split-ratio` | `0.70` | Prefix/suffix split ratio |
| `--probe-cache` | `eval_cache/` | Directory for cached LLM-as-Judge probe sets |
| `--eval-output` | — | Export results as JSON |

### plot

Generate Pareto plots from evaluation result JSON files.

```
uv run python compact.py plot results1.json [results2.json ...] [-o output.png]
```

## Evaluation Framework

supercompact includes a multi-dimensional evaluation framework for comparing compaction methods.

### Entity Coverage

The primary metric. Inspired by [Łajewska et al. (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.0/) — entity preservation is the most discriminating metric for compression quality.

Extracts structured entities from text using regex patterns:

| Entity Type | Weight | Examples |
|-------------|--------|----------|
| file_path | 1.0 | `/home/user/src/foo.py` |
| error | 1.0 | `ModuleNotFoundError` |
| exception | 0.9 | `TypeError`, `ValueError` |
| url | 0.8 | `http://localhost:8080/api` |
| port | 0.8 | `:3000`, `port 5432` |
| command | 0.7 | `git commit`, `docker build` |
| package | 0.7 | `httpx`, `transformers` |
| http_status | 0.6 | `404 Not Found` |
| function | 0.5 | `parse_jsonl()`, `build_query()` |
| class_name | 0.4 | `ScoredTurn`, `EntitySet` |
| env_var | 0.4 | `OPENROUTER_API_KEY` |

Coverage is computed as: what fraction of entities referenced in the suffix (future conversation) are preserved in the compacted prefix.

### Evidence Coverage (LLM-as-Judge)

An optional second metric using LLM-generated probes. An LLM reads the full conversation and generates ~25 probe questions across five dimensions:

| Dimension | Weight | What it tests |
|-----------|--------|---------------|
| error_solution | 0.30 | Can the agent recall failures, root causes, and fixes? |
| instruction | 0.25 | Are user requirements and preferences preserved? |
| progress | 0.25 | Does the agent know what's done, what failed, what's next? |
| environment | 0.15 | File paths, ports, configs, tool versions — exact factual recall |
| noise | 0.05 | Can the agent summarize verbose output without retaining raw noise? |

Probes are cached per (conversation, split_ratio) pair. Evidence coverage measures whether the turns containing probe answers are kept by each method.

## Claude Code Integration

supercompact ships as a Claude Code slash command. Add it to your project or install globally:

### Usage

In any Claude Code session:

```
/supercompact           # Compact with default 80k budget
/supercompact 60000     # Compact with custom budget
```

The slash command:
1. Finds the current session's JSONL transcript
2. Runs EITF compaction at the specified budget
3. Replaces the session JSONL (with backup)
4. Restarts Claude Code to load the compacted context

The command file is at `.claude/commands/supercompact.md`.

## Architecture

```
compact.py              # CLI entry point (compact / evaluate / plot subcommands)
lib/
├── parser.py           # JSONL parsing, Turn dataclass, text extraction
├── tokenizer.py        # Token counting via Qwen3 tokenizer
├── types.py            # ScoredTurn, build_query, random_scores
├── selector.py         # Budget-constrained greedy turn selection
├── formatter.py        # Output formatting (JSONL, summary text, CSV)
├── scorer_base.py      # Scorer protocol, registry, method resolution
├── eitf.py             # EITF scorer (entity-frequency inverse turn frequency)
├── setcover.py         # SetCover scorer (EITF + exclusivity bonus)
├── dedup.py            # Suffix automaton dedup scorer
├── scorer.py           # PyTorch embedding scorer (Qwen3-Embedding-0.6B)
├── llama_embed.py      # llama.cpp embedding scorer (HTTP)
├── llama_rerank.py     # llama.cpp reranker scorer (HTTP)
├── llm_compact.py      # Claude LLM summarization baseline
├── pareto.py           # Pareto plot generation
├── fitness.py          # Legacy fitness evaluation
└── eval/
    ├── probes.py       # LLM-as-Judge probe generation
    ├── entity_coverage.py  # Entity extraction and coverage metrics
    ├── evidence_coverage.py # Evidence turn coverage computation
    ├── cache.py        # Probe set caching
    ├── judge.py        # LLM judge for probe answering
    ├── aggregate.py    # Result aggregation
    └── report.py       # Evaluation reporting
```

### Pipeline

1. **Parse**: Read JSONL, group into alternating user/system turns
2. **Tokenize**: Count tokens per turn using Qwen3 tokenizer
3. **Score**: Run the selected method to assign relevance scores (0-1) to each long system turn
4. **Select**: Greedily pick highest-scoring turns that fit in budget, with a 0.15 recency bonus. User turns, short system turns, and the most recent system turn are always kept.
5. **Output**: Write compacted JSONL (or summary text) preserving original turn structure

## License

MIT
