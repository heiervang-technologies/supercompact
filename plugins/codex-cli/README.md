# supercompact for Codex CLI

Entity-preservation conversation compaction for [OpenAI Codex CLI](https://github.com/openai/codex). Replaces Codex's built-in LLM summarization with EITF (Entity-frequency Inverse Turn Frequency) scoring — **~400x faster**, **2x better entity retention**, **zero API calls**.

## How It Works

Codex CLI stores sessions as JSONL rollout files at `~/.codex/sessions/`. When conversations get long, Codex compacts them using LLM summarization, which is slow and loses important entities (file paths, function names, config values).

supercompact replaces this with a local scoring algorithm that:
1. Parses the Codex rollout JSONL format
2. Scores each assistant turn by entity importance (EITF)
3. Selects the highest-scored turns that fit within the token budget
4. Preserves the original JSONL format for seamless Codex compatibility

## Integration Approaches

This plugin provides three integration levels — use whichever fits your workflow:

### 1. Manual Compaction (`codex-compact`)

Run compaction on-demand, like Codex's built-in `/compact` but faster and better:

```bash
codex-compact                      # Compact latest session
codex-compact --budget 120000      # Custom token budget
codex-compact --method setcover    # Different scoring method
codex-compact --session FILE       # Specific session file
codex-compact --dry-run            # Preview without changes
codex-compact --list               # List recent sessions
codex-compact --verbose            # Detailed score breakdown
```

### 2. Enhanced Compact Prompt (`experimental_compact_prompt_file`)

The least invasive option — improves Codex's built-in LLM compaction by using an entity-preservation-optimized prompt. This still uses the LLM but produces better summaries:

```bash
./install.sh --prompt-only
```

This sets `experimental_compact_prompt_file` in `~/.codex/config.toml` to point to an EITF-optimized prompt that instructs the model to preserve file paths, function names, and other critical entities.

### 3. Auto-Intercept Daemon (`codex-compact-watch`)

Background watcher that detects when Codex performs compaction and automatically replaces the result with supercompact output:

```bash
codex-compact-watch                # Start watching latest session
codex-compact-watch --status       # Check if running
codex-compact-watch --stop         # Stop the daemon
```

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.11+
- [Codex CLI](https://github.com/openai/codex) installed

### Install

```bash
git clone https://github.com/user/supercompact.git
cd supercompact
./plugins/codex-cli/install.sh
```

This will:
1. Install `codex-compact` and `codex-compact-watch` to `~/.local/bin/`
2. Set up Python dependencies via uv
3. Configure Codex's `experimental_compact_prompt_file` for better entity preservation

### Uninstall

```bash
./plugins/codex-cli/uninstall.sh
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPERCOMPACT_METHOD` | `eitf` | Scoring method: `eitf`, `setcover`, `dedup` |
| `SUPERCOMPACT_BUDGET` | `80000` | Target token budget |

### Codex Config (`~/.codex/config.toml`)

The installer adds:

```toml
experimental_compact_prompt_file = "~/.codex/supercompact-prompt.md"
```

This improves the built-in LLM compaction prompt. It works independently of the `codex-compact` command.

## Scoring Methods

| Method | Speed | Quality | Description |
|--------|-------|---------|-------------|
| `eitf` | ~0.2s | Best | Entity-frequency inverse turn frequency scoring |
| `setcover` | ~0.3s | Good | Greedy entity coverage with forward-reference weighting |
| `dedup` | ~0.1s | Fair | Suffix automaton dedup, unique content ratio |

## Codex Session Format

Codex stores sessions as JSONL with these record types:

| Type | Description |
|------|-------------|
| `session_meta` | Session metadata (first line) |
| `turn_context` | Per-turn metadata (model, cwd, policies) |
| `response_item` | Model responses (messages, function calls, outputs) |
| `compacted` | Compaction summaries |
| `event_msg` | UI events (skipped) |

The `codex_parser.py` module handles this format and converts it to the Turn structure used by supercompact's scoring pipeline.

## File Structure

```
plugins/codex-cli/
├── install.sh              # Installer
├── uninstall.sh            # Uninstaller
├── codex-compact           # Manual compaction CLI
├── codex-compact-watch     # Auto-intercept daemon
├── compact_codex.py        # Python entry point
├── codex_parser.py         # Codex JSONL parser
├── compact-prompt.md       # EITF-optimized compact prompt
├── test_codex_parser.py    # Parser tests
└── README.md               # This file
```

## How It Differs from Claude Code Plugin

The Claude Code plugin patches `cli.js` to intercept compaction at the code level. Codex CLI is a Rust binary, so patching isn't practical. Instead, this plugin:

1. **Parses Codex's JSONL format** — different from Claude Code's (Codex uses `response_item`/`turn_context`/`compacted` types vs Claude Code's `user`/`assistant` types)
2. **Operates on session files** — reads and writes the rollout JSONL directly
3. **Provides config-level integration** — via `experimental_compact_prompt_file`
4. **Auto-intercepts via file watching** — detects compaction events by monitoring file changes

## Troubleshooting

### "No Codex sessions found"

Codex stores sessions at `~/.codex/sessions/`. If this directory doesn't exist, run Codex CLI first to create a session.

### Parser errors

If the Codex JSONL format changes in a future version, the parser may need updating. Check the session file format:

```bash
head -5 ~/.codex/sessions/2025/01/01/rollout-*.jsonl | python3 -m json.tool
```

### Backup and recovery

Every compaction creates a `.pre-supercompact` backup:

```bash
# Restore from backup
cp ~/.codex/sessions/.../rollout-xxx.jsonl.pre-supercompact \
   ~/.codex/sessions/.../rollout-xxx.jsonl
```
