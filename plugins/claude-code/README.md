# Supercompact — Claude Code Plugin

Entity-preservation conversation compaction for Claude Code. Replaces the built-in LLM-based `/compact` with EITF scoring — **~400x faster** and **2x better entity retention**.

## Quick Install

```bash
git clone https://github.com/yourusername/supercompact.git
cd supercompact/plugins/claude-code
./install.sh
```

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

## What It Does

When Claude Code compacts your conversation (either automatically or via `/compact`), it normally calls an LLM to summarize — slow (~30s) and lossy. Supercompact replaces this with **EITF** (Entity-frequency Inverse Turn Frequency), a zero-model algorithm that:

1. Extracts structured entities (file paths, errors, functions, URLs, etc.)
2. Scores each conversation turn by entity importance × rarity
3. Greedily selects the most information-dense turns within a token budget
4. Preserves original turn content (no summarization distortion)

Result: compaction in **~0.2 seconds** with **2x better retention** of file paths, error messages, and architectural context.

## How It Works

The installer sets up three integration points:

1. **cli.js patch** — Replaces the LLM API call in Claude Code's main compaction function with a subprocess call to supercompact. Falls back to the original LLM on error.

2. **PreCompact hook** — Backs up the full transcript before any compaction runs, and produces a supercompact alternative alongside Claude's built-in result.

3. **`/supercompact` command** — Manual on-demand compaction with configurable method and budget.

## Configuration

Settings via environment variables (no restart needed to change):

| Variable | Default | Description |
|----------|---------|-------------|
| `PLUGIN_SETTING_METHOD` | `eitf` | Scoring method: `eitf`, `setcover`, `dedup` |
| `PLUGIN_SETTING_BUDGET` | `80000` | Target token budget (or 40% of pre-compact tokens, min 40k) |
| `PLUGIN_SETTING_FALLBACK_TO_BUILTIN` | `true` | Fall back to LLM compaction on error |

## Methods

| Method | Speed | Description |
|--------|-------|-------------|
| `eitf` | ~0.2s | Entity-frequency × inverse turn frequency (recommended) |
| `setcover` | ~0.2s | EITF + adaptive normalization + exclusivity bonus |
| `dedup` | ~0.3s | Suffix automaton unique-content scoring |

## Commands

### `/supercompact [budget] [--method name]`

Manual compaction. Examples:
```
/supercompact                    # Default (eitf, 80k budget)
/supercompact 120000             # Custom budget
/supercompact --method setcover  # Custom method
```

## Install Options

```bash
./install.sh                # Full install (plugin + cli.js patch)
./install.sh --no-patch     # Plugin only (no cli.js modification)
./install.sh --patch-only   # Patch cli.js only (plugin must be installed first)
```

## Uninstall

```bash
./uninstall.sh              # Full removal (restore cli.js + remove files)
./uninstall.sh --unpatch-only  # Only restore cli.js
./uninstall.sh --keep-patch    # Remove files but keep cli.js patch
```

## Files

```
~/.local/share/supercompact/claude-code/
├── supercompact/          # Bundled supercompact library
│   ├── compact.py         # CLI entry point
│   ├── lib/               # Core library
│   ├── pyproject.toml
│   └── uv.lock
└── plugin/                # Claude Code plugin
    ├── .claude-plugin/
    │   └── plugin.json    # Plugin metadata & settings
    ├── commands/
    │   └── supercompact.md  # /supercompact slash command
    ├── hooks/
    │   └── hooks.json     # PreCompact hook registration
    ├── hooks-handlers/
    │   └── supercompact-precompact.sh
    └── scripts/
        ├── patcher.py     # cli.js patching logic
        └── patch-compaction.sh
```

## Logs

Hook activity is logged to `~/.cache/supercompact/hook.log`.

## Troubleshooting

**Compaction not working after Claude Code update:**
The cli.js patch targets a specific code pattern. If Claude Code updates change the minified structure, re-run:
```bash
./install.sh --patch-only
```

**Falling back to LLM compaction:**
Check `~/.cache/supercompact/hook.log` for errors. Common causes:
- Python/uv not in PATH during compaction
- Supercompact directory removed or corrupted

**Verify patch status:**
```bash
grep -c "SUPERCOMPACT_EITF" "$(readlink -f "$(which claude)" | sed 's|[^/]*$|cli.js|')"
# 1 = patched, 0 = not patched
```
