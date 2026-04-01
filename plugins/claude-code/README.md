# Supercompact — Claude Code Plugin

Entity-preservation conversation compaction for Claude Code. **~400x faster** and **2x better entity retention** than the built-in LLM-based `/compact`.

## Quick Install

```bash
git clone https://github.com/heiervang-technologies/supercompact.git
cd supercompact/plugins/claude-code
./install.sh
```

The installer automatically registers the plugin in `~/.claude/settings.json`. Restart Claude Code, then use `/supercompact`.

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), jq

## What It Does

When Claude Code compacts your conversation (either automatically or via `/compact`), it normally calls an LLM to summarize — slow (~30s) and lossy. Supercompact uses **EITF** (Entity-frequency Inverse Turn Frequency), a zero-model algorithm that:

1. Extracts structured entities (file paths, errors, functions, URLs, etc.)
2. Scores each conversation turn by entity importance × rarity
3. Greedily selects the most information-dense turns within a token budget
4. Preserves original turn content (no summarization distortion)

Result: compaction in **~0.2 seconds** with **2x better retention** of file paths, error messages, and architectural context.

## How It Works

The plugin provides three integration points:

1. **`/supercompact` command** — On-demand compaction. Replaces the session with a compacted version and restarts. This is the primary interface.

2. **PreCompact hook** — When Claude's built-in compaction triggers, the hook backs up the full transcript before it's lost. The backup is saved as `*.pre-compact-full` alongside the session JSONL.

3. **cli.js patch** *(npm installations only)* — Replaces the LLM API call in Claude Code's compaction function with supercompact. Falls back to the original LLM on error. Not available on standalone binary installations.

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

## Update

```bash
cd supercompact
git pull
./plugins/claude-code/install.sh
```

Re-running the installer is safe — it replaces all files and is fully idempotent.

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
    │   └── supercompact-precompact.sh  # Backup-only hook
    └── scripts/
        ├── compact-session.sh  # Main compaction script
        ├── patcher.py          # cli.js patching logic
        └── patch-compaction.sh
```

## Logs

Hook activity is logged to `~/.cache/supercompact/hook.log`.

## Standalone Binary Installation

If Claude Code is installed as a standalone binary (not via npm), the cli.js patch cannot be applied. The installer detects this automatically, skips patching, and configures `settings.json` for you.

In standalone mode:
- **`/supercompact`** — Works fully. This is the primary way to compact.
- **`/compact`** — Still uses Claude's built-in LLM compaction (cannot be replaced without cli.js patch).
- **PreCompact hook** — Backs up the full transcript before Claude's built-in compaction runs.

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

**Verify patch status (npm installations only):**
```bash
grep -c "SUPERCOMPACT_EITF" "$(readlink -f "$(which claude)" | sed 's|[^/]*$|cli.js|')"
# 1 = patched, 0 = not patched
```
