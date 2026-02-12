#!/usr/bin/env bash
# patch-compaction.sh - Patch Claude Code to use EITF for compaction instead of LLM
#
# Replaces the LLM API call in MW1 (main compaction function) with a call
# to supercompact's EITF algorithm. Falls back to original LLM compaction on error.
#
# Arguments:
#   $1 — path to the supercompact directory (where compact.py lives)
#
# Environment:
#   CLAUDE_BIN — override Claude Code binary location

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERCOMPACT_DIR="${1:?Usage: $0 <supercompact_dir>}"

if [[ ! -f "$SUPERCOMPACT_DIR/compact.py" ]]; then
    echo "Error: supercompact not found at $SUPERCOMPACT_DIR/compact.py"
    exit 1
fi

# Find Claude Code cli.js
CLAUDE_BIN="${CLAUDE_BIN:-$(which claude 2>/dev/null || echo "")}"
if [[ -z "$CLAUDE_BIN" ]]; then
    echo "Error: Claude Code not found in PATH"
    exit 1
fi

resolve_symlink() {
    local path="$1"
    local max_depth=20
    local depth=0
    while [[ -L "$path" ]]; do
        if (( depth++ >= max_depth )); then
            echo "Error: Too many symlink levels" >&2
            return 1
        fi
        local dir="$(dirname "$path")"
        local link="$(readlink "$path")"
        if [[ "$link" == /* ]]; then
            path="$link"
        else
            path="$(cd "$dir" && cd "$(dirname "$link")" && pwd -P)/$(basename "$link")"
        fi
    done
    echo "$path"
}

CLAUDE_REAL="$(resolve_symlink "$CLAUDE_BIN")"
CLI_JS="$CLAUDE_REAL"
[[ "$(basename "$CLAUDE_REAL")" != "cli.js" ]] && CLI_JS="$(dirname "$CLAUDE_REAL")/cli.js"

if [[ ! -f "$CLI_JS" ]]; then
    NPM_ROOT=$(npm root -g 2>/dev/null || echo "")
    [[ -n "$NPM_ROOT" ]] && CLI_JS="$NPM_ROOT/@anthropic-ai/claude-code/cli.js"
fi

if [[ ! -f "$CLI_JS" ]]; then
    echo "Error: cli.js not found"
    exit 1
fi

echo "Patching compaction in: $CLI_JS"

# Create backup
BACKUP="$CLI_JS.pre-eitf"
if [[ ! -f "$BACKUP" ]]; then
    cp "$CLI_JS" "$BACKUP"
    echo "Backup: $BACKUP"
fi

# Apply patch via Python (avoids shell escaping issues with minified JS)
python3 "$SCRIPT_DIR/patcher.py" "$CLI_JS" "$SUPERCOMPACT_DIR"
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Restoring backup..."
    cp "$BACKUP" "$CLI_JS"
    exit $EXIT_CODE
fi

echo "Restart Claude Code to apply."
