#!/usr/bin/env bash
# uninstall.sh — Remove supercompact integration for Codex CLI.
#
# Removes:
#   1. CLI tool symlinks from ~/.local/bin
#   2. Compact prompt file from ~/.codex/
#   3. Config entries from ~/.codex/config.toml
#   4. Stops the watcher daemon if running
#
# Does NOT remove:
#   - The supercompact repo itself
#   - Session backups (.pre-supercompact files)
#   - The log directory (~/.cache/supercompact/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERCOMPACT_DIR="$(cd "${SCRIPT_DIR}/../../" && pwd)"
BIN_DIR="${HOME}/.local/bin"
CODEX_CONFIG_DIR="${HOME}/.codex"
PID_FILE="/tmp/codex-compact-watch.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
info() { echo -e "${YELLOW}[info]${NC} $*"; }

echo ""
echo "Uninstalling supercompact for Codex CLI..."
echo ""

# Step 1: Stop the watcher if running
if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}" 2>/dev/null || true
        ok "Stopped codex-compact-watch (PID ${PID})"
    fi
    rm -f "${PID_FILE}"
fi

# Step 2: Remove symlinks
for cmd in codex-compact codex-compact-watch; do
    if [[ -L "${BIN_DIR}/${cmd}" ]]; then
        rm "${BIN_DIR}/${cmd}"
        ok "Removed ${BIN_DIR}/${cmd}"
    elif [[ -f "${BIN_DIR}/${cmd}" ]]; then
        info "Skipping ${BIN_DIR}/${cmd} (not a symlink — remove manually if desired)"
    fi
done

# Step 3: Remove compact prompt
INSTALLED_PROMPT="${CODEX_CONFIG_DIR}/supercompact-prompt.md"
if [[ -f "${INSTALLED_PROMPT}" ]]; then
    rm "${INSTALLED_PROMPT}"
    ok "Removed ${INSTALLED_PROMPT}"
fi

# Step 4: Clean up Codex config
CODEX_CONFIG_FILE="${CODEX_CONFIG_DIR}/config.toml"
if [[ -f "${CODEX_CONFIG_FILE}" ]]; then
    if grep -q "supercompact" "${CODEX_CONFIG_FILE}" 2>/dev/null; then
        # Remove supercompact-related lines (comment + config line)
        # Use a temp file to avoid sed -i portability issues
        TMPFILE=$(mktemp)
        grep -v -E '(supercompact|experimental_compact_prompt_file.*supercompact)' \
            "${CODEX_CONFIG_FILE}" > "${TMPFILE}" || true
        mv "${TMPFILE}" "${CODEX_CONFIG_FILE}"
        ok "Removed supercompact entries from ${CODEX_CONFIG_FILE}"
    fi
fi

# Step 5: Remove plugin symlink
PLUGIN_LINK="${SUPERCOMPACT_DIR}/plugins/codex_cli"
if [[ -L "${PLUGIN_LINK}" ]]; then
    rm "${PLUGIN_LINK}"
    ok "Removed plugin symlink ${PLUGIN_LINK}"
fi

echo ""
echo "Uninstall complete."
echo ""
echo "Not removed (clean up manually if desired):"
echo "  - Session backups: ~/.codex/sessions/**/*.pre-supercompact"
echo "  - Logs: ~/.cache/supercompact/"
echo "  - Supercompact repo: ${SUPERCOMPACT_DIR}"
