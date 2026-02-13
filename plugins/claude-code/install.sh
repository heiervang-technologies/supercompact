#!/usr/bin/env bash
# install.sh — Single-command installer for supercompact Claude Code plugin
#
# Installs supercompact as a Claude Code plugin with:
#   - Bundled supercompact library (copied, not symlinked)
#   - Python dependencies via uv
#   - cli.js patch for automatic compaction replacement
#   - /supercompact slash command
#   - PreCompact hook for backup
#
# Usage:
#   ./install.sh                  # Install (default)
#   ./install.sh --patch-only     # Only patch cli.js (skip plugin install)
#   ./install.sh --no-patch       # Install plugin without patching cli.js
#
# Prerequisites: Python 3.11+, uv

set -euo pipefail

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

info()  { echo -e "${BLUE}→${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
err()   { echo -e "${RED}✗${NC} $*" >&2; }
fatal() { err "$*"; exit 1; }

# ------------------------------------------------------------------
# Locate the source repo (this script lives in plugins/claude-code/)
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f "${REPO_ROOT}/compact.py" ]]; then
    fatal "Cannot find supercompact repo root (expected compact.py at ${REPO_ROOT}/compact.py)"
fi

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
DO_PATCH=true
DO_INSTALL=true

for arg in "$@"; do
    case "$arg" in
        --patch-only) DO_INSTALL=false ;;
        --no-patch)   DO_PATCH=false ;;
        --help|-h)
            echo "Usage: $0 [--patch-only | --no-patch | --help]"
            echo ""
            echo "Options:"
            echo "  --patch-only   Only patch cli.js (skip plugin install)"
            echo "  --no-patch     Install plugin without patching cli.js"
            echo "  --help         Show this help"
            exit 0
            ;;
        *) fatal "Unknown argument: $arg" ;;
    esac
done

# ------------------------------------------------------------------
# Check prerequisites
# ------------------------------------------------------------------
info "Checking prerequisites..."

# Python 3.11+
if ! command -v python3 &>/dev/null; then
    fatal "Python 3 not found. Install Python 3.11+ first."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 11) )); then
    fatal "Python 3.11+ required (found $PYTHON_VERSION)"
fi
ok "Python $PYTHON_VERSION"

# uv
if ! command -v uv &>/dev/null; then
    fatal "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
ok "uv $(uv --version 2>/dev/null | head -1)"

# jq (required by PreCompact hook handler)
if ! command -v jq &>/dev/null; then
    fatal "jq not found. Install it: sudo pacman -S jq (Arch) / sudo apt install jq (Debian) / brew install jq (macOS)"
fi
ok "jq found"

# Claude Code
if ! command -v claude &>/dev/null; then
    fatal "Claude Code not found in PATH"
fi
ok "Claude Code found"

# ------------------------------------------------------------------
# Install plugin
# ------------------------------------------------------------------
INSTALL_DIR="${HOME}/.local/share/supercompact/claude-code"

if [[ "$DO_INSTALL" == true ]]; then
    info "Installing supercompact plugin to ${INSTALL_DIR}..."

    # Create install directory
    mkdir -p "${INSTALL_DIR}"

    # Copy the supercompact library
    info "Copying supercompact library..."
    SUPERCOMPACT_DEST="${INSTALL_DIR}/supercompact"
    rm -rf "${SUPERCOMPACT_DEST}"
    mkdir -p "${SUPERCOMPACT_DEST}"

    # Copy only what's needed (lib/, compact.py, pyproject.toml, uv.lock)
    cp "${REPO_ROOT}/compact.py" "${SUPERCOMPACT_DEST}/"
    cp "${REPO_ROOT}/pyproject.toml" "${SUPERCOMPACT_DEST}/"
    cp "${REPO_ROOT}/uv.lock" "${SUPERCOMPACT_DEST}/"
    cp -r "${REPO_ROOT}/lib" "${SUPERCOMPACT_DEST}/"
    ok "Supercompact library copied"

    # Copy the plugin structure
    info "Copying plugin files..."
    PLUGIN_DEST="${INSTALL_DIR}/plugin"
    rm -rf "${PLUGIN_DEST}"
    mkdir -p "${PLUGIN_DEST}"

    # Copy plugin structure
    cp -r "${SCRIPT_DIR}/.claude-plugin" "${PLUGIN_DEST}/"
    cp -r "${SCRIPT_DIR}/commands" "${PLUGIN_DEST}/"
    cp -r "${SCRIPT_DIR}/hooks" "${PLUGIN_DEST}/"
    cp -r "${SCRIPT_DIR}/hooks-handlers" "${PLUGIN_DEST}/"
    cp -r "${SCRIPT_DIR}/scripts" "${PLUGIN_DEST}/"

    # Ensure scripts are executable
    chmod +x "${PLUGIN_DEST}/hooks-handlers/"*.sh
    chmod +x "${PLUGIN_DEST}/scripts/"*.sh 2>/dev/null || true

    ok "Plugin files copied"

    # Install Python dependencies
    info "Installing Python dependencies..."
    cd "${SUPERCOMPACT_DEST}"
    if uv sync --quiet 2>&1; then
        ok "Python dependencies installed"
    else
        warn "uv sync had warnings (may still work)"
    fi

    # Verify the install
    info "Verifying installation..."
    cd "${SUPERCOMPACT_DEST}"
    if uv run python -c "from lib.eitf import eitf_scores; print('EITF scorer OK')" 2>/dev/null; then
        ok "EITF scorer verified"
    else
        warn "Could not verify EITF scorer (may work anyway)"
    fi

    ok "Plugin installed to ${INSTALL_DIR}"

    # Auto-configure settings.json to load the plugin
    SETTINGS_FILE="${HOME}/.claude/settings.json"
    info "Configuring Claude Code to load plugin..."
    mkdir -p "$(dirname "${SETTINGS_FILE}")"

    if [[ ! -f "${SETTINGS_FILE}" ]]; then
        # Create settings.json with pluginDirs
        echo '{"pluginDirs":["'"${PLUGIN_DEST}"'"]}' | jq . > "${SETTINGS_FILE}"
        ok "Created ${SETTINGS_FILE} with pluginDirs"
    elif jq -e '.pluginDirs' "${SETTINGS_FILE}" >/dev/null 2>&1; then
        # pluginDirs exists — check if our path is already there
        if jq -e --arg p "${PLUGIN_DEST}" '.pluginDirs | index($p)' "${SETTINGS_FILE}" >/dev/null 2>&1; then
            ok "Plugin already registered in settings.json"
        else
            # Add our path to existing pluginDirs array
            jq --arg p "${PLUGIN_DEST}" '.pluginDirs += [$p]' "${SETTINGS_FILE}" > "${SETTINGS_FILE}.tmp" \
                && mv "${SETTINGS_FILE}.tmp" "${SETTINGS_FILE}"
            ok "Added plugin to existing pluginDirs in settings.json"
        fi
    else
        # settings.json exists but no pluginDirs key — add it
        jq --arg p "${PLUGIN_DEST}" '. + {pluginDirs: [$p]}' "${SETTINGS_FILE}" > "${SETTINGS_FILE}.tmp" \
            && mv "${SETTINGS_FILE}.tmp" "${SETTINGS_FILE}"
        ok "Added pluginDirs to settings.json"
    fi
fi

# ------------------------------------------------------------------
# Patch cli.js
# ------------------------------------------------------------------
PATCH_APPLIED=false
STANDALONE_BINARY=false

if [[ "$DO_PATCH" == true ]]; then
    SUPERCOMPACT_DEST="${INSTALL_DIR}/supercompact"

    if [[ ! -f "${SUPERCOMPACT_DEST}/compact.py" ]]; then
        fatal "Supercompact not installed at ${SUPERCOMPACT_DEST}. Run install first (without --patch-only)."
    fi

    # Detect standalone binary vs npm installation
    CLAUDE_BIN="${CLAUDE_BIN:-$(which claude 2>/dev/null || echo "")}"
    CLAUDE_REAL=""
    if [[ -n "$CLAUDE_BIN" ]]; then
        CLAUDE_REAL="$(readlink -f "$CLAUDE_BIN" 2>/dev/null || echo "$CLAUDE_BIN")"
    fi

    if [[ -n "$CLAUDE_REAL" ]] && head -c 4 "$CLAUDE_REAL" 2>/dev/null | grep -q "ELF\|MZ"; then
        STANDALONE_BINARY=true
        echo ""
        warn "Claude Code is installed as a standalone binary (not via npm)"
        warn "cli.js patching is not available for standalone installations"
        info "The /supercompact slash command and PreCompact hook will still work"
        info "Use '/supercompact' for on-demand compaction"
    else
        echo ""
        info "Patching Claude Code cli.js..."
        bash "${INSTALL_DIR}/plugin/scripts/patch-compaction.sh" "${SUPERCOMPACT_DEST}"
        EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 0 ]]; then
            ok "cli.js patched — compaction now uses supercompact"
            PATCH_APPLIED=true
        else
            warn "cli.js patching failed (exit code $EXIT_CODE)"
            warn "The /supercompact slash command and PreCompact hook will still work"
        fi
    fi
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Supercompact installed successfully!${NC}"
echo ""
echo "What's installed:"
echo "  • Supercompact library at ${INSTALL_DIR}/supercompact/"
echo "  • Plugin at ${INSTALL_DIR}/plugin/"
echo "  • Plugin registered in ~/.claude/settings.json"
if [[ "$PATCH_APPLIED" == true ]]; then
    echo "  • cli.js patched for automatic compaction replacement"
fi
echo ""
echo "Usage:"
if [[ "$PATCH_APPLIED" == true ]]; then
    echo "  /compact and /supercompact both use supercompact now."
    echo "  Restart Claude Code to activate."
else
    echo "  /supercompact           # On-demand entity-preservation compaction"
    echo "  /supercompact 120000    # Custom token budget"
fi
echo ""
echo "To update later:  git pull && ./install.sh"
echo "To uninstall:     ./uninstall.sh"
