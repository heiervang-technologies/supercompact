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

    # Print plugin-dir usage
    echo ""
    info "To load the plugin, use one of:"
    echo "    claude --plugin-dir ${PLUGIN_DEST}"
    echo ""
    echo "  Or add to ~/.claude/settings.json:"
    echo "    { \"pluginDirs\": [\"${PLUGIN_DEST}\"] }"
    echo ""
fi

# ------------------------------------------------------------------
# Patch cli.js
# ------------------------------------------------------------------
if [[ "$DO_PATCH" == true ]]; then
    SUPERCOMPACT_DEST="${INSTALL_DIR}/supercompact"

    if [[ ! -f "${SUPERCOMPACT_DEST}/compact.py" ]]; then
        fatal "Supercompact not installed at ${SUPERCOMPACT_DEST}. Run install first (without --patch-only)."
    fi

    echo ""
    info "Patching Claude Code cli.js..."
    bash "${INSTALL_DIR}/plugin/scripts/patch-compaction.sh" "${SUPERCOMPACT_DEST}"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        ok "cli.js patched — compaction now uses supercompact"
    else
        err "Patching failed (exit code $EXIT_CODE)"
        exit $EXIT_CODE
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
if [[ "$DO_PATCH" == true ]]; then
    echo "  • cli.js patched for automatic compaction replacement"
fi
echo ""
echo "Configuration (via environment variables or plugin settings):"
echo "  PLUGIN_SETTING_METHOD=eitf             # eitf, setcover, dedup"
echo "  PLUGIN_SETTING_BUDGET=80000            # token budget"
echo "  PLUGIN_SETTING_FALLBACK_TO_BUILTIN=true  # fall back to LLM on error"
echo ""
if [[ "$DO_PATCH" == true ]]; then
    echo "Restart Claude Code to activate the patch."
fi
