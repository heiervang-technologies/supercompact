#!/usr/bin/env bash
# install.sh — Install supercompact integration for Codex CLI.
#
# Installs:
#   1. codex-compact CLI tool (manual compaction command)
#   2. codex-compact-watch daemon (auto-intercept compaction)
#   3. Codex config for EITF-optimized compact prompt
#   4. Python dependencies via uv
#
# The install creates symlinks from ~/.local/bin to the plugin directory,
# so updates to the supercompact repo are immediately available.
#
# Usage:
#   ./install.sh              # Full install
#   ./install.sh --prompt-only  # Only configure Codex compact prompt
#   ./install.sh --uninstall    # Remove everything (same as uninstall.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERCOMPACT_DIR="$(cd "${SCRIPT_DIR}/../../" && pwd)"
BIN_DIR="${HOME}/.local/bin"
CODEX_CONFIG_DIR="${CODEX_HOME:-${HOME}/.codex}"
LOG_DIR="${HOME}/.cache/supercompact"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[info]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
err()   { echo -e "${RED}[error]${NC} $*"; }

PROMPT_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-only)
            PROMPT_ONLY=true
            shift
            ;;
        --uninstall)
            exec "${SCRIPT_DIR}/uninstall.sh"
            ;;
        --help|-h)
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Install supercompact integration for Codex CLI."
            echo ""
            echo "Options:"
            echo "  --prompt-only   Only configure the Codex compact prompt (no CLI tools)"
            echo "  --uninstall     Remove the integration (runs uninstall.sh)"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "====================================="
echo " supercompact for Codex CLI"
echo "====================================="
echo ""

# Verify supercompact repo
if [[ ! -f "${SUPERCOMPACT_DIR}/compact.py" ]]; then
    err "supercompact repo not found at ${SUPERCOMPACT_DIR}"
    err "Expected compact.py at: ${SUPERCOMPACT_DIR}/compact.py"
    exit 1
fi
ok "Found supercompact at ${SUPERCOMPACT_DIR}"

# Check for uv
if ! command -v uv &>/dev/null; then
    err "uv is required but not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
ok "Found uv"

# Check for python
if ! uv run --directory "${SUPERCOMPACT_DIR}" python3 --version &>/dev/null 2>&1; then
    warn "Python environment not ready. Setting up..."
    (cd "${SUPERCOMPACT_DIR}" && uv sync) || {
        err "Failed to set up Python environment"
        exit 1
    }
fi
ok "Python environment ready"

# Step 1: Install Python dependencies
info "Checking supercompact dependencies..."
(cd "${SUPERCOMPACT_DIR}" && uv sync --quiet) || {
    err "Failed to sync dependencies"
    exit 1
}
ok "Dependencies installed"

if [[ "${PROMPT_ONLY}" == false ]]; then
    # Step 2: Create symlinks for CLI tools
    mkdir -p "${BIN_DIR}"

    # codex-compact
    if [[ -L "${BIN_DIR}/codex-compact" ]]; then
        rm "${BIN_DIR}/codex-compact"
    fi
    ln -s "${SCRIPT_DIR}/codex-compact" "${BIN_DIR}/codex-compact"
    ok "Installed codex-compact -> ${BIN_DIR}/codex-compact"

    # codex-compact-watch
    if [[ -L "${BIN_DIR}/codex-compact-watch" ]]; then
        rm "${BIN_DIR}/codex-compact-watch"
    fi
    ln -s "${SCRIPT_DIR}/codex-compact-watch" "${BIN_DIR}/codex-compact-watch"
    ok "Installed codex-compact-watch -> ${BIN_DIR}/codex-compact-watch"

    # Verify PATH
    if ! echo "${PATH}" | tr ':' '\n' | grep -q "^${BIN_DIR}$"; then
        warn "${BIN_DIR} is not in your PATH"
        warn "Add to your shell profile: export PATH=\"\${HOME}/.local/bin:\${PATH}\""
    fi
fi

# Step 3: Configure Codex compact prompt (if Codex config exists or we should create it)
CODEX_CONFIG_FILE="${CODEX_CONFIG_DIR}/config.toml"
COMPACT_PROMPT_FILE="${SCRIPT_DIR}/compact-prompt.md"

if [[ -d "${CODEX_CONFIG_DIR}" ]] || command -v codex &>/dev/null; then
    mkdir -p "${CODEX_CONFIG_DIR}"

    # Install the compact prompt file
    INSTALLED_PROMPT="${CODEX_CONFIG_DIR}/supercompact-prompt.md"
    cp "${COMPACT_PROMPT_FILE}" "${INSTALLED_PROMPT}"
    ok "Installed compact prompt -> ${INSTALLED_PROMPT}"

    # Check if config.toml exists and update it
    if [[ -f "${CODEX_CONFIG_FILE}" ]]; then
        # Check if experimental_compact_prompt_file is already set
        if grep -q "experimental_compact_prompt_file" "${CODEX_CONFIG_FILE}" 2>/dev/null; then
            # Check if it already points to our prompt
            if grep -q "supercompact-prompt.md" "${CODEX_CONFIG_FILE}" 2>/dev/null; then
                ok "Codex config already configured for supercompact prompt"
            else
                warn "Codex config has a different experimental_compact_prompt_file set"
                warn "To use supercompact's prompt, update ${CODEX_CONFIG_FILE}:"
                warn "  experimental_compact_prompt_file = \"${INSTALLED_PROMPT}\""
            fi
        else
            # Append the config
            echo "" >> "${CODEX_CONFIG_FILE}"
            echo "# supercompact: EITF-optimized compact prompt for better entity preservation" >> "${CODEX_CONFIG_FILE}"
            echo "experimental_compact_prompt_file = \"${INSTALLED_PROMPT}\"" >> "${CODEX_CONFIG_FILE}"
            ok "Added experimental_compact_prompt_file to Codex config"
        fi
    else
        # Create a minimal config
        cat > "${CODEX_CONFIG_FILE}" << TOML
# Codex CLI configuration
# See: https://github.com/openai/codex

# supercompact: EITF-optimized compact prompt for better entity preservation
experimental_compact_prompt_file = "${INSTALLED_PROMPT}"
TOML
        ok "Created Codex config with supercompact prompt"
    fi
else
    info "Codex CLI not found. Skipping config setup."
    info "After installing Codex, run: install.sh --prompt-only"
fi

# Step 4: Create log directory
mkdir -p "${LOG_DIR}"

# Step 5: Make the codex_parser module importable from supercompact
PLUGIN_LINK="${SUPERCOMPACT_DIR}/plugins/codex_cli"
if [[ ! -e "${PLUGIN_LINK}" ]]; then
    ln -s "${SCRIPT_DIR}" "${PLUGIN_LINK}"
fi

echo ""
echo "====================================="
echo " Installation complete!"
echo "====================================="
echo ""
if [[ "${PROMPT_ONLY}" == false ]]; then
    echo "Commands installed:"
    echo "  codex-compact           Compact the current Codex session"
    echo "  codex-compact --help    Show all options"
    echo "  codex-compact --list    List recent sessions"
    echo "  codex-compact-watch     Auto-intercept Codex compaction"
    echo ""
fi
echo "Codex config:"
echo "  Compact prompt: ${CODEX_CONFIG_DIR}/supercompact-prompt.md"
echo "  Config file: ${CODEX_CONFIG_FILE}"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""
echo "To compact a Codex session:"
echo "  codex-compact                  # Latest session, EITF, 80k budget"
echo "  codex-compact --budget 120000  # Custom budget"
echo "  codex-compact --dry-run        # Preview without changes"
echo ""
echo "To auto-intercept Codex's built-in compaction:"
echo "  codex-compact-watch            # Start watching latest session"
echo "  codex-compact-watch --stop     # Stop the watcher"
