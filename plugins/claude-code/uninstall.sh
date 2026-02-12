#!/usr/bin/env bash
# uninstall.sh — Clean removal of supercompact Claude Code plugin
#
# Restores the original cli.js from backup and removes all plugin files.
#
# Usage:
#   ./uninstall.sh                # Full uninstall
#   ./uninstall.sh --unpatch-only # Only restore cli.js (keep plugin files)
#   ./uninstall.sh --keep-patch   # Remove plugin files but keep cli.js patch

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

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
DO_UNPATCH=true
DO_REMOVE=true

for arg in "$@"; do
    case "$arg" in
        --unpatch-only) DO_REMOVE=false ;;
        --keep-patch)   DO_UNPATCH=false ;;
        --help|-h)
            echo "Usage: $0 [--unpatch-only | --keep-patch | --help]"
            echo ""
            echo "Options:"
            echo "  --unpatch-only   Only restore cli.js (keep plugin files)"
            echo "  --keep-patch     Remove plugin files but keep cli.js patch"
            echo "  --help           Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $arg"; exit 1 ;;
    esac
done

INSTALL_DIR="${HOME}/.local/share/supercompact/claude-code"

# ------------------------------------------------------------------
# Restore cli.js from backup
# ------------------------------------------------------------------
if [[ "$DO_UNPATCH" == true ]]; then
    info "Restoring original cli.js..."

    # Find Claude Code cli.js
    CLAUDE_BIN="${CLAUDE_BIN:-$(which claude 2>/dev/null || echo "")}"

    if [[ -n "$CLAUDE_BIN" ]]; then
        # Resolve symlinks (portable — no readlink -f which is GNU-only)
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

        BACKUP="$CLI_JS.pre-eitf"
        if [[ -f "$BACKUP" ]]; then
            cp "$BACKUP" "$CLI_JS"
            rm -f "$BACKUP"
            ok "cli.js restored from backup"
        elif [[ -f "$CLI_JS" ]]; then
            # Check if it's actually patched
            if grep -q "SUPERCOMPACT_EITF" "$CLI_JS" 2>/dev/null; then
                warn "cli.js is patched but no backup found at $BACKUP"
                warn "You may need to reinstall Claude Code: npm install -g @anthropic-ai/claude-code"
            else
                ok "cli.js is not patched (nothing to restore)"
            fi
        else
            warn "Could not locate cli.js"
        fi
    else
        warn "Claude Code not found in PATH — skipping cli.js restore"
    fi
fi

# ------------------------------------------------------------------
# Remove plugin files
# ------------------------------------------------------------------
if [[ "$DO_REMOVE" == true ]]; then
    if [[ -d "$INSTALL_DIR" ]]; then
        info "Removing plugin files from ${INSTALL_DIR}..."
        rm -rf "${INSTALL_DIR}"
        ok "Plugin files removed"

        # Clean up parent dir if empty
        PARENT_DIR="$(dirname "${INSTALL_DIR}")"
        if [[ -d "$PARENT_DIR" ]] && [[ -z "$(ls -A "$PARENT_DIR")" ]]; then
            rmdir "$PARENT_DIR"
        fi
    else
        ok "No plugin files found at ${INSTALL_DIR}"
    fi

    # Remove log directory
    LOG_DIR="${HOME}/.cache/supercompact"
    if [[ -d "$LOG_DIR" ]]; then
        info "Removing log directory..."
        rm -rf "${LOG_DIR}"
        ok "Logs removed"
    fi
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Supercompact uninstalled.${NC}"
if [[ "$DO_UNPATCH" == true ]]; then
    echo "Restart Claude Code for changes to take effect."
fi
