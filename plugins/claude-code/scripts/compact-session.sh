#!/usr/bin/env bash
# compact-session.sh - Self-contained supercompact session compaction
#
# Finds the current Claude Code session JSONL, runs supercompact,
# backs up the original, and replaces it with the compacted version.
#
# Usage: compact-session.sh [budget] [--method name]
#
# Environment:
#   CLAUDE_PROJECT_DIR          Project dir set by Claude Code (preferred for JSONL lookup)
#   PLUGIN_SETTING_METHOD       Scoring method (default: eitf)
#   PLUGIN_SETTING_BUDGET       Token budget (default: 80000)

set -euo pipefail

# --- Resolve supercompact installation ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_ROOT="$(cd "${PLUGIN_ROOT}/.." && pwd)"
SUPERCOMPACT_DIR="${INSTALL_ROOT}/supercompact"

if [[ ! -f "${SUPERCOMPACT_DIR}/compact.py" ]]; then
    # Dev mode: repo layout
    if [[ -f "${PLUGIN_ROOT}/../../supercompact/compact.py" ]]; then
        SUPERCOMPACT_DIR="$(cd "${PLUGIN_ROOT}/../../supercompact" && pwd)"
    else
        echo "ERROR: supercompact not found at ${SUPERCOMPACT_DIR}"
        exit 1
    fi
fi

# --- Parse arguments ---
METHOD="${PLUGIN_SETTING_METHOD:-eitf}"
BUDGET="${PLUGIN_SETTING_BUDGET:-80000}"
prev=""
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then BUDGET="$arg"; fi
    if [[ "$prev" == "--method" ]]; then METHOD="$arg"; fi
    prev="$arg"
done

# --- Find the session JSONL ---
find_project_dir() {
    # Prefer CLAUDE_PROJECT_DIR if set
    if [[ -n "${CLAUDE_PROJECT_DIR:-}" && -d "${CLAUDE_PROJECT_DIR}" ]]; then
        echo "${CLAUDE_PROJECT_DIR}"
        return
    fi
    # Fallback: derive from PWD (same logic Claude Code uses)
    local derived
    derived="${HOME}/.claude/projects/$(echo "${PWD}" | sed 's|/|-|g')"
    if [[ -d "${derived}" ]]; then
        echo "${derived}"
        return
    fi
    echo ""
}

PROJECT_DIR="$(find_project_dir)"
if [[ -z "${PROJECT_DIR}" ]]; then
    echo "ERROR: Could not find Claude project directory"
    echo "  Tried CLAUDE_PROJECT_DIR=${CLAUDE_PROJECT_DIR:-<unset>}"
    echo "  Tried PWD-derived=${HOME}/.claude/projects/$(echo "${PWD}" | sed 's|/|-|g')"
    exit 1
fi

JSONL_FILE="$(ls -t "${PROJECT_DIR}"/*.jsonl 2>/dev/null | head -1)"
if [[ -z "${JSONL_FILE}" || ! -f "${JSONL_FILE}" ]]; then
    echo "ERROR: No .jsonl files found in ${PROJECT_DIR}"
    exit 1
fi

LINES_BEFORE=$(wc -l < "${JSONL_FILE}")
echo "Session JSONL: ${JSONL_FILE}"
echo "Lines before:  ${LINES_BEFORE}"
echo "Method:        ${METHOD}"
echo "Budget:        ${BUDGET}"
echo ""

# --- Run supercompact ---
SC_OUTPUT="/tmp/supercompact-output-$$.jsonl"
trap 'rm -f "${SC_OUTPUT}"' EXIT

START_TIME=$(date +%s%N)

cd "${SUPERCOMPACT_DIR}"
SC_STDOUT=$(uv run python compact.py compact "${JSONL_FILE}" \
    --method "${METHOD}" \
    --budget "${BUDGET}" \
    --output "${SC_OUTPUT}" \
    --verbose 2>&1) || {
    echo ""
    echo "${SC_STDOUT}"
    echo ""
    echo "ERROR: supercompact failed"
    exit 1
}

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

echo "${SC_STDOUT}"

if [[ ! -f "${SC_OUTPUT}" ]]; then
    # Already within budget — not an error
    echo ""
    echo "Session is already within budget. No compaction needed."
    exit 0
fi

# --- Backup and replace ---
BACKUP_FILE="${JSONL_FILE}.pre-supercompact"
cp "${JSONL_FILE}" "${BACKUP_FILE}"
mv "${SC_OUTPUT}" "${JSONL_FILE}"
trap - EXIT  # output file moved, no cleanup needed

LINES_AFTER=$(wc -l < "${JSONL_FILE}")

# --- Report ---
if [[ ${LINES_BEFORE} -gt 0 ]]; then
    REDUCTION=$(( (LINES_BEFORE - LINES_AFTER) * 100 / LINES_BEFORE ))
else
    REDUCTION=0
fi

echo ""
echo "=== Compaction Complete ==="
echo "Lines before:  ${LINES_BEFORE}"
echo "Lines after:   ${LINES_AFTER}"
echo "Reduction:     ${REDUCTION}%"
echo "Time:          ${ELAPSED_MS}ms"
echo "Backup:        ${BACKUP_FILE}"
