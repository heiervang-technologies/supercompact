#!/usr/bin/env bash
# supercompact-precompact.sh - PreCompact hook (backup-only)
#
# The PreCompact hook CANNOT block or replace Claude's built-in compaction —
# it is notification-only. Running supercompact here is wasted work since
# Claude's LLM compaction overwrites the result anyway.
#
# Instead, we just:
#   1. Back up the full transcript before Claude's summarization loses detail
#   2. Log the event
#   3. Clean up old backups

set -euo pipefail

LOG_DIR="${HOME}/.cache/supercompact"
mkdir -p "${LOG_DIR}"

# Read hook input from stdin (JSON with transcript_path, session_id, trigger, etc.)
HOOK_INPUT=$(cat)

TRIGGER=$(echo "${HOOK_INPUT}" | jq -r '.trigger // "unknown"')
JSONL_FILE=$(echo "${HOOK_INPUT}" | jq -r '.transcript_path // empty')

echo "$(date -Iseconds) PreCompact hook triggered (trigger=${TRIGGER})" >> "${LOG_DIR}/hook.log"

if [[ -z "${JSONL_FILE}" || ! -f "${JSONL_FILE}" ]]; then
    echo "$(date -Iseconds) ERROR: No transcript_path in hook input or file missing" >> "${LOG_DIR}/hook.log"
    exit 0
fi

JSONL_SIZE=$(wc -l < "${JSONL_FILE}")
echo "$(date -Iseconds) Transcript: ${JSONL_FILE} (${JSONL_SIZE} lines)" >> "${LOG_DIR}/hook.log"

# Back up the full transcript before Claude's compaction destroys detail
BACKUP_FILE="${JSONL_FILE}.pre-compact-full"
cp "${JSONL_FILE}" "${BACKUP_FILE}"
echo "$(date -Iseconds) Full backup saved: ${BACKUP_FILE}" >> "${LOG_DIR}/hook.log"

# Clean up old backups (keep last 3)
ls -t "${JSONL_FILE}.pre-compact-full"* 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true
ls -t "${JSONL_FILE}.pre-supercompact"* 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true

echo "$(date -Iseconds) Backup-only hook complete (use /supercompact for manual compaction)" >> "${LOG_DIR}/hook.log"

exit 0
