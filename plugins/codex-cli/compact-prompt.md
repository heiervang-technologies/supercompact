You are performing a CONTEXT CHECKPOINT COMPACTION using EITF (Entity-frequency Inverse Turn Frequency) principles.

CRITICAL: Preserve ALL of the following entity types in your summary, listed by priority:
1. File paths, module names, and directory structures
2. Function/method names, class names, variable names
3. Error messages, stack traces, and their resolutions
4. Configuration values, environment variables, URLs
5. Git branches, commit hashes, PR numbers
6. Architecture decisions and their rationale
7. User preferences and constraints expressed during the conversation

Create a structured handoff summary for another LLM that will resume this task. The summary MUST:

- List every file path that was read, created, modified, or deleted
- Include the current state of any in-progress work (what's done, what's next)
- Preserve exact names: function names, variable names, config keys, CLI flags
- Record key decisions with their reasoning
- Note any constraints, gotchas, or edge cases discovered
- Include tool call patterns that worked (command + flags that succeeded)

Format:
## Files Touched
[List every file path with what was done to it]

## Current State
[What's working, what's broken, what's in progress]

## Key Entities
[Function names, config keys, variable names, error messages — anything that would be needed to continue]

## Decisions & Context
[Why things were done a certain way, user preferences, constraints]

## Next Steps
[What remains to be done, in priority order]

Be exhaustive on entity names and paths. Be concise on prose. The next LLM needs facts, not narrative.
