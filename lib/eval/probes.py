"""Probe generation for LLM-as-Judge evaluation.

Generates ~25 probes from the full conversation, tagged by dimension and tier.
Probes are generated once per (conversation, split_ratio) pair and cached to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import anthropic

from ..parser import Turn, extract_text


DIMENSIONS = {
    "error_solution": 0.30,
    "instruction": 0.25,
    "progress": 0.25,
    "environment": 0.15,
    "noise": 0.05,
}

PROBE_GEN_MODEL = "claude-opus-4-5-20251101"


@dataclass
class Probe:
    id: str                         # "esr_001"
    dimension: str                  # error_solution | instruction | progress | environment | noise
    tier: str                       # factual | comprehension
    question: str                   # What to ask
    gold_answer: str                # Reference answer from full conversation
    evidence_turns: list[int] = field(default_factory=list)  # Turn indices containing evidence
    difficulty: str = "medium"      # easy | medium | hard


@dataclass
class ProbeSet:
    probes: list[Probe] = field(default_factory=list)
    conv_hash: str = ""
    split_ratio: float = 0.70
    version: str = "1"

    def to_dict(self) -> dict:
        return {
            "conv_hash": self.conv_hash,
            "split_ratio": self.split_ratio,
            "version": self.version,
            "probes": [asdict(p) for p in self.probes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProbeSet:
        probes = [Probe(**p) for p in data["probes"]]
        return cls(
            probes=probes,
            conv_hash=data.get("conv_hash", ""),
            split_ratio=data.get("split_ratio", 0.70),
            version=data.get("version", "1"),
        )


_PROBE_GEN_PROMPT = """\
You are an evaluation designer for context compaction systems. You will read a \
conversation between a user and an AI coding assistant, then generate probes \
(questions) that test whether a compacted version of this conversation preserves \
the important information.

The conversation has been split into a PREFIX (turns 0 through {split_idx}) and a \
SUFFIX (the rest). A compaction method will compress the PREFIX. Your probes test \
whether the compacted prefix retains enough information for an agent to continue \
working correctly.

Generate exactly {num_probes} probes across these dimensions:

1. **error_solution** (weight 0.30): Can the agent recall failures, understand root \
causes, and apply lessons? Focus on debugging patterns, error messages, and fixes \
that were discovered during the conversation.

2. **instruction** (weight 0.25): Are user requirements preserved? Include both \
explicit instructions and implicit context (e.g., coding style preferences, \
architecture decisions the user approved).

3. **progress** (weight 0.25): Does the agent know what's done, what failed, what's \
next, and why? Task sequencing, completion status, remaining work.

4. **environment** (weight 0.15): File paths, ports, configs, tool versions, URLs — \
exact factual recall of system details mentioned in the conversation.

5. **noise** (weight 0.05): Can the agent summarize verbose output (like long tool \
results or log dumps) without retaining the raw noise? Test gist comprehension.

For each probe, classify it as:
- **factual**: Tests whether a specific detail exists in the compacted context \
(e.g., "What port is the server running on?")
- **comprehension**: Tests whether the agent can reason from the context \
(e.g., "If you hit a similar error on a different server, how would you diagnose it?")

And set difficulty:
- **easy**: Answer is stated explicitly in a single turn
- **medium**: Answer requires connecting information across 2-3 turns
- **hard**: Answer requires synthesizing a pattern or reasoning about implications

Distribution targets:
- ~8 error_solution probes (mix of factual + comprehension)
- ~6 instruction probes (mostly comprehension)
- ~6 progress probes (mix)
- ~3 environment probes (mostly factual)
- ~2 noise probes (comprehension)

IMPORTANT: The gold_answer must be answerable ONLY from the PREFIX turns \
(turns 0 through {split_idx}). Do not reference information that only appears in the suffix.

Respond with a JSON array of probe objects. Each object has:
- "id": string like "esr_001", "ins_001", "prg_001", "env_001", "noi_001"
- "dimension": one of "error_solution", "instruction", "progress", "environment", "noise"
- "tier": "factual" or "comprehension"
- "question": the probe question
- "gold_answer": the correct answer derivable from the prefix
- "evidence_turns": array of turn indices (0-based) in the prefix that contain the evidence
- "difficulty": "easy", "medium", or "hard"

Output ONLY the JSON array, no other text."""


def _format_turns_for_prompt(turns: list[Turn], max_chars: int = 150_000) -> str:
    """Format turns into a readable string for the probe generator."""
    parts = []
    total = 0
    for t in turns:
        text = extract_text(t)
        header = f"--- Turn {t.index} ({t.kind}) ---"
        entry = f"{header}\n{text}\n"
        if total + len(entry) > max_chars:
            parts.append(f"\n[...truncated at {max_chars} chars, {len(turns) - len(parts)} turns shown...]")
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def generate_probes(
    prefix_turns: list[Turn],
    suffix_turns: list[Turn],
    split_idx: int,
    conv_hash: str,
    split_ratio: float = 0.70,
    num_probes: int = 25,
    model: str | None = None,
) -> ProbeSet:
    """Generate probes from the full conversation using Claude API.

    Args:
        prefix_turns: The prefix portion of the conversation.
        suffix_turns: The suffix portion (used for context only).
        split_idx: The turn index where the split occurs.
        conv_hash: Hash identifying this conversation + split.
        split_ratio: The split ratio used.
        num_probes: Target number of probes to generate.
        model: Override the default probe generation model.
    """
    model = model or PROBE_GEN_MODEL
    client = anthropic.Anthropic()

    # Format the full conversation for the probe generator
    all_turns = prefix_turns + suffix_turns
    conversation_text = _format_turns_for_prompt(all_turns)

    prompt = _PROBE_GEN_PROMPT.format(
        split_idx=split_idx,
        num_probes=num_probes,
    )

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n<conversation>\n{conversation_text}\n</conversation>",
            }
        ],
    )

    # Parse the response
    text = response.content[0].text.strip()
    # Handle potential markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove opening fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    probes_data = json.loads(text)

    probes = []
    for p in probes_data:
        probes.append(Probe(
            id=p["id"],
            dimension=p["dimension"],
            tier=p["tier"],
            question=p["question"],
            gold_answer=p["gold_answer"],
            evidence_turns=p.get("evidence_turns", []),
            difficulty=p.get("difficulty", "medium"),
        ))

    return ProbeSet(
        probes=probes,
        conv_hash=conv_hash,
        split_ratio=split_ratio,
        version="1",
    )
