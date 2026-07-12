# Context Rot Compaction Targets — Executive Summary

## What this is

A single pure function: `compact_to(current_tokens, model, task, dial) → int`

Given how many tokens you have, which model you're using, what kind of task you're doing, and where you want to sit on the recall/space tradeoff, it returns the number of tokens you should compact down to. It doesn't know or care how your compactor works. It just gives you the number.

The output is always strictly less than the input.

## What it's built on

The function encodes empirical degradation curves from seven published sources:

| Source | Date | Key finding |
|--------|------|-------------|
| Chroma "Context Rot" | Jul 2025 | All 18 tested models degrade at every input length increment |
| Mejba Ahmed testing | Mar 2026 | Opus 4.6: ~2%/100K degradation (100K→2%, 200K→4%, 500K→10%, 1M→14%) |
| Elvex Benchmarks | 2026 | Claude Sonnet 4 <5% degradation at 200K; most models drop sharply ~130K |
| LongCodeBench | Feb 2026 | Gemini 2.5 Pro >90% at 512K with MC options, ~50% without |
| Paulsen MECW | Jan 2026 | Effective context window is task-specific; complex reasoning fails at 1/100th of simple retrieval |
| Shi et al. | Feb 2025 | Optimal context length is bounded by training dataset size |
| Anthropic benchmarks | Mar 2026 | Opus 4.6 NIAH 91.9→78.3 across 1M; MRCR v2 76% at 1M |

## How it works

**Step 1: Sweet spot.** For each (model, task) pair, binary-search the degradation curve for the token count where estimated quality retention = 90%. This is the "sweet spot" — the point where the model is still performing well but is starting to bend. Because harder tasks amplify degradation (via a task multiplier), the sweet spot shifts lower for harder tasks. For example, Claude Opus 4.6's sweet spot for simple retrieval is ~500K tokens, but for coding it's ~200K, and for multi-hop reasoning it's ~167K.

**Step 2: Dial.** The dial (0.0–1.0) slides the target between three anchors:

| Dial | Target | Meaning |
|------|--------|---------|
| 0.0 | `floor` (8K default) | Maximum compression. Nuke it. |
| 0.5 | sweet spot | Balanced. Quality ~90% at this point. |
| 1.0 | `current × 0.90` | Minimal trim. ~10% space savings. |

The mapping is piecewise-linear: `[0, 0.5]` interpolates between floor and sweet spot, `[0.5, 1.0]` interpolates between sweet spot and the 90% ceiling.

**Step 3: Clamp.** Output is clamped to `[floor, current_tokens - 1]`.

## Key numbers

Balanced targets (dial=0.5) for coding at common context lengths:

| Model | 128K → | 256K → | 500K → | 1M → |
|-------|--------|--------|--------|------|
| Claude Opus 4.6 | 96K | 200K | 200K | 200K |
| Claude Sonnet 4.6 | 96K | 150K | 150K | 150K |
| Gemini 2.5 Pro | 82K | 82K | 82K | 82K |
| Gemini 2.5 Flash | 64K | 64K | 64K | 64K |
| Gemini 3.0 Pro | 100K | 100K | 100K | 100K |
| GLM-4.7 | 64K | — | — | — |
| GLM-4.6 | 51K | — | — | — |

Note how Opus 4.6's target scales with input size (because its sweet spot is high enough to move with the input), while Gemini's targets saturate early (sweet spot is low, so even at 1M you'd compact back to ~82K at balanced).

## Confidence levels

- **Claude Opus 4.6 / Sonnet 4.6**: High. Multiple independent benchmarks (Mejba, Elvex, Anthropic MRCR/NIAH, Chroma).
- **Gemini 2.5 Pro / 3.0 Pro**: Medium-high. LongCodeBench and Chroma data; Flash has less independent testing.
- **GLM-4.7 / 4.6**: Low. No published RULER/MRCR at scale. Estimates based on model class and 200K window boundary.
- **Task multipliers**: Medium. Derived from Chroma's semantic vs lexical gap (~2×), LongCodeBench's MC vs open gap, and Paulsen's task-specificity findings. The exact multipliers (1.0–3.0) are interpolated, not directly measured per-model.

## API

```python
from compaction_target import compact_to, quality_at

# Known model
target = compact_to(250_000, model="claude-opus-4.6", task="coding", dial=0.5)

# Unknown model — just pass the context window size
target = compact_to(250_000, max_context=512_000, task="coding", dial=0.5)

# Unknown model name with fallback
target = compact_to(250_000, model="deepseek-r2", max_context=128_000, task="coding")

# Aggressive
target = compact_to(250_000, model="claude-opus-4.6", task="coding", dial=0.0)  # → 8,000

# Conservative
target = compact_to(250_000, model="claude-opus-4.6", task="coding", dial=1.0)  # → 225,000

# Quality inspection
q = quality_at(250_000, task="coding", model="claude-opus-4.6")  # → 0.858
q = quality_at(250_000, task="coding", max_context=512_000)      # → generic estimate

# All dial points at once
from compaction_target import compact_to_range
targets = compact_to_range(250_000, model="gemini-2.5-pro", task="coding")
```

## Unknown models: the `max_context` parameter

For models not in the profile database, pass `max_context` (the model's advertised
context window in tokens). This generates a conservative generic degradation curve:

```
quality = 1.0 - 0.30 × (tokens / max_context)^0.6
```

This shape sits between the best (Claude Opus) and worst (Gemini Flash) profiled
models — a deliberate "assume median" stance. If `model` is given but not recognized
and `max_context` is also given, the generic profile is used as fallback.

The generic profile is intentionally pessimistic for harder tasks. At 50% of max
context, the generic model estimates 80% quality for retrieval, 50% for coding, and
41% for reasoning. This means unknown models get aggressive compaction targets — which
is the right default when you don't have benchmark data to justify keeping more context.

## What this doesn't do

- It doesn't call your compactor. It just gives you a number.
- It doesn't model compaction quality. It doesn't know if your compactor is good or bad.
- It doesn't decide *when* to compact. It only answers "to how many tokens."
- It doesn't handle the trigger logic ("am I past the threshold?"). That's your call.
