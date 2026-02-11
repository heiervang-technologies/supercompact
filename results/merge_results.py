#!/usr/bin/env python3
"""Merge LLM-as-Judge composite scores with speed data for Pareto plotting."""

import json
from pathlib import Path

# --- Load all data sources ---

# 8K LLM eval (has speed_s, kept_tokens, composite)
llm_8k = json.loads(Path("llm_eval_results_8k.json").read_text())

# 3K LLM eval (earlier run, no speed data)
llm_3k = json.loads(Path("llm_eval_results.json").read_text())

# Original recall-based results (has speed_s, kept_tokens, recall)
recall = json.loads(Path("pareto_fixed_results.json").read_text())

# --- Build merged results ---
# Focus on "capable" (Opus-4.5) model for the primary Pareto plot
# We can add cheap model as dimmed secondary points

merged = []

# 8K results (dedup + llama-embed only — llama-rerank was corrupted)
for r in llm_8k:
    if r["method"] == "llama-rerank":
        continue  # corrupted by 402 errors
    merged.append({
        "method": r["method"],
        "budget": r["budget"],
        "model_key": r["model_key"],
        "model_label": r["model_label"],
        "composite": r["composite"],
        "speed_s": r["speed_s"],
        "kept_tokens": r["kept_tokens"],
        "total_tokens": r["total_tokens"],
    })

# 3K results — add speed from recall data (dedup at 2K budget is closest)
recall_speeds = {(r["method"], r["budget"]): r["speed_s"] for r in recall}
for r in llm_3k:
    # Use dedup 2K speed as proxy for 3K (same mandatory floor)
    speed = recall_speeds.get((r["method"], 2000), 1.2)
    merged.append({
        "method": r["method"],
        "budget": r["budget"],
        "model_key": r["model_key"],
        "model_label": r["model_label"],
        "composite": r["composite"],
        "speed_s": speed,
        "kept_tokens": 7144,  # mandatory floor
        "total_tokens": 81696,
    })

# Add recall-based 20K results with estimated composite
# (we couldn't run LLM eval due to credits, but we have speed + recall)
for r in recall:
    if r["budget"] == 20000:
        merged.append({
            "method": r["method"],
            "budget": r["budget"],
            "model_key": "recall_only",
            "model_label": "TF-IDF (no composite)",
            "composite": None,
            "recall": r["recall"],
            "speed_s": r["speed_s"],
            "kept_tokens": r["kept_tokens"],
            "total_tokens": r["total_tokens"],
        })

out = Path("llm_eval_merged.json")
out.write_text(json.dumps(merged, indent=2))
print(f"Wrote {len(merged)} entries to {out}")

# Print summary
print("\nComposite scores (Opus-4.5):")
for r in merged:
    if r["model_key"] == "capable":
        print(f"  {r['method']:15s} budget={r['budget']:6,}  composite={r['composite']:.3f}  speed={r['speed_s']:.1f}s  kept={r['kept_tokens']:,}")

print("\nComposite scores (Kimi-K2.5):")
for r in merged:
    if r["model_key"] == "cheap":
        print(f"  {r['method']:15s} budget={r['budget']:6,}  composite={r['composite']:.3f}  speed={r['speed_s']:.1f}s")

print("\nRecall-only (20K, no composite):")
for r in merged:
    if r["model_key"] == "recall_only":
        print(f"  {r['method']:15s} budget={r['budget']:6,}  recall={r['recall']:.3f}  speed={r['speed_s']:.1f}s")
