from __future__ import annotations

import copy
from rich.console import Console

from .parser import Turn
from .tokenizer import turn_tokens
from .selector import SelectionResult

console = Console()

def nuclear_compact(turns: list[Turn], budget: int) -> SelectionResult:
    """Aggressively squash and truncate turns to fit within the budget.
    
    Maintains the structural integrity of the JSONL (UUIDs, turn types)
    but severely truncates long tool outputs and text blocks.
    """
    console.print(f"Running NUCLEAR compaction with budget {budget:,} tokens...")
    
    kept_turns = copy.deepcopy(turns)
    
    total_tokens_before = sum(turn_tokens(t) for t in turns)
    
    # Simple strategy: iterative truncation
    # First, truncate anything over 10,000 chars. Then 5000, 2000, 1000, 500
    # Stop when we fit the budget.
    
    current_tokens = total_tokens_before
    
    for max_len in [10000, 5000, 2000, 1000, 500, 250, 100]:
        if current_tokens <= budget:
            break
            
        console.print(f"  Over budget ({current_tokens:,} > {budget:,}). Squashing to max {max_len} chars per block...")
        
        for i, turn in enumerate(kept_turns):
            for record in turn.lines:
                if not isinstance(record, dict):
                    continue
                    
                msg = record.get("message", {})
                if not isinstance(msg, dict):
                    continue
                    
                content = msg.get("content")
                if isinstance(content, str):
                    if len(content) > max_len:
                        msg["content"] = content[:max_len//2] + f"\n\n...[TRUNCATED BY NUCLEAR (was {len(content)} chars)]...\n\n" + content[-max_len//2:]
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if len(text) > max_len:
                                    block["text"] = text[:max_len//2] + f"\n\n...[TRUNCATED BY NUCLEAR (was {len(text)} chars)]...\n\n" + text[-max_len//2:]
                            elif block.get("type") == "tool_result":
                                res_content = block.get("content")
                                if isinstance(res_content, str) and len(res_content) > max_len:
                                    block["content"] = res_content[:max_len//2] + f"\n\n...[TRUNCATED BY NUCLEAR (was {len(res_content)} chars)]...\n\n" + res_content[-max_len//2:]
                                elif isinstance(res_content, list):
                                    for sub in res_content:
                                        if isinstance(sub, dict) and sub.get("type") == "text":
                                            text = sub.get("text", "")
                                            if len(text) > max_len:
                                                sub["text"] = text[:max_len//2] + f"\n\n...[TRUNCATED BY NUCLEAR (was {len(text)} chars)]...\n\n" + text[-max_len//2:]

        current_tokens = sum(turn_tokens(t) for t in kept_turns)

    result = SelectionResult(
        kept_turns=kept_turns,
        budget=budget,
        total_input_tokens=total_tokens_before,
        scored_kept_tokens=current_tokens,
    )
    result.user_tokens = 0
    result.short_system_tokens = 0
    
    return result
