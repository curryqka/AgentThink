# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Render Tool-CoT structures into the text formats used for training.

Keeping the renderers separate from the data builders makes the exact training
serialisation easy to inspect and to unit test.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

ANSWER_MARKERS = (
    "**Final Answer**:",
    "**Final Answer:**",
    "The final answer is:",
    "Final Answer",
    "<CONCLUSION>",
)


def _split_reference(item: Dict[str, Any]) -> tuple[str, str]:
    """Return ``(reasoning, final_answer)`` from a dataset item."""
    if item.get("final_answer"):
        return item.get("steps", ""), item["final_answer"]

    answer = item.get("answer", "")
    reasoning = answer
    final = ""
    if "**Step-by-Step Reasoning**:" in answer:
        reasoning = answer.split("**Step-by-Step Reasoning**:", 1)[1]
    for marker in ANSWER_MARKERS:
        if marker in reasoning:
            before, _, after = reasoning.partition(marker)
            reasoning, final = before, after
            break
    return reasoning.strip(), final.strip()


def render_plain(item: Dict[str, Any]) -> str:
    """Reference CoT without tool calls (baseline SFT format)."""
    reasoning, final = _split_reference(item)
    return (
        "**Step-by-Step Reasoning**:\n\n"
        f"{reasoning.strip()}\n\n"
        f"**Final Answer**: {final.strip()}"
    )


def render_tool_cot(cot_data: Dict[str, Any]) -> str:
    """Serialise the structured chain exactly as the planner emits it."""
    return json.dumps(cot_data, indent=4, ensure_ascii=False)


def _render_parameters(parameters: Iterable[Any]) -> str:
    return json.dumps(list(parameters or []), ensure_ascii=False)


def render_trace(
    cot_data: Dict[str, Any],
    tool_results: Optional[List[Optional[dict]]] = None,
    *,
    include_observations: bool = True,
) -> str:
    """Render a chain plus its tool observations as a step-by-step trace.

    Args:
        cot_data: structured Tool-CoT object.
        tool_results: aligned list of ``{"name", "args", "prompt"}`` records
            produced by :mod:`scripts.tools.gen_tool_result_data`.
        include_observations: set to ``False`` for SFT phase 1, where tool
            outputs are intentionally hidden.
    """
    tool_results = tool_results or []
    lines = ["**Step-by-Step Reasoning**:"]

    for index, node in enumerate(cot_data.get("Chain", []) or [], start=1):
        if not isinstance(node, dict):
            continue
        sub = str(node.get("Sub", "")).strip()
        lines.append(f"Step {index}: {sub}" if sub else f"Step {index}:")

        tool = node.get("Tool") if isinstance(node.get("Tool"), dict) else None
        if tool and tool.get("function_name"):
            parameters = _render_parameters(tool.get("parameters") or [])
            lines.append(f"Tool: {tool['function_name']}({parameters})")

            if include_observations and index - 1 < len(tool_results):
                observation = (tool_results[index - 1] or {}).get("prompt") or ""
                if observation.strip():
                    lines.append(f"Tool Output: {observation.strip()}")

    final_answer = str(cot_data.get("final_answer", "")).strip()
    lines.extend(["", f"**Final Answer**: {final_answer}"])
    return "\n".join(lines)
