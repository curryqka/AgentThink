# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Parsing and validation of the structured Tool-CoT output.

The data-generation pipeline (GPT-4o) and ``evaluation/inference_withtool.py``
both use the schema introduced in Sec. 3.1 of the paper::

    {
      "Question": "...",
      "Chain": [
        {
          "Tool": {"function_name": "...", "parameters": [...]},
          "Sub": "...",
          "Guess_Answer": "...",
          "key_words": ["..."],
          "Missing_flag": "False",
          "next_action": "continue reasoning" | "conclude"
        },
        ...
      ],
      "final_answer_keywords": ["..."],
      "final_answer": "..."
    }

The helpers here are dependency-free so they can be reused by the data
pipeline, the inference scripts and the unit tests.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

REQUIRED_NODE_KEYS = ("Tool", "Sub")
VALID_NEXT_ACTIONS = ("continue reasoning", "conclude")


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def extract_json_object(text: str) -> Optional[str]:
    """Return the outermost ``{...}`` object found in ``text``.

    Models occasionally prefix the JSON with a sentence even when asked not to,
    so a strict ``json.loads`` on the whole response is too brittle.
    """
    if not text:
        return None
    text = strip_markdown_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_tool_chain(text: str) -> Optional[Dict[str, Any]]:
    """Parse a model response into a Tool-CoT dictionary.

    Returns ``None`` when the response is not a JSON object.
    """
    candidate = extract_json_object(text)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_tool_chain(data: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable schema violations (empty when valid)."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["response is not a JSON object"]

    chain = data.get("Chain")
    if not isinstance(chain, list) or not chain:
        errors.append('missing or empty "Chain" list')
        return errors

    for index, node in enumerate(chain):
        prefix = f"Chain[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{prefix} is not an object")
            continue
        for key in REQUIRED_NODE_KEYS:
            if key not in node:
                errors.append(f'{prefix} is missing "{key}"')
        tool = node.get("Tool")
        if not isinstance(tool, dict):
            errors.append(f'{prefix}."Tool" is not an object')
        else:
            if not isinstance(tool.get("function_name"), str):
                errors.append(f'{prefix}."Tool"."function_name" must be a string')
            if not isinstance(tool.get("parameters", []), list):
                errors.append(f'{prefix}."Tool"."parameters" must be a list')
        if node.get("next_action") not in VALID_NEXT_ACTIONS:
            errors.append(
                f'{prefix}."next_action" must be one of {VALID_NEXT_ACTIONS}'
            )

    if not isinstance(data.get("final_answer", ""), str):
        errors.append('"final_answer" must be a string')
    return errors


def iter_tool_calls(chain: Iterable[Dict[str, Any]]) -> Iterable[Tuple[str, List[Any]]]:
    """Yield ``(function_name, parameters)`` for every step of a chain."""
    for node in chain:
        tool = node.get("Tool") if isinstance(node, dict) else None
        if not isinstance(tool, dict):
            continue
        name = tool.get("function_name")
        if not isinstance(name, str) or not name:
            continue
        parameters = tool.get("parameters") or []
        if not isinstance(parameters, list):
            parameters = [parameters]
        yield name, parameters
