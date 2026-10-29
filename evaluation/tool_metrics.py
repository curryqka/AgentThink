# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Tool-use metrics introduced in Sec. 3.3 / Table 2 of the paper.

Three LLM-as-Judge scores complement the DriveLMM-o1 reasoning metrics:

* Tool Usage Appropriateness
* Tool Chain Coherence
* Perception-Guided Tool Alignment

They are used by ``evaluation/evaluation_script.py --tool-metrics`` and
reproduce the ``Tool Usage Appro. / Tool Chain Coh. / Percep-Guided Tool Align.``
columns of Table 5.
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Dict, Optional

from evaluation.prompts import TOOL_JUDGE_RESPONSE_FORMAT, TOOL_JUDGE_SYSTEM_PROMPT
from scripts.tools.tool_chain import extract_json_object

TOOL_METRIC_KEYS = (
    "Tool Usage Appropriateness",
    "Tool Chain Coherence",
    "Perception-Guided Tool Alignment",
)


def parse_tool_scores(content: str) -> Optional[Dict[str, float]]:
    if not content:
        return None
    candidate = extract_json_object(content) or content
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores = {
        key: float(data[key])
        for key in TOOL_METRIC_KEYS
        if isinstance(data.get(key), (int, float))
    }
    if len(scores) != len(TOOL_METRIC_KEYS):
        return None
    scores["Overall Tool Score"] = mean(scores.values())
    return scores


def judge_tool_usage(
    client,
    model: str,
    *,
    question: str,
    reference: str,
    response: str,
) -> Optional[Dict[str, float]]:
    """Return the three tool-use scores for one response, or ``None``."""
    messages = [
        {"role": "system", "content": TOOL_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Reference reasoning: {reference}\n\n"
                f"Model response:\n{response}"
            ),
        },
    ]
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=TOOL_JUDGE_RESPONSE_FORMAT,
                max_tokens=400,
                temperature=0.0,
            )
            scores = parse_tool_scores(completion.choices[0].message.content)
            if scores is not None:
                return scores
        except Exception as exc:
            if attempt == 2:
                print(f"Tool judge failed after 3 attempts: {exc}")
    return None
