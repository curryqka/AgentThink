# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Structured GRPO rewards for AgentThink (paper Table 1).

Three reward families are implemented as pure, deterministic functions so that
they can be used directly as ms-swift GRPO reward plugins:

* :func:`final_answer_reward` - task-level correctness.
* :func:`step_reasoning_reward` - step matching and chain coherence.
* :func:`tool_use_reward` - output-format compliance and tool/reasoning
  integration quality.

:func:`agentthink_reward` combines them with the weights below.  The paper
specifies the three components but not their relative weights; the defaults are
the values used in our GRPO runs (``final=0.4, step=0.3, tool=0.3``) and can be
overridden with the ``AGENTTHINK_REWARD_WEIGHTS`` environment variable
(``"0.4,0.3,0.3"``).

All functions accept the ms-swift convention ``(completions, **kwargs)`` and
return a list of floats, one per completion.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from scripts.tools.tool_chain import parse_tool_chain, validate_tool_chain

DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)  # final answer, step reasoning, tool use

ANSWER_MARKERS = (
    "The final answer is:",
    "**Final Answer:**",
    "**Final Answer**:",
    "Final Answer",
    "Answer",
    "**Final Decision**:",
    "Final Step:",
    "<CONCLUSION>",
)
OPTION_PATTERN = re.compile(r"\b([A-F])\)")
WORD_PATTERN = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(WORD_PATTERN.findall((text or "").lower()))


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between two strings."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _as_list(value: Any) -> List[Any]:
    """Normalise the many shapes a dataset column can take."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def extract_option_label(text: str) -> Optional[str]:
    """Return the first ``A)``-style option label in ``text``."""
    match = OPTION_PATTERN.search(text or "")
    return match.group(1) if match else None


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a CoT string or a Tool-CoT JSON object."""
    if not text:
        return ""
    chain = parse_tool_chain(text)
    if chain and isinstance(chain.get("final_answer"), str):
        return chain["final_answer"].strip()

    for marker in ANSWER_MARKERS:
        if marker in text:
            return text.split(marker)[-1].strip()
    return ""


def extract_reasoning(text: str) -> str:
    """Return only the reasoning part of a completion."""
    chain = parse_tool_chain(text)
    if chain and isinstance(chain.get("Chain"), list):
        parts = []
        for node in chain["Chain"]:
            if not isinstance(node, dict):
                continue
            parts.append(str(node.get("Sub", "")))
            if node.get("Guess_Answer"):
                parts.append(str(node["Guess_Answer"]))
        return "\n".join(parts)

    marker = "**Step-by-Step Reasoning**:"
    if marker in text:
        reasoning = text.split(marker, 1)[1]
        for stop in ANSWER_MARKERS:
            if stop in reasoning:
                reasoning = reasoning.split(stop)[0]
        return reasoning
    return text


def tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from a Tool-CoT JSON object or ``Tool(...)`` text."""
    chain = parse_tool_chain(text)
    if chain and isinstance(chain.get("Chain"), list):
        return [
            node["Tool"]
            for node in chain["Chain"]
            if isinstance(node, dict) and isinstance(node.get("Tool"), dict)
        ]
    return [
        {"function_name": name, "parameters": []}
        for name in re.findall(r"Tool\(\s*([A-Za-z_][\w]*)", text or "")
    ]


# ---------------------------------------------------------------------------
# 1. Final answer reward
# ---------------------------------------------------------------------------

def final_answer_reward(
    completions: Sequence[str],
    solution: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Compare the predicted final answer with the reference.

    Multiple-choice samples are scored on the option label; free-form answers
    use exact match with a token-F1 fallback.
    """
    references = _as_list(solution)
    rewards = []
    for index, completion in enumerate(completions):
        reference = references[index] if index < len(references) else references[0] if references else ""
        reference = extract_final_answer(reference) or reference

        prediction = extract_final_answer(completion)
        pred_label = extract_option_label(prediction)
        ref_label = extract_option_label(reference)
        if pred_label and ref_label:
            rewards.append(float(pred_label == ref_label))
            continue

        pred_norm = _normalize(prediction)
        ref_norm = _normalize(reference)
        if not pred_norm or not ref_norm:
            rewards.append(0.0)
            continue
        if pred_norm == ref_norm:
            rewards.append(1.0)
        else:
            rewards.append(token_f1(prediction, reference))
    return rewards


# ---------------------------------------------------------------------------
# 2. Step reasoning reward
# ---------------------------------------------------------------------------

def _iter_key_groups(key_steps: Any) -> List[List[str]]:
    """Normalise the ``key_steps`` column into groups of alternative keywords."""
    groups: List[List[str]] = []
    for item in _as_list(key_steps):
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                groups.append([item])
                continue
        if isinstance(item, dict):
            item = item.get("Steps", [])
        for group in _as_list(item):
            groups.append([str(alternative) for alternative in _as_list(group)])
    return groups


def step_matching_score(reasoning: str, key_steps: Any) -> float:
    """Fraction of reference key-step groups covered by the reasoning."""
    groups = _iter_key_groups(key_steps)
    if not groups:
        return 0.0
    reasoning_lower = (reasoning or "").lower()
    matched = sum(
        1
        for group in groups
        if any(alternative.lower() in reasoning_lower for alternative in group)
    )
    return matched / len(groups)


def coherence_score(completion: str) -> float:
    """Heuristic for smooth, ordered reasoning transitions (0-1).

    A chain is coherent when every reasoning step is non-empty, the final step
    concludes, and steps are not duplicated.
    """
    chain = parse_tool_chain(completion)
    if not chain or not isinstance(chain.get("Chain"), list):
        text = extract_reasoning(completion)
        sentences = [s.strip() for s in re.split(r"[\n.]", text) if s.strip()]
        if not sentences:
            return 0.0
        return 1.0 if len(sentences) >= 2 else 0.5

    steps = [node for node in chain["Chain"] if isinstance(node, dict)]
    if not steps:
        return 0.0
    non_empty = sum(1 for node in steps if str(node.get("Sub", "")).strip())
    unique = len({_normalize(str(node.get("Sub", ""))) for node in steps})
    ends_with_conclude = steps[-1].get("next_action") == "conclude"
    score = non_empty / len(steps)
    score *= unique / len(steps)
    if not ends_with_conclude:
        score *= 0.8
    return score


def step_reasoning_reward(
    completions: Sequence[str],
    solution: Optional[Sequence[str]] = None,
    key_steps: Any = None,
    **kwargs: Any,
) -> List[float]:
    """Mean of step matching (0.7) and chain coherence (0.3)."""
    references = _as_list(solution)
    rewards = []
    for index, completion in enumerate(completions):
        reference = references[index] if index < len(references) else ""
        reasoning = extract_reasoning(completion)
        matching = step_matching_score(reasoning, key_steps)
        if not _iter_key_groups(key_steps) and reference:
            matching = token_f1(reasoning, extract_reasoning(reference))
        rewards.append(0.7 * matching + 0.3 * coherence_score(completion))
    return rewards


# ---------------------------------------------------------------------------
# 3. Tool use reward
# ---------------------------------------------------------------------------

def tool_format_score(completion: str) -> float:
    """1.0 for a schema-valid Tool-CoT object, 0.0 otherwise."""
    chain = parse_tool_chain(completion)
    if chain is None:
        return 0.0
    return 0.0 if validate_tool_chain(chain) else 1.0


def tool_integration_score(completion: str, key_steps: Any = None) -> float:
    """Measure whether tool outputs are woven into the reasoning.

    A tool call is well integrated when it belongs to a step with a concrete
    sub-question and either a guess answer or keywords.  When ``key_steps`` are
    available, coverage of the reference steps is also required.
    """
    chain = parse_tool_chain(completion)
    if not chain or not isinstance(chain.get("Chain"), list):
        return 0.0
    steps = [node for node in chain["Chain"] if isinstance(node, dict)]
    if not steps:
        return 0.0

    integrated = 0
    for node in steps:
        tool = node.get("Tool")
        if not isinstance(tool, dict) or not tool.get("function_name"):
            continue
        has_sub = bool(str(node.get("Sub", "")).strip())
        has_evidence = bool(node.get("Guess_Answer")) or bool(node.get("key_words"))
        if has_sub and has_evidence:
            integrated += 1
    tool_ratio = integrated / len(steps)

    if _iter_key_groups(key_steps):
        coverage = step_matching_score(extract_reasoning(completion), key_steps)
        return 0.7 * tool_ratio + 0.3 * coverage
    return tool_ratio


def tool_use_reward(
    completions: Sequence[str],
    solution: Optional[Sequence[str]] = None,
    key_steps: Any = None,
    **kwargs: Any,
) -> List[float]:
    """Mean of format compliance (0.4) and integration quality (0.6)."""
    return [
        0.4 * tool_format_score(completion)
        + 0.6 * tool_integration_score(completion, key_steps)
        for completion in completions
    ]


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def get_weights() -> tuple[float, float, float]:
    raw = os.getenv("AGENTTHINK_REWARD_WEIGHTS", "")
    if not raw:
        return DEFAULT_WEIGHTS
    parts = [float(value) for value in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("AGENTTHINK_REWARD_WEIGHTS must contain three values")
    return tuple(parts)  # type: ignore[return-value]


def agentthink_reward(
    completions: Sequence[str],
    solution: Optional[Sequence[str]] = None,
    key_steps: Any = None,
    **kwargs: Any,
) -> List[float]:
    """Weighted combination of the three reward families (paper Table 1)."""
    final_rewards = final_answer_reward(completions, solution)
    step_rewards = step_reasoning_reward(completions, solution, key_steps)
    tool_rewards = tool_use_reward(completions, solution, key_steps)
    w_final, w_step, w_tool = get_weights()
    return [
        w_final * final + w_step * step + w_tool * tool
        for final, step, tool in zip(final_rewards, step_rewards, tool_rewards)
    ]
