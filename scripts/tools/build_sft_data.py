#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Build tool-augmented SFT / GRPO data for AgentThink (paper Sec. 3.2).

Input: the JSON produced by ``scripts/tools/gen_tool_result_data.py`` (one item
per question with ``cot_data``, ``tool_result`` and ``system_prompts``).

Three assistant formats are supported:

* ``plain``       - reference reasoning and final answer, no tool calls.
* ``tool-cot``    - the structured Tool-CoT JSON chain (what the planner emits).
* ``agentthink``  - the execution trace, optionally including tool outputs.  This
  is the format used for the two SFT phases in the paper (phase 1 without
  observations, phase 2 with observations).

Examples:
    # Phase 1 SFT: actions only, tool observations masked out
    python scripts/tools/build_sft_data.py --input ... --output ... \
        --format agentthink --target sft --no-observations

    # Phase 2 SFT: reasoning + observations
    python scripts/tools/build_sft_data.py --input ... --output ... \
        --format agentthink --target sft

    # GRPO prompts (reference answer and key steps become dataset columns)
    python scripts/tools/build_sft_data.py --input ... --output ... \
        --format agentthink --target grpo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.chain_render import render_plain, render_tool_cot, render_trace


def build_record(
    item: Dict[str, Any],
    *,
    fmt: str,
    target: str,
    image_dir: str,
    include_observations: bool,
) -> Optional[dict]:
    idx = item.get("idx") or item.get("id")
    question = item.get("question")
    if not idx or not question:
        return None

    if fmt == "plain":
        content = render_plain(item)
    elif fmt == "tool-cot":
        content = render_tool_cot(item.get("cot_data") or {})
    else:
        content = render_trace(
            item.get("cot_data") or {},
            item.get("tool_result") or [],
            include_observations=include_observations,
        )

    # For tool-aware formats the ego-state context is part of the prompt.
    if fmt != "plain" and item.get("system_prompts"):
        question = f"{str(item['system_prompts']).strip()}\n{question}"

    image_path = str(Path(image_dir) / f"{str(idx).rsplit('_', 1)[0]}.png")
    messages = [{"role": "user", "content": f"<image>{question}"}]
    if target == "sft":
        messages.append({"role": "assistant", "content": content})

    record = {
        "messages": messages,
        "images": [image_path],
        "solution": item.get("final_answer", ""),
        "key_steps": key_steps_from_chain(item.get("cot_data") or {}),
    }
    return record


def key_steps_from_chain(cot_data: Dict[str, Any]) -> List[List[str]]:
    """Use the per-step ``key_words`` of the Tool-CoT as reference groups."""
    groups = []
    for node in cot_data.get("Chain", []) or []:
        if not isinstance(node, dict):
            continue
        words = [str(word) for word in node.get("key_words") or [] if str(word).strip()]
        if not words and node.get("Guess_Answer"):
            words = [str(node["Guess_Answer"])]
        if words:
            groups.append(words)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AgentThink SFT/GRPO data")
    parser.add_argument("--input", required=True, help="Tool-result JSON.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--image-dir", default="data/image2concat")
    parser.add_argument("--format", choices=["plain", "tool-cot", "agentthink"], default="agentthink")
    parser.add_argument("--target", choices=["sft", "grpo"], default="sft")
    parser.add_argument(
        "--no-observations",
        action="store_true",
        help="Omit tool outputs from the trace (SFT phase 1).",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    if args.limit:
        items = items[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            record = build_record(
                item,
                fmt=args.format,
                target=args.target,
                image_dir=args.image_dir,
                include_observations=not args.no_observations,
            )
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path} "
          f"(format={args.format}, target={args.target})")


if __name__ == "__main__":
    main()
