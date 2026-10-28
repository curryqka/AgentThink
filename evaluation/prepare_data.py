#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Convert a tool-result JSON file into the JSONL format used by inference.

Input: the output of ``scripts/tools/gen_tool_result_data.py`` (a JSON list
whose items carry ``question``, ``steps``, ``final_answer``, ``tool_result``
and ``system_prompts``).

Output: one JSON object per line with the fields expected by
``evaluation/runner.py``::

    {"id", "image", "conversations": [human question, gpt answer, gpt steps],
     "tool_result", "system_prompts"}

Example:
    python evaluation/prepare_data.py \
        --input data/tool_results/cot_val_Qwen2.5-VL.json \
        --output data/DriveLMMo1_TEST_tool_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def image_name(idx: str, pattern: str = "{stem}.png") -> str:
    """Map a DriveLMM-o1 ``idx`` to its stitched six-view image name."""
    stem = idx.rsplit("_", 1)[0]
    return pattern.format(stem=stem)


def build_records(records: Iterable[dict], image_pattern: str) -> List[dict]:
    """Build JSONL records, skipping items without a question or answer."""
    output = []
    for item in records:
        idx = item.get("idx") or item.get("id")
        if not idx:
            continue
        question = item.get("question")
        final_answer = item.get("final_answer")
        steps = item.get("steps", "")
        if not question or final_answer is None:
            continue
        output.append(
            {
                "id": idx,
                "image": image_name(str(idx), image_pattern),
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": final_answer},
                    {"from": "gpt", "value": steps},
                ],
                "tool_result": item.get("tool_result", []),
                "system_prompts": item.get("system_prompts", ""),
            }
        )
    return output


def write_jsonl(records: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tool-result JSONL for evaluation")
    parser.add_argument("--input", required=True, help="Tool-result JSON (list of samples).")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--image-pattern",
        default="{stem}.png",
        help="Filename pattern for the stitched image (default: {stem}.png).",
    )
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    output_records = build_records(records, args.image_pattern)
    count = write_jsonl(output_records, Path(args.output))
    print(f"Wrote {count} records to {args.output} "
          f"(skipped {len(records) - count}).")


if __name__ == "__main__":
    main()
