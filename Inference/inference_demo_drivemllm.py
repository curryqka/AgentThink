#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Demo: single camera + open-vocabulary tools (DriveMLLM style samples).

The example data already contains the tool chain to execute; the script runs the
tools with :class:`scripts.tools.tool_libraries_simple.FuncAgent`, inserts their
outputs into the prompt, and asks AgentThink for the final answer.

Example:
    python Inference/inference_demo_drivemllm.py \
        --model-path pretrained_model/AgentThink
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Inference.qwen2_5vl import Qwen25VLInterface, build_tool_prompt  # noqa: E402
from scripts.tools.tool_libraries_simple import FuncAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveMLLM single-view demo")
    parser.add_argument("--model-path", default="pretrained_model/AgentThink")
    parser.add_argument(
        "--data-file",
        default=str(REPO_ROOT / "Inference" / "inference_demo_data_drivemllm.json"),
    )
    parser.add_argument(
        "--image-root",
        default=str(REPO_ROOT),
        help="Prefix prepended to the relative image paths stored in the demo data.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON file for the outputs.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.data_file, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    func_agent = FuncAgent()
    interface = Qwen25VLInterface(args.model_path)
    results = []

    for sample in samples:
        image_path = Path(args.image_root) / sample["image"][0]
        tool_prompt = build_tool_prompt(
            func_agent, sample.get("tool_result"), verbose=args.verbose
        )
        full_prompt = (
            f"{sample.get('system_prompts', '').strip()}\n"
            f"{sample['question']}\n"
            f"Tool results:{tool_prompt}"
        )

        image = Image.open(image_path).convert("RGB")
        output = interface.infer_with_retry(
            image, full_prompt, max_tokens=args.max_new_tokens
        )
        print(f"[{image_path.name}] {output}\n")

        results.append(
            {
                "image": str(image_path),
                "question": sample["question"],
                "answer": output,
                "gt_answers": sample.get("final_answer"),
            }
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=4, ensure_ascii=False)
        print(f"Saved {len(results)} outputs to {args.output}")


if __name__ == "__main__":
    main()
