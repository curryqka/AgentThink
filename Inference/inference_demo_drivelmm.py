#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Demo: six-view stitched image + driving-centric tools (DriveLMM-o1 style).

Tool results come from the Agent-Driver perception pickles (``val.pkl``), which
are required for :class:`scripts.tools.tool_libraries.FuncAgent`.

Example:
    python Inference/inference_demo_drivelmm.py \
        --model-path pretrained_model/AgentThink \
        --tool-data-path data/tool_results
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
from scripts.tools.gen_tool_result_data import AgentThink  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveLMM-o1 six-view demo")
    parser.add_argument("--model-path", default="pretrained_model/AgentThink")
    parser.add_argument(
        "--data-file",
        default=str(REPO_ROOT / "Inference" / "inference_demo_data_drivelmm.json"),
    )
    parser.add_argument(
        "--tool-data-path",
        default="data/tool_results",
        help="Directory containing Agent-Driver val/*.pkl files.",
    )
    parser.add_argument(
        "--image-root",
        default=str(REPO_ROOT / "data" / "image2concat"),
        help="Directory of stitched six-view images.",
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.data_file, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    interface = Qwen25VLInterface(args.model_path)
    results = []

    for sample in samples:
        sample_idx = sample["idx"]
        frame_token = sample_idx.split("_")[1]
        image_path = Path(args.image_root) / f"{sample_idx.rsplit('_', 1)[0]}.png"

        agent = AgentThink(
            token=frame_token,
            split=args.split,
            data_path=args.tool_data_path,
            model_name="Qwen2.5-VL",
            verbose=args.verbose,
            json_data_dict={"image": sample.get("image", [])},
        )
        tool_prompt = build_tool_prompt(
            agent.func_agent, sample.get("tool_result"), verbose=args.verbose
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
        print(f"[{sample_idx}] {output}\n")

        results.append(
            {
                "id": sample_idx,
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
