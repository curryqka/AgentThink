#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Convert DriveLMM-o1 JSON into ms-swift JSONL (plain CoT).

Used for the supervised warm-up runs that do **not** include tool calls.  For
tool-augmented traces use ``scripts/tools/build_sft_data.py`` instead.

Example:
    python scripts/tools/prepare_swift_data.py \
        --input data/DriveLMMo1/DriveLMMo1_TRAIN.json \
        --output data/DriveLMMo1_swift_TRAIN.jsonl \
        --image-dir data/image2concat --split-type train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_records(data, split_type: str, image_dir: str, ratio: float = 1.0):
    chosen = data[: round(ratio * len(data))] if ratio < 1.0 else data
    records = []
    for item in chosen:
        question = item["question"]
        image_path = str(Path(image_dir) / f"{item['idx'].rsplit('_', 1)[0]}.png")

        if split_type == "train":
            answer_content = item["answer"]
        else:
            answer_content = (
                "**Step-by-Step Reasoning**:\n\n"
                f"{item['steps'].strip()}\n\n"
                f"**Final Answer**: {item['final_answer']}"
            )

        records.append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>{question}"},
                    {"role": "assistant", "content": answer_content},
                ],
                "images": [image_path],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift JSONL from DriveLMM-o1")
    parser.add_argument("--input", required=True, help="DriveLMMo1_TRAIN/TEST.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--image-dir", required=True, help="Directory of stitched images")
    parser.add_argument("--split-type", choices=["train", "val"], default="train")
    parser.add_argument("--ratio", type=float, default=1.0, help="Fraction of the dataset to keep")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = build_records(data, args.split_type, args.image_dir, args.ratio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
