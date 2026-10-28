#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""AgentThink tool-augmented inference (Stage 3 of the paper pipeline).

The model answers the driving question after the retrieval results of the
planned tool chain have been inserted into the prompt.  This is the script used
to obtain the numbers reported for AgentThink in Table 3.

Input: a JSONL file that contains a ``tool_result`` field per sample, e.g.
``data/DriveLMMo1_TEST_tool_results.jsonl`` (built with
``scripts/tools/gen_tool_result_data.py`` and ``evaluation/prepare_data.py``).

Example:
    python evaluation/inference_agentthink.py \
        --checkpoint pretrained_model/AgentThink \
        --output-name AgentThink
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prompts, runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentThink tool-augmented inference")
    runner.add_common_arguments(
        parser, default_test_file=runner.DEFAULT_TOOL_RESULT_TEST_FILE
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=prompts.EVAL_SYSTEM_MESSAGE,
        help="System message used for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner.validate_args(args)
    random.seed(args.seed)

    runner.init_distributed()
    engine, _, _ = runner.load_engine(args.checkpoint)

    dataset = runner.DriveLMMo1Dataset(
        args.test_file,
        prompts.PLAIN_COT_PROMPT,
        args.image_root,
        with_tool_results=True,
        tool_result_style=args.tool_result_style,
        limit=args.limit,
    )
    outputs = runner.evaluate_model(
        engine, dataset, args, system_message=args.system_message
    )
    runner.gather_and_save(outputs, args, dataset_name="DriveLMMo1")


if __name__ == "__main__":
    main()
