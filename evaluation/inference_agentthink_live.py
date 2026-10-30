#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""AgentThink inference with live tool execution.

The model plans tool calls, the tools are executed against the Agent-Driver
perception pickle, observations are appended to the context, and generation
continues for up to ``--max-turns`` turns.

Use ``inference_agentthink.py`` instead when you want the released pipeline with
precomputed ``tool_result`` observations.

Example:
    python evaluation/inference_agentthink_live.py \
        --checkpoint pretrained_model/AgentThink \
        --tool-data-path data/tool_results --pickle-split val \
        --max-turns 6 --output-name AgentThink_live
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import live_runner, prompts, runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentThink live tool inference")
    runner.add_common_arguments(
        parser, default_test_file=runner.DEFAULT_TOOL_RESULT_TEST_FILE
    )
    parser.add_argument(
        "--tool-data-path",
        default="data/tool_results",
        help="Agent-Driver pickle root.",
    )
    parser.add_argument(
        "--pickle-split",
        choices=["val", "train"],
        default="val",
        help="Pickle folder for the evaluation split (TEST uses val).",
    )
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--system-message",
        default=prompts.EVAL_SYSTEM_MESSAGE,
        help="System message used for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner.validate_args(args)
    if args.max_turns <= 0:
        raise ValueError("--max-turns must be positive")
    random.seed(args.seed)

    runner.init_distributed()
    engine, _, _ = runner.load_engine(args.checkpoint)

    dataset = runner.DriveLMMo1Dataset(
        args.test_file,
        prompts.LIVE_TOOL_PROMPT,
        args.image_root,
        limit=args.limit,
    )
    outputs = live_runner.evaluate_live(
        engine, dataset, args, system_message=args.system_message
    )
    runner.gather_and_save(outputs, args, dataset_name="DriveLMMo1")


if __name__ == "__main__":
    main()
