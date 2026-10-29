#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Baseline inference: plain chain-of-thought, no tool use.

This reproduces the ``Qwen2.5-VL`` / ``Qwen + CoT`` rows of the benchmark
tables.  For the full AgentThink pipeline use ``inference_agentthink.py``.

Example:
    python evaluation/inference.py --checkpoint /path/to/ckpt --output-name qwen7b_cot

Use ``torchrun`` for multi-GPU evaluation (see scripts/inference_scripts/).
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
    parser = argparse.ArgumentParser(description="AgentThink baseline CoT inference")
    runner.add_common_arguments(parser)
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
        limit=args.limit,
    )
    outputs = runner.evaluate_model(engine, dataset, args)
    runner.gather_and_save(outputs, args, dataset_name="DriveLMMo1")


if __name__ == "__main__":
    main()
