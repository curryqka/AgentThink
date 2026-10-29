#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Structured Tool-CoT generation (tool planning, no execution).

The model receives the tool catalog and is asked to produce the structured
reasoning chain defined in Sec. 3.1 of the paper.  The output JSON is parsed
and validated; the resulting ``tool_chain`` can be executed with
``scripts/tools/gen_tool_result_data.py`` to obtain the tool observations that
AgentThink conditions on.

This script does **not** answer the driving question.  Use
``inference_agentthink.py`` for tool-augmented answer generation and
``evaluation_script.py`` for scoring.

Example:
    python evaluation/inference_withtool.py \
        --checkpoint pretrained_model/AgentThink \
        --output-name toolcot
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prompts, runner  # noqa: E402
from scripts.tools.function_schema import build_tool_catalog  # noqa: E402
from scripts.tools.tool_chain import parse_tool_chain, validate_tool_chain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured Tool-CoT generation")
    runner.add_common_arguments(parser)
    parser.add_argument(
        "--tool-groups",
        type=str,
        default="",
        help="Comma-separated subset of visual,detection,prediction,occupancy,map "
        "(default: all tools).",
    )
    return parser.parse_args()


def postprocess(prediction: str) -> Dict[str, Any]:
    """Attach the parsed and validated tool chain to the output record."""
    chain = parse_tool_chain(prediction)
    if chain is None:
        return {
            "tool_chain": None,
            "format_valid": False,
            "format_errors": ["response is not a JSON object"],
        }
    errors = validate_tool_chain(chain)
    return {
        "tool_chain": chain,
        "format_valid": not errors,
        "format_errors": errors,
    }


def main() -> None:
    args = parse_args()
    runner.validate_args(args)
    random.seed(args.seed)

    runner.init_distributed()
    engine, _, _ = runner.load_engine(args.checkpoint)

    # The tool catalog is rendered once and reused for every sample.
    if args.tool_groups:
        from scripts.tools.function_schema import (
            format_function_prompt,
            list_function_infos,
        )

        groups = [group.strip() for group in args.tool_groups.split(",") if group.strip()]
        catalog = format_function_prompt(list_function_infos(groups=groups))
    else:
        catalog = build_tool_catalog()
    prompt = prompts.build_tool_cot_prompt(catalog)

    dataset = runner.DriveLMMo1Dataset(
        args.test_file,
        prompt,
        args.image_root,
        limit=args.limit,
    )

    outputs = runner.evaluate_model(engine, dataset, args, postprocess=postprocess)
    runner.gather_and_save(outputs, args, dataset_name="DriveLMMo1")

    if torch_distributed_rank() == 0:
        valid = sum(1 for record in outputs if record["format_valid"])
        print(f"Schema-valid tool chains: {valid}/{len(outputs)}")


def torch_distributed_rank() -> int:
    import torch

    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


if __name__ == "__main__":
    main()
