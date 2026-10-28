#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Execute Planned Tool Chains (Stage 1c of the paper pipeline).

Given the structured Tool-CoT produced by ``agentthink_data_generater_pipeline``
(or by ``evaluation/inference_withtool.py``), this script runs every tool call on
the Agent-Driver perception results and stores the observations next to the
sample.  The resulting JSON is converted to JSONL by
``evaluation/prepare_data.py`` and consumed by
``evaluation/inference_agentthink.py``.

The tool observations are **not** produced by the VLM: they come from the
Agent-Driver ``val/*.pkl`` perception files.  This is why ``tool_result`` and
``DriveLMMo1_TEST_tool_results.jsonl`` exist in the released data.

Example:
    python scripts/tools/gen_tool_result_data.py \
        --input cot_test_gpt-4o-mini.json \
        --output data/tool_results/cot_val_Qwen2.5-VL.json \
        --tool-data-path data/tool_results \
        --split val
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.tool_chain import parse_tool_chain  # noqa: E402
from scripts.tools.tool_libraries import FuncAgent  # noqa: E402
from scripts.tools.tool_prompts import (  # noqa: E402
    get_ego_prompts,
    get_system_prompt,
)


class AgentThink:
    """Bind a Tool-CoT chain to one nuScenes frame."""

    #: Keyword -> category used to enforce the per-category call limits.
    CATEGORY_KEYWORDS = {
        "detection": ("detection",),
        "prediction": ("trajector", "future"),
        "occupancy": ("occupancy",),
        "map": ("drivable", "lane", "shoulder", "crossing"),
    }

    def __init__(
        self,
        token: str,
        split: str = "train",
        data_path: str = "data/tool_results",
        model_name: str = "Qwen2.5-VL",
        verbose: bool = False,
    ) -> None:
        self.token = token
        self.split = split
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.verbose = verbose

        folder = Path("val") if "val" in split else Path("train")
        self.file_name = self.data_path / folder / f"{token}.pkl"
        with self.file_name.open("rb") as handle:
            self.data_dict = pickle.load(handle)

        self.func_agent = FuncAgent(self.data_dict)

        # Safety limits on repeated tool categories (Sec. 3.1 of the paper).
        self.max_calls = {
            "detection": 3,
            "prediction": 1,
            "occupancy": 1,
            "map": 1,
        }

    # -- argument handling ---------------------------------------------------
    @staticmethod
    def _coerce_argument(value: Any, name: str) -> Any:
        """Convert the loose JSON values produced by an LLM to Python types."""
        if name in ("locations", "location"):
            if isinstance(value, (list, tuple)):
                if value and isinstance(value[0], (list, tuple)):
                    return [tuple(point) for point in value]
                if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                    return [tuple(value)]
            return [value] if not isinstance(value, list) else value
        if name in ("object_ids", "object_names") and isinstance(value, str):
            return [value]
        if name == "timestep":
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return value

    def _build_kwargs(self, method, parameters: List[Any]) -> Optional[Dict[str, Any]]:
        """Bind the model's positional parameter list to the method signature.

        Parameters with defaults (e.g. ``return_score``) are optional, so a
        shorter list from the planner is still valid.
        """
        signature_params = [
            parameter
            for parameter in inspect.signature(method).parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        required = [
            parameter.name
            for parameter in signature_params
            if parameter.default is inspect.Parameter.empty
        ]
        optional = [
            parameter.name
            for parameter in signature_params
            if parameter.default is not inspect.Parameter.empty
        ]

        if len(parameters) == len(required):
            names = required
        elif len(parameters) == len(required) + len(optional):
            names = required + optional
        elif len(required) == 1:
            names = required
        else:
            if self.verbose:
                print(
                    f"Mismatched arguments for {method.__name__}: "
                    f"got {parameters}, expected {required}"
                )
            return None

        if len(names) == 1 and len(parameters) != 1:
            return {names[0]: self._coerce_argument(parameters, names[0])}
        return {
            name: self._coerce_argument(value, name)
            for name, value in zip(names, parameters)
        }

    def tool_call(self, node: Dict[str, Any]) -> Optional[dict]:
        """Execute one ``Chain`` node and return ``{name, args, prompt}``."""
        tool = node.get("Tool") if isinstance(node, dict) else None
        if not isinstance(tool, dict):
            return None
        tool_name = tool.get("function_name") or ""
        if not tool_name or tool_name == "none":
            return None

        parameters = tool.get("parameters") or []
        if parameters == [""]:
            parameters = []

        method = getattr(self.func_agent, tool_name, None)
        if method is None or not callable(method):
            if self.verbose:
                print(f"Unknown tool: {tool_name}")
            return None

        kwargs = self._build_kwargs(method, parameters)
        if kwargs is None:
            return None

        try:
            tool_prompt, _ = method(**kwargs)
        except Exception as exc:
            if self.verbose:
                print(f"Tool {tool_name} failed: {exc}")
            return None

        if self.verbose:
            print(f"Tool: {tool_name}\nArgs: {kwargs}\nPrompt: {tool_prompt}")
        return {"name": tool_name, "args": kwargs, "prompt": tool_prompt or ""}

    @classmethod
    def category_for(cls, tool_name: str) -> Optional[str]:
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(keyword in tool_name for keyword in keywords):
                return category
        return None

    # -- chain execution -----------------------------------------------------
    def get_tool_results(
        self,
        cot_data: Dict[str, Any],
        ego_prompts: str = "",
    ) -> Tuple[List[Optional[dict]], str]:
        """Run every tool of a chain, respecting the per-category call limits.

        Returns the list of tool responses and the system prompt (tool
        observations are rendered for the VLM by ``evaluation/runner.py``).
        """
        system_message = get_system_prompt() + "\n" + (ego_prompts or "") + "\n"
        tool_responses: List[Optional[dict]] = []
        call_counts = {key: 0 for key in self.max_calls}

        for node in cot_data.get("Chain", []):
            tool = node.get("Tool") if isinstance(node, dict) else None
            tool_name = (tool or {}).get("function_name", "")
            category = self.category_for(tool_name)
            if category:
                call_counts[category] += 1
                if call_counts[category] > self.max_calls[category]:
                    tool_responses.append(None)
                    continue
            if not tool_name:
                tool_responses.append(None)
                continue

            tool_responses.append(self.tool_call(node))

        return tool_responses, system_message


def load_cot_data(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Accept either a ``cot_data`` field or a raw model answer string."""
    if isinstance(sample.get("cot_data"), dict):
        return sample["cot_data"]
    answer = sample.get("answer") or sample.get("llm-response")
    if isinstance(answer, str):
        return parse_tool_chain(answer)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Tool-CoT chains")
    parser.add_argument("--input", required=True, help="JSON with cot_data or model answers.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--tool-data-path",
        default="data/tool_results",
        help="Directory containing Agent-Driver val/ and train/ pickles.",
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--model-name", default="Qwen2.5-VL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    outputs, skipped = [], 0
    for sample in tqdm(samples, desc="Executing tool chains"):
        idx = sample.get("idx") or sample.get("id")
        cot_data = load_cot_data(sample)
        if not idx or not cot_data:
            skipped += 1
            continue

        parts = str(idx).split("_")
        if len(parts) < 2:
            skipped += 1
            continue
        frame_token = parts[1]

        try:
            agent = AgentThink(
                token=frame_token,
                split=args.split,
                data_path=args.tool_data_path,
                model_name=args.model_name,
                verbose=args.verbose,
            )
        except FileNotFoundError:
            skipped += 1
            if args.verbose:
                print(f"Missing perception pickle for frame {frame_token}")
            continue

        try:
            ego_prompts = get_ego_prompts(agent.data_dict)
        except (KeyError, IndexError, TypeError) as exc:
            if args.verbose:
                print(f"Incomplete ego state for frame {frame_token}: {exc}")
            ego_prompts = ""

        tool_responses, system_prompts = agent.get_tool_results(
            cot_data, ego_prompts=ego_prompts
        )

        sample["cot_data"] = cot_data
        sample["tool_result"] = tool_responses
        sample["system_prompts"] = system_prompts
        outputs.append(sample)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=4, ensure_ascii=False)

    print(
        f"Wrote {len(outputs)} samples to {output_path} "
        f"(skipped {skipped}/{len(samples)})."
    )


if __name__ == "__main__":
    main()
