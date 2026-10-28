#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Generate structured Tool-CoT data with a GPT judge (paper Sec. 3.1).

Two-stage prompting per sample:

1. **Tool planning** - the model sees the question, the reference answer and the
   tool catalog, and selects at most four tools.
2. **Chain generation** - the model decomposes the reference reasoning into
   atomic steps and attaches a tool call, sub-question, guess answer,
   uncertainty flag and next action to every step.

Example:
    export OPENAI_API_KEY=sk-...
    python scripts/tools/agentthink_data_generater_pipeline.py \
        --input data/DriveLMMo1_TRAIN.json --split train --model-name gpt-4o-mini

The resulting ``final_cot_<split>_<model>.json`` is consumed by
``scripts/tools/gen_tool_result_data.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.function_schema import (  # noqa: E402
    format_function_prompt,
    list_function_infos,
)
from scripts.tools.tool_chain import extract_json_object  # noqa: E402

LOGGER = logging.getLogger("agentthink.data")


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def get_client():
    """Create the OpenAI client lazily from environment variables."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running the pipeline."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        timeout=120.0,
    )


def chat_completion(client, **kwargs):
    """Call the chat API with exponential backoff (3 attempts)."""
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # network / rate limit
            last_error = exc
            wait = 2 ** attempt
            LOGGER.warning("OpenAI call failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)
    raise RuntimeError("OpenAI call failed after 3 attempts") from last_error


def run_one_round_conversation(
    client,
    system_message: str,
    user_message: str,
    *,
    model_name: str,
    temperature: float = 0.0,
) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    response = chat_completion(
        client, model=model_name, messages=messages, temperature=temperature
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def ask_for_json(client, system_prompt: str, user_prompt: str, **kwargs) -> Any:
    """Ask for JSON, retrying when the model returns malformed output."""
    last_response = ""
    for attempt in range(1, 6):
        response = run_one_round_conversation(
            client, system_prompt, user_prompt, **kwargs
        )
        last_response = response
        candidate = extract_json_object(response)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        LOGGER.warning(
            "Attempt %d/5: invalid JSON (%s...)", attempt, response[:160].replace("\n", " ")
        )
    raise ValueError(f"Could not parse JSON after 5 attempts: {last_response[:200]}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    return """
**A Language Agent for Autonomous Driving**
Role: You are the brain of an autonomous vehicle (a.k.a. ego-vehicle).
Your task is to extract necessary information from the driving scenario that will be useful for driving question answering.

Necessary information might include:
- Visual infos: Visual information from specific cameras
- Detections: Detected objects that require attention
- Predictions: Estimated future motions of detected objects
- Maps: Traffic lanes and road boundaries
- Occupancy: Whether locations are occupied by other objects

Task:
- Determine what types of information (Visual info, Detections, Predictions, Maps, Occupancy) to extract
- Prioritize Detections and Predictions for motion planning questions
- Focus on Maps information for lane maintenance and road layout questions
"""


def get_user_prompt(raw_question: str, raw_answer: str) -> str:
    """Prompt for the first stage (tool selection)."""
    function_list = list_function_infos()
    tool_catalog = format_function_prompt(function_list)
    output_format = """
        Please choose tools to answer this problem and support the final answer.
        Return a list of tool names (no more than 4) with "tool_name" and related "parameters".
        If parameters are blank, use [""] as the value.

        Output format example:
        [
            {"tool_name": "function_name1", "parameters": ["arg1", "arg2"]},
            {"tool_name": "function_name2", "parameters": ["arg1", ["option1", "option2"]]}
        ]

        STRICTLY FOLLOW THE JSON RESPONSE FORMAT.
        RESPONSE MUST START WITH "[{" AND END WITH "}]".
        DO NOT START WITH "```json" OR ANY MARKDOWN.
"""
    return (
        f"The question you need to answer is: {raw_question}\n"
        f"The final answer to this question is: {raw_answer}\n"
        f"Available tools:\n{tool_catalog}\n{output_format}"
    )


def get_cot_system_prompt() -> str:
    return """You're an autonomous driving inference optimization expert.
Your task is to decompose short CoT data into finer-grained atomic steps and reorganize them for optimal reasoning path.
Break down reasoning into minimal units, dynamically cluster atomic steps, extract and label key steps, add tool invocations,
and form the optimal reasoning path for the current problem."""


def get_cot_user_prompt(
    raw_question: str,
    reason_part: str,
    final_answer: str,
    tool_choose_list: List[Dict],
) -> str:
    """Prompt for the second stage (structured chain generation)."""
    function_list = list_function_infos()
    selected = {
        tool.get("tool_name") for tool in tool_choose_list if isinstance(tool, dict)
    }
    selected_infos = [info for info in function_list if info["name"] in selected]
    tool_use_prompt = format_function_prompt(selected_infos, title="Selected tools")

    output_format = """
        Also generate:
        - Sub-question for each action/reasoning step (perception, prediction, planning)
        - Guess answer for each Sub if you are confident
        - Keywords (2-5 synonyms/alternatives) related to the Guess Answer
        - Missing_flag: "False" if you can answer, "True" otherwise
        - next_action: "continue reasoning" or "conclude"
        - Continue until reasoning chain is complete
        - Final answer and its keywords

        Output format example:
        {
            "Question": "",
            "Chain": [
                {
                    "Tool": {"function_name": "tool1", "parameters": ["param1"]},
                    "Sub": "Sub-question 1",
                    "Guess_Answer": "Answer 1",
                    "key_words": ["word1", "word2"],
                    "Missing_flag": "False",
                    "next_action": "continue reasoning"
                }
            ],
            "final_answer_keywords": ["keyword1", "keyword2"],
            "final_answer": "Final answer"
        }

        STRICTLY FOLLOW THE JSON RESPONSE FORMAT.
        RESPONSE MUST START WITH "{".
        DO NOT START WITH "```json" OR ANY MARKDOWN.
"""
    return (
        f"Raw question: {raw_question}\n"
        f"Existing CoT data: {reason_part}\n"
        f"Final answer: {final_answer}\n\n"
        "Please reconstruct the raw CoT data into atomic CoT data:\n"
        "For each atomic step, choose appropriate tools and parameters.\n"
        f"{tool_use_prompt}\n"
        f"{output_format}"
    )


# ---------------------------------------------------------------------------
# Reference-answer parsing (train split only)
# ---------------------------------------------------------------------------

START_MARKER = "**Step-by-Step Reasoning**:"
STOP_MARKERS = (
    "The final answer is:",
    "**Final Answer:**",
    "**Final Answer**:",
    "Final Answer",
    "Answer",
    "Why take this action?:",
    "**Final Answer**",
    "**Final Decision**:",
    "Final Step:",
    "<CONCLUSION>",
)


def extract_key_steps(text: str) -> str:
    """Extract the reasoning part of a DriveLMM-o1 reference answer."""
    if START_MARKER not in text:
        return ""
    relevant = text.split(START_MARKER, 1)[1].strip()
    for marker in STOP_MARKERS:
        if marker in relevant:
            relevant = relevant.split(marker)[0].strip()
    return relevant


def extract_final_answer(text: str) -> str:
    """Extract the final answer after the last known marker."""
    for marker in STOP_MARKERS:
        if marker in text:
            return text.split(marker)[-1].strip()
    return ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_sample_prompts(sample: dict, split: str) -> Tuple[str, str, str]:
    """Return ``(question, reasoning, final_answer)`` for one sample."""
    question = sample["question"]
    if split == "train":
        reference = sample.get("answer", "")
        return question, extract_key_steps(reference), extract_final_answer(reference)
    return question, sample.get("steps", ""), sample.get("final_answer", "")


def process_sample(
    client,
    sample: dict,
    split: str,
    model_name: str,
    temperature: float,
) -> dict:
    question, reasoning, final_answer = build_sample_prompts(sample, split)

    # Stage 1: tool planning.
    tool_choose_list = ask_for_json(
        client,
        get_system_prompt(),
        get_user_prompt(question, final_answer),
        model_name=model_name,
        temperature=temperature,
    )
    if not isinstance(tool_choose_list, list):
        raise ValueError("Tool planning did not return a JSON list")

    # Stage 2: structured chain generation.
    cot_data = ask_for_json(
        client,
        get_cot_system_prompt(),
        get_cot_user_prompt(question, reasoning, final_answer, tool_choose_list),
        model_name=model_name,
        temperature=temperature,
    )
    if not isinstance(cot_data, dict):
        raise ValueError("Chain generation did not return a JSON object")

    sample["cot_data"] = cot_data
    return sample


def load_progress(progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    with progress_path.open("r", encoding="utf-8") as handle:
        return int(json.load(handle).get("processed_index", -1)) + 1


def save_progress(progress_path: Path, index: int) -> None:
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump({"processed_index": index}, handle, indent=4)


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output or f"final_cot_{args.split}_{args.model_name}.json")
    progress_path = Path(args.progress_file or f"progress_{args.split}.json")

    with input_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    start_index = load_progress(progress_path)
    if start_index:
        LOGGER.info("Resuming from sample %d", start_index)

    client = get_client()
    if output_path.exists() and start_index:
        with output_path.open("r", encoding="utf-8") as handle:
            try:
                new_samples = json.load(handle)
            except json.JSONDecodeError:
                new_samples = []
    else:
        new_samples = []

    for index in tqdm(range(start_index, len(samples)), initial=start_index):
        sample = samples[index]
        try:
            new_samples.append(
                process_sample(
                    client, sample, args.split, args.model_name, args.temperature
                )
            )
            output_path.write_text(
                json.dumps(new_samples, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            save_progress(progress_path, index)
        except Exception as exc:
            LOGGER.error("Sample %d failed: %s", index, exc)
            if args.debug:
                raise

    output_path.write_text(
        json.dumps(new_samples, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Wrote %d samples to %s", len(new_samples), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tool-CoT training data")
    parser.add_argument("--input", default="data/DriveLMMo1_TRAIN.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"gen_{args.split}_data_{args.model_name}.log"),
            logging.StreamHandler(),
        ],
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
