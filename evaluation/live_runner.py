# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Live tool-inference loop.

Unlike ``inference_agentthink.py`` (which reads precomputed ``tool_result``
observations), this runner lets the model call tools during generation:

    model turn -> Tool: name([...]) -> execute -> append observation -> next turn

The loop shares ToolRuntime with the GRPO multi-turn scheduler, so training and
inference execute tools through exactly the same code path.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from evaluation import runner
from scripts.tools.tool_chain import extract_tool_calls
from scripts.tools.tool_runtime import ToolRuntime


def evaluate_live(
    engine: Any,
    dataset: runner.DriveLMMo1Dataset,
    args: Any,
    *,
    system_message: Optional[str] = None,
) -> List[dict]:
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=runner.InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(runner.collate_fn, tokenizer=None),
    )

    from swift.llm import InferRequest, RequestConfig

    runtime = ToolRuntime()
    outputs: List[dict] = []
    for images, questions, answers, _, data_ids, metadata in tqdm(
        dataloader, desc="Live rollout", disable=torch.distributed.get_rank() != 0
    ):
        image = images[0]
        question = questions[0]
        data_id = data_ids[0]
        data = dict(metadata[0])
        data["pickle_split"] = args.pickle_split
        data["tool_data_path"] = args.tool_data_path

        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        ego_state = str(data.get("system_prompts") or "").strip()
        if ego_state:
            question = f"{ego_state}\n{question}"
        messages.append({"role": "user", "content": f"<image>{question}"})

        assistant_turns: List[str] = []
        tool_trace: List[dict] = []
        for turn in range(1, args.max_turns + 1):
            request = InferRequest(messages=messages, images=[image])
            response = engine.infer(
                [request],
                RequestConfig(
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                ),
            )[0]
            completion = response.choices[0].message.content or ""
            assistant_turns.append(completion)
            # Mirror the GRPO scheduler: continuation turns extend the same
            # assistant message instead of creating consecutive assistant turns.
            if messages[-1]["role"] == "assistant":
                messages[-1]["content"] += completion
            else:
                messages.append({"role": "assistant", "content": completion})

            calls = extract_tool_calls(completion)
            if "**Final Answer**" in completion or not calls:
                break

            observations = runtime.execute_calls(data, calls)
            observation_text = "\n".join(
                f"Tool Output: {observation}" for observation in observations
            )
            # Keep the same assistant-continuation convention as the GRPO
            # scheduler: the observation becomes part of the assistant trace.
            messages[-1]["content"] += "\n" + observation_text
            tool_trace.append(
                {"turn": turn, "calls": calls, "observations": observations}
            )

        outputs.append(
            {
                "question": question,
                "answer": "\n".join(assistant_turns),
                "gt_answers": answers[0],
                "id": data_id,
                "tool_trace": tool_trace,
                "num_turns": len(assistant_turns),
                "transcript": messages,
            }
        )
    return outputs
