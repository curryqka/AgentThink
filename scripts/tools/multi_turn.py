# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Multi-turn GRPO rollout for AgentThink.

During GRPO the model can emit one or more ``Tool: name([...])`` calls per turn.
This scheduler executes them against the Agent-Driver perception pickle, appends
the observations to the assistant context (masked from the loss), and lets the
model continue reasoning in the next turn.  It is registered as
``agentthink_tool_scheduler`` for ms-swift's ``--multi_turn_scheduler``.

The dataset must carry the execution metadata produced by
``scripts/tools/build_sft_data.py --target grpo``:
``frame_token``, ``pickle_split``, ``tool_data_path`` and ``camera_paths``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swift.plugin import MultiTurnScheduler, multi_turns  # noqa: E402

from scripts.tools.tool_chain import extract_tool_calls  # noqa: E402
from scripts.tools.tool_runtime import ToolRuntime  # noqa: E402


class AgentThinkToolScheduler(MultiTurnScheduler):
    """Execute Tool-CoT calls and feed observations back to the model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.runtime = ToolRuntime()

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        completion = response_choice.message.content or ""
        calls = extract_tool_calls(completion)
        if os.getenv("AGENTTHINK_DEBUG_ROLLOUT") == "1":
            print(
                f"[AgentThinkToolScheduler] turn={current_turn} "
                f"calls={[call['function_name'] for call in calls]} "
                f"completion={completion[:200]!r}",
                flush=True,
            )
        if "**Final Answer**" in completion:
            return True
        if not calls:
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn: int) -> Dict[str, Any]:
        completion = response_choice.message.content or ""
        calls = extract_tool_calls(completion)
        data = infer_request.data_dict or {}

        observations = self.runtime.execute_calls(data, calls)
        observation_text = "\n".join(
            f"Tool Output: {observation}" for observation in observations
        )

        result: Dict[str, Any] = {
            "infer_request": infer_request,
            "rollout_infos": {
                "tool_calls": calls,
                "tool_results": observations,
                "num_turns": current_turn,
            },
        }
        if not observation_text:
            return result

        messages = infer_request.messages
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = (
                (messages[-1].get("content") or "") + "\n" + observation_text
            )
        else:
            messages.append({"role": "assistant", "content": observation_text})

        # Mask tool observations out of the policy loss while keeping them in
        # the context for the next turn.
        tokenizer = self.tokenizer
        if tokenizer is None:
            return result
        base_ids = response_choice.token_ids
        if base_ids is None:
            base_ids = tokenizer.encode(completion, add_special_tokens=False)
        base_ids = [int(token_id) for token_id in base_ids]
        observation_ids = tokenizer.encode(
            "\n" + observation_text, add_special_tokens=False
        )
        result["response_token_ids"] = base_ids + observation_ids
        result["response_loss_mask"] = [1] * len(base_ids) + [0] * len(observation_ids)
        return result


multi_turns["agentthink_tool_scheduler"] = AgentThinkToolScheduler
