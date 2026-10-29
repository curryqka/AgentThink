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

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swift.plugin import MultiTurnScheduler, multi_turns  # noqa: E402

from scripts.tools.gen_tool_result_data import AgentThink  # noqa: E402
from scripts.tools.tool_chain import extract_tool_calls  # noqa: E402


class AgentThinkToolScheduler(MultiTurnScheduler):
    """Execute Tool-CoT calls and feed observations back to the model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Agent-Driver pickles are expensive to load; keep a small LRU cache.
        self._agents: "OrderedDict[tuple, AgentThink]" = OrderedDict()
        self.max_cached_agents = int(os.getenv("AGENTTHINK_AGENT_CACHE", "64"))

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [value]
        return [str(item) for item in value] if isinstance(value, (list, tuple)) else []

    def _get_agent(self, data: Dict[str, Any]):
        token = str(data.get("frame_token") or "")
        if not token:
            return None, "missing frame_token in the dataset row"
        split = str(data.get("pickle_split") or "train")
        data_path = str(data.get("tool_data_path") or "data/tool_results")
        key = (token, split, data_path)
        if key not in self._agents:
            camera_paths = self._as_list(data.get("camera_paths"))
            try:
                self._agents[key] = AgentThink(
                    token=token,
                    split=split,
                    data_path=data_path,
                    json_data_dict={"image": camera_paths} if camera_paths else None,
                )
            except Exception as exc:  # missing/corrupt pickle
                return None, f"cannot load {data_path}/{split}/{token}.pkl: {exc}"
            while len(self._agents) > max(1, self.max_cached_agents):
                self._agents.popitem(last=False)
        else:
            self._agents.move_to_end(key)
        return self._agents[key], None

    def _execute(self, data: Dict[str, Any], call: Dict[str, Any]) -> str:
        agent, error = self._get_agent(data)
        if agent is None:
            return f"Tool environment unavailable: {error}"
        response = agent.tool_call({"Tool": call})
        if response is None:
            return f"Tool {call['function_name']} failed or is unavailable."
        return response.get("prompt") or (
            f"Tool {call['function_name']} returned no observation."
        )

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

        observations = [self._execute(data, call) for call in calls]
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
