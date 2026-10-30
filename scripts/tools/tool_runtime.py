# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Shared tool-execution runtime for GRPO rollout and live inference.

Both the multi-turn GRPO scheduler and ``evaluation/inference_agentthink_live.py``
need the same behaviour: load the Agent-Driver pickle for a frame, execute a
Tool-CoT call through :class:`AgentThink`, and return the natural-language
observation.  Keeping it here avoids two diverging implementations.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List

from scripts.tools.gen_tool_result_data import AgentThink


class ToolRuntime:
    """Execute tool calls against Agent-Driver perception pickles."""

    def __init__(self, max_cached_agents: int | None = None) -> None:
        self._agents: "OrderedDict[tuple, AgentThink]" = OrderedDict()
        self.max_cached_agents = (
            max_cached_agents
            if max_cached_agents is not None
            else int(os.getenv("AGENTTHINK_AGENT_CACHE", "64"))
        )

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [value]
        return [str(item) for item in value] if isinstance(value, (list, tuple)) else []

    def get_agent(self, data: Dict[str, Any]):
        """Return ``(agent, error)`` for the frame described by ``data``."""
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

    def execute(self, data: Dict[str, Any], call: Dict[str, Any]) -> str:
        """Execute one ``{"function_name", "parameters"}`` call."""
        agent, error = self.get_agent(data)
        if agent is None:
            return f"Tool environment unavailable: {error}"
        response = agent.tool_call({"Tool": call})
        if response is None:
            return f"Tool {call['function_name']} failed or is unavailable."
        return response.get("prompt") or (
            f"Tool {call['function_name']} returned no observation."
        )

    def execute_calls(
        self, data: Dict[str, Any], calls: List[Dict[str, Any]]
    ) -> List[str]:
        return [self.execute(data, call) for call in calls]
