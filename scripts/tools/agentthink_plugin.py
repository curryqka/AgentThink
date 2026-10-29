# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""ms-swift plugin entry point for AgentThink.

Registers both the GRPO reward (``agentthink_reward``) and the multi-turn tool
scheduler (``agentthink_tool_scheduler``) with one ``--custom_register_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import multi_turn  # noqa: E402,F401
from scripts.tools import rewards  # noqa: E402,F401
