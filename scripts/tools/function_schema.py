# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Tool-library catalog and prompt formatting.

The paper organises the tool library into five driving-centric modules plus
single-view vision utilities (Appendix A).  Both the data-generation pipeline
and ``evaluation/inference_withtool.py`` need to show this catalog to the model,
so the formatting lives here instead of inside the generation script.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

#: Attribute names on :class:`scripts.tools.tool_libraries.FuncAgent`.
FUNCTION_GROUPS: Dict[str, str] = {
    "visual": "visual_func_infos",
    "detection": "detection_func_infos",
    "prediction": "prediction_func_infos",
    "occupancy": "occupancy_func_infos",
    "map": "map_func_infos",
}

GROUP_TITLES: Dict[str, str] = {
    "visual": "Visual info",
    "detection": "Detection",
    "prediction": "Prediction",
    "occupancy": "Occupancy",
    "map": "Map",
}


def default_func_agent():
    """Instantiate the tool library lazily (keeps imports light)."""
    from scripts.tools.tool_libraries import FuncAgent

    return FuncAgent()


def list_function_infos(
    func_agent=None,
    groups: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Return the JSON-schema style metadata of every available tool."""
    agent = func_agent or default_func_agent()
    infos: List[dict] = []
    for group in groups or FUNCTION_GROUPS:
        infos.extend(getattr(agent, FUNCTION_GROUPS[group], []))
    return infos


def format_function_prompt(
    function_infos: Iterable[dict],
    *,
    title: str = "Available functions",
) -> str:
    """Render tool metadata as the compact list used in prompts."""
    lines = [f"{title}:"]
    for info in function_infos:
        parameters = info.get("parameters", {})
        required = parameters.get("required") or []
        signature = ", ".join(required)
        lines.append(f"- {info['name']}({signature}) # {info['description']}")
    return "\n".join(lines) + "\n"


def format_selected_functions(
    function_infos: Iterable[dict],
    tool_names: Iterable[str],
    *,
    title: str = "Selected tools",
) -> str:
    """Render only the tools selected by the first-stage planner."""
    by_name = {info["name"]: info for info in function_infos}
    infos = [by_name[name] for name in tool_names if name in by_name]
    return format_function_prompt(infos, title=title)


def build_tool_catalog(func_agent=None) -> str:
    """Return the full catalog grouped by module (Appendix A layout)."""
    agent = func_agent or default_func_agent()
    sections = []
    for group, attribute in FUNCTION_GROUPS.items():
        infos = getattr(agent, attribute, [])
        if infos:
            sections.append(
                format_function_prompt(infos, title=f"{GROUP_TITLES[group]} tools")
            )
    return "\n".join(sections)
