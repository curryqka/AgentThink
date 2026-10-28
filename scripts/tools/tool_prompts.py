# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
# Originally written by Jiageng Mao (Agent-Driver).
"""Ego-state and system prompts for the tool-augmented pipeline."""

from __future__ import annotations


def extract_ego_inputs(data_dict):
    """Render the ego-vehicle state and return it together with the raw fields.

    Ego-state layout (Agent-Driver):
        ego_states: [vx, vy, ?, ?, v_yaw (rad/s), length, width, v0, Kappa]
    """
    ego_prompts = "*" * 5 + "Ego States:" + "*" * 5 + "\n"

    vx = data_dict["ego_states"][0] * 0.5
    vy = data_dict["ego_states"][1] * 0.5
    v_yaw = data_dict["ego_states"][4]
    ax = data_dict["ego_hist_traj_diff"][-1, 0] - data_dict["ego_hist_traj_diff"][-2, 0]
    ay = data_dict["ego_hist_traj_diff"][-1, 1] - data_dict["ego_hist_traj_diff"][-2, 1]
    cx = data_dict["ego_states"][2]
    cy = data_dict["ego_states"][3]
    vhead = data_dict["ego_states"][7] * 0.5
    steering = data_dict["ego_states"][8]

    ego_prompts += "Current State:\n"
    ego_prompts += f" - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n"
    ego_prompts += f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n"
    ego_prompts += f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n"
    ego_prompts += f" - Can Bus: ({cx:.2f},{cy:.2f})\n"
    ego_prompts += f" - Heading Speed: ({vhead:.2f})\n"
    ego_prompts += f" - Steering: ({steering:.2f})\n"

    # Historical trajectory: ego_hist_traj is [5, 2] (last 2 seconds).
    xh1, yh1 = data_dict["ego_hist_traj"][0]
    xh2, yh2 = data_dict["ego_hist_traj"][1]
    xh3, yh3 = data_dict["ego_hist_traj"][2]
    xh4, yh4 = data_dict["ego_hist_traj"][3]
    ego_prompts += "Historical Trajectory (last 2 seconds):"
    ego_prompts += (
        f" [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), "
        f"({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n"
    )

    right, left, forward = data_dict["goal"]
    if right > 0:
        mission_goal = "RIGHT"
    elif left > 0:
        mission_goal = "LEFT"
    else:
        assert forward > 0, "goal vector has no positive component"
        mission_goal = "FORWARD"
    ego_prompts += f"Mission Goal: {mission_goal}\n"

    keys_to_extract = ["token", "ego_states", "ego_hist_traj_diff", "ego_hist_traj", "goal"]
    ego_data = {key: data_dict[key] for key in keys_to_extract if key in data_dict}
    return ego_prompts, ego_data


def get_ego_prompts(data_dict):
    """Return only the ego-state prompt (used as initial context)."""
    ego_prompts, _ = extract_ego_inputs(data_dict)
    return ego_prompts


def get_system_prompt():
    return """
    **A Language Agent for Autonomous Driving**
    Role: You are the brain of an autonomous vehicle (a.k.a. ego-vehicle).
    """


def get_detection_prompt():
    return """
        The detected objects that you need to pay attention to.
    """
