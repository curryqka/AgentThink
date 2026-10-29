#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Inspect one Agent-Driver perception pickle.

Example:
    python scripts/tools/debug_tools/read_pkl.py \
        --path data/tool_results/val/<frame_token>.pkl
"""

from __future__ import annotations

import argparse
import pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the keys of a tool-result pickle")
    parser.add_argument("--path", required=True)
    parser.add_argument("--full", action="store_true", help="Print the whole dictionary")
    args = parser.parse_args()

    with open(args.path, "rb") as handle:
        data = pickle.load(handle)

    if args.full:
        print(data)
    else:
        print(sorted(data.keys()))
        for key, value in data.items():
            shape = getattr(value, "shape", None)
            print(f"- {key}: {type(value).__name__}" + (f" shape={shape}" if shape else ""))


if __name__ == "__main__":
    main()
