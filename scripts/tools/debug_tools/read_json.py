#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Pretty-print one record of a JSON / JSONL file.

Example:
    python scripts/tools/debug_tools/read_json.py \
        --path data/DriveLMMo1_TEST_tool_results.jsonl --index 0
"""

from __future__ import annotations

import argparse
import json


def load_record(path: str, index: int) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            for position, line in enumerate(handle):
                if position == index:
                    return json.loads(line)
            raise IndexError(f"{path} has fewer than {index + 1} records")
        data = json.load(handle)
    if not isinstance(data, list):
        return data
    return data[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a JSON/JSONL record")
    parser.add_argument("--path", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(load_record(args.path, args.index), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
