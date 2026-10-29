#!/usr/bin/env bash
# Convert the validation split of the tool-result JSON into JSONL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/tools/prepare_swift_data.py \
  --input ./data/tool_results/cot_val_Qwen2.5-VL.json \
  --output ./data/DriveLMMo1_swift_VAL.jsonl \
  --split-type val \
  --image-dir ./data/image2concat
