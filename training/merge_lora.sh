#!/usr/bin/env bash
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
#
# Merge a LoRA adapter into the base model so that it can be loaded by
# evaluation/inference*.py (which expect a merged checkpoint).
#
# Usage:
#   bash training/merge_lora.sh OUTPUTS/agentthink_sft_phase2/v2-xxx/checkpoint-1000 \
#       pretrained_model/AgentThink
set -euo pipefail

ADAPTER="${1:?usage: merge_lora.sh <adapter_dir> <output_dir>}"
OUTPUT="${2:?usage: merge_lora.sh <adapter_dir> <output_dir>}"

swift export \
  --adapters "${ADAPTER}" \
  --merge_lora true \
  --output_dir "${OUTPUT}"
