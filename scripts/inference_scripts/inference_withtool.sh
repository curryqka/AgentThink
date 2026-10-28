#!/usr/bin/env bash
# Structured Tool-CoT generation (planner output, no tool execution).
# Usage: bash scripts/inference_scripts/inference_withtool.sh <checkpoint> <output-name> [extra args]
set -euo pipefail

CKPT="${1:?usage: inference_withtool.sh <checkpoint> <output-name> [extra args]}"
OUTPUT="${2:?usage: inference_withtool.sh <checkpoint> <output-name> [extra args]}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

torchrun --nnodes=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port="${MASTER_PORT:-63668}" \
         --nproc_per_node="${NPROC_PER_NODE:-1}" \
         evaluation/inference_withtool.py \
         --checkpoint "${CKPT}" \
         --output-name "${OUTPUT}" \
         "$@"
