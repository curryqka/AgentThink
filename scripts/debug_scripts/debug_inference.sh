#!/usr/bin/env bash
# Debug helper: baseline inference on a small subset.
# Usage: bash scripts/debug_scripts/debug_inference.sh <checkpoint> <output-name> [limit]
set -euo pipefail

CKPT="${1:?usage: debug_inference.sh <checkpoint> <output-name> [limit]}"
OUTPUT="${2:?usage: debug_inference.sh <checkpoint> <output-name> [limit]}"
LIMIT="${3:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

torchrun --nnodes=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port="${MASTER_PORT:-63668}" \
         --nproc_per_node="${NPROC_PER_NODE:-1}" \
         evaluation/inference.py \
         --checkpoint "${CKPT}" \
         --output-name "${OUTPUT}" \
         --out-dir results/debug \
         --limit "${LIMIT}"
