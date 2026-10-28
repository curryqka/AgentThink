#!/usr/bin/env bash
# Multi-GPU / multi-node plain-CoT inference.
#
# On a single node set NPROC_PER_NODE.  On a cluster that exposes the MLP_*
# variables the script uses them automatically.
#
# Usage: bash scripts/inference_scripts/inference_multigpu.sh <checkpoint> <output-name> [extra args]
set -euo pipefail

CKPT="${1:?usage: inference_multigpu.sh <checkpoint> <output-name> [extra args]}"
OUTPUT="${2:?usage: inference_multigpu.sh <checkpoint> <output-name> [extra args]}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${MLP_WORKER_NUM:-}" ]]; then
  LAUNCH_ARGS=(
    --nnodes "${MLP_WORKER_NUM}"
    --node_rank "${MLP_ROLE_INDEX}"
    --master_addr "${MLP_WORKER_0_HOST}"
    --master_port "${MLP_WORKER_0_PORT}"
    --nproc_per_node "${MLP_WORKER_GPU}"
  )
else
  LAUNCH_ARGS=(
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port="${MASTER_PORT:-63668}"
    --nproc_per_node="${NPROC_PER_NODE:-8}"
  )
fi

torchrun "${LAUNCH_ARGS[@]}" \
         evaluation/inference.py \
         --checkpoint "${CKPT}" \
         --output-name "${OUTPUT}" \
         "$@"
