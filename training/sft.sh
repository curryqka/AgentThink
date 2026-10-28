#!/usr/bin/env bash
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
#
# Two-phase SFT warm-up (paper Sec. 3.2.1, Appendix B.1).
# Hyperparameters below mirror Appendix B.1:
#   LoRA r=8, alpha=32, 20 epochs, lr 1e-4, bs 2 x grad-accum 16,
#   bf16, ViT frozen, max_length 4096, ZeRO-2, warmup 0.05.
#
# Usage:
#   bash training/sft.sh phase1 [DATA_JSONL] [OUTPUT_DIR]
#   bash training/sft.sh phase2 [DATA_JSONL] [OUTPUT_DIR]
set -euo pipefail

PHASE="${1:-phase1}"
DATA="${2:-data/agentthink_sft_phase1.jsonl}"
OUTPUT="${3:-outputs/agentthink_sft_${PHASE}}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
NPROC="${NPROC_PER_NODE:-8}"

if [[ "${PHASE}" == "phase2" && "${DATA}" == "data/agentthink_sft_phase1.jsonl" ]]; then
  DATA="data/agentthink_sft_phase2.jsonl"
fi

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${VAL_DATASET:-}" ]]; then
  EXTRA_ARGS+=(--val_dataset "${VAL_DATASET}" --eval_steps 1000 --per_device_eval_batch_size 2)
fi

swift sft \
  --model "${MODEL}" \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --dataset "${DATA}" \
  "${EXTRA_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --freeze_vit true \
  --max_length 4096 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --logging_steps 5 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"reentrancy": false}' \
  --deepspeed zero2 \
  --output_dir "${OUTPUT}" \
  --nproc_per_node "${NPROC}"
