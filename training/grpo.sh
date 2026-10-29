#!/usr/bin/env bash
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
#
# GRPO reinforcement fine-tuning with the three structured rewards of paper
# Table 1 (final answer / step reasoning / tool use).  Hyperparameters follow
# Appendix B.2: lr 1e-5, 8 epochs, bs 1 x grad-accum 8, 2 generations at
# temperature 1.2, max_length 2048 / completion 1024, beta 0.001, ZeRO-3.
#
# Usage:
#   ADAPTERS=/path/to/sft_adapter bash training/grpo.sh [MODEL] [DATA_JSONL] [OUTPUT_DIR]
#
# Start from the phase-2 SFT adapter, e.g.
#   ADAPTERS=outputs/agentthink_sft_phase2/v2-xxxx/checkpoint-xxxx \
#     bash training/grpo.sh Qwen/Qwen2.5-VL-7B-Instruct
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA="${2:-data/agentthink_grpo.jsonl}"
OUTPUT="${3:-outputs/agentthink_grpo}"
NPROC="${NPROC_PER_NODE:-8}"
ADAPTERS="${ADAPTERS:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
USE_VLLM="${USE_VLLM:-false}"
DEEPSPEED="${DEEPSPEED-zero3}"   # set DEEPSPEED="" to disable DeepSpeed
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MULTI_TURN="${MULTI_TURN:-true}"
MAX_TURNS="${MAX_TURNS:-6}"

if [[ ! -f "${DATA}" ]]; then
  echo "Dataset not found: ${DATA}" >&2
  echo "Build it with scripts/tools/build_sft_data.py --target grpo." >&2
  exit 1
fi

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
# ms-swift reads the world size from NPROC_PER_NODE; it is not a CLI flag.
export NPROC_PER_NODE="${NPROC}"

EXTRA_ARGS=()
if [[ -n "${ADAPTERS}" ]]; then
  EXTRA_ARGS+=(--adapters "${ADAPTERS}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${VAL_DATASET:-}" ]]; then
  EXTRA_ARGS+=(--val_dataset "${VAL_DATASET}" --eval_steps 500)
fi
if [[ -n "${DEEPSPEED}" ]]; then
  EXTRA_ARGS+=(--deepspeed "${DEEPSPEED}")
fi
if [[ "${MULTI_TURN}" == "true" ]]; then
  EXTRA_ARGS+=(--multi_turn_scheduler agentthink_tool_scheduler --max_turns "${MAX_TURNS}")
  if [[ "${USE_VLLM}" == "true" ]]; then
    EXTRA_ARGS+=(--vllm_server_pass_dataset true)
  fi
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --dataset "${DATA}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --reward_funcs agentthink_reward \
  --custom_register_path scripts/tools/agentthink_plugin.py \
  --system "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer." \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --num_train_epochs 8 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate 1e-5 \
  --num_generations 2 \
  --temperature 1.2 \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --use_vllm "${USE_VLLM}" \
  --beta 0.001 \
  --eval_steps 500 \
  --save_steps 100 \
  --save_total_limit 20 \
  --logging_steps 1 \
  --warmup_ratio 0.01 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --output_dir "${OUTPUT}"
