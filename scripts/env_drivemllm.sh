#!/usr/bin/env bash
# Extra dependencies used by the DriveMLLM / Qwen2.5-VL demo path.
#
# Required environment:
#   QWEN2_5_VL_DIR  checkout of https://github.com/QwenLM/Qwen2.5-VL
#
# This script only installs packages; it does not modify ~/.bashrc.
# Set PYTHONPATH to the repository root before running the scripts:
#   export PYTHONPATH="$PWD:$PYTHONPATH"
set -euo pipefail

: "${QWEN2_5_VL_DIR:?set QWEN2_5_VL_DIR to the Qwen2.5-VL checkout}"

pip install -r "${QWEN2_5_VL_DIR}/requirements_web_demo.txt"
pip install 'qwen-vl-utils[decord]'

# FlashAttention is optional but strongly recommended (used by Inference/).
pip install flash-attn --no-build-isolation --no-cache-dir || \
  echo "flash-attn install failed; install a matching wheel manually." >&2
