#!/usr/bin/env bash
# Extra dependencies used by the Qwen2.5-VL demo path.
#
# The demo loads Qwen2.5-VL through transformers, so no Qwen repository checkout
# is required. Set PYTHONPATH to the repository root before running the scripts:
#   export PYTHONPATH="$PWD:$PYTHONPATH"
set -euo pipefail

pip install 'qwen-vl-utils[decord]'

# FlashAttention is optional but strongly recommended (used by Inference/).
pip install flash-attn --no-build-isolation --no-cache-dir || \
  echo "flash-attn install failed; install a matching wheel manually." >&2
