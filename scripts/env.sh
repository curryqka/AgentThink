#!/usr/bin/env bash
# Install ms-swift (training + inference engine).
#
# AgentThink uses the ms-swift 3.x API (`swift.llm.PtEngine`, `swift.plugin.ORM`).
# 3.11.3 is the version validated with transformers==4.50.1, trl==0.18.2 and
# qwen-vl-utils==0.0.14:
#   * ms-swift 3.12.x requires trl>=0.20, which requires transformers>=4.53.2;
#   * ms-swift 3.5.x expects an older qwen-vl-utils (IMAGE_FACTOR attribute).
set -euo pipefail
pip install 'ms-swift==3.11.3'
