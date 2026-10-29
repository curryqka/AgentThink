#!/usr/bin/env bash
# Install ms-swift (training + inference engine).
#
# AgentThink uses the ms-swift 3.x API (`swift.llm.PtEngine`, `swift.plugin.ORM`).
# ms-swift 4.0 restructured the package and removed `swift.llm`, so the
# inference scripts in this repository require 3.x.
set -euo pipefail
pip install 'ms-swift[all]>=3.5,<4.0' -U
