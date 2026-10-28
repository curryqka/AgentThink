<div align="center">

<h1><b>🚗 AgentThink：面向自动驾驶的工具增强链式推理统一框架</b></h1>

<img src="assets/AgentThink2.png" alt="AgentThink Logo" width="360"/>

**中文** ｜ [English](README.md)

<p>
  <a href="https://curryqka.github.io/AgentThink.github.io/">🌐 项目主页</a> •
  <a href="https://arxiv.org/pdf/2505.15298">📄 论文（arXiv）</a> •
  <a href="https://aclanthology.org/2025.findings-emnlp.564/">📚 EMNLP 2025</a> •
  <a href="https://github.com/curryqka/AgentThink">🔖 代码</a> •
  <a href="LICENSE">🪪 Apache-2.0</a>
</p>
<p>
  <a href="https://drive.google.com/drive/folders/1RsgJlo3K7xRCUqphWvWECgyOcCvOa9Hx?usp=sharing">🪚 模型</a> |
  <a href="https://drive.google.com/drive/folders/1RsgJlo3K7xRCUqphWvWECgyOcCvOa9Hx?usp=sharing">🔢 数据</a> |
  <a href="https://github.com/curryqka/AgentThink/issues/2">🤗 Hugging Face 镜像（跟踪中）</a>
</p>

<p>
  邮箱：<a href="mailto:qka23@mails.tsinghua.edu.cn">qka23@mails.tsinghua.edu.cn</a> ·
  <a href="mailto:sicong.jiang@mail.mcgill.ca">sicong.jiang@mail.mcgill.ca</a> ·
  <a href="mailto:luoziang0122@163.com">luoziang0122@163.com</a> ·
  <a href="mailto:zhongyangtony@163.com">zhongyangtony@163.com</a>
</p>

</div>

在开发 AgentThink 的过程中，我们深受古代智慧的启发。正如荀子所言：

> 📜「君子生非异也，善假于物也。」

这句话意指：君子并非天资异禀，而是善于借助外物。AgentThink 的设计理念正是如此——
把多种驾驶工具与显式推理链结合起来，应对复杂的自动驾驶问答。

---

## 目录

- [亮点](#-亮点)
- [动态](#-动态)
- [框架](#-框架)
- [环境安装](#-环境安装)
- [模型与数据](#-模型与数据)
- [流程总览](#-流程总览)
- [1. Tool-CoT 数据生成](#1-tool-cot-数据生成)
- [2. 两阶段训练（SFT → GRPO）](#2-两阶段训练sft--grpo)
- [3. 推理](#3-推理)
- [4. 评测](#4-评测)
- [复现检查表](#-复现检查表)
- [实验结果](#-实验结果)
- [仓库结构](#-仓库结构)
- [常见问题](#-常见问题)
- [开源协议与引用](#-开源协议与引用)

## ✨ 亮点

- 🔧 **工具增强推理** —— 视觉、检测、预测、占据、地图五大驾驶工具模块，外加单视图视觉工具。
- 🧠 **显式 Tool-CoT** —— 每个推理步都带有 `Tool`、`Sub`、`Guess_Answer`、
  `key_words`、`Missing_flag`、`next_action`。
- 🎯 **两阶段训练** —— SFT 预热推理与工具调用，GRPO 用终答、步骤、工具三类奖励继续优化。
- 📊 **面向工具的评测** —— DriveLMM-o1 推理指标 + 三个 LLM-as-Judge 工具使用指标。
- 🚀 **效果** —— 在 DriveLMM-o1 上，整体推理较 Qwen2.5-VL-7B 提升 51.9%，MCQ 提升 33.5%。

<div align="center">
  <img src="assets/AgentThink.png" width="92%" alt="AgentThink 框架图"/>
</div>

## 📰 动态

- **2025.10** —— 仓库重构：共享推理入口、补齐训练代码、GRPO 奖励、工具指标、单元测试与
  完整复现指南（[变更摘要](docs/REFACTOR_SUMMARY.zh-CN.md)）。
- **2025.08.20** —— AgentThink 被 **EMNLP 2025 Findings** 接收。
- **2025.07.02** —— v1.1 发布，包含 demo 与样例数据。
- **2025.05.22** —— 论文上传 arXiv。

## 🧭 框架

对应论文第 3 节，AgentThink 由三部分组成：

| 模块 | 作用 | 代码 |
|---|---|---|
| **1. 数据生成** | GPT-4o 把参考 CoT 改写成结构化 Tool-CoT；在 Agent-Driver 感知结果上执行驾驶工具 | `scripts/tools/` |
| **2. 两阶段训练** | LoRA SFT（阶段一：动作；阶段二：动作 + 观测）后接 GRPO 三类奖励 | `training/` |
| **3. 推理与评测** | 推理时动态调用工具；DriveLMM-o1 Judge + 工具使用指标 | `evaluation/` |

## ⚙️ 环境安装

### 环境要求

| 组件 | 版本 |
|---|---|
| 操作系统 | Ubuntu 20.04（Linux） |
| Python | 3.10 |
| CUDA Toolkit | 12.4 |
| PyTorch | 2.6.0 |
| GPU | NVIDIA A800 80 GB（论文使用 16 卡） |

### 安装步骤

```bash
git clone https://github.com/curryqka/AgentThink.git
cd AgentThink

conda create -n agentthink python=3.10 -y
conda activate agentthink
pip install -r requirements.txt

# ms-swift（推理与训练引擎）
bash scripts/env.sh

# 可选：Qwen2.5-VL demo 依赖
export QWEN2_5_VL_DIR=/path/to/Qwen2.5-VL
bash scripts/env_drivemllm.sh

# 让仓库可被 import
export PYTHONPATH="$PWD:$PYTHONPATH"
```

第三方工具仓库需单独克隆：

```bash
cd third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
git clone https://github.com/AILab-CVC/YOLO-World.git
git clone https://github.com/modelscope/ms-swift.git ms-swift-main
```

把 `depth_anything_v2_vitb.pth` 与 `yolov8x-worldv2.pt` 放到 `pretrained_model/` 下。

## 📦 模型与数据

| 产物 | 位置 | 说明 |
|---|---|---|
| AgentThink 合并权重（Qwen2.5-VL-7B） | [Google Drive](https://drive.google.com/drive/folders/1RsgJlo3K7xRCUqphWvWECgyOcCvOa9Hx?usp=sharing) | 放到 `pretrained_model/AgentThink` |
| DriveLMM-o1 训练/测试 JSON | [Google Drive](https://drive.google.com/drive/folders/1RsgJlo3K7xRCUqphWvWECgyOcCvOa9Hx?usp=sharing) | `data/DriveLMMo1/` |
| 处理后测试 JSONL 与 Tool-CoT 数据 | [Google Drive](https://drive.google.com/drive/folders/1RsgJlo3K7xRCUqphWvWECgyOcCvOa9Hx?usp=sharing) | `data/` |
| Agent-Driver 工具 pickle | [Agent-Driver](https://github.com/USC-GVL/Agent-Driver) | `data/tool_results/{val,train}/` |
| Hugging Face 镜像 | [issue #2](https://github.com/curryqka/AgentThink/issues/2) 跟踪 | 欢迎社区协助 |

推荐的目录结构：

```text
AgentThink/
├── data/
│   ├── DriveLMMo1/
│   │   ├── DriveLMMo1_TRAIN.json
│   │   └── DriveLMMo1_TEST.json
│   ├── DriveLMMo1_TEST.jsonl                 # 纯测试集
│   ├── DriveLMMo1_TEST_tool_results.jsonl    # 带工具观测的测试集
│   ├── image2concat/                         # 3 x 2 六视图拼接图
│   │   └── <scene_token>_<frame_token>.png
│   ├── tool_results/
│   │   ├── val/<frame_token>.pkl
│   │   └── train/<frame_token>.pkl
│   └── agentthink_sft_phase1.jsonl           # build_sft_data.py 生成
├── pretrained_model/
│   ├── AgentThink/
│   ├── depth_anything_v2_vitb.pth
│   └── yolov8x-worldv2.pt
├── demo_image/
├── docs/
├── evaluation/
├── Inference/
├── scripts/
├── tests/
├── third_party/
└── training/
```

## 🚀 流程总览

```text
DriveLMM-o1 问答 + nuScenes 图像 + Agent-Driver pickle
        │
        ├─(1a) GPT-4o 生成 Tool-CoT ───────► final_cot_train_<model>.json
        ├─(1b) 六视图拼接 ────────────────► data/image2concat/*.png
        ├─(1c) 执行规划出的工具 ──────────► cot_val_Qwen2.5-VL.json
        ├─(1d) 转成 JSONL ────────────────► DriveLMMo1_TEST_tool_results.jsonl
        └─(1e) 构建 SFT/GRPO 数据 ────────► data/agentthink_*.jsonl
        │
        ├─(2) SFT 阶段一 → SFT 阶段二 → GRPO ──► 合并后的 checkpoint
        ├─(3) 推理（基线 / 规划器 / AgentThink）
        └─(4) LLM-as-Judge 评测
```

每个产物的生产者/消费者、复现检查表与 FAQ 见
**[docs/pipeline.md](docs/pipeline.md)**。

## 1. Tool-CoT 数据生成

```bash
export OPENAI_API_KEY=...            # 可选 OPENAI_BASE_URL=...

# (1a) 从参考答案生成结构化 Tool-CoT
python scripts/tools/agentthink_data_generater_pipeline.py \
    --input data/DriveLMMo1/DriveLMMo1_TRAIN.json \
    --split train --model-name gpt-4o-mini \
    --output final_cot_train_gpt-4o-mini.json

# (1b) 六个相机拼成一张图
python scripts/tools/data_image_converter.py \
    --json-file-path data/DriveLMMo1/DriveLMMo1_TEST.json \
    --image-root-dir /path/to/nuscenes \
    --savepath-img data/image2concat

# (1c) 在 Agent-Driver 感知结果上执行规划出的工具
python scripts/tools/gen_tool_result_data.py \
    --input final_cot_train_gpt-4o-mini.json \
    --output data/tool_results/cot_val_Qwen2.5-VL.json \
    --tool-data-path data/tool_results --split val

# (1d) 转成评测用 JSONL
python evaluation/prepare_data.py \
    --input data/tool_results/cot_val_Qwen2.5-VL.json \
    --output data/DriveLMMo1_TEST_tool_results.jsonl
```

> JSONL 里的 `tool_result` **不是模型输出**，而是第 (1c) 步执行驾驶工具得到的观测；
> `system_prompts` 是自车状态描述。详见
> [docs/pipeline.md](docs/pipeline.md#1c-execute-the-planned-tools)。

## 2. 两阶段训练（SFT → GRPO）

训练脚本按论文附录 B 固化（LoRA r=8、α=32；SFT 20 epoch、lr 1e-4；
GRPO 8 epoch、lr 1e-5、每题 2 次 rollout、β=0.001）。

```bash
export PYTHONPATH=$PWD

# 构建带工具的 SFT/GRPO 数据
python scripts/tools/build_sft_data.py \
    --input data/tool_results/cot_val_Qwen2.5-VL.json \
    --output data/agentthink_sft_phase1.jsonl \
    --format agentthink --target sft --no-observations
python scripts/tools/build_sft_data.py \
    --input data/tool_results/cot_val_Qwen2.5-VL.json \
    --output data/agentthink_sft_phase2.jsonl \
    --format agentthink --target sft
python scripts/tools/build_sft_data.py \
    --input data/tool_results/cot_val_Qwen2.5-VL.json \
    --output data/agentthink_grpo.jsonl \
    --format agentthink --target grpo

# 训练
NPROC_PER_NODE=8 bash training/sft.sh phase1 data/agentthink_sft_phase1.jsonl
NPROC_PER_NODE=8 bash training/sft.sh phase2 data/agentthink_sft_phase2.jsonl
NPROC_PER_NODE=8 bash training/grpo.sh <phase2_adapter> data/agentthink_grpo.jsonl

# 合并适配器供推理使用
bash training/merge_lora.sh <grpo_adapter> pretrained_model/AgentThink
```

1/2/4/8 卡硬件建议、奖励细节与排错见 **[training/README.md](training/README.md)**。

## 3. 推理

三种模式回答三个不同的问题：

| 命令 | 输入 | 用途 |
|---|---|---|
| `evaluation/inference.py` | `DriveLMMo1_TEST.jsonl` | 纯 CoT 基线 |
| `evaluation/inference_withtool.py` | `DriveLMMo1_TEST.jsonl` + 工具目录 | 结构化**工具规划**（不执行） |
| `evaluation/inference_agentthink.py` | `DriveLMMo1_TEST_tool_results.jsonl` | **AgentThink**：基于工具观测作答 |

```bash
CKPT=pretrained_model/AgentThink

bash scripts/inference_scripts/inference.sh            "$CKPT" baseline
bash scripts/inference_scripts/inference_withtool.sh   "$CKPT" toolcot
bash scripts/inference_scripts/inference_agentthink.sh "$CKPT" AgentThink
```

所有脚本都支持 `--limit N` 做冒烟测试，并可通过 `NPROC_PER_NODE` 或 `torchrun`
多卡运行；单卡也可以直接 `python evaluation/inference.py ...`。

交互式 Demo：

```bash
python Inference/inference_demo_drivemllm.py --model-path "$CKPT"
python Inference/inference_demo_drivelmm.py  --model-path "$CKPT" \
    --tool-data-path data/tool_results
```

## 4. 评测

```bash
export OPENAI_API_KEY=...
python evaluation/evaluation_script.py \
    --input results/agentthink/DriveLMMo1_AgentThink.json \
    --dataset data/DriveLMMo1/DriveLMMo1_TEST.json \
    --output results/agentthink_eval \
    --tool-metrics
```

输出 DriveLMM-o1 的 12 项分数、论文 Table 3 的 7 列、MCQ 准确率；加
`--tool-metrics` 后还会输出 Table 2 的三个工具使用指标。结果会写成 `.json`、
`.csv` 与 `.summary.json`。

## ✅ 复现检查表

1. 使用**合并后的 AgentThink checkpoint**，不是基座模型。
2. 使用**六视图拼接图**（`data/image2concat/<stem>.png`），不是单相机图。
3. AgentThink 推理必须使用 `DriveLMMo1_TEST_tool_results.jsonl`。
4. `--max-new-tokens` 保持 2000，避免终答被截断。
5. Judge 使用 `gpt-4o-mini`、temperature 0；分数通常有 ±1–2 的漂移。
6. 参考值：基座 Qwen2.5-VL-7B MCQ = 37.81，DriveLMM-o1 = 62.36，AgentThink = 71.35。

更完整的失败模式见
[docs/pipeline.md](docs/pipeline.md#5-reproduction-checklist-issue-6)。

## 📊 实验结果

### DriveLMM-o1

| 视觉语言模型 | Risk Assess. (%)↑ | Rule Adh. (%)↑ | Scene Aware. (%)↑ | Relevance (%)↑ | Missing (%)↑ | Reason. (%)↑ | MCQ (%)↑ |
|---|---|---|---|---|---|---|---|
| [GPT-4o](https://github.com/gpt4o-image/GPT-4o) | 71.32 | 80.72 | 72.96 | 76.65 | 71.43 | 72.52 | 57.84 |
| [Ovis1.5-Gemma2-9B](https://huggingface.co/AIDC-AI/Ovis1.5-Gemma2-9B) | 51.34 | 66.36 | 54.74 | 55.72 | 55.74 | 55.62 | 48.85 |
| [Mulberry-7B](https://github.com/HJYao00/Mulberry) | 51.89 | 63.66 | 56.68 | 57.27 | 57.45 | 57.65 | 52.86 |
| [LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT) | 57.62 | 69.01 | 60.84 | 62.72 | 60.67 | 61.41 | 49.27 |
| [LlamaV-o1](https://github.com/mbzuai-oryx/LlamaV-o1) | 60.20 | 73.52 | 62.67 | 64.66 | 63.41 | 63.13 | 50.02 |
| [InternVL2.5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) | 69.02 | 78.43 | 71.52 | 75.80 | 70.54 | 71.62 | 54.87 |
| [Qwen2.5-VL-7B](https://github.com/QwenLM/Qwen2.5-VL) | 46.44 | 60.45 | 51.02 | 50.15 | 52.19 | 51.77 | 37.81 |
| [DriveLMM-o1](https://github.com/ayesha-ishaq/DriveLMM-o1) | 73.01 | 81.56 | 75.39 | 79.42 | 74.49 | 75.24 | 62.36 |
| **AgentThink（本文）** | **80.51** | **84.98** | **82.11** | **84.99** | **79.56** | **79.68** | **71.35** |

### 工具使用指标（Table 5）

| 模型 | Tool Usage Appro. | Tool Chain Coh. | Percep-Guided Tool Align. | Overall Tool Score |
|---|---|---|---|---|
| Base + DirectTool | 59.61 | 73.29 | 69.71 | 67.54 |
| Base + SFT | 62.38 | 78.19 | 75.78 | 72.12 |
| Base + GRPO | 68.44 | 80.73 | 80.82 | 76.66 |
| **AgentThink（本文）** | **70.92** | **82.16** | **84.25** | **79.11** |

### DriveMLLM

| 类型 | 模型 | L/R | F/B | RHD | RD | PPos | BBox | CVD | CD | AccS | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | GPT-4o | 91.72 | 67.60 | 9.58 | 14.69 | 40.90 | 4.07 | 46.11 | 70.65 | 43.16 | 25.63 |
| | GPT-4o-mini | 67.67 | 50.13 | 70.44 | 0.00 | 29.28 | 3.78 | 0.00 | 46.40 | 33.46 | 16.68 |
| | LLaVA-ov-72B | 85.42 | 49.48 | 13.76 | 45.27 | 16.46 | 0.00 | 42.97 | 27.09 | 35.06 | 21.10 |
| | Qwen2.5-VL-7B | 76.55 | 55.24 | 7.14 | 17.11 | 55.97 | 38.31 | 55.94 | 51.52 | 44.72 | 13.36 |
| | Qwen + CoT | 87.06 | 63.09 | 16.69 | 22.56 | 52.51 | 38.87 | 76.90 | 38.71 | 49.55 | 19.31 |
| | Qwen + DirectTool | 78.95 | 48.96 | 58.43 | 67.57 | 58.20 | 42.22 | 51.76 | 51.38 | 57.18 | 24.05 |
| | **AgentThink（本文）** | 82.33 | 54.40 | 56.14 | 61.45 | 70.45 | 56.23 | 23.09 | 51.60 | 56.96 | 26.52 |
| One-shot | GPT-4o | 91.08 | 69.37 | 36.51 | 71.17 | 42.44 | 5.10 | 0.00 | 63.88 | 47.44 | 33.17 |
| | GPT-4o-mini | 66.00 | 48.95 | 83.02 | 58.47 | 25.71 | 3.97 | 52.73 | 55.23 | 49.26 | 22.13 |
| | LLaVA-ov-72B | 79.12 | 62.97 | 49.26 | 68.04 | 28.57 | 2.20 | 53.12 | 60.90 | 50.52 | 36.66 |
| | Qwen2.5-VL-7B | 80.30 | 53.14 | 36.96 | 39.13 | 62.69 | 22.63 | 49.88 | 48.32 | 49.13 | 33.53 |
| | Qwen + CoT | 86.35 | 59.95 | 43.29 | 31.81 | 53.64 | 26.93 | 51.02 | 42.30 | 49.41 | 32.06 |
| | Qwen + DirectTool | 84.57 | 55.50 | 67.32 | 59.54 | 85.58 | 26.07 | 52.34 | 53.25 | 60.52 | 42.27 |
| | **AgentThink（本文）** | 78.71 | 48.46 | 60.64 | 60.71 | 72.36 | 64.46 | 52.26 | 52.04 | 61.21 | 47.24 |

## 🎬 Demo 演示

<p align="center">
  <img src="assets/demo.gif" alt="AgentThink Demo" width="640"/>
</p>

<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
  <img src="assets/planning.png" alt="白天规划" style="width: 48%; max-width: 480px;" />
  <img src="assets/planning-night.png" alt="夜间规划" style="width: 48%; max-width: 480px;" />
</div>

<p align="center">
  <img src="assets/zero-visual.png" alt="零样本空间理解" style="max-width: 90%;"/>
</p>

## 📁 仓库结构

```text
AgentThink/
├── assets/                       # 图片与演示素材
├── demo_image/                   # demo 相机图像
├── docs/
│   ├── pipeline.md               # 产物链路 + 复现检查表
│   └── REFACTOR_SUMMARY.zh-CN.md # 本次整理与修复摘要
├── evaluation/
│   ├── prompts.py                # 所有模型可见 prompt
│   ├── runner.py                 # 共享数据集 / 分布式 / 生成逻辑
│   ├── inference.py              # 纯 CoT
│   ├── inference_withtool.py     # 结构化工具规划
│   ├── inference_agentthink.py   # 基于工具观测作答
│   ├── evaluation_script.py      # DriveLMM-o1 Judge + MCQ
│   ├── tool_metrics.py           # Table 2 工具指标
│   └── prepare_data.py           # 工具结果 JSON -> 评测 JSONL
├── Inference/
│   ├── qwen2_5vl.py              # 共享 Qwen2.5-VL 接口
│   └── inference_demo_*.py       # 交互 Demo
├── scripts/
│   ├── env.sh / env_drivemllm.sh
│   ├── inference_scripts/        # torchrun 启动脚本
│   ├── debug_scripts/
│   └── tools/
│       ├── tool_libraries.py     # 驾驶工具库
│       ├── tool_libraries_simple.py
│       ├── function_schema.py    # 工具目录渲染
│       ├── tool_chain.py         # Tool-CoT 解析与校验
│       ├── chain_render.py       # 训练文本渲染
│       ├── agentthink_data_generater_pipeline.py
│       ├── gen_tool_result_data.py
│       ├── build_sft_data.py
│       ├── prepare_swift_data.py
│       ├── data_image_converter.py
│       ├── rewards.py            # GRPO 奖励（Table 1）
│       ├── utils/
│       └── debug_tools/
├── tests/                        # 30 个无重依赖单测
├── training/
│   ├── README.md                 # 硬件 + 数据 + 命令
│   ├── sft.sh / grpo.sh
│   └── merge_lora.sh
├── third_party/                  # YOLO-World、Depth-Anything、ms-swift
├── LICENSE
├── requirements.txt
├── README.md
└── README_CN.md
```

## ❓ 常见问题

**只有 2 张或 4 张 A800 能跑吗？**  
推理可以，7B 合并模型单张 80 GB 足够。SFT 在 4 卡上可用默认配置，2 卡需要调小
`--max_length`。GRPO 受生成吞吐限制，4×80 GB 是实际下限（缩短 completion 长度），
论文使用 16×A800。详见 [training/README.md](training/README.md#2-hardware)。

**`inference_withtool.py` 为什么不直接回答？**  
它是工具**规划器**，输出的是交给 `gen_tool_result_data.py` 执行的结构化链；
真正作答的是 `inference_agentthink.py`。这正是论文中数据生成与推理两个阶段的区别。

**`tool_result` 从哪里来？**  
来自在 Agent-Driver 感知 pickle 上执行规划出的工具，不是 VLM 输出。详见
[docs/pipeline.md](docs/pipeline.md#1c-execute-the-planned-tools)。

**复现不出 MCQ 71.35 怎么办？**  
按 [docs/pipeline.md](docs/pipeline.md#5-reproduction-checklist-issue-6) 的六项检查表逐条排查。
最常见原因：用了基座模型、用了单相机图而非拼接图、`tool_result` 为空、生成被截断。

## 🔗 相关工作

| 名称 | 说明 | 链接 |
|---|---|---|
| Depth-Anything-V2 | 单目深度估计 | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| YOLO-World | 开放词表目标检测 | [GitHub](https://github.com/AILab-CVC/YOLO-World) |
| all-MiniLM-L6-v2 | 句子语义相似度 | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Agent-Driver | 驾驶工具结果与 pickle | [GitHub](https://github.com/USC-GVL/Agent-Driver) |
| DriveLMM-o1 | 基准与评测协议 | [GitHub](https://github.com/ayesha-ishaq/DriveLMM-o1) |

## 🪪 开源协议与引用

本项目基于 [Apache License 2.0](LICENSE) 开源。

```bibtex
@inproceedings{qian2025agentthink,
  title     = {{AgentThink}: A Unified Framework for Tool-Augmented Chain-of-Thought
               Reasoning in Vision-Language Models for Autonomous Driving},
  author    = {Qian, Kangan and Jiang, Sicong and Zhong, Yang and Luo, Ziang and
               Huang, Zilin and Zhu, Tianze and Jiang, Kun and Yang, Mengmeng and
               Fu, Zheng and Miao, Jinyu and others},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  pages     = {10663--10682},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-emnlp.564/}
}
```

早期版本见 [arXiv:2505.15298](https://arxiv.org/abs/2505.15298)。

> **作者：** Kangan Qian\*、Sicong Jiang\*、Yang Zhong\*、Ziang Luo\*、Zilin Huang、
> Tianze Zhu、Kun Jiang†、Mengmeng Yang、Zheng Fu、Jinyu Miao、Yining Shi、
> He Zhe Lim、Li Liu、Tianbao Zhou、Huang Yu、Yifei Hu、Guang Li、Guang Chen、
> Hao Ye†、Lijun Sun、Diange Yang†
>
> **单位：** ¹ 清华大学 · ² 麦吉尔大学 · ³ 小米公司 · ⁴ 威斯康星大学麦迪逊分校
>
> \* 同等贡献 · † 通讯作者
