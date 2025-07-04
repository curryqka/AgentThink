# AgentThink：用于自动驾驶的工具增强视觉语言推理框架


在开发AgentThink的过程中，我们深受古代智慧的启发。正如荀子所言：
> 📜君子生非异也，善假于物也。
>
> 这句话意指君子之所以能成为君子，并不是因为他们天生就与众不同，而是因为他们善于借助外物的力量来提升自己。这与我们的设计理念不谋而合——通过结合多种工具和模型，AgentThink能够更好地理解和应对复杂的自动驾驶场景。

---

**中文** ｜ [English](README.md)

<div align="center">

<img src="assets/AgentThink.png" alt="AgentThink Logo" width="560"/>

<p>
  <a href="https://agentthink.github.io">🌐 项目主页</a> •
  <a href="https://arxiv.org/pdf/2505.15298">📄 论文链接</a> •
  <a href="https://github.com/agentthink/agentthink/releases/tag/v1.1">🔖 最新版本 v1.1</a> •
  <a href="LICENSE">🪪 开源协议</a>
</p>

</div>

## 🎬 Demo演示

欢迎观看AgentThink的实际操作演示！以下我们将展示一段视频，介绍系统在自动驾驶场景下的表现。同时，我们也会提供一些关键功能的可视化结果截图，帮助您更好地理解AgentThink的工作流程和技术细节。

### 视频演示

请观看以下视频以了解AgentThink在复杂交通环境中的环境认知过程：

<!-- <p align="center">
  <a href="https://youtu.be/your_video_id" target="_blank">
    <img src="assets/thumbnail_agentthink_demo.png" alt="AgentThink Demo Video Thumbnail" width="640"/>
  </a>
</p> -->
<p align="center">
  <img src="assets/demo.gif" alt="Demo GIF" width="640"/>
</p>


> **提示**: 点击上方图片播放视频。如果您无法查看YouTube链接，请访问我们的[B站](#)频道获取更多观看选项。

### 可视化展示
<!-- 横向对比图 -->
<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
  <img src="assets/planning.png" alt="Daytime Planning" style="width: 48%; max-width: 480px; height: auto;" />
  <img src="assets/planning-night.png" alt="Nighttime Planning" style="width: 48%; max-width: 480px; height: auto;" />
</div>

<!-- 单张展示图 -->
<p style="text-align: center; margin: 25px 0;">
  <img src="assets/zero-visual.png" alt="AgentThink zero-shot learning" 
       style="max-width: 90%; height: auto; display: block; margin: 0 auto;" />
</p>

除了视频演示外，我们还准备了一系列图像来展示AgentThink处理不同驾驶情境的能力。这些图像是从实际测试中截取的，展示了模型对道路状况、障碍物检测及路径规划的理解。

| 场景 | 描述 | 图像 |
| --- | --- | --- |
| 障碍物识别 | 展现AgentThink在检测并分类前方障碍物时的表现 | ![障碍物识别](assets/demo_obstacle_detection.png) |
| 空间理解 | 解释AgentThink如何调用工具求解空间关系 | ![空间理解](assets/demo_path_planning.png) |
| 天气适应性 | 演示AgentThink在不同天气条件下（如雨天、雾天）的性能 | ![天气适应性](assets/demo_weather_adaptability.png) |

每个示例都详细说明了AgentThink是如何利用其独特的框架结构和算法来实现高效准确的决策制定。通过结合视觉信息与语言指令，AgentThink能够为未来的自动驾驶技术带来新的突破。

---

## 目录
- [✨ 项目亮点](#-项目亮点)
- [📰 项目动态](#-项目动态)
- [🚀 快速导航](#-快速导航)
- [⚙️ 开始使用](#️-开始使用)
- [🚀 快速上手](#-快速上手)
- [📋 TODO 列表](#-todo-列表)
- [📂 仓库结构](#-仓库结构)
- [📊 Benchmark结果](#-benchmark结果)
- [🖼️ 论文结果](#️-论文结果)
- [📜 许可证与引用](#-许可证与引用)

## ✨ 项目亮点

- 🔧 **工具增强推理**：结合视觉、预测、占用、地图等工具进行多模态感知与推理
- 🧠 **推理链+工具调用**：将任务拆解为细粒度推理步骤，显式调用工具辅助决策
- 🎯 **GRPO训练机制**：设计终答奖励、步骤奖励、工具使用奖励三重信号
- 🚀 **领先性能**：相比传统 VLM 模型，推理准确率提升超 53.91%

<div align="center">
  <img src="assets/poster.png" width="95%">
</div>

## 📰 项目动态

- 📄 [2025.05.22] AgentThink 论文发布于 arXiv
- 🚀 [2025.07.02] 发布 v1.1，支持 Demo 和样例数据
- 🎥 Web Demo 与 Swift 全流程训练即将上线

当然可以，以下是与你提供的英文版 `README` 内容**一一对应**的中文翻译版本，可直接命名为 `README_CN.md` 中的相关章节内容。

---


## 🚀 快速导航

| 模块                | 描述                                   | 跳转链接                                  |
|---------------------|----------------------------------------|-------------------------------------------|
| 环境配置            | 安装依赖并配置环境                     | [环境配置](#环境配置)                     |
| 启动训练            | 使用 Swift 进行 SFT/RLFT 训练          | [启动训练](#启动训练)                     |
| Demo 推理           | 在测试集上运行推理脚本                 | [Demo 推理](#demo-推理)                   |
| 评估与指标          | 使用 LLM-as-Judge 评估性能              | [评估与指标](#评估与指标)                 |
| Benchmark 结果      | 与现有方法的性能对比                   | [Benchmark 结果](#benchmark-结果)         |

---

## 🛠️ 环境配置

安装依赖并配置虚拟环境：

```bash
conda create -n agentthink python=3.10 -y
conda activate agentthink
pip install -r requirements.txt
```

使用 OpenAI 接口，请配置密钥：

```bash
export OPENAI_API_KEY="你的密钥"
export OPENAI_API_BASE="https://你的代理地址"
```

---

## 🏋️ 启动训练

单卡调试训练（设置模型路径、数据路径、输出路径）：

```bash
python train/sft_debug.py \
  --model "/path/to/your/model" \
  --dataset "/path/to/your/dataset" \
  --output_dir "/path/to/output/directory"
```

多卡（≥8 张 GPU）训练脚本如下：

```bash
# 阶段1：监督微调（SFT）
bash train/script/sft_drivelmm_8gpu.sh

# 阶段2：强化学习微调（RLFT）
bash train/script/rlft_drivelmm_8gpu.sh
```

---

## 🎬 Demo 推理

使用训练好的 checkpoint 进行测试集推理：

```bash
# 推理脚本
bash scripts/inference_swift.sh [你的CKPT路径] [你的输出目录]
```

---

## 📊 评估与指标

使用 LLM-as-Judge 自动评估模型表现：

```bash
# 步骤1：评估推理能力与选择题准确率
python evaluation/evaluation_reasoning_script.py

# 步骤2：评估工具使用情况
python evaluation/evaluation_tool_script.py
```

---

## 🏆 Benchmark 结果

查看 [Benchmark 结果](#benchmark-结果) 或论文海报，了解 AgentThink 在多个评测维度上的领先性能。


## ⚙️ 环境配置
### 基础环境
| 组件 | 版本 | 验证命令 |
|------|------|----------|
| 操作系统 | Ubuntu 20.04 | `cat /etc/issue` |
| Python | 3.10.12 | `python --version` |
| CUDA Toolkit | 12.4 | `nvcc --version` |
| GPU驱动 | 535.129.03 | `nvidia-smi | grep "Driver Version"` |
| PyTorch | 2.6.0 | `print(torch.__version__)` |

### 环境设置
```bash
# 创建虚拟环境
conda create -n agentthink python=3.10
conda activate agentthink

# 安装依赖
pip install -r requirements.txt

# 安装ms-swift
bash scripts/env.sh

# 安装drivemllm依赖
bash scripts/env_drivemllm.sh
```

### 克隆ms-swift
```bash
cd third_party
git clone https://github.com/modelscope/ms-swift.git
```

## 🚀快速上手
### 下载模型
我们的AgentThink模型基于Qwen2.5-VL-7B[AgentThink](xx).

### 下载工具模型
Clone the depth anythingv2[DAM]: (https://github.com/DepthAnything/Depth-Anything-V2)
```
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

Clone the YoloWorld[YoloWorld]: (https://github.com/AILab-CVC/YOLO-World)
```
git clone https://github.com/AILab-CVC/YOLO-World
```
Then download the pretrain models in the [YoloWorld](https://docs.ultralytics.com/zh/models/yolo-world/) and [DepthAnything](https://huggingface.co/depth-anything/Depth-Anything-V2-Base)

### Demo
```bash
# drivemllm
python Inference/inference_demo_drivemllm.py

# drivelmm-o1
python Inference/inference_demo_drivelmm.py
```
---

## 📋 TODO 列表

### 🔧 近期开发任务
| 状态 | 任务描述                   | 
|------|----------------------------|
| ✅   | AgentThink demo      | 
| 🔜   | AgentThink 通用推理评价指标计算        | 
| 🔜   | AgentThink 工具评价指标计算        | 
| 🔜   | AgentThink 数据预处理         | 
| ✅   | AgentThink 示例debug      | 
| 🔜   | AgentThink 多阶段训练          | 
| 🔜   | AgentThink 工具库函数交互环境     | 

---

## 📊 Benchmark结果
### DriveLMM-o1
| Vision Language Models | Risk Assess. (%)↑ | Rule Adh. (%)↑ | Scene Aware. (%)↑ | Relevance (%)↑ | Missing (%)↑ | Reason. (%)↑ | MCQ (%)↑ |
|------------------------|-------------------|----------------|--------------------|-----------------|--------------|--------------|----------|
| [GPT-4o](https://github.com/example/GPT-4o) [16] | 71.32             | 80.72          | 72.96              | 76.65           | 71.43        | 72.52        | 57.84    |
| [Ovis1.5-Gemma2-9B](https://github.com/example/Ovis1.5-Gemma2-9B) [21] | 51.34            | 66.36          | 54.74              | 55.72           | 55.74        | 55.62        | 48.85    |
| [Mulberry-7B](https://github.com/example/Mulberry-7B) [45] | 51.89            | 63.66          | 56.68              | 57.27           | 57.45        | 57.65        | 52.86    |
| [LLaVA-CoT](https://github.com/example/LLaVA-CoT) [43] | 57.62            | 69.01          | 60.84              | 62.72           | 60.67        | 61.41        | 49.27    |
| [LlamaV-o1](https://github.com/example/LlamaV-o1) [34] | 60.20            | 73.52          | 62.67              | 64.66           | 63.41        | 63.13        | 50.02    |
| [InternVL2.5-8B](https://github.com/example/InternVL2.5-8B) [4] | 69.02           | 78.43          | 71.52              | 75.80           | 70.54        | 71.62        | 54.87    |
| [Qwen2.5-VL-7B](https://github.com/example/Qwen2.5-VL-7B) [1] | 46.44           | 60.45          | 51.02              | 50.15           | 52.19        | 51.77        | 37.81    |
| [DriveLMM-o1](https://github.com/example/DriveLMM-o1) [15] | 73.01           | 81.56          | 75.39              | 79.42           | 74.49        | 75.24        | 62.36    |
| **AgentThink (Ours)** | **80.51**         | **84.98**      | **82.11**          | **84.99**       | **79.56**    | **79.68**    | **71.35** |

## 📊 性能对比分析
从上表可以看出，AgentThink 在各个评估指标上均表现出色，特别是在风险评估、规则遵守和场景感知方面，分别达到了 **80.51%**, **84.98%**, 和 **82.11%** 的高分。此外，在场景细节的相关性和缺失信息检测上也取得了显著优势，分别为 **84.99%** 和 **79.56%**。总体评分和多项选择题（MCQ）得分分别为 **79.68%** 和 **71.35%**，进一步证明了 AgentThink 在自动驾驶视觉语言推理任务中的优越性能。

---

### DriveMLLM

| Type       | Model                                                                 | L/R    | F/B    | RHD   | RD     | PPos  | BBox  | CVD   | CD     | AccS  | Overall |
|------------|-----------------------------------------------------------------------|--------|--------|-------|--------|-------|-------|-------|--------|-------|---------|
| Zero-shot  | [GPT-4o](https://github.com/example/GPT-4o) [16]                     | 91.72  | 67.60  | 9.58  | 14.69  | 40.90 | 4.07  | 46.11 | 70.65  | 43.16 | 25.63   |
|            | [GPT-4o-mini](https://github.com/example/GPT-4o-mini)                | 67.67  | 50.13  | 70.44 | 0.00   | 29.28 | 3.78  | 0.00  | 46.40  | 33.46 | 16.68   |
|            | [LLaVA-ov-72B](https://github.com/example/LLaVA-ov-72B) [19]          | 85.42  | 49.48  | 13.76 | 45.27  | 16.46 | 0.00  | 42.97 | 27.09  | 35.06 | 21.10   |
|            | [Qwen2.5-VL-7B](https://github.com/example/Qwen2.5-VL-7B) [1]         | 76.55  | 55.24  | 7.14  | 17.11  | 55.97 | 38.31 | 55.94 | 51.52  | 44.72 | 13.36   |
|            | [Qwen + CoT](https://github.com/example/Qwen-CoT)                    | 87.06  | 63.09  | 16.69 | 22.56  | 52.51 | 38.87 | 76.90 | 38.71  | 49.55 | 19.31   |
|            | [Qwen + DirectTool](https://github.com/example/Qwen-DirectTool)       | 78.95  | 48.96  | 58.43 | 67.57  | 58.20 | 42.22 | 51.76 | 51.38  | 57.18 | 24.05   |
|            | **AgentThink (Ours)**                                                 | 82.33  | 54.40  | 56.14 | 61.45  | 70.45 | 56.23 | 23.09 | 51.60  | 56.96 | 26.52   |
| One-shot   | [GPT-4o](https://github.com/example/GPT-4o)                           | 91.08  | 69.37  | 36.51 | 71.17  | 42.44 | 5.10  | 0.00  | 63.88  | 47.44 | 33.17   |
|            | [GPT-4o-mini](https://github.com/example/GPT-4o-mini)                 | 66.00  | 48.95  | 83.02 | 58.47  | 25.71 | 3.97  | 52.73 | 55.23  | 49.26 | 22.13   |
|            | [LLaVA-ov-72B](https://github.com/example/LLaVA-ov-72B) [19]           | 79.12  | 62.97  | 49.26 | 68.04  | 28.57 | 2.20  | 53.12 | 60.90  | 50.52 | 36.66   |
|            | [Qwen2.5-VL-7B](https://github.com/example/Qwen2.5-VL-7B) [1]         | 80.30  | 53.14  | 36.96 | 39.13  | 62.69 | 22.63 | 49.88 | 48.32  | 49.13 | 33.53   |
|            | [Qwen + CoT](https://github.com/example/Qwen-CoT)                    | 86.35  | 59.95  | 43.29 | 31.81  | 53.64 | 26.93 | 51.02 | 42.30  | 49.41 | 32.06   |
|            | [Qwen + DirectTool](https://github.com/example/Qwen-DirectTool)       | 84.57  | 55.50  | 67.32 | 59.54  | 85.58 | 26.07 | 52.34 | 53.25  | 60.52 | 42.27   |
|            | **AgentThink (Ours)**                                                 | 78.71  | 48.46  | 60.64 | 60.71  | 72.36 | 64.46 | 52.26 | 52.04  | 61.21 | 47.24   |

## 性能对比分析
从上表可以看出，在零样本（Zero-shot）和单样本（One-shot）两种设置下，AgentThink在多个评估指标上均表现出色。特别是在“PPos”、“BBox”、“AccS”等关键指标上取得了显著优势，分别达到了**70.45%**、**56.23%**和**56.96%**（零样本），以及**72.36%**、**64.46%**和**61.21%**（单样本）。这表明AgentThink在自动驾驶视觉语言推理任务中具有强大的泛化能力和适应性。

## License and Citation

### License

本项目遵循[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)开源协议发布。这意味着您可以在遵守许可条款的前提下自由使用、修改和分发此软件。请确保在您的应用中包含适当的版权声明和免责声明。

除非适用法律要求或书面同意，软件根据“原样”提供，不附带任何明示或暗示的担保责任。具体详情，请参阅许可文件中的语言规定权限和限制。

### Citation

如果您在研究中使用了AgentThink框架，请按照以下格式引用我们的工作：

```bibtex
@misc{qian2025agentthinkunifiedframeworktoolaugmented,
      title={AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving}, 
      author={Kangan Qian and Sicong Jiang and Yang Zhong and Ziang Luo and Zilin Huang and Tianze Zhu and Kun Jiang and Mengmeng Yang and Zheng Fu and Jinyu Miao and Yining Shi and He Zhe Lim and Li Liu and Tianbao Zhou and Huang Yu and Yifei Hu and Guang Li and Guang Chen and Hao Ye and Lijun Sun and Diange Yang},
      year={2025},
      eprint={2505.15298},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.15298}, 
}

