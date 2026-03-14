# VLA从零入门 | VLA (Vision-Language-Action) from Scratch

**视觉-语言-动作模型**完整入门教程，从基础原理到完整可运行实例。

> 适合人群：熟悉Transformer/LLM基础，想要快速入门机器人VLA研究方向

---

## 什么是VLA？

VLA = **V**ision + **L**anguage + **A**ction

**一句话定义**：给机器人装上"眼睛"（视觉）和"语言理解"（语言），让它能执行自然语言指令的模型。

```
输入：📷 图像（机器人摄像头） + 💬 "把苹果放到碗里"
输出：🤖 [x=0.3, y=-0.1, z=0.5, roll=0.0, pitch=0.8, yaw=-0.2, gripper=-0.8]
              ↑ 机器人末端执行器7维动作
```

**核心思想**：把机器人控制变成**下一个Token预测**问题！
- 连续动作 → 离散Token（就像NLP中的词）
- 视觉信息 → Patch Token（就像ViT）
- 语言指令 → 词Token（就像LLM）
- 统一送入Transformer，预测动作Token

---

## 项目结构

```
vla-from-scratch/
├── 01_basics/                    # 基础概念（先从这里开始！）
│   ├── 01_action_tokenization.py # 动作离散化：连续值→Token ID
│   ├── 02_vision_encoder.py      # 视觉编码器：图像→Patch Token
│   └── 03_cross_modal_attention.py # 跨模态注意力：视觉+语言融合
│
├── 02_minimal_vla/               # 完整可训练VLA模型
│   ├── model.py                  # MinimalVLA架构实现
│   ├── train.py                  # 在合成数据集上训练
│   └── inference.py              # 推理+可视化自回归生成
│
├── 03_rt2_style/                 # RT-2风格：动作=文本Token
│   └── rt2_demo.py               # RT-2架构演示+对比传统VLA
│
├── 04_advanced/                  # 进阶：主流VLA全景对比
│   └── compare_vlas.py           # RT-2/OpenVLA/π0/Octo对比
│
├── assets/                       # 生成的可视化图像
└── requirements.txt
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 第一课：动作离散化（约1分钟）
python 01_basics/01_action_tokenization.py

# 3. 第二课：视觉编码器（约1分钟）
python 01_basics/02_vision_encoder.py

# 4. 第三课：跨模态注意力（约1分钟）
python 01_basics/03_cross_modal_attention.py

# 5. 训练完整VLA模型（CPU约5-10分钟）
python 02_minimal_vla/train.py

# 6. 推理演示（需先训练）
python 02_minimal_vla/inference.py

# 7. RT-2风格演示
python 03_rt2_style/rt2_demo.py

# 8. 主流VLA架构对比
python 04_advanced/compare_vlas.py
```

---

## 核心概念速查

### 1. 动作离散化（Action Tokenization）

```python
# 连续动作值 → 离散Token（RT-2/OpenVLA的做法）
action = [0.3, -0.1, 0.5, 0.0, 0.8, -0.2, 1.0]  # 7维连续动作
tokens = [166, 115, 191,  128, 230, 102, 255]      # 7个Token ID（256个桶）
```

### 2. 视觉Token化

```python
# ViT: 图像 → Patch Token序列
# 224×224图像, patch_size=16
# → (224/16)² = 196个Patch Token
# → 每个Token: [embed_dim] 维向量
```

### 3. VLA输入格式

```
[视觉Token×196] [语言Token序列]     [动作Token×7]
────────────    ─────────────────   ────────────
<img_0>         Pick up the apple   128 89 201 0 180 130 255
...             ↑ 语言指令           ↑ 自回归预测
<img_195>
```

---

## 主流VLA对比

| 模型 | 参数量 | 动作表示 | 开源 | 核心创新 |
|------|--------|----------|------|----------|
| **RT-2** (Google, 2023) | 562B | 离散Token | ✗ | 动作=文本，Web知识迁移 |
| **OpenVLA** (Stanford, 2024) | 7B | 离散Token | ✓ | 开源RT-2，多机器人泛化 |
| **π0** (PhysAI, 2024) | 3B | Flow Matching | ✗ | 高精度灵巧操作 |
| **Octo** (Berkeley, 2024) | 90M | 扩散模型 | ✓ | 轻量通用策略 |
| **RoboFlamingo** (2023) | 9B | MLP头 | ✓ | Cross-Attn融合 |

---

## 学习路线

```
本项目基础 → 阅读OpenVLA论文 → 在LIBERO/CALVIN上微调 → 提出改进
    ↓              ↓                    ↓                  ↓
理解原理      看懂完整实现          跑基准测试           发论文！
```

**推荐论文阅读顺序**：
1. [RT-2 (2023)](https://arxiv.org/abs/2307.15818) - VLA开山之作
2. [OpenVLA (2024)](https://arxiv.org/abs/2406.09246) - 最佳开源基线
3. [Octo (2024)](https://arxiv.org/abs/2405.12213) - 轻量通用策略
4. [π0 (2024)](https://arxiv.org/abs/2410.24164) - Flow Matching动作头

---

## 可视化输出示例

运行各脚本后会在 `assets/` 目录生成：
- `01_action_tokenization.png` - 动作离散化过程
- `02_vision_encoder.png` - Patch切割示意
- `03_cross_modal_attention.png` - 跨模态注意力热图
- `04_training_curve.png` - 训练损失曲线
- `05_predictions.png` - 预测动作 vs 真实动作
- `06_autoregressive_generation.png` - 自回归生成逐步展示
- `07_rt2_vs_traditional.png` - RT-2 vs 传统VLA架构图
- `08_vla_landscape.png` - 主流VLA全景对比

---

## 关键参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `action_dim` | 动作维度（通常是7: x,y,z,rx,ry,rz,gripper） | 7 |
| `num_bins` | 每个维度的离散化桶数 | 256 |
| `patch_size` | ViT的Patch大小 | 16 (ViT-B/16) |
| `embed_dim` | Token嵌入维度 | 768 (ViT-B), 1024 (ViT-L) |

---

*参考项目：[diffusion-from-scratch](https://github.com/Ody-trek/diffusion-from-scratch) | [transformer-from-scratch](https://github.com/Ody-trek/transformer-from-scratch)*
