"""
主流VLA架构对比
===============
横向对比当前最重要的VLA模型：
  RT-2, OpenVLA, π0 (pi-zero), Octo, RoboFlamingo

运行方式：
    python 04_advanced/compare_vlas.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────
# 主流VLA架构数据
# ─────────────────────────────────────────────

VLA_MODELS = {
    "RT-2\n(Google, 2023)": {
        "vision_backbone": "ViT-4B (PaLI-X)",
        "language_backbone": "PaLM-E 562B",
        "action_representation": "离散Token（词表扩展）",
        "action_output": "自回归文本生成",
        "training_data": "Web数据 + 机器人演示",
        "params": 562,  # B
        "key_innovation": "动作=文本Token\nWeb知识迁移",
        "open_source": False,
        "color": "#4285F4",
        "zero_shot": 9,   # 1-10分
        "generalization": 9,
        "inference_speed": 3,
        "data_efficiency": 6,
    },
    "OpenVLA\n(Stanford, 2024)": {
        "vision_backbone": "DINOv2 + SigLIP",
        "language_backbone": "Llama 2 7B",
        "action_representation": "离散Token（词表扩展）",
        "action_output": "自回归文本生成",
        "training_data": "Open X-Embodiment (970K演示)",
        "params": 7,
        "key_innovation": "开源RT-2\n多机器人泛化",
        "open_source": True,
        "color": "#EA4335",
        "zero_shot": 7,
        "generalization": 8,
        "inference_speed": 4,
        "data_efficiency": 7,
    },
    "π0 (pi-zero)\n(PhysAI, 2024)": {
        "vision_backbone": "SigLIP",
        "language_backbone": "PaliGemma 3B",
        "action_representation": "连续动作（Flow Matching）",
        "action_output": "扩散/Flow Matching解码",
        "training_data": "机器人演示 + 预训练VLM",
        "params": 3,
        "key_innovation": "Flow Matching动作头\n高精度灵巧操作",
        "open_source": False,
        "color": "#34A853",
        "zero_shot": 6,
        "generalization": 7,
        "inference_speed": 5,
        "data_efficiency": 8,
    },
    "Octo\n(Berkeley, 2024)": {
        "vision_backbone": "ViT（从头训练）",
        "language_backbone": "小型Transformer",
        "action_representation": "连续动作 + Diffusion",
        "action_output": "扩散模型解码",
        "training_data": "Open X-Embodiment (800K+)",
        "params": 0.09,
        "key_innovation": "轻量高效\n多任务通用策略",
        "open_source": True,
        "color": "#FBBC04",
        "zero_shot": 5,
        "generalization": 7,
        "inference_speed": 7,
        "data_efficiency": 8,
    },
    "RoboFlamingo\n(2023)": {
        "vision_backbone": "CLIP ViT-L",
        "language_backbone": "OPT-9B",
        "action_representation": "连续动作",
        "action_output": "MLP动作头",
        "training_data": "CALVIN演示数据",
        "params": 9,
        "key_innovation": "Cross-Attention融合\n语言引导操作",
        "open_source": True,
        "color": "#FF6D00",
        "zero_shot": 5,
        "generalization": 6,
        "inference_speed": 6,
        "data_efficiency": 7,
    },
}


def print_comparison_table():
    """打印对比表格"""
    print("=" * 85)
    print("  主流VLA模型对比")
    print("=" * 85)

    # 表头
    headers = ["模型", "参数量", "开源", "动作表示", "核心创新"]
    print(f"{'模型':<20} {'参数量':>8} {'开源':>6} {'动作表示':<22} 核心创新")
    print("-" * 85)

    for name, info in VLA_MODELS.items():
        clean_name = name.replace('\n', ' ')
        params_str = f"{info['params']}B" if info['params'] >= 1 else f"{int(info['params']*1000)}M"
        open_str = "✓" if info['open_source'] else "✗"
        action_str = info['action_representation'][:20]
        innovation = info['key_innovation'].replace('\n', ' / ')
        print(f"{clean_name:<20} {params_str:>8} {open_str:>6} {action_str:<22} {innovation}")

    print("=" * 85)


def visualize_vla_landscape():
    """可视化VLA研究版图"""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    fig.suptitle("主流VLA架构全景对比 (2023-2024)", fontsize=15, fontweight='bold')

    models = list(VLA_MODELS.keys())
    colors = [VLA_MODELS[m]['color'] for m in models]

    # ── 图1：参数量对比
    ax1 = fig.add_subplot(gs[0, 0])
    params = [VLA_MODELS[m]['params'] for m in models]
    short_names = [m.split('\n')[0] for m in models]
    bars = ax1.bar(range(len(models)), params, color=colors, alpha=0.85, edgecolor='gray')
    ax1.set_yscale('log')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel("参数量 (B)")
    ax1.set_title("模型参数量对比", fontsize=10)
    for bar, p in zip(bars, params):
        label = f"{p}B" if p >= 1 else f"{int(p*1000)}M"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 label, ha='center', fontsize=8)

    # ── 图2：雷达图（能力对比）
    ax2 = fig.add_subplot(gs[0, 1:], polar=True)
    categories = ['零样本\n泛化', '跨平台\n泛化', '推理\n速度', '数据\n效率']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for model_name, color in zip(models, colors):
        info = VLA_MODELS[model_name]
        values = [info['zero_shot'], info['generalization'],
                  info['inference_speed'], info['data_efficiency']]
        values += values[:1]
        label = model_name.split('\n')[0]
        ax2.plot(angles, values, 'o-', linewidth=1.5, color=color, label=label, alpha=0.8)
        ax2.fill(angles, values, color=color, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=9)
    ax2.set_ylim(0, 10)
    ax2.set_yticks([2, 4, 6, 8, 10])
    ax2.set_yticklabels(['2', '4', '6', '8', '10'], size=7)
    ax2.set_title("能力雷达图（主观评分1-10）", size=10, pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    # ── 图3：动作表示方式分类
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 5)
    ax3.set_title("动作表示方式对比", fontsize=10)

    # 离散Token方式
    ax3.add_patch(plt.Rectangle((0.5, 3.0), 5, 1.5, facecolor='#E3F2FD',
                                 edgecolor='#1565C0', linewidth=2))
    ax3.text(3.0, 4.1, "离散Token方式 (RT-2, OpenVLA)", fontsize=9,
             ha='center', fontweight='bold', color='#1565C0')
    ax3.text(3.0, 3.7, "连续值 → 256桶 → Token ID → LLM词表", fontsize=8, ha='center')
    ax3.text(3.0, 3.35, "优点: 完全利用LLM能力 | 缺点: 精度受桶数量限制", fontsize=7,
             ha='center', style='italic')

    # 连续动作 + Flow Matching
    ax3.add_patch(plt.Rectangle((6.5, 3.0), 5, 1.5, facecolor='#E8F5E9',
                                 edgecolor='#2E7D32', linewidth=2))
    ax3.text(9.0, 4.1, "连续动作 + Flow Matching (π0)", fontsize=9,
             ha='center', fontweight='bold', color='#2E7D32')
    ax3.text(9.0, 3.7, "LLM特征 → Flow Matching → 连续动作", fontsize=8, ha='center')
    ax3.text(9.0, 3.35, "优点: 高精度 | 缺点: 推理时需多步迭代", fontsize=7,
             ha='center', style='italic')

    # 扩散模型方式
    ax3.add_patch(plt.Rectangle((0.5, 1.0), 5, 1.5, facecolor='#FFF8E1',
                                 edgecolor='#F57F17', linewidth=2))
    ax3.text(3.0, 2.1, "连续动作 + Diffusion (Octo)", fontsize=9,
             ha='center', fontweight='bold', color='#F57F17')
    ax3.text(3.0, 1.7, "特征 → 扩散模型解码 → 连续动作轨迹", fontsize=8, ha='center')
    ax3.text(3.0, 1.35, "优点: 多模态动作分布 | 缺点: 速度慢", fontsize=7,
             ha='center', style='italic')

    # MLP方式
    ax3.add_patch(plt.Rectangle((6.5, 1.0), 5, 1.5, facecolor='#FCE4EC',
                                 edgecolor='#880E4F', linewidth=2))
    ax3.text(9.0, 2.1, "MLP动作头 (RoboFlamingo等早期)", fontsize=9,
             ha='center', fontweight='bold', color='#880E4F')
    ax3.text(9.0, 1.7, "VLM特征 → MLP → 连续动作", fontsize=8, ha='center')
    ax3.text(9.0, 1.35, "优点: 简单快速 | 缺点: 表达能力有限", fontsize=7,
             ha='center', style='italic')
    ax3.axis('off')

    # ── 图4：VLA发展时间线
    ax4 = fig.add_subplot(gs[1, 2])
    timeline = [
        (2023.0, "RT-2\n(Google)", "#4285F4"),
        (2023.5, "RoboFlamingo\n(MSRA)", "#FF6D00"),
        (2024.1, "Octo\n(Berkeley)", "#FBBC04"),
        (2024.3, "OpenVLA\n(Stanford)", "#EA4335"),
        (2024.7, "π0\n(PhysAI)", "#34A853"),
    ]
    ax4.set_xlim(2022.8, 2025.2)
    ax4.set_ylim(-0.5, len(timeline))
    ax4.axvline(2023, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(2024, color='gray', linestyle='--', alpha=0.5)

    for i, (year, name, color) in enumerate(timeline):
        ax4.scatter(year, i, color=color, s=150, zorder=5)
        ax4.text(year + 0.05, i, name, fontsize=8, va='center', color=color)

    ax4.set_xlabel("时间")
    ax4.set_title("VLA发展时间线", fontsize=10)
    ax4.set_yticks([])
    ax4.axhline(-0.5, color='black', linewidth=0.5)

    # ── 图5：选择指南
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_xlim(0, 16)
    ax5.set_ylim(0, 4)
    ax5.set_title("选型指南：根据你的研究方向选择VLA", fontsize=11, fontweight='bold')

    guide_items = [
        (0.5, 2.5, "#4285F4", "想要最强零样本能力?", "→ RT-2 (或等待开源复现)"),
        (0.5, 1.5, "#EA4335", "想要可复现/可微调的基线?", "→ OpenVLA (HuggingFace可下载)"),
        (5.5, 2.5, "#34A853", "需要高精度灵巧操作?", "→ π0 + Flow Matching动作头"),
        (5.5, 1.5, "#FBBC04", "资源有限/快速验证?", "→ Octo (90M参数，轻量)"),
        (10.5, 2.5, "#FF6D00", "研究跨模态注意力?", "→ RoboFlamingo (Cross-Attn)"),
        (10.5, 1.5, "purple",  "自己从头实现?", "→ 本项目 MinimalVLA ✓"),
    ]

    for x, y, color, question, answer in guide_items:
        ax5.add_patch(plt.Rectangle((x, y - 0.35), 4.5, 0.75,
                                     facecolor=color, alpha=0.15,
                                     edgecolor=color, linewidth=1.5))
        ax5.text(x + 0.1, y + 0.1, question, fontsize=8.5, fontweight='bold', color=color)
        ax5.text(x + 0.1, y - 0.2, answer, fontsize=8, color='black')

    ax5.axis('off')

    plt.savefig("assets/08_vla_landscape.png", dpi=150, bbox_inches='tight')
    print("VLA全景图已保存: assets/08_vla_landscape.png")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    print_comparison_table()
    visualize_vla_landscape()

    print("\n" + "=" * 60)
    print("  研究路线建议（适合VLA入门研究者）")
    print("=" * 60)
    print("""
第1阶段（本项目）：理解基础
  ✓ Action Tokenization → 01_basics/01_action_tokenization.py
  ✓ Vision Encoder (ViT) → 01_basics/02_vision_encoder.py
  ✓ Cross-Modal Attention → 01_basics/03_cross_modal_attention.py

第2阶段：跑通完整VLA
  ✓ 训练 MinimalVLA → 02_minimal_vla/train.py
  ✓ RT-2风格架构 → 03_rt2_style/rt2_demo.py

第3阶段：调研主流模型
  → 阅读 OpenVLA 论文 + GitHub（最推荐，代码开源）
  → 阅读 Octo 论文（轻量，适合快速实验）
  → 尝试在 LIBERO / CALVIN 基准上微调

第4阶段：创新方向
  → 更高效的动作表示（连续 vs 离散）
  → 更好的视觉-语言融合（如π0的VLM初始化）
  → 数据高效学习（few-shot adaptation）
  → 实时推理优化（量化/蒸馏）
    """)
