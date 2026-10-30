"""
VLA入门 第1课：动作离散化（Action Tokenization）
===============================================
VLA的核心创新之一：把机器人的连续动作（连续值）转化为离散Token
这样语言模型就可以直接预测"下一个动作Token"，就像预测下一个词一样！

RT-2 / OpenVLA 的做法：
  连续动作值 [-1, 1]  →  均匀分成256个桶  →  每个桶对应一个整数Token

运行方式：
    python 01_basics/01_action_tokenization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 默认的 DejaVu Sans 不含中文字形；优先使用系统中的中文字体。
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei',
    'SimHei', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans',
]


# ─────────────────────────────────────────────
# 1. 动作离散化器（Action Tokenizer）
# ─────────────────────────────────────────────

class ActionTokenizer:
    """
    将连续动作值离散化为Token ID

    机器人动作通常是7维：[x, y, z, roll, pitch, yaw, gripper]
    每个维度的值范围是 [-1, 1]（归一化后）

    RT-2 把每个维度分成 256 个桶，所以一个7维动作
    变成 7 个Token，可以直接被语言模型预测！
    """

    def __init__(self, action_dim: int = 7, num_bins: int = 256,
                 min_val: float = -1.0, max_val: float = 1.0):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # 创建均匀分布的桶边界
        self.bins = np.linspace(min_val, max_val, num_bins + 1)
        print(f"[ActionTokenizer] {action_dim}维动作 → {action_dim}个Token")
        print(f"  桶数量: {num_bins}, 桶宽度: {(max_val - min_val) / num_bins:.4f}")

    def encode(self, action: np.ndarray) -> np.ndarray:
        """连续动作 → Token ID"""
        # np.digitize: 找到每个值所在的桶
        token_ids = np.digitize(action, self.bins[1:-1])  # [0, num_bins-1]
        return token_ids.astype(np.int32)

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """Token ID → 连续动作（取桶中心值）"""
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        return bin_centers[token_ids]

    def reconstruction_error(self, action: np.ndarray) -> float:
        """计算量化误差"""
        reconstructed = self.decode(self.encode(action))
        return np.mean(np.abs(action - reconstructed))


# ─────────────────────────────────────────────
# 2. 可视化演示
# ─────────────────────────────────────────────

def demo_tokenization():
    print("=" * 60)
    print("  动作离散化（Action Tokenization）演示")
    print("=" * 60)

    tokenizer = ActionTokenizer(action_dim=7, num_bins=256)

    # 模拟一个机器人抓取动作
    # [x=0.3, y=-0.1, z=0.5, roll=0.0, pitch=0.8, yaw=-0.2, gripper=1.0]
    action = np.array([0.3, -0.1, 0.5, 0.0, 0.8, -0.2, 1.0])
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    print(f"\n原始连续动作（7维）:")
    for name, val in zip(action_names, action):
        print(f"  {name:>8}: {val:+.4f}")

    # 编码为Token
    token_ids = tokenizer.encode(action)
    print(f"\n离散化Token ID（7个）:")
    for name, tid in zip(action_names, token_ids):
        print(f"  {name:>8}: Token#{tid:3d}  (桶范围: [{tokenizer.bins[tid]:.4f}, {tokenizer.bins[tid+1]:.4f}])")

    # 解码回连续值
    reconstructed = tokenizer.decode(token_ids)
    error = tokenizer.reconstruction_error(action)
    print(f"\n重建后的动作（Token→连续值）:")
    for name, orig, recon in zip(action_names, action, reconstructed):
        print(f"  {name:>8}: {orig:+.4f} → {recon:+.4f}  (误差: {abs(orig-recon):.4f})")
    print(f"\n平均量化误差: {error:.4f} (桶宽的一半 ≈ {1/256:.4f})")

    return tokenizer, action, token_ids


def visualize_tokenization(tokenizer, action, token_ids):
    """可视化离散化过程"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VLA动作离散化 (Action Tokenization)", fontsize=14, fontweight='bold')

    # 左图：连续值 vs Token ID
    ax1 = axes[0]
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    colors = plt.cm.Set2(np.linspace(0, 1, 7))

    for i, (name, val, tid, c) in enumerate(zip(action_names, action, token_ids, colors)):
        ax1.barh(i, val, color=c, alpha=0.8, label=f"{name}={val:.2f}→T#{tid}")

    ax1.set_yticks(range(7))
    ax1.set_yticklabels(action_names)
    ax1.set_xlabel("连续动作值")
    ax1.set_title("连续动作值 [-1, 1]")
    ax1.axvline(0, color='k', linewidth=0.5)
    ax1.set_xlim(-1.2, 1.2)
    ax1.legend(loc='lower right', fontsize=8)

    # 右图：Token ID分布（量化桶示意）
    ax2 = axes[1]
    x = np.linspace(-1, 1, 1000)
    ax2.plot(x, np.zeros_like(x), 'k-', linewidth=0.5, alpha=0.3)

    # 画桶边界
    for b in tokenizer.bins[::32]:
        ax2.axvline(b, color='gray', linewidth=0.3, alpha=0.5)

    # 标注每个动作维度对应的Token桶
    for i, (name, val, tid, c) in enumerate(zip(action_names, action, token_ids, colors)):
        bin_center = (tokenizer.bins[tid] + tokenizer.bins[tid + 1]) / 2
        ax2.scatter(val, i * 0.1, color=c, s=100, zorder=5, label=f"{name}")
        ax2.scatter(bin_center, i * 0.1, color=c, s=50, marker='x', zorder=6)
        ax2.annotate(f"T#{tid}", (bin_center, i * 0.1 + 0.03), fontsize=7, ha='center')

    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-0.1, 0.8)
    ax2.set_xlabel("值范围 [-1, 1]")
    ax2.set_title(f"量化桶示意（共256桶，每桶宽≈{2/256:.3f}）")
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig("assets/01_action_tokenization.png", dpi=150, bbox_inches='tight')
    print("\n图像已保存: assets/01_action_tokenization.png")
    plt.show()


def demo_sequence():
    """演示一段动作序列的tokenization，类比NLP中的句子"""
    print("\n" + "=" * 60)
    print("  类比NLP：动作序列 = 句子，动作Token = 词Token")
    print("=" * 60)

    tokenizer = ActionTokenizer(action_dim=7, num_bins=256)

    # 模拟一个抓取任务的5步动作序列
    # 就像NLP中的句子有多个词，机器人任务有多步动作
    action_sequence = np.array([
        [0.0,  0.0,  0.8,  0.0,  0.0,  0.0,  1.0],  # 步骤1: 悬停
        [0.3, -0.1,  0.8,  0.0,  0.0,  0.0,  1.0],  # 步骤2: 移向目标
        [0.3, -0.1,  0.3,  0.0,  0.2,  0.0,  1.0],  # 步骤3: 下降
        [0.3, -0.1,  0.3,  0.0,  0.2,  0.0, -0.5],  # 步骤4: 抓取（夹爪关闭）
        [0.3, -0.1,  0.8,  0.0,  0.0,  0.0, -0.5],  # 步骤5: 提起
    ])

    step_names = ["悬停", "移向目标", "下降", "抓取", "提起"]

    print("\n任务: '抓取桌上的苹果'")
    print("\n动作序列 → Token序列（类比：句子 → 词ID序列）")
    print(f"{'步骤':<8} {'动作描述':<10} {'Token序列（7个值）':<45} {'可解读为'}")
    print("-" * 85)

    all_tokens = []
    for i, (act, name) in enumerate(zip(action_sequence, step_names)):
        tokens = tokenizer.encode(act)
        all_tokens.extend(tokens.tolist())
        token_str = "[" + ", ".join(f"{t:3d}" for t in tokens) + "]"
        print(f"步骤{i+1:<4} {name:<10} {token_str:<45}")

    print(f"\n完整Token序列（共{len(all_tokens)}个）:")
    print(all_tokens)
    print(f"\n👆 语言模型的任务：给定视觉观测+语言指令，预测这{len(all_tokens)}个Token！")
    print("  这就是VLA的核心：把机器人控制变成一个'下一个Token预测'问题")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    tokenizer, action, token_ids = demo_tokenization()
    visualize_tokenization(tokenizer, action, token_ids)
    demo_sequence()

    print("\n" + "=" * 60)
    print("  关键要点总结")
    print("=" * 60)
    print("1. 连续动作 → 离散Token：让LLM能直接预测动作")
    print("2. 量化误差很小（≈1/256）：足够精确用于机器人控制")
    print("3. 动作序列 ≈ 词序列：VLA = 会预测动作的语言模型")
    print("4. RT-2、OpenVLA 都用了这个简单而有效的设计")
