"""
RT-2 风格 VLA 演示
==================
RT-2（Robotics Transformer 2）的核心创新：
  把动作直接表示成"文本"！
  不需要单独的动作解码器，直接用LLM生成动作字符串

例如：
  输入: [图像] + "Pick up the cup. What action should the robot take?"
  输出: "256 89 201 0 180 130 255"  ← 7个数字，就是7维动作Token！

这样VLA就完全是一个文本生成任务，直接用LLM处理！

运行方式：
    python 03_rt2_style/rt2_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_minimal_vla"
))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from einops import rearrange


# 默认的 DejaVu Sans 不含中文字形；优先使用系统中的中文字体。
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei',
    'SimHei', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans',
]


# ─────────────────────────────────────────────
# RT-2 的动作表示方式
# ─────────────────────────────────────────────

class RT2ActionRepresentation:
    """
    RT-2 把动作Token直接嵌入到LLM的词表中

    做法：
      1. 原词表大小 = V（例如256000）
      2. 为每个动作维度×每个桶值添加一个新Token
      3. 新词表大小 = V + action_dim × num_bins

    训练时：
      - 正常的语言Token用原始词表
      - 动作Token用新增的特殊Token

    推理时：
      - LLM自回归生成，遇到动作特殊Token就解码为动作值
    """

    def __init__(self, base_vocab_size=1000, action_dim=7, num_bins=256):
        self.base_vocab_size = base_vocab_size
        self.action_dim = action_dim
        self.num_bins = num_bins

        # 动作Token从 base_vocab_size 开始编号
        self.action_token_start = base_vocab_size
        self.total_vocab_size = base_vocab_size + action_dim * num_bins

        # 桶边界
        self.bins = np.linspace(-1, 1, num_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        print(f"[RT-2 Action Representation]")
        print(f"  基础词表大小: {base_vocab_size:,}")
        print(f"  新增动作Token数: {action_dim * num_bins:,}")
        print(f"  总词表大小: {self.total_vocab_size:,}")
        print(f"  动作Token范围: [{self.action_token_start}, {self.total_vocab_size-1}]")

    def action_to_tokens(self, action: np.ndarray, dim: int) -> int:
        """
        将某一维度的连续动作值 → 全局词表Token ID

        dim: 动作维度索引 (0=x, 1=y, ..., 6=gripper)
        """
        # 找到对应的桶
        bin_idx = np.digitize(action, self.bins[1:-1])
        # 转换为全局Token ID
        global_token_id = self.action_token_start + dim * self.num_bins + bin_idx
        return int(global_token_id)

    def token_to_action(self, global_token_id: int, dim: int) -> float:
        """全局词表Token ID → 连续动作值"""
        local_bin_idx = global_token_id - self.action_token_start - dim * self.num_bins
        return float(self.bin_centers[local_bin_idx])

    def encode_action_sequence(self, action: np.ndarray) -> list:
        """7维动作 → 7个全局Token ID（就是LLM要生成的"词"）"""
        return [self.action_to_tokens(action[dim], dim) for dim in range(self.action_dim)]

    def decode_token_sequence(self, token_ids: list) -> np.ndarray:
        """7个全局Token ID → 7维连续动作"""
        return np.array([self.token_to_action(tid, dim)
                         for dim, tid in enumerate(token_ids)])

    def format_as_text(self, action_tokens: list) -> str:
        """把动作Token格式化成'文本'（RT-2的做法）"""
        return " ".join([f"<act_{t}>" for t in action_tokens])


# ─────────────────────────────────────────────
# RT-2 风格的输入格式
# ─────────────────────────────────────────────

def demo_rt2_input_format():
    """演示RT-2的输入输出格式"""
    print("=" * 65)
    print("  RT-2 输入/输出格式演示")
    print("=" * 65)

    repr = RT2ActionRepresentation(base_vocab_size=1000, action_dim=7, num_bins=256)

    # 模拟一个抓取动作
    action = np.array([0.3, -0.1, 0.5, 0.0, 0.8, -0.2, 1.0])
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    print("\n示例：机器人执行 'pick up the cup' 任务")
    print("\n方式一：传统VLA（单独动作头）")
    print("  输入: [图像] + 'pick up the cup'")
    print("  输出: [-0.3, 0.1, 0.5, 0.0, 0.8, -0.2, 1.0]  ← 专用动作解码器输出")

    print("\n方式二：RT-2（动作=文本Token）")
    print("  输入: [图像] + 'pick up the cup. Output: '")
    action_tokens = repr.encode_action_sequence(action)
    action_text = " ".join(str(t) for t in action_tokens)
    print(f"  输出: '{action_text}'  ← LLM直接生成的文本！")

    print(f"\n Token ID 解析:")
    for i, (name, true_val, tok_id) in enumerate(zip(action_names, action, action_tokens)):
        decoded_val = repr.token_to_action(tok_id, i)
        print(f"  {name:>8}: {true_val:+.3f} → Token#{tok_id:5d} → {decoded_val:+.3f}")

    print(f"\n RT-2 的词表结构:")
    print(f"  Token 0 ~ {repr.base_vocab_size-1:4d}:  正常语言词（'the', 'cup', 'pick'...）")
    print(f"  Token {repr.base_vocab_size:4d} ~ {repr.base_vocab_size+255:4d}:  x轴动作桶 (dim=0)")
    print(f"  Token {repr.base_vocab_size+256:4d} ~ {repr.base_vocab_size+511:4d}:  y轴动作桶 (dim=1)")
    print(f"  ... 以此类推")
    print(f"  Token {repr.action_token_start+6*256:4d} ~ {repr.total_vocab_size-1:4d}:  gripper桶 (dim=6)")

    return repr, action, action_tokens


# ─────────────────────────────────────────────
# RT-2 风格的完整模型
# ─────────────────────────────────────────────

class RT2StyleVLA(nn.Module):
    """
    RT-2 风格的VLA：动作就是文本Token

    关键区别于 MinimalVLA：
    - 没有单独的动作解码器
    - 所有模态（视觉/语言/动作）用同一个LLM处理
    - 词表扩展了：加入了动作特殊Token
    """

    def __init__(self, img_size=64, patch_size=8, embed_dim=256,
                 base_vocab_size=1000, action_dim=7, num_bins=256):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.base_vocab_size = base_vocab_size
        total_vocab = base_vocab_size + action_dim * num_bins

        # 视觉编码器（和MinimalVLA相同）
        from model import MiniViT
        self.vision_encoder = MiniViT(img_size, patch_size, embed_dim, num_heads=4, num_layers=3)
        self.vision_proj = nn.Linear(embed_dim, embed_dim)

        # 扩展词表：语言Token + 动作Token共用一个Embedding
        self.token_embed = nn.Embedding(total_vocab, embed_dim)
        self.pos_embed = nn.Embedding(256, embed_dim)

        # 统一的Transformer（语言+动作一起生成）
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(embed_dim)

        # 输出头：预测整个（扩展后的）词表
        self.lm_head = nn.Linear(embed_dim, total_vocab)

        self.action_repr = RT2ActionRepresentation(base_vocab_size, action_dim, num_bins)
        print(f"[RT2StyleVLA] 总词表: {total_vocab:,} (语言:{base_vocab_size} + 动作:{action_dim*num_bins})")

    def encode_vision(self, images):
        feats = self.vision_encoder(images)  # [B, n_patches, D]
        return self.vision_proj(feats)

    def forward(self, images, text_tokens, target_tokens=None):
        """
        images: [B, 3, H, W]
        text_tokens: [B, L_text] 语言指令Token IDs
        target_tokens: [B, L_target] 目标序列（语言回答+动作Token），训练时用
        """
        B = images.shape[0]

        # 视觉上下文
        visual_memory = self.encode_vision(images)  # [B, n_patches, D]

        if target_tokens is None:
            # 推理：只用语言Token
            target_tokens = text_tokens

        L = target_tokens.shape[1]
        pos = torch.arange(L, device=images.device).unsqueeze(0)
        tgt = self.token_embed(target_tokens) + self.pos_embed(pos)

        # 因果掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=images.device)

        out = self.transformer(tgt, visual_memory, tgt_mask=tgt_mask)
        logits = self.lm_head(self.norm(out))  # [B, L, total_vocab]

        return logits

    @torch.no_grad()
    def generate_action(self, images, lang_tokens, max_new_tokens=7):
        """自回归生成动作Token（和文本生成完全一样！）"""
        self.eval()
        B = images.shape[0]
        visual_memory = self.encode_vision(images)

        # 以语言Token作为输入的开始
        generated = lang_tokens.clone()

        action_token_ids = []
        for step in range(max_new_tokens):
            L = generated.shape[1]
            pos = torch.arange(L, device=images.device).unsqueeze(0)
            tgt = self.token_embed(generated) + self.pos_embed(pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=images.device)

            out = self.transformer(tgt, visual_memory, tgt_mask=tgt_mask)
            logits = self.lm_head(self.norm(out))

            # 只从动作Token范围内采样（约束解码）
            action_dim_start = self.base_vocab_size + step * self.num_bins
            action_dim_end = action_dim_start + self.num_bins

            action_logits = logits[:, -1, action_dim_start:action_dim_end]
            next_token_local = action_logits.argmax(dim=-1)  # [B]
            next_token_global = next_token_local + action_dim_start  # 转回全局ID

            action_token_ids.append(next_token_global)
            generated = torch.cat([generated, next_token_global.unsqueeze(1)], dim=1)

        # 解码动作
        action_ids = torch.stack(action_token_ids, dim=1).cpu().numpy()  # [B, 7]
        actions = []
        for b in range(B):
            action = self.action_repr.decode_token_sequence(action_ids[b].tolist())
            actions.append(action)

        return np.array(actions), action_ids


# ─────────────────────────────────────────────
# 可视化 RT-2 vs 传统VLA
# ─────────────────────────────────────────────

def visualize_rt2_vs_traditional(repr):
    """对比RT-2和传统VLA的架构差异"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("RT-2 vs 传统VLA 架构对比", fontsize=14, fontweight='bold')

    def draw_box(ax, x, y, w, h, text, color, fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='gray',
                              linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, label, fontsize=7, color='gray')

    # ── 左图：传统VLA（MinimalVLA/OpenVLA）
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title("传统VLA (OpenVLA风格)\n独立动作解码器", fontsize=11)

    draw_box(ax, 1, 10.5, 8, 1.0, "输入: 图像 + 语言指令", '#E3F2FD', 10)

    draw_box(ax, 1, 8.5, 3.5, 1.5, "视觉编码器\n(ViT)", '#BBDEFB')
    draw_box(ax, 5.5, 8.5, 3.5, 1.5, "语言编码器\n(LLM)", '#C8E6C9')

    draw_arrow(ax, 2.75, 10.5, 2.75, 10.0)
    draw_arrow(ax, 7.25, 10.5, 7.25, 10.0)

    draw_box(ax, 1, 6.5, 8, 1.5, "视觉Token + 语言Token 拼接\n→ Transformer", '#FFF9C4')
    draw_arrow(ax, 2.75, 8.5, 2.75, 8.0)
    draw_arrow(ax, 7.25, 8.5, 7.25, 8.0)

    draw_box(ax, 1, 4.5, 8, 1.5, "动作解码器\n（独立MLP/Transformer）", '#FFE0B2')
    draw_arrow(ax, 5, 6.5, 5, 6.0)

    draw_box(ax, 1, 2.8, 8, 1.2, "动作Token预测\n词表大小: 256 (per dim)", '#FFCCBC')
    draw_arrow(ax, 5, 4.5, 5, 4.0)

    draw_box(ax, 1, 1.0, 8, 1.5, "输出: [0.3, -0.1, 0.5, 0.0, 0.8, -0.2, 1.0]\n7维连续动作", '#F8BBD9', 9)
    draw_arrow(ax, 5, 2.8, 5, 2.5)
    ax.axis('off')

    # ── 右图：RT-2
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title("RT-2\n动作 = 文本Token（无独立解码器）", fontsize=11)

    draw_box(ax, 1, 10.5, 8, 1.0, "输入: 图像 + '...Output: '", '#E3F2FD', 10)

    draw_box(ax, 1, 8.5, 3.5, 1.5, "视觉编码器\n(ViT)", '#BBDEFB')
    draw_arrow(ax, 2.75, 10.5, 2.75, 10.0)

    draw_box(ax, 1, 5.5, 8, 2.5,
             "统一LLM\n(词表 = 语言词表 + 动作Token)\n\n词表大小: 256000 + 7×256\n       原词表    +  新增动作Token",
             '#DCEDC8', 9)
    draw_arrow(ax, 2.75, 8.5, 2.75, 8.0)
    draw_arrow(ax, 7.25, 10.5, 7.25, 8.0)

    draw_box(ax, 1, 3.5, 8, 1.5,
             "自回归生成（和文本生成完全相同！）\n'<act_1153> <act_2137> <act_2761>...'",
             '#FFF9C4', 8)
    draw_arrow(ax, 5, 5.5, 5, 5.0)

    draw_box(ax, 1, 1.8, 8, 1.2, "Token解码 → [0.3, -0.1, 0.5, 0.0, 0.8, -0.2, 1.0]", '#FFCCBC', 9)
    draw_arrow(ax, 5, 3.5, 5, 3.0)

    # 优势说明
    ax.text(5, 0.7, "✓ 无需额外动作头  ✓ 可直接用预训练LLM  ✓ 零样本泛化能力强",
            ha='center', fontsize=8, color='green',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("assets/07_rt2_vs_traditional.png", dpi=150, bbox_inches='tight')
    print("RT-2对比图已保存: assets/07_rt2_vs_traditional.png")
    plt.show()


def demo_rt2_model():
    """演示RT2StyleVLA的前向传播"""
    print("\n" + "=" * 60)
    print("  RT-2 风格模型前向传播演示")
    print("=" * 60)

    model = RT2StyleVLA(
        img_size=64, patch_size=8, embed_dim=128,
        base_vocab_size=200, action_dim=7, num_bins=64  # 小词表便于CPU测试
    )

    B = 2
    images = torch.randn(B, 3, 64, 64)
    lang_tokens = torch.randint(0, 200, (B, 6))  # 6个语言Token
    target_tokens = torch.randint(0, 200 + 7 * 64, (B, 9))  # 语言+动作Token

    with torch.no_grad():
        logits = model(images, lang_tokens, target_tokens)
        print(f"\n输入: 图像{list(images.shape)}, 语言{list(lang_tokens.shape)}")
        print(f"输出logits: {list(logits.shape)}")
        print(f"  → [batch={B}, seq_len=9, vocab={200+7*64}]")

        # 自回归生成
        actions, action_ids = model.generate_action(images, lang_tokens)
        print(f"\n生成的动作（自回归）:")
        print(f"  Token IDs: {action_ids[0]}")
        print(f"  连续动作: {actions[0].round(3)}")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    # RT-2 输入格式演示
    repr, action, action_tokens = demo_rt2_input_format()

    # 架构对比可视化
    visualize_rt2_vs_traditional(repr)

    # RT-2 模型演示
    demo_rt2_model()

    print("\n" + "=" * 60)
    print("  RT-2 核心要点总结")
    print("=" * 60)
    print("1. 动作Token嵌入到LLM词表：扩展词表实现统一处理")
    print("2. 和文本生成完全相同的框架：自回归预测下一个Token")
    print("3. 约束解码：生成动作位置时，只从动作Token范围采样")
    print("4. 优势：直接利用预训练LLM的语义理解能力")
    print("5. RT-2 = PaLI-X/PaLM-E (视觉语言模型) + 动作Token微调")
