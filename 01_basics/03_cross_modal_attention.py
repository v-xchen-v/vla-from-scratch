"""
VLA入门 第3课：跨模态注意力（Cross-Modal Attention）
====================================================
VLA的大脑核心：让视觉Token和语言Token互相"对话"

标准Transformer使用自注意力（Self-Attention）
VLA中需要让视觉和语言信息融合，有两种主要方式：
  1. 早期融合（Early Fusion）：拼接后一起做Self-Attention（LLaVA/OpenVLA方式）
  2. 交叉注意力（Cross-Attention）：语言作Query，视觉作Key-Value（Flamingo方式）

运行方式：
    python 01_basics/03_cross_modal_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


# ─────────────────────────────────────────────
# 1. 基础多头注意力
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    标准多头注意力，展示注意力权重用于可视化
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, return_attn=False):
        B, Lq, D = query.shape
        Lkv = key.shape[1]

        # 投影 + 分头
        Q = self.q_proj(query).reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力权重
        attn = (Q @ K.transpose(-2, -1)) / self.scale  # [B, heads, Lq, Lkv]
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = (attn @ V).transpose(1, 2).reshape(B, Lq, D)
        out = self.out_proj(out)

        if return_attn:
            return out, attn
        return out


# ─────────────────────────────────────────────
# 2. 方式一：早期融合（LLaVA/OpenVLA方式）
# ─────────────────────────────────────────────

class EarlyFusionBlock(nn.Module):
    """
    OpenVLA/LLaVA 的做法：
    视觉Token + 语言Token 直接拼接 → Self-Attention

    优点：简单，利用预训练LLM的全部能力
    缺点：序列变长，计算量大
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, visual_tokens, language_tokens):
        # 拼接：[视觉Token, 语言Token]
        combined = torch.cat([visual_tokens, language_tokens], dim=1)

        # Self-Attention（视觉和语言互相关注）
        attn_out, attn_weights = self.self_attn(
            self.norm1(combined), self.norm1(combined), self.norm1(combined),
            return_attn=True
        )
        combined = combined + attn_out
        combined = combined + self.ffn(self.norm2(combined))

        # 分开
        n_visual = visual_tokens.shape[1]
        return combined[:, :n_visual], combined[:, n_visual:], attn_weights


# ─────────────────────────────────────────────
# 3. 方式二：交叉注意力（Flamingo方式）
# ─────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Flamingo / Otter 的做法：
    语言Token作为Query，视觉Token作为Key-Value
    语言"主动询问"视觉：'我现在说的这个词，对应图像的哪个区域？'

    优点：不增加序列长度，视觉信息插入更灵活
    缺点：需要额外的Cross-Attention层
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_self = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, language_tokens, visual_tokens):
        # Cross-Attention：语言关注视觉
        cross_out, cross_weights = self.cross_attn(
            self.norm_q(language_tokens),
            self.norm_kv(visual_tokens),
            self.norm_kv(visual_tokens),
            return_attn=True
        )
        language_tokens = language_tokens + cross_out

        # Self-Attention：语言内部关注
        self_out, _ = self.self_attn(
            self.norm_self(language_tokens),
            self.norm_self(language_tokens),
            self.norm_self(language_tokens),
            return_attn=True
        )
        language_tokens = language_tokens + self_out
        language_tokens = language_tokens + self.ffn(self.norm_ffn(language_tokens))

        return language_tokens, cross_weights


# ─────────────────────────────────────────────
# 4. 演示和可视化
# ─────────────────────────────────────────────

def demo_attention_patterns():
    print("=" * 60)
    print("  跨模态注意力演示")
    print("=" * 60)

    embed_dim = 64
    num_heads = 4
    n_visual = 16  # 4×4 Patch
    n_lang = 6     # "pick up the red apple"

    # 模拟输入
    torch.manual_seed(42)
    visual_tokens = torch.randn(1, n_visual, embed_dim)
    language_tokens = torch.randn(1, n_lang, embed_dim)

    print(f"\n视觉Token: {list(visual_tokens.shape)}  ({n_visual}个Patch)")
    print(f"语言Token: {list(language_tokens.shape)}  ({n_lang}个词)")

    # ── 方式一：早期融合
    print(f"\n方式一（早期融合 - OpenVLA方式）:")
    early_fusion = EarlyFusionBlock(embed_dim, num_heads)
    with torch.no_grad():
        v_out, l_out, attn_w = early_fusion(visual_tokens, language_tokens)

    print(f"  拼接后序列长度: {n_visual + n_lang}")
    print(f"  注意力矩阵大小: {list(attn_w.shape)}  [batch, heads, seq, seq]")
    print(f"  视觉输出: {list(v_out.shape)}")
    print(f"  语言输出: {list(l_out.shape)}")

    # 取第一个head的注意力权重
    early_attn_map = attn_w[0].mean(0).detach().numpy()  # [seq, seq]

    # ── 方式二：交叉注意力
    print(f"\n方式二（交叉注意力 - Flamingo方式）:")
    cross_attn = CrossAttentionBlock(embed_dim, num_heads)
    with torch.no_grad():
        l_out_cross, cross_w = cross_attn(language_tokens, visual_tokens)

    print(f"  语言Token 关注 视觉Token")
    print(f"  Cross-Attention矩阵: {list(cross_w.shape)}  [batch, heads, lang, visual]")
    print(f"  语言输出: {list(l_out_cross.shape)}")

    cross_attn_map = cross_w[0].mean(0).detach().numpy()  # [n_lang, n_visual]

    return early_attn_map, cross_attn_map, n_visual, n_lang


def visualize_attention(early_attn_map, cross_attn_map, n_visual, n_lang):
    """可视化注意力模式"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VLA跨模态注意力可视化", fontsize=13, fontweight='bold')

    lang_labels = ["pick", "up", "the", "red", "apple", "<act>"]
    visual_labels = [f"P{i}" for i in range(n_visual)]

    # 图1：早期融合的注意力矩阵
    ax1 = axes[0]
    all_labels = [f"V{i}" for i in range(n_visual)] + lang_labels
    im1 = ax1.imshow(early_attn_map, cmap='Blues', aspect='auto')
    ax1.set_title("早期融合 Self-Attention\n(OpenVLA/LLaVA 方式)", fontsize=10)
    ax1.set_xlabel("Key (被关注的Token)")
    ax1.set_ylabel("Query (正在关注的Token)")
    ax1.set_xticks(range(len(all_labels)))
    ax1.set_yticks(range(len(all_labels)))
    ax1.set_xticklabels(all_labels, rotation=45, fontsize=6)
    ax1.set_yticklabels(all_labels, fontsize=6)

    # 标注视觉区域和语言区域
    rect_v = plt.Rectangle((-0.5, -0.5), n_visual, n_visual,
                             fill=False, edgecolor='green', linewidth=2)
    rect_l = plt.Rectangle((n_visual - 0.5, n_visual - 0.5), n_lang, n_lang,
                             fill=False, edgecolor='orange', linewidth=2)
    rect_vl = plt.Rectangle((n_visual - 0.5, -0.5), n_lang, n_visual,
                              fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect_v)
    ax1.add_patch(rect_l)
    ax1.add_patch(rect_vl)
    ax1.text(n_visual // 2, -1.5, "视觉→视觉", ha='center', color='green', fontsize=7)
    ax1.text(n_visual + n_lang // 2, -1.5, "语言→语言", ha='center', color='orange', fontsize=7)
    plt.colorbar(im1, ax=ax1)

    # 图2：交叉注意力矩阵
    ax2 = axes[1]
    im2 = ax2.imshow(cross_attn_map, cmap='Reds', aspect='auto')
    ax2.set_title("交叉注意力 Cross-Attention\n(Flamingo方式：语言查询视觉)", fontsize=10)
    ax2.set_xlabel("Key: 视觉Patch位置")
    ax2.set_ylabel("Query: 语言Token")
    ax2.set_xticks(range(n_visual))
    ax2.set_yticks(range(n_lang))
    ax2.set_xticklabels(visual_labels, rotation=45, fontsize=7)
    ax2.set_yticklabels(lang_labels, fontsize=9)
    plt.colorbar(im2, ax=ax2)

    # 图3：概念图解
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_title("VLA架构对比", fontsize=10)

    # OpenVLA方式
    ax3.text(2.5, 9.2, "OpenVLA / LLaVA", ha='center', fontsize=9, fontweight='bold')
    ax3.add_patch(plt.Rectangle((0.5, 7.5), 4, 1.4, facecolor='lightgreen', alpha=0.7))
    ax3.text(2.5, 8.2, "视觉Token + 语言Token", ha='center', fontsize=8)
    ax3.text(2.5, 7.8, "↓ 拼接 + Self-Attention", ha='center', fontsize=7, style='italic')
    ax3.annotate("", xy=(2.5, 6.5), xytext=(2.5, 7.5),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.add_patch(plt.Rectangle((0.5, 5.5), 4, 0.9, facecolor='lightblue', alpha=0.7))
    ax3.text(2.5, 5.95, "LLM (预测动作Token)", ha='center', fontsize=8)

    # Flamingo方式
    ax3.text(7.5, 9.2, "Flamingo / Otter", ha='center', fontsize=9, fontweight='bold')
    ax3.add_patch(plt.Rectangle((5.5, 7.5), 4, 1.4, facecolor='lightyellow', alpha=0.7))
    ax3.text(7.5, 8.2, "视觉Token (固定)", ha='center', fontsize=8)
    ax3.text(7.5, 7.8, "↓ Cross-Attention", ha='center', fontsize=7, style='italic')
    ax3.add_patch(plt.Rectangle((5.5, 6.0), 4, 1.2, facecolor='lightcoral', alpha=0.7))
    ax3.text(7.5, 6.6, "语言Token", ha='center', fontsize=8)
    ax3.text(7.5, 6.2, "↓ Self-Attention", ha='center', fontsize=7, style='italic')
    ax3.annotate("", xy=(7.5, 5.0), xytext=(7.5, 6.0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.add_patch(plt.Rectangle((5.5, 4.1), 4, 0.8, facecolor='lightblue', alpha=0.7))
    ax3.text(7.5, 4.5, "LLM (预测动作Token)", ha='center', fontsize=8)

    # 共同特征
    ax3.text(5, 3.0, "共同点：最终都预测动作Token [x,y,z,rx,ry,rz,grip]",
             ha='center', fontsize=8, color='purple',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig("assets/03_cross_modal_attention.png", dpi=150, bbox_inches='tight')
    print("图像已保存: assets/03_cross_modal_attention.png")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    early_attn_map, cross_attn_map, n_visual, n_lang = demo_attention_patterns()
    visualize_attention(early_attn_map, cross_attn_map, n_visual, n_lang)

    print("\n关键要点:")
    print("1. Self-Attention（早期融合）：视觉和语言Token拼一起，OpenVLA/LLaVA用这种")
    print("2. Cross-Attention：语言向视觉'询问'信息，Flamingo用这种")
    print("3. 现代VLA（OpenVLA）基本都用早期融合，因为能直接用LLM的预训练权重")
    print("4. 注意力权重揭示了'apple'这个词主要关注图像中苹果所在的Patch区域")
