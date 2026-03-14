"""
VLA入门 第2课：视觉编码器（Vision Encoder）
==========================================
VLA的眼睛：把图像转化为Token序列

VLA中的视觉编码器通常是 ViT（Vision Transformer）
把图像切成小块（Patch），每个Patch变成一个视觉Token
然后这些视觉Token和语言Token一起送入LLM！

运行方式：
    python 01_basics/02_vision_encoder.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from einops import rearrange


# ─────────────────────────────────────────────
# 1. Patch Embedding（图像→Patch Token）
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    把图像切成固定大小的Patch，然后投影成Token向量

    例如：224×224图像，每个Patch 16×16
         → (224/16)² = 196 个 Patch Token
         → 每个Token维度 = embed_dim

    这是ViT和VLA视觉编码器的第一步！
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 用卷积实现Patch切割+线性投影（等价于：把每个Patch展平后做线性变换）
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        print(f"[PatchEmbedding] {img_size}×{img_size} 图像")
        print(f"  Patch大小: {patch_size}×{patch_size}")
        print(f"  Patch数量: {self.num_patches}")
        print(f"  每个Token维度: {embed_dim}")

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)           # [B, embed_dim, H/P, W/P]
        x = rearrange(x, 'b d h w -> b (h w) d')  # [B, num_patches, embed_dim]
        return x


# ─────────────────────────────────────────────
# 2. 位置编码（Positional Encoding）
# ─────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """
    给每个Patch Token加上位置信息
    让模型知道"这个Patch在图像的哪个位置"
    """

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # 可学习的位置编码（ViT原版做法）
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        # +1 是因为还有一个 [CLS] Token

    def forward(self, x):
        return x + self.pos_embed[:, 1:, :]  # 跳过CLS Token的位置编码


# ─────────────────────────────────────────────
# 3. 简化版 ViT 视觉编码器
# ─────────────────────────────────────────────

class SimpleViTEncoder(nn.Module):
    """
    简化版Vision Transformer编码器
    VLA中通常使用预训练的ViT-B/16或ViT-L/14（CLIP版本）

    结构：
      图像 → Patch Embedding → [CLS] Token拼接 → Transformer Layers → 视觉特征
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=256,
                 num_heads=8, num_layers=6, in_channels=3):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS Token：代表整张图像的全局特征
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True  # Pre-LN更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch Embedding
        patch_tokens = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 2. 拼接 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # [B, num_patches+1, embed_dim]

        # 3. 加位置编码
        tokens = tokens + self.pos_embed

        # 4. Transformer编码
        features = self.transformer(tokens)
        features = self.norm(features)

        # 返回：[CLS特征, Patch特征序列]
        cls_feature = features[:, 0]            # [B, embed_dim] 全局特征
        patch_features = features[:, 1:]        # [B, num_patches, embed_dim] 局部特征

        return cls_feature, patch_features


# ─────────────────────────────────────────────
# 4. VLA中的视觉-语言接口
# ─────────────────────────────────────────────

class VisionProjector(nn.Module):
    """
    把视觉Token投影到LLM的词嵌入空间

    这是VLA的关键桥梁：
      视觉特征 (visual_dim) → LLM词嵌入维度 (llm_dim)

    LLaVA / OpenVLA 用的就是这种 MLP Projector！
    """

    def __init__(self, visual_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, visual_tokens):
        return self.proj(visual_tokens)


# ─────────────────────────────────────────────
# 5. 演示
# ─────────────────────────────────────────────

def demo_vision_encoder():
    print("=" * 60)
    print("  VLA视觉编码器演示")
    print("=" * 60)

    # 创建编码器（小尺寸，CPU可运行）
    encoder = SimpleViTEncoder(
        img_size=64,     # 简化为64×64（实际VLA用224×224）
        patch_size=8,    # 8×8 Patch
        embed_dim=128,   # 嵌入维度
        num_heads=4,
        num_layers=3
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 模拟机器人摄像头拍到的图像（batch_size=2）
    batch_size = 2
    images = torch.randn(batch_size, 3, 64, 64)  # [B, C, H, W]
    print(f"\n输入图像形状: {list(images.shape)}")

    with torch.no_grad():
        cls_feat, patch_feats = encoder(images)

    print(f"\n输出:")
    print(f"  CLS全局特征: {list(cls_feat.shape)}  ← 代表整张图像")
    print(f"  Patch局部特征: {list(patch_feats.shape)}  ← 每个Patch的特征")
    print(f"    (batch={batch_size}, num_patches={(64//8)**2}, embed_dim=128)")

    # 视觉→语言投影
    projector = VisionProjector(visual_dim=128, llm_dim=512)
    llm_visual_tokens = projector(patch_feats)
    print(f"\n投影到LLM空间后:")
    print(f"  视觉Token形状: {list(llm_visual_tokens.shape)}")
    print(f"  → 这{(64//8)**2}个视觉Token将和语言Token拼接，一起送入LLM！")

    return encoder, images


def visualize_patching():
    """可视化Patch切割过程"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("VLA视觉处理：图像 → Patch Token 序列", fontsize=13, fontweight='bold')

    # 模拟机器人看到的场景（用随机颜色块代表物体）
    np.random.seed(42)
    img = np.ones((64, 64, 3)) * 0.9  # 白色背景

    # 画桌面
    img[40:64, :] = [0.6, 0.4, 0.2]  # 棕色桌面

    # 画一个"苹果"（红色圆）
    for y in range(20, 40):
        for x in range(20, 45):
            if (x - 32) ** 2 + (y - 30) ** 2 < 100:
                img[y, x] = [0.9, 0.1, 0.1]  # 红色

    # 画一个"杯子"（蓝色矩形）
    img[25:42, 45:58] = [0.1, 0.3, 0.9]  # 蓝色

    # 图1：原始图像
    axes[0].imshow(img)
    axes[0].set_title("原始图像\n(机器人摄像头拍摄)", fontsize=11)
    axes[0].axis('off')

    # 图2：切成Patch
    axes[1].imshow(img)
    patch_size = 8
    num_patches_per_side = 64 // patch_size
    for i in range(num_patches_per_side + 1):
        axes[1].axhline(i * patch_size, color='yellow', linewidth=1, alpha=0.8)
        axes[1].axvline(i * patch_size, color='yellow', linewidth=1, alpha=0.8)

    # 标注一些Patch编号
    for py in range(num_patches_per_side):
        for px in range(num_patches_per_side):
            patch_idx = py * num_patches_per_side + px
            if patch_idx % 8 == 0:  # 只标注部分
                axes[1].text(px * patch_size + patch_size // 2,
                             py * patch_size + patch_size // 2,
                             str(patch_idx), fontsize=5,
                             ha='center', va='center', color='yellow')

    axes[1].set_title(f"切割成Patch\n({num_patches_per_side}×{num_patches_per_side}={num_patches_per_side**2}个Patch)", fontsize=11)
    axes[1].axis('off')

    # 图3：Token序列示意
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, num_patches_per_side ** 2 + 2)
    ax3.set_title(f"Patch Token序列\n(送入Transformer)", fontsize=11)

    # 展示前10个Token
    token_colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']
    for i, color in enumerate(token_colors):
        rect = patches.FancyBboxPatch((1, i * 0.85 + 0.1), 8, 0.7,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='gray', alpha=0.8)
        ax3.add_patch(rect)
        ax3.text(5, i * 0.85 + 0.45, f"Token #{i+1}: [0.23, -0.11, ..., 0.87]",
                 ha='center', va='center', fontsize=7)

    ax3.text(5, 10.5, "... (共64个Token)", ha='center', va='center',
             fontsize=9, style='italic', color='gray')

    # CLS Token
    rect_cls = patches.FancyBboxPatch((1, 11.2), 8, 0.7,
                                       boxstyle="round,pad=0.05",
                                       facecolor='gold', edgecolor='orange', linewidth=2)
    ax3.add_patch(rect_cls)
    ax3.text(5, 11.55, "[CLS] Token: 全局图像特征", ha='center', va='center',
             fontsize=8, fontweight='bold')

    ax3.axis('off')

    plt.tight_layout()
    plt.savefig("assets/02_vision_encoder.png", dpi=150, bbox_inches='tight')
    print("\n图像已保存: assets/02_vision_encoder.png")
    plt.show()


def demo_vla_input_format():
    """演示VLA完整输入格式"""
    print("\n" + "=" * 60)
    print("  VLA输入格式：视觉Token + 语言Token + 动作Token")
    print("=" * 60)
    print("""
VLA输入序列结构（类比GPT的输入）：

  [视觉Token×64]  [语言Token序列]          [动作Token×7]
  ───────────     ──────────────────────   ────────────
  <img_0>         Pick  up  the  apple     128 89 201 0 180 130 255
  <img_1>         →                        ↑   ↑   ↑  ↑   ↑   ↑   ↑
  ...             语言指令                  x   y   z  rx  ry  rz  grip
  <img_63>

  完整序列长度 = 64(视觉) + N(语言) + 7(动作) 个Token

  LLM的任务：给定视觉+语言，自回归预测7个动作Token！
    """)


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    encoder, images = demo_vision_encoder()
    visualize_patching()
    demo_vla_input_format()

    print("\n关键要点:")
    print("1. ViT把图像切成Patch，每个Patch变成一个Token（类比NLP里的词）")
    print("2. 视觉Token通过MLP Projector投影到LLM的嵌入空间")
    print("3. VLA = 视觉Token + 语言Token + 动作Token → 一个统一的序列")
    print("4. 实际VLA（如OpenVLA）用预训练CLIP ViT-L/14，不从头训练视觉编码器")
