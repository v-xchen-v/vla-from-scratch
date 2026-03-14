"""
最小化VLA模型（Minimal VLA）
============================
把前三课学到的知识整合成一个完整可训练的VLA模型

架构：
  视觉编码器（Mini ViT）→ MLP Projector
         ↓
  [视觉Token, 语言Token] → Transformer LLM → 动作Token预测

这是OpenVLA核心架构的简化版本，可在CPU上运行。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# ─────────────────────────────────────────────
# 视觉编码器
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b d h w -> b (h w) d')


class MiniViT(nn.Module):
    """轻量级视觉编码器"""

    def __init__(self, img_size=64, patch_size=8, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = num_patches

    def forward(self, x):
        tokens = self.patch_embed(x) + self.pos_embed
        return self.norm(self.transformer(tokens))  # [B, num_patches, embed_dim]


# ─────────────────────────────────────────────
# 动作离散化器
# ─────────────────────────────────────────────

class ActionTokenizer:
    """连续动作 ↔ 离散Token"""

    def __init__(self, action_dim=7, num_bins=256):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.bins = torch.linspace(-1, 1, num_bins + 1)

    def encode(self, action: torch.Tensor) -> torch.Tensor:
        """连续动作 → Token ID  [B, action_dim] → [B, action_dim]"""
        bins = self.bins.to(action.device)
        return torch.bucketize(action, bins[1:-1])  # [0, num_bins-1]

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token ID → 连续动作"""
        bin_centers = ((self.bins[:-1] + self.bins[1:]) / 2).to(token_ids.device)
        return bin_centers[token_ids]


# ─────────────────────────────────────────────
# 语言编码器（简化版Word Embedding）
# ─────────────────────────────────────────────

class SimpleLanguageEncoder(nn.Module):
    """
    简化语言编码器（实际VLA用LLaMA/Mistral等预训练LLM）
    这里用简单的词嵌入+位置编码模拟
    """

    def __init__(self, vocab_size=1000, embed_dim=128, max_seq_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        return self.norm(self.embed(token_ids) + self.pos_embed(positions))


# ─────────────────────────────────────────────
# 核心：VLA解码器（自回归预测动作Token）
# ─────────────────────────────────────────────

class VLADecoder(nn.Module):
    """
    自回归Transformer解码器
    给定[视觉Token, 语言Token]，预测动作Token序列

    使用因果掩码（Causal Mask）保证自回归生成
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=4,
                 vocab_size=1000, action_vocab_size=256, action_dim=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_vocab_size = action_vocab_size

        # 动作Token嵌入
        self.action_embed = nn.Embedding(action_vocab_size, embed_dim)
        self.action_pos_embed = nn.Embedding(action_dim, embed_dim)

        # Transformer层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # 输出头：预测每个动作维度的Token概率
        self.output_head = nn.Linear(embed_dim, action_vocab_size)

    def forward(self, memory, action_tokens=None):
        """
        memory: [B, seq_len, embed_dim]  视觉+语言Token
        action_tokens: [B, t] 已生成的动作Token（训练时用ground truth）

        返回 logits: [B, t, action_vocab_size]
        """
        B = memory.shape[0]

        if action_tokens is None:
            # 推理时：teacher forcing off，使用BOS Token开始
            action_tokens = torch.zeros(B, 1, dtype=torch.long, device=memory.device)

        L = action_tokens.shape[1]
        positions = torch.arange(L, device=memory.device).unsqueeze(0)

        # 动作Token嵌入 + 位置编码
        tgt = self.action_embed(action_tokens) + self.action_pos_embed(
            positions.clamp(max=self.action_dim - 1)
        )

        # 因果掩码（不能看到未来的动作Token）
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=memory.device)

        # 解码
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        out = self.norm(out)
        logits = self.output_head(out)  # [B, L, action_vocab_size]

        return logits


# ─────────────────────────────────────────────
# 完整VLA模型
# ─────────────────────────────────────────────

class MinimalVLA(nn.Module):
    """
    最小化VLA（Minimal Vision-Language-Action）模型

    输入：RGB图像 + 语言指令Token IDs
    输出：预测的动作Token logits（7维，每维256类）

    对应真实VLA：
      MiniViT → CLIP ViT-L/14
      SimpleLanguageEncoder → LLaMA-2 / Mistral-7B
      VLADecoder → 预训练LLM + 动作输出头
    """

    def __init__(self, img_size=64, patch_size=8, embed_dim=128,
                 vocab_size=512, action_dim=7, action_bins=256):
        super().__init__()
        self.action_dim = action_dim
        self.action_bins = action_bins

        # 各模块
        self.vision_encoder = MiniViT(img_size, patch_size, embed_dim)
        self.vision_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.language_encoder = SimpleLanguageEncoder(vocab_size, embed_dim)
        self.decoder = VLADecoder(embed_dim, num_heads=4, num_layers=4,
                                   action_vocab_size=action_bins,
                                   action_dim=action_dim)
        self.action_tokenizer = ActionTokenizer(action_dim, action_bins)

    def encode_context(self, images, lang_tokens):
        """编码视觉+语言上下文"""
        # 视觉编码
        visual_feats = self.vision_encoder(images)          # [B, n_patches, D]
        visual_feats = self.vision_projector(visual_feats)  # [B, n_patches, D]

        # 语言编码
        lang_feats = self.language_encoder(lang_tokens)     # [B, L_lang, D]

        # 早期融合：拼接视觉+语言Token
        context = torch.cat([visual_feats, lang_feats], dim=1)  # [B, n_patches+L_lang, D]
        return context

    def forward(self, images, lang_tokens, action_tokens=None):
        """
        训练时：提供 action_tokens（teacher forcing）
        推理时：action_tokens=None，自回归生成

        返回：logits [B, action_dim, action_bins]
        """
        context = self.encode_context(images, lang_tokens)
        logits = self.decoder(context, action_tokens)  # [B, L, action_bins]
        return logits

    @torch.no_grad()
    def predict_action(self, images, lang_tokens):
        """推理：给定图像+语言，预测7维动作"""
        self.eval()
        context = self.encode_context(images, lang_tokens)

        B = images.shape[0]
        action_token_ids = []

        # 自回归生成 action_dim 个Token
        generated = torch.zeros(B, 1, dtype=torch.long, device=images.device)
        for step in range(self.action_dim):
            logits = self.decoder(context, generated)   # [B, step+1, action_bins]
            next_token = logits[:, -1].argmax(dim=-1)   # 贪心解码 [B]
            action_token_ids.append(next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        action_ids = torch.stack(action_token_ids, dim=1)  # [B, action_dim]
        continuous_action = self.action_tokenizer.decode(action_ids)  # [B, action_dim]

        return continuous_action, action_ids

    def compute_loss(self, images, lang_tokens, target_actions):
        """
        计算训练损失

        target_actions: [B, action_dim] 连续动作值 ([-1, 1])
        """
        # 将目标动作离散化
        target_ids = self.action_tokenizer.encode(target_actions)  # [B, action_dim]

        # Teacher forcing：用ground truth action token作为decoder输入
        # 输入：[BOS, a_0, a_1, ..., a_5]（最后一个不输入）
        bos = torch.zeros(target_ids.shape[0], 1, dtype=torch.long, device=target_ids.device)
        decoder_input = torch.cat([bos, target_ids[:, :-1]], dim=1)  # [B, action_dim]

        # 前向传播
        logits = self.forward(images, lang_tokens, decoder_input)  # [B, action_dim, bins]

        # 交叉熵损失（每个维度独立预测）
        loss = F.cross_entropy(
            logits.reshape(-1, self.action_bins),
            target_ids.reshape(-1)
        )
        return loss


# ─────────────────────────────────────────────
# 模型信息工具
# ─────────────────────────────────────────────

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model):
    total, trainable = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  MinimalVLA 模型摘要")
    print(f"{'='*50}")
    print(f"  总参数量:      {total:>10,} ({total/1e6:.2f}M)")
    print(f"  可训练参数量:  {trainable:>10,} ({trainable/1e6:.2f}M)")
    print(f"{'='*50}")

    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<25}: {n_params:>8,} params")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 快速测试
    model = MinimalVLA()
    print_model_summary(model)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    lang = torch.randint(0, 512, (B, 8))
    actions = torch.FloatTensor(B, 7).uniform_(-1, 1)

    # 测试损失计算
    loss = model.compute_loss(images, lang, actions)
    print(f"训练损失: {loss.item():.4f}")

    # 测试推理
    pred_action, pred_ids = model.predict_action(images, lang)
    print(f"预测动作形状: {list(pred_action.shape)}")
    print(f"预测动作: {pred_action[0].numpy().round(3)}")
    print("模型测试通过！")
