"""
MinimalVLA 训练脚本
===================
在合成数据集上训练VLA模型，理解训练流程

合成任务：
  - 图像：包含不同颜色/位置物体的64×64图像
  - 指令：'pick red'、'pick blue'、'pick green' 等
  - 动作：根据物体位置生成的7维动作（x,y,z,roll,pitch,yaw,gripper）

运行方式：
    python 02_minimal_vla/train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MinimalVLA, print_model_summary


# 默认的 DejaVu Sans 不含中文字形；优先使用系统中的中文字体。
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei',
    'SimHei', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans',
]


# ─────────────────────────────────────────────
# 合成机器人数据集
# ─────────────────────────────────────────────

class SyntheticRobotDataset(Dataset):
    """
    合成机器人抓取数据集

    任务："pick [color]"
    - 图像中有一个指定颜色的物体
    - 机器人需要移动到物体位置并抓取

    这是真实数据集（如LIBERO、OpenX-Embodiment）的简化版本
    """

    # 简单词表
    VOCAB = {
        '<pad>': 0, '<bos>': 1, '<eos>': 2,
        'pick': 3, 'up': 4, 'the': 5, 'object': 6,
        'red': 7, 'green': 8, 'blue': 9,
        'left': 10, 'right': 11, 'center': 12,
    }

    COLORS = {
        'red':   ([0.9, 0.1, 0.1], 7),
        'green': ([0.1, 0.8, 0.1], 8),
        'blue':  ([0.1, 0.3, 0.9], 9),
    }

    def __init__(self, size=2000, img_size=64, seed=42):
        self.size = size
        self.img_size = img_size
        np.random.seed(seed)

        # 预生成数据
        self.data = [self._generate_sample() for _ in range(size)]

    def _generate_sample(self):
        """生成一个(图像, 语言指令, 动作)样本"""
        img_size = self.img_size

        # 随机选择目标颜色和位置
        color_name = np.random.choice(['red', 'green', 'blue'])
        color_rgb, color_token = self.COLORS[color_name]

        # 物体在图像中的位置（归一化到[0.2, 0.8]）
        obj_x = np.random.uniform(0.2, 0.8)  # 图像X轴位置
        obj_y = np.random.uniform(0.2, 0.8)  # 图像Y轴位置

        # 生成图像
        image = self._render_scene(color_rgb, obj_x, obj_y, img_size)

        # 语言指令：'pick up the [color] object'
        lang_tokens = [
            self.VOCAB['pick'], self.VOCAB['up'],
            self.VOCAB['the'], color_token, self.VOCAB['object']
        ]
        # 填充到固定长度8
        lang_tokens = lang_tokens + [self.VOCAB['<pad>']] * (8 - len(lang_tokens))

        # 目标动作：根据物体位置生成
        # 将图像坐标[0,1]映射到机器人动作空间[-1,1]
        action = np.array([
            obj_x * 2 - 1,      # x: 物体水平位置
            obj_y * 2 - 1,      # y: 物体垂直位置（对应机器人前后）
            0.3,                 # z: 下降高度（固定）
            0.0,                 # roll
            0.15,                # pitch（向下倾斜抓取）
            0.0,                 # yaw
            -0.8,                # gripper: 关闭（-1=完全关闭）
        ], dtype=np.float32)

        return {
            'image': torch.FloatTensor(image),
            'lang': torch.LongTensor(lang_tokens),
            'action': torch.FloatTensor(action),
            'color': color_name,
            'obj_pos': (obj_x, obj_y),
        }

    def _render_scene(self, obj_color, obj_x, obj_y, size):
        """渲染一个简单的桌面场景"""
        img = np.ones((3, size, size), dtype=np.float32) * 0.85  # 灰色背景

        # 桌面
        img[:, size // 2:] = np.array([0.55, 0.35, 0.15]).reshape(3, 1, 1)

        # 物体（圆形）
        cx = int(obj_x * size)
        cy = int(obj_y * size)
        radius = size // 10

        for y in range(max(0, cy - radius), min(size, cy + radius)):
            for x in range(max(0, cx - radius), min(size, cx + radius)):
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    img[:, y, x] = obj_color

        # 添加少量噪声
        img += np.random.randn(*img.shape) * 0.02
        img = np.clip(img, 0, 1)
        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────

class VLATrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0
        for batch in loader:
            images = batch['image'].to(self.device)
            lang = batch['lang'].to(self.device)
            actions = batch['action'].to(self.device)

            optimizer.zero_grad()
            loss = self.model.compute_loss(images, lang, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        for batch in loader:
            images = batch['image'].to(self.device)
            lang = batch['lang'].to(self.device)
            actions = batch['action'].to(self.device)

            loss = self.model.compute_loss(images, lang, actions)
            pred_action, _ = self.model.predict_action(images, lang)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(pred_action - actions)).item()

        return total_loss / len(loader), total_mae / len(loader)

    def train(self, train_loader, val_loader, num_epochs=30, lr=1e-3):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

        print(f"\n{'='*55}")
        print(f"  开始训练 VLA 模型")
        print(f"  设备: {self.device}  轮次: {num_epochs}  学习率: {lr}")
        print(f"{'='*55}")

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, val_mae = self.evaluate(val_loader)
            scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "02_minimal_vla/best_model.pt")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{num_epochs}  "
                      f"Train Loss: {train_loss:.4f}  "
                      f"Val Loss: {val_loss:.4f}  "
                      f"Val MAE: {val_mae:.4f}")

        print(f"\n最佳验证损失: {best_val_loss:.4f}")
        print("模型已保存: 02_minimal_vla/best_model.pt")
        return self.train_losses, self.val_losses


# ─────────────────────────────────────────────
# 可视化训练结果
# ─────────────────────────────────────────────

def visualize_training(train_losses, val_losses):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MinimalVLA 训练结果", fontsize=13, fontweight='bold')

    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', color='#2196F3')
    axes[0].plot(val_losses, label='验证损失', color='#FF5722')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("训练/验证损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 理论最优损失
    theoretical_min = np.log(256)  # 随机猜测的损失 ≈ ln(256) ≈ 5.5
    axes[0].axhline(theoretical_min, color='gray', linestyle='--',
                    alpha=0.7, label=f'随机基线 (ln256≈{theoretical_min:.1f})')
    axes[0].legend()

    # 损失降幅
    improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    axes[1].bar(['初始损失', '最终损失'],
                [train_losses[0], train_losses[-1]],
                color=['#FF5722', '#4CAF50'])
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title(f"训练效果\n损失降低 {improvement:.1f}%")
    for i, v in enumerate([train_losses[0], train_losses[-1]]):
        axes[1].text(i, v + 0.05, f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("assets/04_training_curve.png", dpi=150, bbox_inches='tight')
    print("训练曲线已保存: assets/04_training_curve.png")
    plt.show()


def visualize_predictions(model, dataset, device='cpu', n_samples=6):
    """可视化模型预测结果"""
    model.eval()
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    fig.suptitle("VLA预测结果：图像+指令 → 动作", fontsize=12, fontweight='bold')

    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]

    with torch.no_grad():
        for i in range(n_samples):
            sample = dataset[i * (len(dataset) // n_samples)]
            image = sample['image'].unsqueeze(0).to(device)
            lang = sample['lang'].unsqueeze(0).to(device)
            true_action = sample['action'].numpy()
            color = sample['color']
            obj_pos = sample['obj_pos']

            pred_action, _ = model.predict_action(image, lang)
            pred_action = pred_action[0].cpu().numpy()

            # 上行：显示图像
            ax_img = axes[0][i]
            img_show = image[0].cpu().permute(1, 2, 0).numpy()
            img_show = np.clip(img_show, 0, 1)
            ax_img.imshow(img_show)
            ax_img.set_title(f"pick {color}\npos=({obj_pos[0]:.2f},{obj_pos[1]:.2f})",
                             fontsize=8)
            ax_img.axis('off')

            # 下行：预测 vs 真实动作
            ax_act = axes[1][i]
            x = np.arange(7)
            ax_act.bar(x - 0.2, true_action, 0.35, label='真实', color='#4CAF50', alpha=0.8)
            ax_act.bar(x + 0.2, pred_action, 0.35, label='预测', color='#2196F3', alpha=0.8)
            ax_act.set_xticks(x)
            ax_act.set_xticklabels(action_names, fontsize=7)
            ax_act.set_ylim(-1.2, 1.2)
            ax_act.axhline(0, color='k', linewidth=0.5)
            if i == 0:
                ax_act.legend(fontsize=7)
            mae = np.mean(np.abs(true_action - pred_action))
            ax_act.set_title(f"MAE={mae:.3f}", fontsize=8)

    plt.tight_layout()
    plt.savefig("assets/05_predictions.png", dpi=150, bbox_inches='tight')
    print("预测可视化已保存: assets/05_predictions.png")
    plt.show()


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建数据集
    print("\n创建合成机器人数据集...")
    train_dataset = SyntheticRobotDataset(size=2000, seed=42)
    val_dataset = SyntheticRobotDataset(size=400, seed=123)
    test_dataset = SyntheticRobotDataset(size=200, seed=456)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    # 创建模型
    model = MinimalVLA(
        img_size=64, patch_size=8, embed_dim=128,
        vocab_size=20, action_dim=7, action_bins=256
    )
    print_model_summary(model)

    # 训练
    trainer = VLATrainer(model, device)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, num_epochs=40, lr=3e-4
    )

    # 可视化
    visualize_training(train_losses, val_losses)
    visualize_predictions(model, test_dataset, device)

    print("\n训练完成！下一步：运行 python 02_minimal_vla/inference.py 进行推理演示")
