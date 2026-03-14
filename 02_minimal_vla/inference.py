"""
MinimalVLA 推理演示
===================
加载训练好的VLA模型，交互式演示推理过程

功能：
1. 加载预训练权重（如果有）
2. 可视化模型"如何看图像"
3. 逐步展示自回归动作生成过程
4. 测试模型泛化性

运行方式：
    python 02_minimal_vla/inference.py
（需要先运行 train.py）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import MinimalVLA
from train import SyntheticRobotDataset


def load_model(checkpoint_path=None):
    model = MinimalVLA(
        img_size=64, patch_size=8, embed_dim=128,
        vocab_size=20, action_dim=7, action_bins=256
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"已加载模型权重: {checkpoint_path}")
    else:
        print("警告: 未找到预训练权重，使用随机初始化模型（请先运行 train.py）")

    model.eval()
    return model


def visualize_autoregressive_generation(model, dataset):
    """
    可视化自回归动作生成过程
    展示VLA如何一个Token一个Token地生成动作
    """
    sample = dataset[5]
    image = sample['image'].unsqueeze(0)
    lang = sample['lang'].unsqueeze(0)
    true_action = sample['action'].numpy()
    color = sample['color']

    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
    action_tokenizer = model.action_tokenizer

    # 逐步生成动作Token
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 8, figure=fig)

    fig.suptitle(f"VLA自回归动作生成过程\n指令: 'pick up the {color} object'",
                 fontsize=13, fontweight='bold')

    # 左上：原始图像
    ax_img = fig.add_subplot(gs[0, :2])
    img_show = image[0].permute(1, 2, 0).numpy()
    ax_img.imshow(np.clip(img_show, 0, 1))
    ax_img.set_title(f"输入图像\n目标: {color}色物体", fontsize=10)
    ax_img.axis('off')

    # 右上：生成过程说明
    ax_desc = fig.add_subplot(gs[0, 2:])
    ax_desc.text(0.05, 0.9, "自回归生成流程:", fontsize=11, fontweight='bold',
                 transform=ax_desc.transAxes)
    ax_desc.text(0.05, 0.75, "步骤1: 视觉编码器 → 64个视觉Token", fontsize=9,
                 transform=ax_desc.transAxes, color='green')
    ax_desc.text(0.05, 0.62, "步骤2: 语言编码器 → 8个语言Token", fontsize=9,
                 transform=ax_desc.transAxes, color='orange')
    ax_desc.text(0.05, 0.49, "步骤3: 拼接 → 72个上下文Token", fontsize=9,
                 transform=ax_desc.transAxes, color='purple')
    ax_desc.text(0.05, 0.36, "步骤4: 逐个预测7个动作Token:", fontsize=9,
                 transform=ax_desc.transAxes, color='red')
    ax_desc.text(0.08, 0.22, "[BOS] → T_x → T_y → T_z → T_rx → T_ry → T_rz → T_grip",
                 fontsize=8, transform=ax_desc.transAxes, color='darkred',
                 fontfamily='monospace')
    ax_desc.text(0.05, 0.08, "步骤5: Token → 连续动作值", fontsize=9,
                 transform=ax_desc.transAxes, color='blue')
    ax_desc.axis('off')

    # 中行：逐步生成每个Token
    model.eval()
    context = model.encode_context(image, lang)
    generated_ids = []
    generated_actions = []

    with torch.no_grad():
        current_input = torch.zeros(1, 1, dtype=torch.long)  # BOS

        for step in range(7):
            ax = fig.add_subplot(gs[1, step])

            # 前向传播
            logits = model.decoder(context, current_input)
            probs = torch.softmax(logits[0, -1], dim=-1).numpy()

            # 选择最高概率的Token
            next_token_id = probs.argmax()
            generated_ids.append(int(next_token_id))

            # 转换为连续值
            cont_val = action_tokenizer.decode(torch.tensor([next_token_id])).item()
            generated_actions.append(cont_val)

            # 可视化概率分布（只显示top区域）
            top_k = 30
            top_tokens = probs.argsort()[-top_k:][::-1]
            top_probs = probs[top_tokens]

            ax.bar(range(top_k), top_probs, color='#2196F3', alpha=0.7, width=0.8)
            ax.axvline(0, color='red', linewidth=2)  # 最高概率
            ax.set_title(f"{action_names[step]}\nT#{next_token_id}\n≈{cont_val:.3f}",
                         fontsize=8)
            ax.set_xlabel("Top-30 Token ID", fontsize=6)
            if step == 0:
                ax.set_ylabel("概率", fontsize=7)
            ax.tick_params(labelsize=6)

            # 更新输入
            current_input = torch.cat([
                current_input,
                torch.tensor([[next_token_id]])
            ], dim=1)

    # 下行：最终动作对比
    ax_compare = fig.add_subplot(gs[2, :])
    x = np.arange(7)
    bars1 = ax_compare.bar(x - 0.25, true_action, 0.45, label='真实动作',
                           color='#4CAF50', alpha=0.85)
    bars2 = ax_compare.bar(x + 0.25, generated_actions, 0.45, label='预测动作',
                           color='#F44336', alpha=0.85)

    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels(
        [f"{n}\n(T#{tid})" for n, tid in zip(action_names, generated_ids)],
        fontsize=9
    )
    ax_compare.set_ylabel("动作值 [-1, 1]")
    ax_compare.set_title("预测结果 vs 真实动作")
    ax_compare.legend()
    ax_compare.axhline(0, color='k', linewidth=0.5)
    ax_compare.set_ylim(-1.3, 1.3)
    ax_compare.grid(True, alpha=0.3, axis='y')

    mae = np.mean(np.abs(np.array(true_action) - np.array(generated_actions)))
    ax_compare.text(0.98, 0.95, f"MAE = {mae:.4f}", transform=ax_compare.transAxes,
                    ha='right', va='top', fontsize=10, color='red',
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig("assets/06_autoregressive_generation.png", dpi=150, bbox_inches='tight')
    print("自回归生成可视化已保存: assets/06_autoregressive_generation.png")
    plt.show()

    return generated_actions, true_action


def test_generalization(model, n_tests=12):
    """测试模型对不同颜色/位置的泛化性"""
    print("\n" + "=" * 60)
    print("  泛化性测试")
    print("=" * 60)

    test_dataset = SyntheticRobotDataset(size=n_tests * 3, seed=999)

    results = {'red': [], 'green': [], 'blue': []}

    model.eval()
    with torch.no_grad():
        for i in range(min(n_tests, len(test_dataset))):
            sample = test_dataset[i]
            image = sample['image'].unsqueeze(0)
            lang = sample['lang'].unsqueeze(0)
            true_action = sample['action'].numpy()
            color = sample['color']
            obj_pos = sample['obj_pos']

            pred_action, _ = model.predict_action(image, lang)
            pred_action = pred_action[0].numpy()

            # 主要关注x,y坐标的预测误差
            xy_mae = np.mean(np.abs(true_action[:2] - pred_action[:2]))
            results[color].append({
                'pos': obj_pos, 'mae': xy_mae,
                'true_xy': true_action[:2], 'pred_xy': pred_action[:2]
            })

    # 打印结果
    for color, color_results in results.items():
        if color_results:
            avg_mae = np.mean([r['mae'] for r in color_results])
            print(f"\n  {color.upper()} 物体 (n={len(color_results)}):")
            print(f"  平均XY误差: {avg_mae:.4f}")
            for r in color_results[:3]:
                print(f"    位置({r['pos'][0]:.2f},{r['pos'][1]:.2f}) "
                      f"真实({r['true_xy'][0]:+.3f},{r['true_xy'][1]:+.3f}) "
                      f"预测({r['pred_xy'][0]:+.3f},{r['pred_xy'][1]:+.3f}) "
                      f"MAE={r['mae']:.4f}")

    return results


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    # 加载模型
    model = load_model("02_minimal_vla/best_model.pt")

    # 创建测试数据集
    test_dataset = SyntheticRobotDataset(size=100, seed=789)

    # 可视化自回归生成过程
    print("\n可视化自回归生成过程...")
    gen_actions, true_actions = visualize_autoregressive_generation(model, test_dataset)

    print("\n逐步生成结果:")
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
    for name, pred, true in zip(action_names, gen_actions, true_actions):
        print(f"  {name:>5}: 预测={pred:+.4f}  真实={true:+.4f}  误差={abs(pred-true):.4f}")

    # 泛化测试
    test_generalization(model)

    print("\n推理演示完成！")
    print("\n下一步：")
    print("  python 03_rt2_style/rt2_demo.py  → RT-2风格VLA（动作作为文本Token）")
    print("  python 04_advanced/compare_vlas.py → 对比主流VLA架构")
