"""
SimVP 预测结果可视化脚本
从保存的 .npy 文件生成可视化图像
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def load_data(result_path='./results/Debug/results/Debug/sv/'):
    """加载预测结果"""
    print("正在加载数据...")
    inputs = np.load(os.path.join(result_path, 'inputs.npy'))
    trues = np.load(os.path.join(result_path, 'trues.npy'))
    preds = np.load(os.path.join(result_path, 'preds.npy'))
    
    print(f"✓ 数据加载成功!")
    print(f"  输入形状: {inputs.shape}")
    print(f"  真实值形状: {trues.shape}")
    print(f"  预测值形状: {preds.shape}")
    
    return inputs, trues, preds


def visualize_sample(inputs, trues, preds, sample_idx=0, save_path='./visualizations/sample.png'):
    """可视化单个样本"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    t_in = inputs.shape[1]
    t_out = trues.shape[1]
    
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(3, max(t_in, t_out), hspace=0.3, wspace=0.1)
    
    # 输入序列
    for t in range(t_in):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(inputs[sample_idx, t, 0], cmap='gray')
        ax.set_title(f'Input t={t}', fontsize=10)
        ax.axis('off')
    
    # 真实未来帧
    for t in range(t_out):
        ax = fig.add_subplot(gs[1, t])
        ax.imshow(trues[sample_idx, t, 0], cmap='gray')
        ax.set_title(f'True t={t_in+t}', fontsize=10)
        ax.axis('off')
    
    # 预测未来帧
    for t in range(t_out):
        ax = fig.add_subplot(gs[2, t])
        ax.imshow(preds[sample_idx, t, 0], cmap='gray')
        ax.set_title(f'Pred t={t_in+t}', fontsize=10)
        ax.axis('off')
    
    fig.text(0.02, 0.83, 'Input', ha='center', va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.50, 'Truth', ha='center', va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.17, 'Pred', ha='center', va='center', fontsize=12, weight='bold')
    
    plt.suptitle(f'Sample {sample_idx + 1}', fontsize=14, weight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_path}")
    plt.close()


def plot_errors(trues, preds, save_path='./visualizations/errors.png'):
    """绘制误差曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    mse_per_frame = np.mean((trues - preds) ** 2, axis=(0, 2, 3, 4))
    mae_per_frame = np.mean(np.abs(trues - preds), axis=(0, 2, 3, 4))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(mse_per_frame, marker='o', linewidth=2)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE per Frame')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mae_per_frame, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE per Frame')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("SimVP 可视化")
    print("="*60)
    
    # 加载数据
    inputs, trues, preds = load_data()
    
    # 生成可视化
    print("\n生成可视化...")
    num_samples = min(5, inputs.shape[0])
    for i in range(num_samples):
        visualize_sample(inputs, trues, preds, i, f'./visualizations/sample_{i+1}.png')
    
    plot_errors(trues, preds)
    
    # 统计信息
    mse = np.mean((trues - preds) ** 2)
    mae = np.mean(np.abs(trues - preds))
    print(f"\n总体 MSE: {mse:.6f}")
    print(f"总体 MAE: {mae:.6f}")
    
    print("\n✓ 完成! 结果保存在 ./visualizations/")
    print("="*60)


if __name__ == '__main__':
    main()
