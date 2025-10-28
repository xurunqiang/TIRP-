import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，适合保存图片
from .common import make_dirs


def save_plots(train_metrics, val_metrics, save_dir, metric_names):
    """保存训练/验证指标曲线图（Loss、PSNR、SSIM）"""
    make_dirs(save_dir)
    epochs = [x[0] for x in train_metrics[0]]  # 提取epoch列表

    # 1. 为每个指标单独绘图
    for i, metric_name in enumerate(metric_names):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [x[1] for x in train_metrics[i]], label=f'Train {metric_name}', marker='o', markersize=3)
        plt.plot(epochs, [x[1] for x in val_metrics[i]], label=f'Val {metric_name}', marker='s', markersize=3)

        plt.title(f'Training and Validation {metric_name}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        save_path = f'{save_dir}/{metric_name.lower()}_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} curve to {save_path}")

    # 2. 所有指标合并绘图
    plt.figure(figsize=(15, 10))
    for i, metric_name in enumerate(metric_names):
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, [x[1] for x in train_metrics[i]], label=f'Train {metric_name}', marker='o', markersize=3)
        plt.plot(epochs, [x[1] for x in val_metrics[i]], label=f'Val {metric_name}', marker='s', markersize=3)
        plt.title(f'{metric_name}', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = f'{save_dir}/all_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined metrics plot to {save_path}")
