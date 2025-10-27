import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

# 导入自定义模块
from utils.common import make_dirs, set_seed
from utils.metrics import psnr, ssim
from utils.plot import save_plots
from data.dataset import PairedImageDataset
from models.prien import PRIEN
from losses.joint_loss import JointLoss


# -------------------------------
# 训练/验证单轮函数
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, grad_clip=None, log_interval=50,
                    writer=None, epoch=0):
    model.train()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train', leave=False)

    for i, (low, gt) in pbar:
        low, gt = low.to(device), gt.to(device)
        optimizer.zero_grad()

        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(low)
                loss = criterion(out, gt)
        else:
            out = model(low)
            loss = criterion(out, gt)

        # 计算指标
        batch_psnr = psnr(out, gt)
        batch_ssim = ssim(out, gt)

        # 反向传播与优化
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 累计指标
        batch_size = low.size(0)
        running_loss += float(loss.item()) * batch_size
        running_psnr += float(batch_psnr.item()) * batch_size
        running_ssim += float(batch_ssim.item()) * batch_size

        # 日志输出
        if i % log_interval == 0:
            pbar.set_postfix(
                {'loss': f"{loss.item():.4f}", 'psnr': f"{batch_psnr.item():.2f}", 'ssim': f"{batch_ssim.item():.4f}"})
            if writer is not None:
                global_step = (epoch - 1) * len(dataloader) + i
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
                writer.add_scalar('train/batch_psnr', batch_psnr.item(), global_step)
                writer.add_scalar('train/batch_ssim', batch_ssim.item(), global_step)

    # 计算 epoch 平均指标
    avg_loss = running_loss / len(dataloader.dataset)
    avg_psnr = running_psnr / len(dataloader.dataset)
    avg_ssim = running_ssim / len(dataloader.dataset)

    # TensorBoard 记录
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_psnr', avg_psnr, epoch)
        writer.add_scalar('train/epoch_ssim', avg_ssim, epoch)

    return avg_loss, avg_psnr, avg_ssim


def validate(model, dataloader, criterion, device, out_dir=None, epoch=0, max_save=4, writer=None):
    model.eval()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, (low, gt) in enumerate(tqdm(dataloader, desc='Val', leave=False)):
            low, gt = low.to(device), gt.to(device)
            out = model(low)

            # 计算指标
            loss = criterion(out, gt)
            batch_psnr = psnr(out, gt)
            batch_ssim = ssim(out, gt)

            # 累计指标
            batch_size = low.size(0)
            running_loss += float(loss.item()) * batch_size
            running_psnr += float(batch_psnr.item()) * batch_size
            running_ssim += float(batch_ssim.item()) * batch_size

            # 保存可视化结果
            if out_dir and i < max_save:
                save_path = os.path.join(out_dir, f'epoch{epoch}_batch{i}.png')
                save_image(torch.cat([low, out, gt], dim=0), save_path, nrow=low.size(0))

    # 计算平均指标
    avg_loss = running_loss / len(dataloader.dataset)
    avg_psnr = running_psnr / len(dataloader.dataset)
    avg_ssim = running_ssim / len(dataloader.dataset)

    # TensorBoard 记录
    if writer is not None:
        writer.add_scalar('val/epoch_loss', avg_loss, epoch)
        writer.add_scalar('val/epoch_psnr', avg_psnr, epoch)
        writer.add_scalar('val/epoch_ssim', avg_ssim, epoch)
        # 保存示例图像
        if out_dir:
            img_path = os.path.join(out_dir, f'epoch{epoch}_batch0.png')
            if os.path.exists(img_path):
                img = Image.open(img_path)
                writer.add_image(f'val/examples_epoch_{epoch}', transforms.ToTensor()(img), epoch)

    return avg_loss, avg_psnr, avg_ssim


# -------------------------------
# 主流程
# -------------------------------
def main(args):
    # 1. 初始化设置
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Set random seed: {args.seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 创建目录
    make_dirs(args.save_dir)
    make_dirs(args.vis_dir) if args.vis_dir else None
    plot_dir = os.path.join(args.save_dir, 'plots')
    log_dir = os.path.join(args.save_dir, 'logs')
    make_dirs(plot_dir)
    make_dirs(log_dir)

    # 3. 初始化日志
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs saved to: {log_dir}")

    # 4. 加载数据
    train_low = os.path.join(args.data_dir, 'train', 'low')
    train_gt = os.path.join(args.data_dir, 'train', 'high')
    val_low = os.path.join(args.data_dir, 'val', 'low')
    val_gt = os.path.join(args.data_dir, 'val', 'high')

    train_ds = PairedImageDataset(train_low, train_gt, image_size=args.image_size)
    val_ds = PairedImageDataset(val_low, val_gt, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 5. 初始化模型、优化器、损失
    model = PRIEN(
        stages=args.stages,
        recursions=args.recursions,
        use_attention_heads=not args.no_att_heads
    ).to(device)
    # 记录模型结构到TensorBoard
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    writer.add_graph(model, dummy_input)

    # 优化器选择
    if args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Optimizer must be 'sgd' or 'adam'")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    # 损失函数（论文配置）
    criterion = JointLoss(
        lambda1=0.85, lambda2=0.15, omega2=1.0, omega3=0.5
    ).to(device)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None
    if scaler is not None:
        print("Enabled mixed precision training (AMP)")

    # 6. 训练循环
    best_val_psnr = -1.0
    # 指标记录列表
    train_metrics = [[], [], []]  # [loss, psnr, ssim]
    val_metrics = [[], [], []]  # [loss, psnr, ssim]

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 训练
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_clip=args.grad_clip,
            log_interval=args.log_interval, writer=writer, epoch=epoch
        )
        # 验证
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device,
            out_dir=args.vis_dir, epoch=epoch, max_save=4, writer=writer
        )
        # 学习率调度
        scheduler.step()

        # 记录指标
        train_metrics[0].append((epoch, train_loss))
        train_metrics[1].append((epoch, train_psnr))
        train_metrics[2].append((epoch, train_ssim))
        val_metrics[0].append((epoch, val_loss))
        val_metrics[1].append((epoch, val_psnr))
        val_metrics[2].append((epoch, val_ssim))

        # 打印结果
        print(f"Train - Loss: {train_loss:.6f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val   - Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        # 保存 checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'prien_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metrics': {'train_loss': train_loss, 'train_psnr': train_psnr, 'val_loss': val_loss,
                            'val_psnr': val_psnr}
            }, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

        # 保存最佳模型（基于验证PSNR）
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = os.path.join(args.save_dir, 'prien_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'best_psnr': best_val_psnr,
                'metrics': {'val_loss': val_loss, 'val_ssim': val_ssim}
            }, best_path)
            print(f"Saved best model (PSNR: {best_val_psnr:.2f}) to: {best_path}")

    # 7. 训练结束：保存曲线、关闭日志
    writer.close()
    metric_names = ['Loss', 'PSNR', 'SSIM']
    save_plots(train_metrics, val_metrics, plot_dir, metric_names)
    print("\nTraining completed!")


# -------------------------------
# 参数解析
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PRIEN Training Script (Modular Version)")
    # 数据配置
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Root of train/val data')
    parser.add_argument('--image_size', type=int, default=360, help='Input image size (H=W)')
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer (sgd/adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60, 90], help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR decay factor')
    # 模型配置
    parser.add_argument('--stages', type=int, default=6, help='Number of PRIEN stages')
    parser.add_argument('--recursions', type=int, default=5, help='Recursions in ResBlocks')
    parser.add_argument('--no_att_heads', action='store_true', help='Disable external AttentionHeads')
    # 工具配置
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save dir')
    parser.add_argument('--vis_dir', type=str, default='./vis', help='Visualization save dir')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision (AMP)')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping norm')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=50, help='Log every N batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)