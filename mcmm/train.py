# =====================================
#  中文注释升级版：训练 + 可视化 + 损失曲线
#  运行示例：
#  python train_S_map_cn_vis.py \
#     --S_low_dir  ./data/S_low \
#     --S_normal_dir ./data/S_normal \
#     --save_dir ./ckpt_vis
# =====================================
import torch, torch.nn as nn, torch.nn.functional as F
import os, numpy as np, argparse, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm

# -------------------- 可视化新增 --------------------
import matplotlib
# 远程服务器无显示器自动切换后端
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def vis_epoch(model, dataset, device, epoch, save_dir, max_show=4):
    """
    随机抽 max_show 张图，可视化对比：
    左：S_low  中：模型预测 S_normal_hat  右：真值 S_normal
    保存路径：save_dir/vis_epoch_{epoch}.png
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(max_show, len(dataset)))
    fig, axes = plt.subplots(3, max_show, figsize=(max_show*4, 6))
    if max_show == 1:  # 单张图维度兼容
        axes = axes.reshape(3, -1)
    with torch.no_grad():
        for col, idx in enumerate(indices):
            s_low, s_normal = dataset[idx]
            s_low_in = s_low.unsqueeze(0).to(device)
            s_normal_hat = model(s_low_in).cpu().squeeze(0)
            # 绘图
            def tensor2np(t):
                return t.squeeze().numpy()
            axes[0, col].imshow(tensor2np(s_low), cmap='gray')
            axes[0, col].set_title('S_low')
            axes[1, col].imshow(tensor2np(s_normal_hat), cmap='gray')
            axes[1, col].set_title('S_normal_hat')
            axes[2, col].imshow(tensor2np(s_normal), cmap='gray')
            axes[2, col].set_title('S_normal(GT)')
            for row in range(3):
                axes[row, col].axis('off')
    plt.suptitle(f'Epoch {epoch} 可视化')
    vis_path = os.path.join(save_dir, f'vis_epoch_{epoch}.png')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    model.train()
    print(f'可视化已保存 → {vis_path}')


def plot_loss_curve(loss_history, save_dir):
    """绘制并保存训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history)+1), loss_history, 'b-', linewidth=2)
    plt.title('训练损失曲线 (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('平均MSE损失')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, len(loss_history))
    plt.ylim(0, max(loss_history) * 1.1)  # 留出一些顶部空间
    
    # 标记最佳损失点
    best_epoch = np.argmin(loss_history) + 1
    best_loss = loss_history[best_epoch - 1]
    plt.scatter(best_epoch, best_loss, color='r', s=80, zorder=5, 
                label=f'最佳损失: {best_loss:.6f} (Epoch {best_epoch})')
    plt.legend()
    
    loss_path = os.path.join(save_dir, 'loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f'损失曲线已保存 → {loss_path}')


# -------------------- 原网络结构不变 --------------------
def conv_block(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
    )

class UNetEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.c1 = conv_block(in_ch, base_ch)
        self.c2 = conv_block(base_ch, base_ch*2)
        self.c3 = conv_block(base_ch*2, base_ch*4)
        self.c4 = conv_block(base_ch*4, base_ch*8)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        e1 = self.c1(x)
        e2 = self.c2(self.pool(e1))
        e3 = self.c3(self.pool(e2))
        e4 = self.c4(self.pool(e3))
        return [e1, e2, e3, e4]

class UNetDecoder(nn.Module):
    def __init__(self, base_ch=32, out_ch=1):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.c4 = conv_block(base_ch*8, base_ch*4)
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.c3 = conv_block(base_ch*4, base_ch*2)
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.c2 = conv_block(base_ch*2, base_ch)
        self.final = nn.Conv2d(base_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feats):
        e1,e2,e3,e4 = feats
        d4 = self.c4(torch.cat([self.up4(e4), e3], 1))
        d3 = self.c3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.c2(torch.cat([self.up2(d3), e1], 1))
        return self.sigmoid(self.final(d2))

class UNetMinChannelEstimator(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        self.decoder = UNetDecoder(base_ch, out_ch)
    def forward(self, x):
        return self.decoder(self.encoder(x))

# -------------------- 数据集 --------------------
class PreExtractedMinChannelDataset(Dataset):
    def __init__(self, S_low_dir, S_normal_dir, image_size=400):
        self.S_low_paths = sorted(glob(os.path.join(S_low_dir, '*.png')))
        self.S_normal_paths = sorted(glob(os.path.join(S_normal_dir, '*.png')))
        assert len(self.S_low_paths) == len(self.S_normal_paths) and len(self.S_low_paths) > 0, '数据集为空或数量不一致'
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.S_low_paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.S_low_paths[idx])), \
               self.transform(Image.open(self.S_normal_paths[idx]))

# -------------------- 训练函数（含可视化和损失曲线） --------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = PreExtractedMinChannelDataset(args.S_low_dir, args.S_normal_dir, args.image_size)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print(f'数据集大小：{len(train_dataset)}')

    model = UNetMinChannelEstimator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    best = float('inf')
    loss_history = []  # 记录每个epoch的损失
    model.train()
    
    for epoch in range(1, args.epochs+1):
        running = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for s_low, s_normal in pbar:
            s_low, s_normal = s_low.to(device), s_normal.to(device)
            out = model(s_low)
            loss = criterion(out, s_normal)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item() * s_low.size(0)
            pbar.set_postfix({'mse': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        avg = running / len(train_dataset)
        loss_history.append(avg)  # 保存当前epoch的平均损失
        scheduler.step()

        # 可视化
        if epoch % args.vis_every == 0 or epoch == args.epochs:
            vis_epoch(model, train_dataset, device, epoch, args.save_dir)

        # 保存最优 & checkpoint
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_min_channel_estimator.pth'))
            print(f'最佳模型更新（MSE={best:.6f}）')
        if epoch % args.save_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 训练结束后保存最终模型、最终可视化和损失曲线
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_min_channel_estimator.pth'))
    vis_epoch(model, train_dataset, device, 'final', args.save_dir, max_show=8)
    plot_loss_curve(loss_history, args.save_dir)  # 绘制损失曲线
    
    print(f'\n训练完成！最佳 MSE={best:.6f}')
    print(f'可视化结果见：{args.save_dir}/vis_epoch_final.png')
    print(f'损失曲线见：{args.save_dir}/loss_curve.png')

# -------------------- 参数解析 --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--S_low_dir', required=True, help='低光 S-map 目录')
    parser.add_argument('--S_normal_dir', required=True, help='正常光 S-map 目录')
    parser.add_argument('--image_size', type=int, default=360)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[100, 150])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_dir', default='./ckpt_vis')
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--vis_every', type=int, default=10, help='每 N 个 epoch 可视化一次')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)

#python train.py --S_low_dir  ./dataset/train/low_cam --S_normal_dir ./dataset/train/high_cam --save_dir ./ckpt_vis --vis_every 5
