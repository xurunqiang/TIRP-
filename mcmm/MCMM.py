import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 你提供的注意力图计算函数
def _dark_channel_torch(img: torch.Tensor, ksize: int = 5):
    pad = ksize // 2
    dark = img.min(dim=1, keepdim=True)[0]
    dark = -F.max_pool2d(-dark, ksize, stride=1, padding=pad)
    return dark


def _atmosphere_torch(img: torch.Tensor, dark: torch.Tensor, ratio: float = 0.01):
    B, C, H, W = img.shape
    num = int(H * W * ratio)
    dark_flat = dark.view(B, -1)
    img_flat = img.view(B, 3, -1)
    _, idx = torch.topk(dark_flat, k=num, dim=1, largest=True, sorted=False)
    idx = idx.unsqueeze(1).expand(-1, 3, -1)
    sel = torch.gather(img_flat, dim=2, index=idx)
    A = sel.mean(dim=2)
    return A


def compute_attention_map(img: torch.Tensor, win_size: int = 5):
    dark = _dark_channel_torch(img, win_size)
    A = _atmosphere_torch(img, dark)
    A = A.view(-1, 3, 1, 1)
    I3 = 1.0 - img / A
    attn = I3.min(dim=1, keepdim=True)[0]
    return attn.clamp(0, 1)


# 自定义数据集处理图片
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path


def process_images(input_dir, output_dir, win_size=5, batch_size=8,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义图片转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    ])

    # 创建数据集和数据加载器
    dataset = ImageDataset(input_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"开始处理图片，共 {len(dataset)} 张图片")
    print(f"使用设备: {device}")

    for batch_idx, (images, paths) in enumerate(dataloader):
        images = images.to(device)

        # 计算注意力图
        with torch.no_grad():
            attn_maps = compute_attention_map(images, win_size)

        # 处理并保存每张图片的结果
        for i, (attn_map, img_path) in enumerate(zip(attn_maps, paths)):
            # 转换为PIL图片格式 (将[0,1]映射到[0,255]的灰度图)
            attn_np = attn_map.squeeze().cpu().numpy() * 255
            attn_img = Image.fromarray(attn_np.astype(np.uint8))

            # 构建输出路径
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_attn{ext}")

            # 保存图片
            attn_img.save(output_path)

            # 打印进度
            if (batch_idx * batch_size + i + 1) % 10 == 0:
                print(f"已处理 {batch_idx * batch_size + i + 1}/{len(dataset)} 张图片")

    print("所有图片处理完成！")


if __name__ == "__main__":
    # 配置输入输出目录
    INPUT_DIR = r"/data/coding/TIPR/dataset/val/high"  # 输入图片文件夹
    OUTPUT_DIR = r"/data/coding/TIPR/dataset/val/high_cam"  # 输出注意力图文件夹

    # 处理参数
    WINDOW_SIZE = 5  # 窗口大小，可根据需要调整
    BATCH_SIZE = 8  # 批处理大小，根据内存情况调整

    # 开始处理
    process_images(INPUT_DIR, OUTPUT_DIR, WINDOW_SIZE, BATCH_SIZE)
