import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageDataset(Dataset):
    """成对低照度/高照度图像数据集"""

    def __init__(self, low_dir, gt_dir, image_size=360, augment=False):
        self.low_paths = sorted(glob(os.path.join(low_dir, '*')))
        self.gt_paths = sorted(glob(os.path.join(gt_dir, '*')))

        # 数据校验
        assert len(self.low_paths) == len(self.gt_paths), \
            f"Low/GT image count mismatch: {len(self.low_paths)} vs {len(self.gt_paths)}"
        if len(self.low_paths) == 0:
            raise RuntimeError(f"No images found in {low_dir} or {gt_dir}")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.augment = augment  # 预留数据增强接口

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        # 读取图像并转换为RGB
        low = Image.open(self.low_paths[idx]).convert('RGB')
        gt = Image.open(self.gt_paths[idx]).convert('RGB')

        # 应用变换
        low = self.transform(low)
        gt = self.transform(gt)
        return low, gt
