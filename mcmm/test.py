#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单张图片推理脚本
Author: You
"""
import argparse
import os

import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# -------------------- 网络结构（与训练端保持一致） --------------------
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

# -------------------- 推理 --------------------
def main():
    parser = argparse.ArgumentParser(description='单张图片快速推理')
    parser.add_argument('--img_low',   required=True, help='低光图片路径（png/jpg）')
    parser.add_argument('--model',     required=True, help='权重路径（.pth）')
    parser.add_argument('--save_dir',  default='./results', help='结果保存目录')
    parser.add_argument('--image_size',type=int, default=360, help='训练时的输入尺寸')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载模型
    model = UNetMinChannelEstimator().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 2. 预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(1),
        transforms.ToTensor()          # -> [0,1]
    ])
    img_low = Image.open(args.img_low).convert('RGB')
    x = transform(img_low).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        pred = model(x).cpu().squeeze(0).squeeze(0).numpy()  # [H,W]

    # 4. 保存对比图
    base_name = os.path.splitext(os.path.basename(args.img_low))[0]
    save_path = os.path.join(args.save_dir, f'{base_name}_compare.png')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_low, cmap='gray')
    plt.title('Low-light Input'); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
    plt.title('Predicted S-normal-hat'); plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'结果已保存 → {save_path}')

if __name__ == '__main__':
    main()


# 1. 单张低光图推理（GPU）
# python test.py \
#   --img_low   ./demo/low_01.png \
#   --model     ./ckpt_vis/best_min_channel_estimator.pth \
#   --save_dir  ./results

# 2. 若在无显卡服务器
# python test.py --cpu \
#   --img_low   ./demo/low_01.png \
#   --model     ./ckpt_vis/best_min_channel_estimator.pth \
#   --save_dir  ./results
