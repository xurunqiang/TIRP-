import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torchvision import transforms


# -------------------------------
# 基础组件：高斯窗口生成
# -------------------------------
def gaussian(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    x = torch.arange(window_size, device=device, dtype=torch.float32) - (window_size // 2)
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
    _1D_window = gaussian(window_size, sigma=1.5, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


# -------------------------------
# 1. MS-SSIM损失（LGE1）
# -------------------------------
class MS_SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, levels: int = 3, reduction: str = 'mean', C1: float = 0.01 ** 2,
                 C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.levels = levels
        self.reduction = reduction
        self.C1 = C1
        self.C2 = C2
        self.channel = 3  # RGB图像
        self.register_buffer('window', torch.Tensor())  # 延迟初始化窗口

    def _get_window(self, device: torch.device) -> torch.Tensor:
        if self.window.numel() == 0 or self.window.device != device:
            self.window = create_window(self.window_size, self.channel, device)
        return self.window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        assert img1.shape == img2.shape, "Input shapes must match"
        device = img1.device
        window = self._get_window(device)
        padding = self.window_size // 2

        total_ssim = 0.0
        current_img1, current_img2 = img1, img2
        for _ in range(self.levels):
            # 计算当前尺度SSIM
            mu1 = F.conv2d(current_img1, window, padding=padding, groups=self.channel)
            mu2 = F.conv2d(current_img2, window, padding=padding, groups=self.channel)
            mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

            sigma1_sq = F.conv2d(current_img1 * current_img1, window, padding=padding, groups=self.channel) - mu1_sq
            sigma2_sq = F.conv2d(current_img2 * current_img2, window, padding=padding, groups=self.channel) - mu2_sq
            sigma12 = F.conv2d(current_img1 * current_img2, window, padding=padding, groups=self.channel) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
                        (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
            total_ssim += ssim_map.mean()

            # 下采样到下一尺度
            current_img1 = F.avg_pool2d(current_img1, kernel_size=2, stride=2)
            current_img2 = F.avg_pool2d(current_img2, kernel_size=2, stride=2)

        ms_ssim = total_ssim / self.levels
        if self.reduction == 'mean':
            return 1 - ms_ssim
        elif self.reduction == 'sum':
            return (1 - ms_ssim) * img1.size(0)
        raise ValueError(f"Unsupported reduction: {self.reduction}")


# -------------------------------
# 2. 感知损失（LLE）
# -------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer: str = 'relu3_3'):
        super().__init__()
        # 加载预训练VGG19并冻结参数
        vgg = models.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)

        # 选择特征层
        layer_indices = {
            'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 'relu3_3': 13, 'relu4_1': 20, 'relu4_3': 22
        }
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_indices[feature_layer] + 1])

        # VGG图像标准化
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # 预处理并提取特征
        pred_norm = self.preprocess(pred)
        gt_norm = self.preprocess(gt)
        pred_feat = self.feature_extractor(pred_norm)
        gt_feat = self.feature_extractor(gt_norm)

        # 计算特征L2损失
        b, c, h, w = pred_feat.shape
        return F.mse_loss(pred_feat, gt_feat) * (1.0 / (h * w))


# -------------------------------
# 3. 联合损失（LTotal）
# -------------------------------
class JointLoss(nn.Module):
    def __init__(self, lambda1: float = 0.85, lambda2: float = 0.15, omega2: float = 1.0, omega3: float = 0.5,
                 ms_ssim_levels: int = 3):
        super().__init__()
        # 全局增强损失组件
        self.ms_ssim_loss = MS_SSIMLoss(levels=ms_ssim_levels)
        self.l1_loss = nn.L1Loss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # 局部感知损失组件
        self.perceptual_loss = PerceptualLoss()

        # 总损失权重
        self.omega2 = omega2
        self.omega3 = omega3

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # 1. 全局增强损失 LGE = λ1*LGE1 + λ2*LGE2
        lge1 = self.ms_ssim_loss(pred, gt)
        lge2 = self.l1_loss(pred, gt)
        lge = self.lambda1 * lge1 + self.lambda2 * lge2

        # 2. 局部感知损失 LLE
        lle = self.perceptual_loss(pred, gt)

        # 3. 总损失 LTotal = ω2*LGE + ω3*LLE
        return self.omega2 * lge + self.omega3 * lle
