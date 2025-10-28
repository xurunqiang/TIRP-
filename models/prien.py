import torch
import torch.nn as nn
import torch.nn.functional as F
from mcmm.MCMM import compute_attention_map
from models.AttentionHeads import UNetMinChannelEstimator


# -------------------------------
# 基础组件
# -------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out


class RecursiveResBlocks(nn.Module):
    def __init__(self, channels=32, recursions=5):
        super().__init__()
        self.recursions = recursions
        self.block = ResBlock(channels)  # 权重共享

    def forward(self, x):
        out = x
        for _ in range(self.recursions):
            out = self.block(out)
        return out


class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_r = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.conv_z.out_channels, x.size(2), x.size(3), device=x.device)
        combined = torch.cat([x, h_prev], dim=1)
        z = self.sigmoid(self.conv_z(combined))
        r = self.sigmoid(self.conv_r(combined))
        combined2 = torch.cat([x, r * h_prev], dim=1)
        h_tilde = self.tanh(self.conv_h(combined2))
        return (1 - z) * h_prev + z * h_tilde


class DualAttention(nn.Module):
    """局部空间注意力 + 通道注意力（轻量版）"""

    def __init__(self, in_channels=3, inter_channels=32, k_reduce=8):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 1)

        # 局部空间注意力（深度可分离卷积）
        self.local_att = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, groups=inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 1),
            nn.Sigmoid()
        )

        # 通道注意力（SE模块）
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, inter_channels // k_reduce)
        self.fc1 = nn.Conv2d(inter_channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, inter_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_in = self.conv_in(x)
        spatial_mask = self.local_att(x_in)
        x_spatial = x_in * spatial_mask

        se = self.channel_pool(x_spatial)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x_spatial * se


# -------------------------------
# PRIEN阶段单元
# -------------------------------
class PRIENStage(nn.Module):
    def __init__(self, in_channels=7, feat_channels=32, gru_hidden=32, recursions=5):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.gru = ConvGRUCell(in_channels=feat_channels, hidden_channels=gru_hidden)
        self.reduce = nn.Conv2d(gru_hidden, feat_channels, 1) if gru_hidden != feat_channels else nn.Identity()
        self.recursive_res = RecursiveResBlocks(channels=feat_channels, recursions=recursions)
        self.conv_out = nn.Conv2d(feat_channels, 3, 3, padding=1)

    def forward(self, inp_concat, h_prev):
        x = self.conv_in(inp_concat)
        x = self.relu(x)
        h_new = self.gru(x, h_prev)
        feat = self.reduce(h_new)
        feat = self.recursive_res(feat)
        return self.conv_out(feat), h_new


# -------------------------------
# PRIEN主模型
# -------------------------------
class PRIEN(nn.Module):
    def __init__(self, stages=6, recursions=5, use_attention_heads=True, unet_ckpt_path="/data/coding/TIPR/ckpt_vis/best_min_channel_estimator.pth"):
        super().__init__()
        self.stages = stages
        self.recursions = recursions
        self.use_attention_heads = use_attention_heads

        # 注意力组件
        self.dual_att = DualAttention(in_channels=3, inter_channels=32, k_reduce=8)
        # 加载预训练的UNet模型作为att_head
        self.att_head = UNetMinChannelEstimator()  # 初始化UNet模型
        if unet_ckpt_path and torch.cuda.is_available():
            # 加载预训练权重
            self.att_head.load_state_dict(
                torch.load(unet_ckpt_path, map_location=torch.device('cuda'))
            )
        elif unet_ckpt_path:
            self.att_head.load_state_dict(
                torch.load(unet_ckpt_path, map_location=torch.device('cpu'))
            )
        # 设置为评估模式（关闭BN和Dropout）
        self.att_head.eval()

        # 阶段单元（输入6通道：lt(3)+att_map(1)+max_lum(1)+illum_tex）
        self.stage_unit = PRIENStage(in_channels=6, feat_channels=32, gru_hidden=32, recursions=recursions)

    def forward(self, L):
        """
        Args:
            L: (B, 3, H, W) 低照度RGB输入，范围[0,1]
        Returns:
            lt: (B, 3, H, W) 增强后图像
        """
        B, C, H, W = L.shape
        # 假设compute_attention_map已实现（计算MCMM特征）
        mincmm = compute_attention_map(L)  # 假设输出为(B, 1, H, W)的单通道特征图
        
        # 禁用att_head的梯度计算（固定预训练模型）
        with torch.no_grad():
            # UNet输入为单通道，需将mincmm（单通道）传入
            unet_output = self.att_head(mincmm)  # 输出形状：(B, 1, H, W)
        
        yi = self.dual_att(L)
        att_map = yi.mean(dim=1, keepdim=True)  # 空间注意力图：(B, 1, H, W)
        max_lum, _ = L.max(dim=1, keepdim=True)  # 最大亮度通道：(B, 1, H, W)

        lt = L.clone()
        h_prev = None
        for _ in range(self.stages):
            # 拼接输入通道
            if self.use_attention_heads and self.att_head is not None:
                # 用UNet输出作为illum，可根据需求调整tex（这里简单复制illum）
                illum_tex = unet_output  # (B, 1, H, W)
                
                # 拼接通道：lt(3) + att_map(1) + max_lum(1) + illum_tex(1) = 6通道
                concat = torch.cat([lt, att_map, max_lum, illum_tex], dim=1)
            else:
                zeros = torch.zeros(B, 1, H, W, device=L.device)
                concat = torch.cat([lt, att_map, max_lum, zeros], dim=1)

            # 阶段前向传播
            out_rgb, h_prev = self.stage_unit(concat, h_prev)
            lt = torch.clamp(lt + out_rgb, 0.0, 1.0)  # 累积输出并截断

        return lt
