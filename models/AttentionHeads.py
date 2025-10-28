import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        # 第一层卷积：输入通道→输出通道
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),  # 批归一化，加速训练
        nn.ReLU(inplace=True),   # ReLU激活，引入非线性
        
        # 第二层卷积：输出通道保持不变（加深特征提取）
        nn.Conv2d(out_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNetEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        # 4个卷积块，通道数逐层翻倍（32 → 64 → 128 → 256）
        self.c1 = conv_block(in_ch, base_ch)         # 输入：1通道（灰度图）→ 32通道
        self.c2 = conv_block(base_ch, base_ch*2)     # 32 → 64
        self.c3 = conv_block(base_ch*2, base_ch*4)   # 64 → 128
        self.c4 = conv_block(base_ch*4, base_ch*8)   # 128 → 256
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化：尺寸减半（H×W → H/2×W/2）

    def forward(self, x):
        e1 = self.c1(x)               # 输出：(B, 32, H, W)
        e2 = self.c2(self.pool(e1))   # 池化后卷积：(B, 64, H/2, W/2)
        e3 = self.c3(self.pool(e2))   # 输出：(B, 128, H/4, W/4)
        e4 = self.c4(self.pool(e3))   # 输出：(B, 256, H/8, W/8)
        return [e1, e2, e3, e4]  # 返回各层特征，用于解码器跳跃连接

class UNetDecoder(nn.Module):
    def __init__(self, base_ch=32, out_ch=1):
        super().__init__()
        # 上采样层：通过转置卷积将特征图尺寸翻倍
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)  # 256 → 128，尺寸×2
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)  # 128 → 64，尺寸×2
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)    # 64 → 32，尺寸×2

        # 卷积块：融合上采样特征与编码器对应层特征（跳跃连接）
        self.c4 = conv_block(base_ch*8, base_ch*4)  # 128（上采样）+128（编码器e3）→ 128
        self.c3 = conv_block(base_ch*4, base_ch*2)  # 64（上采样）+64（编码器e2）→ 64
        self.c2 = conv_block(base_ch*2, base_ch)    # 32（上采样）+32（编码器e1）→ 32

        # 最终输出层：将特征映射到目标通道（1通道灰度图）
        self.final = nn.Conv2d(base_ch, out_ch, 1)  # 32 → 1
        self.sigmoid = nn.Sigmoid()  # 输出归一化到[0,1]

    def forward(self, feats):
        e1, e2, e3, e4 = feats  # 接收编码器的四层特征
        
        # 第一层上采样：融合e4的上采样结果与e3
        d4 = self.c4(torch.cat([self.up4(e4), e3], dim=1))  # 尺寸：H/4×W/4，通道128
        
        # 第二层上采样：融合d4的上采样结果与e2
        d3 = self.c3(torch.cat([self.up3(d4), e2], dim=1))  # 尺寸：H/2×W/2，通道64
        
        # 第三层上采样：融合d3的上采样结果与e1
        d2 = self.c2(torch.cat([self.up2(d3), e1], dim=1))  # 尺寸：H×W，通道32
        
        # 输出最终结果（归一化到[0,1]）
        return self.sigmoid(self.final(d2))  # 尺寸：H×W，通道1（灰度图）

class UNetMinChannelEstimator(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)  # 实例化编码器
        self.decoder = UNetDecoder(base_ch, out_ch)  # 实例化解码器

    def forward(self, x):
        # 输入x：(B, 1, H, W) 低光S-map（灰度图）
        feats = self.encoder(x)  # 提取编码器特征
        output = self.decoder(feats)  # 解码器输出正常光S-map
        return output  # 输出：(B, 1, H, W)，范围[0,1]
