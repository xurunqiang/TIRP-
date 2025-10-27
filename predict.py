import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.prien import PRIEN
from mcmm.MCMM import compute_attention_map
from torchvision.utils import save_image

# -------------------------------
# 单张图片测试函数
# -------------------------------
def test_single_image(image_path, save_dir, model_ckpt=None, device='cpu'):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 读取与预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 2. 加载模型
    model = PRIEN(
        stages=6,
        recursions=5,
        use_attention_heads=True,
        unet_ckpt_path=r"C:\Users\31392\Desktop\net\TIPRf\TIPR\ckpt_vis\best_min_channel_estimator.pth"
    ).to(device)
    model.eval()

    if model_ckpt and os.path.exists(model_ckpt):
        print(f"Loading trained model weights from: {model_ckpt}")
        state_dict = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state_dict['model_state'] if 'model_state' in state_dict else state_dict)

    # 3. 前向推理
    with torch.no_grad():
        # 计算最小通道约束图
        mincmm = compute_attention_map(img_tensor)  # (B,1,H,W)
        # UNet 输出（照度/纹理估计）
        unet_output = model.att_head(mincmm)
        # 空间注意力图
        yi = model.dual_att(img_tensor)
        att_map = yi.mean(dim=1, keepdim=True)
        # 最大亮度图
        max_lum, _ = img_tensor.max(dim=1, keepdim=True)
        # 最终增强结果
        output = model(img_tensor)

    # 4. 保存结果
    save_image(unet_output, os.path.join(save_dir, "unet_output.png"))
    save_image(att_map, os.path.join(save_dir, "att_map.png"))
    save_image(max_lum, os.path.join(save_dir, "max_lum.png"))
    save_image(output, os.path.join(save_dir, "enhanced.png"))

    print(f"✅ 结果已保存至: {save_dir}")
    print("包含以下文件：unet_output.png, att_map.png, max_lum.png, enhanced.png")


# -------------------------------
# 主函数
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single Image Test for PRIEN")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save output')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to trained PRIEN checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda or cpu)')
    args = parser.parse_args()

    test_single_image(args.image, args.save_dir, args.ckpt, args.device)
