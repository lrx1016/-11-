import torch
import segmentation_models_pytorch as smp

# 1. 选择计算设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. 创建 U-Net
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

# 3. 创建模拟 RGB 图像
image = torch.randn(
    2, 3, 256, 256
).to(device)

# 4. 模型预测
model.eval()

with torch.no_grad():
    output = model(image)

print("输入尺寸:", image.shape)
print("输出尺寸:", output.shape)
print("运行设备:", output.device)
