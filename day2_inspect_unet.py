import torch
import segmentation_models_pytorch as smp

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(device)

model.eval()

# 模拟一批农田 RGB 图像
x = torch.randn(1, 3, 288, 384).to(device)

print("输入形状:", x.shape)

with torch.no_grad():

    # 获取编码器各阶段特征
    features = model.encoder(x)

    print("\nEncoder 各阶段输出：")

    for i, feature in enumerate(features):
        print(
            f"Stage {i}: {tuple(feature.shape)}"
        )

    # 完整 U-Net 前向传播
    output = model(x)

print("\n最终输出形状:", output.shape)
print("运行设备:", device)
