import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# =================================
# 1. Channel Attention
# =================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两个池化分支共享同一个 MLP
        self.mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.mlp(self.avg_pool(x))

        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(
            avg_out + max_out
        )

        return x * attention


# =================================
# 2. Spatial Attention
# =================================

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 沿通道维度求平均
        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        # 沿通道维度求最大值
        max_out = torch.max(
            x,
            dim=1,
            keepdim=True
        )[0]

        # 拼接
        combined = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# =================================
# 3. 完整 CBAM
# =================================

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x

class CBAMEncoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        channels = encoder.out_channels

        # 第0层是原始输入图像，不做注意力处理
        self.attention = nn.ModuleList([
            nn.Identity()
            if i == 0
            else CBAM(c)
            for i, c in enumerate(channels)
        ])

        # 保留 SMP U-Net 所需的通道配置
        self.out_channels = channels

    def forward(self, x):

        features = self.encoder(x)

        enhanced_features = []

        for feature, attention in zip(
            features,
            self.attention
        ):

            enhanced_features.append(
                attention(feature)
            )

        return enhanced_features


# =================================
# 创建原始 U-Net
# =================================

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

# 替换 Encoder
model.encoder = CBAMEncoder(model.encoder)

print("CBAM U-Net 创建成功！")

# =================================
# 测试前向传播
# =================================

x = torch.randn(1, 3, 288, 384)

model.eval()

with torch.no_grad():

    output = model(x)

print("输入形状:", x.shape)
print("输出形状:", output.shape)

print(
    "模型总参数量:",
    sum(p.numel() for p in model.parameters()) / 1e6,
    "M"
)
