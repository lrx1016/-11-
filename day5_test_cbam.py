
import torch
import torch.nn as nn


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


# =================================
# 4. 测试 CBAM
# =================================

if __name__ == "__main__":

    # 模拟 U-Net Encoder 输出
    x = torch.randn(1, 64, 36, 48)

    model = CBAM(
        channels=64,
        reduction=16
    )

    output = model(x)

    print("输入形状:", x.shape)

    print("输出形状:", output.shape)

    print(
        "CBAM 参数量:",
        sum(p.numel() for p in model.parameters())
    )
