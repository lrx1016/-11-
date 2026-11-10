import torch
import torch.nn as nn


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch, channels, _, _ = x.size()

        y = self.avg_pool(x)
        y = y.view(batch, channels)

        y = self.fc(y)
        y = y.view(batch, channels, 1, 1)

        return x * y


# 模拟 Encoder 输出
x = torch.randn(1, 64, 36, 48)

# 创建 SE
se = SEBlock(
    channels=64,
    reduction=16
)

# 前向传播
output = se(x)

print("输入形状:", x.shape)
print("输出形状:", output.shape)

print("SE 参数量:",
      sum(p.numel() for p in se.parameters()))