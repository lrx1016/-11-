import torch

# 创建一个需要学习的参数
w = torch.tensor(
    1.0,
    requires_grad=True
)

# 设置学习率
learning_rate = 0.1

# 计算损失
loss = (w - 3) ** 2

print("初始参数:", w.item())
print("初始损失:", loss.item())

# 自动计算梯度
loss.backward()

print("计算得到的梯度:", w.grad.item())

# 根据梯度更新参数
with torch.no_grad():
    w -= learning_rate * w.grad

# 清除梯度
w.grad = None

print("更新后的参数:", w.item())

# 再次计算损失
new_loss = (w - 3) ** 2

print("更新后的损失:", new_loss.item())
