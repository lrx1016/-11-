# U-Net++ 实验记录（2026-11-06）

## 1. 实验目的

学习 U-Net++ 网络结构，理解嵌套跳跃连接，并在 CWFID 数据集上完成植被二分类语义分割。

## 2. 实验环境

- Python：3.11.18
- PyTorch：2.15.0+cu132
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- 数据集：CWFID
- 图像数量：60

## 3. 数据划分

- 随机种子：42
- 训练集：48张
- 验证集：12张
- 输入尺寸：288×384

与 U-Net 使用相同的固定数据划分。

## 4. 模型及训练参数

- 网络：U-Net++
- Encoder：ResNet18
- Encoder预训练权重：None
- 输入通道：3
- 输出通道：1
- Batch Size：4
- Epoch：30
- Optimizer：Adam
- Learning Rate：0.0001
- Loss：BCE + Dice

## 5. 实验结果

- 最佳 Epoch：25
- 平均验证 Dice：0.9288
- 平均验证 IoU：0.8672
- 参数量：15.970 M

最佳权重：`best_epoch_25.pth`

## 6. 逐图评价

- 验证图像数量：12
- 最高 Dice：0.9475（018_image.png）
- 最低 Dice：0.9158（002_image.png）

逐图评价结果：`per_image_metrics.csv`

## 7. 与 U-Net 对比

| 模型 | Dice | IoU | 参数量 |
|---|---:|---:|---:|
| U-Net | 0.9305 | 0.8701 | 15.970 M |
| U-Net++ | 0.9288 | 0.8672 | 15.970 M |

本次实验中 U-Net 的验证指标略高于 U-Net++，两者表现接近。该结果仅代表当前固定划分下的初步实验。

## 8. 后续计划

学习 DeepLabV3+、ASPP 和空洞卷积，继续开展农业图像语义分割实验。