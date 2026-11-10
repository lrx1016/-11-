# Day 4 实验记录：FCN（Fully Convolutional Network）

## 1. 实验目的

学习 FCN 全卷积网络的基本结构，理解全连接层替换为卷积层、上采样（Upsampling）以及跳跃融合（Skip Fusion）的作用。

使用 CWFID 作物图像数据集完成植被与背景二分类语义分割实验，并与之前的 U-Net、U-Net++ 和 DeepLabV3+ 进行初步比较。

## 2. 实验环境

- 操作系统：Windows
- Python：3.11.18
- PyTorch：2.15.0+cu132
- Torchvision：0.29.2+cu132
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- 数据集：CWFID
- 图像数量：60
- 训练集：48
- 验证集：12
- 数据划分随机种子：42

## 3. 模型配置

- 网络：FCN
- 具体实现：Torchvision FCN-ResNet50
- Backbone：ResNet50
- 预训练权重：未使用
- 输入通道：3（RGB）
- 输出类别：1（二分类）
- 输入尺寸：288×384
- Batch Size：4
- Epochs：30
- Optimizer：Adam
- Learning Rate：0.0001
- Loss Function：BCEWithLogitsLoss + Dice Loss
- 总参数量：32.947M
- 可训练参数量：32.947M

### 模型输出处理

Torchvision 的 FCN 返回字典，因此使用 `get_logits()` 函数提取预测结果：

```python
def get_logits(model, images):

    output = model(images)

    if isinstance(output, dict):
        return output["out"]

    return output
```

模型前向传播输出尺寸：

```text
torch.Size([1, 1, 288, 384])
```

## 4. 训练结果

| 指标 | 结果 |
|---|---:|
| 最终 Train Loss | 0.3442 |
| 最终 Val Loss | 0.3741 |
| 最佳验证 Dice | 0.7791 |
| 最佳验证 IoU | 0.6394 |
| 最佳 Epoch | 30 |
| 模型参数量 | 32.947M |

### 最后五轮训练记录

| Epoch | Train Loss | Val Loss | Dice | IoU |
|---:|---:|---:|---:|---:|
| 26 | 0.3571 | 0.3867 | 0.7670 | 0.6234 |
| 27 | 0.3530 | 0.3802 | 0.7723 | 0.6303 |
| 28 | 0.3508 | 0.3792 | 0.7737 | 0.6323 |
| 29 | 0.3439 | 0.3995 | 0.7699 | 0.6269 |
| 30 | 0.3442 | 0.3741 | 0.7791 | 0.6394 |

## 5. 验证集逐图评价

| 图像 | Dice | IoU |
|---|---:|---:|
| 052 | 0.8106 | 0.6815 |
| 044 | 0.7988 | 0.6650 |
| 007 | 0.7795 | 0.6387 |
| 057 | 0.7827 | 0.6430 |
| 009 | 0.7594 | 0.6121 |
| 015 | 0.8162 | 0.6894 |
| 016 | 0.7789 | 0.6379 |
| 018 | 0.8457 | 0.7326 |
| 048 | 0.7164 | 0.5582 |
| 002 | 0.7686 | 0.6242 |
| 008 | 0.7451 | 0.5938 |
| 041 | 0.7473 | 0.5965 |

平均 Dice：0.7791

平均 IoU：0.6394

### Dice 最低的三张图像

1. 048_image.png：Dice = 0.7164
2. 008_image.png：Dice = 0.7451
3. 041_image.png：Dice = 0.7473

## 6. 实验现象与分析

### 6.1 训练过程

1. 第1～4轮 Dice 均为0，说明初期模型预测概率尚未达到0.5的二值化阈值。
2. 第5轮开始，模型逐渐识别出植被区域。
3. 随着训练进行，Train Loss 和 Val Loss 总体下降。
4. 后期验证 Dice 在0.76～0.78附近波动。
5. 第30轮取得最佳验证 Dice，为0.7791。

### 6.2 模型性能分析

本次 FCN 使用 ResNet50 作为 Backbone，总参数量为32.947M。

尽管参数量明显高于此前的 U-Net 和 DeepLabV3+，但验证 Dice 仅为0.7791。

说明模型参数量并不能直接决定语义分割性能。

可能的影响因素包括：

- 数据集规模较小，仅有48张训练图像。
- 未使用预训练权重。
- FCN 的特征融合与上采样结构。
- Backbone 与其他实验模型不同。

这些因素目前尚未通过消融实验单独验证，因此不能确定具体原因。

### 6.3 困难样本分析

FCN 的最低 Dice 样本为048、008、041。

与此前模型的逐图评价结果比较：

- 048_image.png 在 U-Net++、DeepLabV3+ 和 FCN 中均表现较差。
- 041_image.png 在三个模型中也均属于困难样本。
- 008_image.png 在 DeepLabV3+ 和 FCN 中表现较差。

说明部分图像可能具有较高的分割难度，值得后续进一步分析。

## 7. 与此前模型的初步比较

| 模型 | Backbone | 参数量 | Dice | IoU |
|---|---|---:|---:|---:|
| U-Net | ResNet18 | 15.970M | 0.9305 | 0.8701 |
| U-Net++ | ResNet18 | 15.970M | 0.9288 | 0.8672 |
| DeepLabV3+ | ResNet18 | 12.329M | 0.8259 | 0.7040 |
| FCN | ResNet50 | 32.947M | 0.7791 | 0.6394 |
| PSPNet | ResNet18 | 待实验 | 待实验 | 待实验 |

注：所有结果均为当前固定验证集上的实验结果，不代表独立测试集性能。由于 Backbone 和网络结构配置不同，当前比较属于初步实验对比。

## 8. 实验文件

训练代码：

- train_cwfid_fcn.py
- day4_evaluate_fcn.py

输出目录：

```text
outputs/
└── cwfid_FCN/
    ├── checkpoints/
    │   └── best_epoch_30.pth
    ├── loss_curve.png
    ├── metrics_curve.png
    ├── prediction_result.png
    └── per_image_metrics.csv
```

## 9. 实验总结

本次实验完成了 FCN-ResNet50 在 CWFID 数据集上的训练、最佳权重保存、训练曲线绘