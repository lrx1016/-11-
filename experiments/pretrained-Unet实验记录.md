# 实验记录：ImageNet 预训练 U-Net（2026-11-09）

## 一、实验目的

本次实验主要研究迁移学习（Transfer Learning）在农业图像语义分割任务中的应用。

此前的原始 U-Net 使用随机初始化的编码器权重，本次实验改为加载 ImageNet 预训练的 ResNet18 编码器，观察预训练权重能否帮助模型更好地提取农业图像中的叶片特征。

本次实验希望回答：

1. 使用 ImageNet 预训练权重是否能够提高分割性能？
2. 预训练权重是否能够帮助模型更快地收敛？
3. 在小规模农业数据集上，迁移学习是否具有优势？

## 二、模型结构

**模型名称：** U-Net（ImageNet Pretrained）

**模型结构：**
- Encoder：ResNet18
- Decoder：U-Net 解码器
- Encoder 权重：ImageNet 预训练权重
- 输出类别：1（二分类）
- 输出形式：Logits

与原始 U-Net 相比，本次实验没有改变 U-Net 的基本编码器—解码器结构，主要改变的是 Encoder 的初始化方式。

原始 U-Net：随机初始化权重。

预训练 U-Net：使用 ImageNet 预训练权重初始化 ResNet18。

## 三、数据集

数据集：CWFID（Corn and Weed Field Image Dataset）

数据集来源：https://github.com/cwfid/dataset

| 项目 | 配置 |
|---|---|
| 原始图像数量 | 60 |
| 图像尺寸 | 966×1296 |
| 训练集 | 48 |
| 验证集 | 12 |
| 随机种子 | 42 |
| 输入尺寸 | 288×384 |
| 标签类型 | 植被 / 背景 |

训练集和验证集沿用此前保存的 `split.json`，保证与原始 U-Net 实验使用相同的数据划分。

## 四、训练参数

| 参数 | 设置 |
|---|---|
| Epochs | 30 |
| Batch Size | 4 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | BCE + Dice |
| Encoder | ResNet18 |
| Encoder Weights | ImageNet |
| 输入归一化 | ImageNet Mean / Std |
| GPU | NVIDIA RTX 5060 Laptop GPU |
| 训练设备 | CUDA |

ImageNet 标准化参数：

- Mean = [0.485, 0.456, 0.406]
- Std = [0.229, 0.224, 0.225]

## 五、训练结果

训练过程中，Train Loss 和 Validation Loss 总体持续下降，模型能够正常收敛。

最终训练结果：

| 指标 | Epoch 30 |
|---|---:|
| Train Loss | 0.5257 |
| Validation Loss | 0.4878 |
| Validation Dice | 0.9146 |
| Validation IoU | 0.8429 |

最佳验证 Dice：0.9146。

最佳验证 IoU：0.8429。

## 六、逐图评价

对固定验证集的12张图像进行逐图评价。

| 图像 | Dice | IoU |
|---|---:|---:|
| 052_image | 0.9336 | 0.8754 |
| 044_image | 0.9342 | 0.8765 |
| 007_image | 0.9354 | 0.8787 |
| 057_image | 0.9275 | 0.8648 |
| 009_image | 0.9302 | 0.8696 |
| 015_image | 0.9312 | 0.8713 |
| 016_image | 0.9318 | 0.8723 |
| 018_image | 0.9475 | 0.9003 |
| 048_image | 0.9191 | 0.8502 |
| 002_image | 0.9158 | 0.8447 |
| 008_image | 0.9215 | 0.8545 |
| 041_image | 0.9177 | 0.8478 |

平均 Dice：0.9146。

平均 IoU：0.8430。

注意：以上逐图评价结果来自后续完成 ImageNet 标准化修正的评价代码。最终复现实验时，应确保训练和评价使用完全一致的预处理。

## 七、与原始 U-Net 对比

| 模型 | 验证 Dice | 验证 IoU |
|---|---:|---:|
| 原始 U-Net | 0.9305 | 0.8701 |
| ImageNet 预训练 U-Net | 0.9146 | 0.8429 |

从当前实验结果来看，预训练 U-Net 没有超过原始 U-Net。

这说明使用预训练权重并不意味着一定能够获得更高的分割精度。

可能的影响因素包括：
1. 当前数据集规模较小。
2. ImageNet 与农业田间图像存在一定的领域差异。
3. 预训练权重的优势可能受到训练策略和超参数影响。

以上属于可能的解释，尚未通过专门消融实验验证。

## 八、实验中遇到的问题

训练期间出现以下提示：

1. Hugging Face 未认证请求提示：不影响本次权重下载和模型训练。
2. Windows 不支持符号链接提示：不影响模型运行，主要影响缓存空间利用。
3. Matplotlib 显示图像时出现数值范围警告：由于输入图像经过 ImageNet 标准化，直接使用 imshow 显示时超出 [0,1] 范围。可通过反标准化后再显示解决。

## 九、实验结论

本次实验成功实现了使用 ImageNet 预训练 ResNet18 编码器的 U-Net 语义分割模型。

通过本次实验，进一步理解了迁移学习、预训练权重、图像标准化等概念。

虽然本次预训练 U-Net 的验证指标没有超过原始 U-Net，但实验说明预训练模型在农业小样本任务中的实际效果仍需要结合数据特点和训练策略进行验证。

## 十、结果文件

实验输出目录：

`outputs/cwfid_unet_pretrained/`

主要文件：
- `loss_curve.png`：训练与验证损失曲线
- `metrics_curve.png`：验证集 Dice、IoU 曲线
- `prediction_result.png`：预测结果可视化
- 模型最佳权重文件（以实际保存文件名为准）

