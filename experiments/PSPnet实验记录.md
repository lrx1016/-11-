# PSPNet 实验记录（Day 4）

## 1. 实验目的

使用 PSPNet 完成 CWFID 植被语义分割任务，理解金字塔池化模块（PPM）的工作原理，并与 U-Net、U-Net++、DeepLabV3+、FCN 进行对比。

## 2. 网络结构

- 网络：PSPNet
- Backbone：ResNet18
- 核心模块：Pyramid Pooling Module（PPM）
- 输入通道：3
- 输出类别：1

模型总参数量：11.330 M  
可训练参数量：11.330 M

## 3. 数据集

- 数据集：CWFID
- 总样本数：60
- 训练集：48
- 验证集：12
- 输入尺寸：288×384
- 使用固定随机种子 42 的数据划分文件。

## 4. 训练参数

| 参数 | 设置 |
|---|---|
| Epochs | 30 |
| Batch Size | 4 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | BCE + Dice |
| Encoder | ResNet18 |
| Encoder Weights | None |

## 5. 实验结果

- 最佳 Epoch：28
- 最佳验证 Dice：0.7671
- 对应验证 IoU：0.6231
- 最终 Epoch Train Loss：0.3477
- 最终 Epoch Val Loss：0.3729

## 6. 结果分析

训练 Loss 整体持续下降，验证 Loss 前期波动较明显，后期趋于稳定。

验证 Dice 在第 28 轮达到最高值 0.7671，后期进入平台期。

预测可视化表明，模型能够识别部分主要植被区域，但对于细小叶片、细长枝叶和相邻叶片边界的分割仍存在不足，预测结果具有一定平滑现象。

## 7. 初步结论

在当前 CWFID 数据集、ResNet18 编码器及训练配置下，PSPNet 的验证 Dice 为 0.7671，低于此前 U-Net、U-Net++、DeepLabV3+ 和 FCN 实验结果。

当前结果说明，PSPNet 在本次实验条件下没有表现出性能优势，但不能据此推断其在其他农业分割数据集上的表现。

## 8. 实验文件

- train_cwfid_pspnet.py
- day4_evaluate_pspnet.py
- outputs/cwfid_PSPnet/checkpoints/
- outputs/cwfid_PSPnet/loss_curve.png
- outputs/cwfid_PSPnet/metrics_curve.png
- outputs/cwfid_PSPnet/prediction_result.png
- outputs/cwfid_PSPnet/per_image_metrics.csv