#实验记录（20261104）
##网络：
Unet:
  Encoder:ResNet18
  loss：BCE+Dice

###数据集：
github上的cvfid数据集：https://github.com/cwfid/dataset
样本数：60
训练集：48
验证集：12
###训练参数：
Epochs: 30
Batch size: 4
Learning rate: 0.0001
###指标：
最后一个epoch的指标：Dice：0.9305，IoU:0.8701
