from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# ==========================================
# 1. 实验配置
# ==========================================

SEED = 42

IMAGE_SIZE = (288, 384)  # 高、宽
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4

ROOT = Path("data/CWFID")
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"

OUTPUT_DIR = Path(
    "outputs/cwfid_deeplabv3plus"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("================================")
print("CWFID DeepLabV3+ 实验")
print("运行设备:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("================================")


# ==========================================
# 2. 检查图像和 Mask 配对
# ==========================================

image_paths = sorted(IMAGE_DIR.glob("*_image.png"))

assert len(image_paths) == 60, (
    f"预期60张图像，实际找到{len(image_paths)}张"
)

for image_path in image_paths:

    number = image_path.stem.replace("_image", "")

    mask_path = MASK_DIR / f"{number}_mask.png"

    assert mask_path.exists(), (
        f"缺少对应Mask: {mask_path}"
    )

print("图像与Mask配对检查通过！")
print("样本总数:", len(image_paths))


# ==========================================
# 3. 数据集读取
# ==========================================

class CWFIDDataset(Dataset):

    def __init__(self, paths, augment=False):

        self.paths = paths
        self.augment = augment

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, index):

        image_path = self.paths[index]

        number = image_path.stem.replace("_image", "")

        mask_path = MASK_DIR / f"{number}_mask.png"

        # 读取原始图像
        image = Image.open(
            image_path
        ).convert("RGB")

        # 读取原始Mask
        mask = Image.open(
            mask_path
        ).convert("L")

        # 调整尺寸
        height, width = IMAGE_SIZE

        image = image.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (width, height),
            Image.Resampling.NEAREST
        )

        # 图像归一化
        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # CWFID：黑色=植被，白色=背景
        # 转换后：1=植被，0=背景
        mask = (
            np.asarray(mask) < 128
        ).astype(np.float32)

        # 数据增强
        # 图像和Mask必须同步变换
        if self.augment:

            if random.random() < 0.5:

                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if random.random() < 0.5:

                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # 转换成 PyTorch Tensor
        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        mask = torch.from_numpy(
            mask[None, :, :].copy()
        )

        return image, mask


# ==========================================
# 4. 划分训练集与验证集
# ==========================================

# ==========================================
# 4. 划分训练集与验证集
# ==========================================

import json

with open(
    "outputs/experiment_split/split.json",
    "r",
    encoding="utf-8"
) as f:
    split = json.load(f)

train_paths = [
    IMAGE_DIR / name
    for name in split["train"]
]

val_paths = [
    IMAGE_DIR / name
    for name in split["val"]
]

# 检查数据划分
assert len(train_paths) == 48
assert len(val_paths) == 12

assert not set(train_paths) & set(val_paths)

print("\n数据划分检查通过！")
print("训练集:", len(train_paths))
print("验证集:", len(val_paths))

# ==========================================
# 4.1 创建 Dataset
# ==========================================

train_dataset = CWFIDDataset(
    train_paths,
    augment=True
)

val_dataset = CWFIDDataset(
    val_paths,
    augment=False
)

print("\nDataset 创建成功！")

# ==========================================
# 4.2 创建 DataLoader
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("\nDataLoader 创建成功！")

print("训练批次数:", len(train_loader))
print("验证批次数:", len(val_loader))


# ==========================================
# 5. 创建 deeplabv3+ 模型
# ==========================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
    decoder_channels=256
).to(DEVICE)

print("DeepLabV3+ 创建成功！")
total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"模型总参数量: {total_params / 1e6:.3f} M"
)

print(
    f"可训练参数量: {trainable_params / 1e6:.3f} M"
)

# ==========================================
# 6. BCE + Dice Loss
# ==========================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = (
        probabilities * targets
    ).sum(dim=dims)

    denominator = (
        probabilities.sum(dim=dims)
        + targets.sum(dim=dims)
    )

    dice = (
        2 * intersection + 1e-7
    ) / (
        denominator + 1e-7
    )

    return 1 - dice.mean()


def combined_loss(logits, targets):

    return (
        bce_loss(logits, targets)
        + dice_loss(logits, targets)
    )


# ==========================================
# 7. Dice 和 IoU
# ==========================================

def calculate_metrics(logits, targets):

    predictions = (
        torch.sigmoid(logits) > 0.5
    ).float()

    dims = (1, 2, 3)

    intersection = (
        predictions * targets
    ).sum(dim=dims)

    pred_sum = predictions.sum(dim=dims)
    target_sum = targets.sum(dim=dims)

    union = pred_sum + target_sum - intersection

    dice = (
        2 * intersection + 1e-7
    ) / (
        pred_sum + target_sum + 1e-7
    )

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return dice.sum().item(), iou.sum().item()


# ==========================================
# 8. 优化器
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_dice = -1.0

# 单独建立检查点文件夹
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 记录最佳模型文件路径
best_checkpoint_path = None

history = {
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": []
}


# ==========================================
# 9. 开始训练
# ==========================================

for epoch in range(EPOCHS):

    # -------- 训练阶段 --------

    model.train()

    total_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item() * images.size(0)
        )

    train_loss = (
        total_train_loss / len(train_dataset)
    )

    # -------- 验证阶段 --------

    model.eval()

    total_val_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            loss = combined_loss(
                logits,
                masks
            )

            dice, iou = calculate_metrics(
                logits,
                masks
            )

            total_val_loss += (
                loss.item() * images.size(0)
            )

            total_dice += dice
            total_iou += iou

    val_loss = (
        total_val_loss / len(val_dataset)
    )

    val_dice = (
        total_dice / len(val_dataset)
    )

    val_iou = (
        total_iou / len(val_dataset)
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)
    history["val_iou"].append(val_iou)

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Dice: {val_dice:.4f} "
        f"IoU: {val_iou:.4f}"
    )

    
# ==========================================
# 保存最佳模型
# ==========================================

    if val_dice > best_dice:

        best_dice = val_dice

        # 每次刷新最佳成绩，使用不同文件名
        save_path = (
            CHECKPOINT_DIR /
            f"best_epoch_{epoch + 1:02d}.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )

        best_checkpoint_path = save_path

        print(
            f"最佳模型已保存: {save_path}"
        )


# ==========================================
# 10. 保存训练曲线
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid()

plt.savefig(
    OUTPUT_DIR / "loss_curve.png",
    dpi=150
)

plt.close()


plt.figure(figsize=(9, 5))

plt.plot(
    history["val_dice"],
    label="Validation Dice"
)

plt.plot(
    history["val_iou"],
    label="Validation IoU"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")

plt.legend()
plt.grid()

plt.savefig(
    OUTPUT_DIR / "metrics_curve.png",
    dpi=150
)

plt.close()


# ==========================================
# 11. 使用最佳模型生成预测
# ==========================================

assert best_checkpoint_path is not None, (
    "训练期间没有成功保存最佳模型！"
)

model.load_state_dict(
    torch.load(
        best_checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )
)

print("成功加载最佳模型:", best_checkpoint_path)

model.eval()

image_tensor, mask_tensor = val_dataset[0]

with torch.no_grad():

    logits = model(
        image_tensor.unsqueeze(0).to(DEVICE)
    )

    prediction = (
        torch.sigmoid(logits) > 0.5
    ).float()

prediction = (
    prediction.squeeze().cpu().numpy()
)

image_show = (
    image_tensor.permute(1, 2, 0).numpy()
)

mask_show = mask_tensor.squeeze().numpy()

# 生成预测对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_show)
axes[0].set_title("Original Image")

axes[1].imshow(mask_show, cmap="gray")
axes[1].set_title("Ground Truth")

axes[2].imshow(prediction, cmap="gray")
axes[2].set_title("DeepLabV3+ Prediction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "prediction_result.png",
    dpi=150
)

plt.close()

print("\n================================")
print("训练完成！")
print("最佳验证 Dice:", best_dice)
print("结果保存位置:", OUTPUT_DIR)
print("================================")
