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
    "outputs/cwfid_unet_pretrained"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("================================")
print("CWFID U-Net pretained 实验")
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
        # ImageNet 标准化
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (image - mean) / std

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

indices = list(range(len(image_paths)))

random.Random(SEED).shuffle(indices)

train_indices = indices[:48]
val_indices = indices[48:]

train_paths = [
    image_paths[i] for i in train_indices
]

val_paths = [
    image_paths[i] for i in val_indices
]

train_dataset = CWFIDDataset(
    train_paths,
    augment=True
)

val_dataset = CWFIDDataset(
    val_paths,
    augment=False
)

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

print("训练集:", len(train_dataset))
print("验证集:", len(val_dataset))


# ==========================================
# 5. 创建 预训练U-Net
# ==========================================

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

print("预训练U-Net 创建成功！")


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

    # 保存最佳模型
    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            OUTPUT_DIR / "best_unet.pth"
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

model.load_state_dict(
    torch.load(
        OUTPUT_DIR / "best_unet.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

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
axes[2].set_title("Pretrained U-Net Prediction")

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
