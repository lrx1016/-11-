from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp

# =====================================
# 1. 基础配置
# =====================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = Path("data/CWFID")
MASK_DIR = DATA_DIR / "masks"

OUTPUT_DIR = Path("outputs/cwfid_FCN")
SPLIT_PATH = Path("outputs/experiment_split/split.json")

# 注意：这里是 (高度, 宽度)
IMAGE_SIZE = (288, 384)

# =====================================
# 2. 读取固定的数据划分
# =====================================

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)

train_names = split["train"]
val_names = split["val"]

assert len(train_names) == 48
assert len(val_names) == 12
assert not set(train_names) & set(val_names)

print("训练集数量:", len(train_names))
print("验证集数量:", len(val_names))
print("数据划分随机种子:", split["seed"])

# =====================================
# 3. 创建 FCN 模型
# =====================================

from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(
    weights=None,
    weights_backbone=None,
    num_classes=1,
    aux_loss=False
).to(DEVICE)

def get_logits(model, images):

    output = model(images)

    if isinstance(output, dict):
        return output["out"]

    return output
# 加载最佳权重
weight_path = (
    Path("outputs/cwfid_FCN/checkpoints/best_epoch_30.pth"))

model.load_state_dict(
    torch.load(
        weight_path,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("\n模型权重加载成功")
print("运行设备:", DEVICE)

# =====================================
# 4. 逐张评价
# =====================================

results = []

for image_name in val_names:

    # 获取图像编号
    number = image_name.replace(
        "_image.png", ""
    )

    image_path = DATA_DIR / "images" / image_name

    mask_path = MASK_DIR / f"{number}_mask.png"

    assert image_path.exists(), image_path
    assert mask_path.exists(), mask_path

    # 读取图像
    image = Image.open(
        image_path
    ).convert("RGB")

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

    # 与原训练 Dataset 保持相同的预处理
    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    mask = (
        np.asarray(mask) < 128
    ).astype(np.uint8)

    # 转换为 Tensor
    image_tensor = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(DEVICE)

    # 模型预测
    with torch.no_grad():

        logits = get_logits(model, image_tensor)

        probability = torch.sigmoid(logits)

        prediction = (
            probability > 0.5
        ).cpu().numpy()[0, 0]

    # 转换为布尔数组
    prediction = prediction.astype(bool)
    ground_truth = mask.astype(bool)

    # 计算交集和并集
    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    pred_sum = prediction.sum()
    gt_sum = ground_truth.sum()

    # Dice
    dice = (
        2 * intersection + 1e-7
    ) / (
        pred_sum + gt_sum + 1e-7
    )

    # IoU
    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    results.append({
        "image": image_name,
        "dice": float(dice),
        "iou": float(iou)
    })

    print(
        f"{image_name} | "
        f"Dice: {dice:.4f} | "
        f"IoU: {iou:.4f}"
    )

# =====================================
# 5. 汇总评价结果
# =====================================

dice_values = [
    item["dice"] for item in results
]

iou_values = [
    item["iou"] for item in results
]

mean_dice = np.mean(dice_values)
mean_iou = np.mean(iou_values)

print("\n========== 验证集评价结果 ==========")

print("验证图像数量:", len(results))

print(f"平均 Dice: {mean_dice:.4f}")

print(f"平均 IoU: {mean_iou:.4f}")

print("\n========== Dice 最低的三张图像 ==========")

for item in sorted(
    results,
    key=lambda x: x["dice"]
)[:3]:

    print(
        item["image"],
        f"Dice={item['dice']:.4f}",
        f"IoU={item['iou']:.4f}"
    )

# =====================================
# 6. 保存 CSV
# =====================================

csv_path = (
    Path("outputs/cwfid_FCN") / "per_image_metrics.csv"

)
with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["image", "dice", "iou"]
    )

    writer.writeheader()
    writer.writerows(results)

print("\n逐图评价结果已保存:")
print(csv_path)
