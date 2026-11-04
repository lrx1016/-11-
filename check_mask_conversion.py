from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = Path("data/CWFID")

image = np.array(
    Image.open(root / "images/001_image.png").convert("RGB")
)

# 保留原始 Mask
mask_raw = np.array(
    Image.open(root / "masks/001_mask.png")
)

# 将植被转换为 1，背景转换为 0
mask_binary = (mask_raw == 0).astype(np.uint8)

# 读取作物与杂草标注
annotation = np.array(
    Image.open(
        root / "annotations/001_annotation.png"
    ).convert("RGB")
)

weed = annotation[:, :, 0] > 127
crop = annotation[:, :, 1] > 127

# 合并作物与杂草
vegetation = crop | weed

# 检查两种标注的差异
difference = np.logical_xor(
    mask_binary == 1,
    vegetation
)

print("原始Mask像素值:", np.unique(mask_raw))
print("转换后Mask像素值:", np.unique(mask_binary))
print("植被像素比例:", mask_binary.mean())
print("作物像素比例:", crop.mean())
print("杂草像素比例:", weed.mean())
print("Mask与Annotation不一致比例:", difference.mean())

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")

axes[0, 1].imshow(mask_binary, cmap="gray")
axes[0, 1].set_title("Corrected Vegetation Mask")

axes[1, 0].imshow(vegetation, cmap="gray")
axes[1, 0].set_title("Crop + Weed")

axes[1, 1].imshow(difference, cmap="gray")
axes[1, 1].set_title("Annotation Difference")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "outputs/mask_conversion_check.png",
    dpi=150
)
plt.show()
