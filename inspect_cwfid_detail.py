from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = Path("data/CWFID")

image_dir = root / "images"
mask_dir = root / "masks"
annotation_dir = root / "annotations"

# 检查三个文件夹的文件数量
print("========== 文件数量 ==========")

for folder in [image_dir, mask_dir, annotation_dir]:
    print(folder, ":", len(list(folder.iterdir())))

# 检查前三组图像
for i in range(1, 4):

    name = f"{i:03d}"

    image_path = image_dir / f"{name}_image.png"
    mask_path = mask_dir / f"{name}_mask.png"
    annotation_path = annotation_dir / f"{name}_annotation.png"
    yaml_path = annotation_dir / f"{name}_annotation.yaml"

    print(f"\n========== 样本 {name} ==========")

    # 读取原始文件，不提前转换颜色模式
    image_pil = Image.open(image_path)
    mask_pil = Image.open(mask_path)
    ann_pil = Image.open(annotation_path)

    image = np.array(image_pil)
    mask = np.array(mask_pil)
    annotation = np.array(ann_pil)

    print("图像模式:", image_pil.mode)
    print("图像尺寸:", image.shape)
    print("Mask模式:", mask_pil.mode)
    print("Mask尺寸:", mask.shape)
    print("Mask像素值:", np.unique(mask))

    print("Annotation模式:", ann_pil.mode)
    print("Annotation尺寸:", annotation.shape)

    # 检查标注图各通道的像素值
    if annotation.ndim == 3:
        for c, channel_name in enumerate(["R", "G", "B"]):
            print(
                f"{channel_name}通道像素值:",
                np.unique(annotation[:, :, c])
            )

    # 展示图像和标注
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Original Mask")

    axes[2].imshow(annotation)
    axes[2].set_title("Crop / Weed Annotation")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"outputs/cwfid_{name}_inspection.png",
        dpi=150
    )

    plt.show()

    # 输出 YAML 标注文件开头
    if yaml_path.exists():
        print("\nYAML标注内容：")

        with open(yaml_path, "r", encoding="utf-8") as f:
            print(f.read()[:1200])
