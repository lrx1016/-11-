from pathlib import Path
import random
import json

SEED = 42

DATA_DIR = Path("data/CWFID")
OUTPUT_DIR = Path("outputs/experiment_split")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    (DATA_DIR / "images").glob("*_image.png")
)

assert len(image_paths) == 60, (
    f"预期60张图像，实际{len(image_paths)}张"
)

indices = list(range(len(image_paths)))

random.Random(SEED).shuffle(indices)

train_paths = [
    image_paths[i].name for i in indices[:48]
]

val_paths = [
    image_paths[i].name for i in indices[48:]
]

# 检查重复
assert not set(train_paths) & set(val_paths)

split = {
    "seed": SEED,
    "train": train_paths,
    "val": val_paths
}

with open(
    OUTPUT_DIR / "split.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        split,
        f,
        ensure_ascii=False,
        indent=4
    )

print("训练集数量:", len(train_paths))
print("验证集数量:", len(val_paths))
print("交集数量:", len(set(train_paths) & set(val_paths)))

print("划分清单已保存:")
print(OUTPUT_DIR / "split.json")
