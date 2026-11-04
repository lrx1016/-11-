from pathlib import Path

root = Path("data/CWFID")

for folder in ["images", "masks", "annotations"]:
    path = root / folder

    print("\n文件夹:", folder)

    if not path.exists():
        print("文件夹不存在，请检查解压位置")
        continue

    files = list(path.iterdir())

    print("文件数量:", len(files))

    for file in files[:8]:
        print(file.name)
