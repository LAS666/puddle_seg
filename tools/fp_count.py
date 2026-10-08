#!/usr/bin/env python3

from pathlib import Path


DATASET_ROOT = Path("./datasets/changcheng_aug")

IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def get_image_keys(image_dir: Path):
    """获取图片相对于 images 目录的路径，去掉扩展名。"""
    image_map = {}

    for path in image_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = path.relative_to(image_dir)
        key = relative.with_suffix("")

        image_map[key] = path

    return image_map


def get_label_keys(label_dir: Path):
    """获取标签相对于 labels 目录的路径，去掉 .txt 扩展名。"""
    label_map = {}

    for path in label_dir.rglob("*.txt"):
        if not path.is_file():
            continue

        relative = path.relative_to(label_dir)
        key = relative.with_suffix("")

        label_map[key] = path

    return label_map


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")

    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"标签目录不存在: {LABEL_DIR}")

    image_map = get_image_keys(IMAGE_DIR)
    label_map = get_label_keys(LABEL_DIR)

    image_keys = set(image_map)
    label_keys = set(label_map)

    matched = image_keys & label_keys

    # 有图片，没有 TXT
    missing_labels = image_keys - label_keys

    # 有 TXT，没有图片
    missing_images = label_keys - image_keys

    print("=" * 70)
    print("数据集匹配统计")
    print("=" * 70)

    print(f"图片总数:       {len(image_map)}")
    print(f"标签总数:       {len(label_map)}")
    print(f"成功匹配:       {len(matched)}")
    print(f"图片缺标签:     {len(missing_labels)}")
    print(f"标签缺图片:     {len(missing_images)}")

    print("\n" + "=" * 70)
    print(f"有图片但没有标签 ({len(missing_labels)})")
    print("=" * 70)

    if missing_labels:
        for key in sorted(missing_labels):
            print(image_map[key])
    else:
        print("无")

    print("\n" + "=" * 70)
    print(f"有标签但没有图片 ({len(missing_images)})")
    print("=" * 70)

    if missing_images:
        for key in sorted(missing_images):
            print(label_map[key])
    else:
        print("无")

    print("\n" + "=" * 70)

    if not missing_labels and not missing_images:
        print("✓ images 和 labels 完全匹配")
    else:
        print("⚠ 存在未匹配文件")


if __name__ == "__main__":
    main()