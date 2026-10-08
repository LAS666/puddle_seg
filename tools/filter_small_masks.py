#!/usr/bin/env python3
"""Copy YOLO segmentation labels while removing masks smaller than a pixel threshold."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_image(dataset: Path, relative_label: Path) -> Path:
    # 只移除最后的 .txt；文件名本身可能包含多个点（例如 xxx_cur.txt）。
    image_name = relative_label.name[:-len(relative_label.suffix)]
    image_stem = dataset / "images" / relative_label.parent / image_name
    for extension in IMAGE_EXTENSIONS:
        candidate = Path(f"{image_stem}{extension}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"找不到标注对应的图片：{relative_label}（已检查 images/ 下常见图片扩展名）"
    )


def polygon_area_pixels(values: list[float], image_shape: tuple[int, int]) -> int:
    height, width = image_shape
    points = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("多边形至少需要 3 个点")
    points = (points * [width, height]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return int(np.count_nonzero(mask))


def filter_label_file(
    label_path: Path, output_path: Path, image_path: Path, min_area: int
) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image_shape = (image.height, image.width)

    kept = []
    removed = 0
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number} 包含非数字标注") from error
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"{label_path}:{line_number} 不是有效 YOLO 多边形标注")
        area = polygon_area_pixels(values, image_shape)
        if area >= min_area:
            kept.append(line)
        else:
            removed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept), removed


def filter_dataset(dataset: Path, min_area: int) -> tuple[int, int, int]:
    labels_root = dataset / "labels"
    output_root = dataset / "labels_300"
    if not labels_root.is_dir():
        raise FileNotFoundError(f"标注目录不存在：{labels_root}")
    if min_area < 0:
        raise ValueError("min-area 必须为非负整数")

    files = sorted(labels_root.rglob("*.txt"))
    processed = removed = kept = 0
    for label_path in files:
        relative = label_path.relative_to(labels_root)
        image_path = find_image(dataset, relative)
        kept_count, removed_count = filter_label_file(
            label_path, output_root / relative, image_path, min_area
        )
        processed += 1
        kept += kept_count
        removed += removed_count
    return processed, kept, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复制 YOLO 分割标注，并过滤像素面积小于阈值的多边形实例。"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed, kept, removed = filter_dataset(args.dataset, args.min_area)
    print(
        f"完成：处理 {processed} 个标注文件，保留 {kept} 个实例，"
        f"过滤 {removed} 个实例；输出：{(args.dataset / 'labels_300').resolve()}"
    )


if __name__ == "__main__":
    main()
