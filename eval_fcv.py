#!/usr/bin/env python3
"""Evaluate each trained cross-validation fold on its own validation manifest."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from tools import fp as fp_logic

COUNT_FIELDS = ["GT_TP", "Prediction_TP", "FP_Count", "Instance_FN", "Small_FP_Ignore", "Fragment_Ignore", "Duplicate_Ignore"]

from eval import (
    INSTANCE_MIOU_DEFINITION,
    InstanceMetrics,
    load_names,
    PixelMetrics,
    combine_rows,
    empty_instances,
    filter_small_instances,
    merge_instances,
    remove_small_regions,
    save_visualizations,
    yolo_txt_to_instances,
)


def label_path_for(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def latest_weight(fold_root: Path, fold: int) -> Path:
    candidates = sorted(
        fold_root.glob(f"fold{fold}*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"找不到 fold{fold} 的 best.pt：{fold_root}")
    return candidates[0]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: "N/A" if value is None else value for key, value in row.items()} for row in rows)


def evaluate_fold(fold: int, fold_root: Path, args: argparse.Namespace) -> list[dict]:
    from ultralytics import YOLO

    fold_dir = fold_root / f"fold{fold}"
    manifest = fold_dir / "manifests/val.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    images = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = yaml.safe_load((fold_dir / "data.yaml").read_text(encoding="utf-8"))
    names = load_names(data)
    model_path = latest_weight(fold_root, fold)
    model = YOLO(str(model_path.resolve()))
    if model.task != "segment":
        raise ValueError(f"不是分割模型：{model_path}")
    model_names = [str(model.names[i]) for i in range(len(model.names))]
    if model_names != names:
        raise ValueError(f"类别不一致：{model_names} != {names}")

    pixel = PixelMetrics(names)
    instance = InstanceMetrics(
        names, args.instance_iou, overlap_threshold=args.mask_overlap_threshold,
        overlap_class_agnostic=args.mask_overlap_class_agnostic,
        min_area=args.min_instance_area, min_fp_area=args.min_fp_area,
    )
    for index, image in enumerate(images, 1):
        with Image.open(image) as opened:
            shape = (opened.height, opened.width)
            if opened.getexif().get(274, 1) != 1:
                raise ValueError(f"存在 EXIF 旋转：{image}")
        gt, gt_masks, gt_classes = yolo_txt_to_instances(
            label_path_for(image), shape, len(names), args.gt_overlap
        )
        # 和 eval.py 一致：像素 GT 不过滤；实例 GT 使用 fp.py 的栅格化。
        gt_masks, gt_classes = fp_logic.load_gt_masks(label_path_for(image), shape, return_classes=True)
        gt_masks = np.asarray(gt_masks, dtype=bool)
        if len(gt_masks) == 0:
            gt_masks = np.zeros((0, *shape), dtype=bool)
        valid = gt != 255
        predict_kwargs = dict(
            source=str(image), conf=args.conf, iou=args.nms_iou,
            max_det=args.max_det, imgsz=args.imgsz, retina_masks=True,
            verbose=False, save=False,
        )
        if args.device is not None:
            predict_kwargs["device"] = args.device
        result = model.predict(**predict_kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            pred = np.zeros(shape, dtype=np.uint8)
            pred_masks, pred_classes, pred_scores = empty_instances(shape)
        else:
            if result.masks is None:
                raise ValueError(f"有检测框但无分割掩码：{image}")
            raw_masks = result.masks.data.cpu().numpy()
            if raw_masks.shape[1:] != shape:
                raise ValueError(f"mask 尺寸错误：{image} {raw_masks.shape[1:]} != {shape}")
            pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            pred_scores = result.boxes.conf.cpu().numpy().astype(np.float64)
            pred_masks = raw_masks > 0.5
            pred_masks, pred_classes, pred_scores = filter_small_instances(
                pred_masks, pred_classes, pred_scores,
                min_area=args.min_instance_area,
            )
            pred = merge_instances(pred_masks, pred_classes, pred_scores, shape, len(names))
            pred = remove_small_regions(pred, args.min_instance_area)
        pixel.update(pred, gt)
        instance.update(pred_masks, pred_classes, pred_scores, gt_masks, gt_classes, valid)
        if args.visualize_dir:
            save_visualizations(
                image,
                pred,
                gt,
                args.visualize_dir / f"fold{fold}",
                Path(image.name),
                len(names),
            )
        print(f"fold{fold}: {index}/{len(images)} {image.name}", flush=True)

    rows = combine_rows(pixel.rows(), instance.rows())
    output_dir = args.output / f"fold{fold}"
    write_rows(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.json").write_text(json.dumps({"fold": fold, "weights": str(model_path), "images": len(images), "per_class": rows, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "gt_coverage_threshold": fp_logic.GT_COVERAGE_THRESHOLD}, ensure_ascii=False, indent=2), encoding="utf-8")
    return [{
        "fold": fold,
        "class_id": cid,
        "name": names[cid],
        "weights": str(model_path),
        "images": len(images),
        "Recall": row["Recall"],
        "IoU": row["IoU"],
        "Pixel_FPR": row["Pixel_FPR"],
        "FP": row["FP"],
        "Instance_Precision": row["Instance_Precision"],
        "Instance_Recall": row["Instance_Recall"],
        "Instance_mIoU": row["Instance_mIoU"],
        **{key: row[key] for key in COUNT_FIELDS},
    } for cid, row in enumerate(rows)]


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, default=project / "runs/data_aug_fcv")
    parser.add_argument("--output", type=Path, default=project / "runs/data_aug_fcv/eval")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--instance-iou", type=float, default=0.5)
    parser.add_argument("--min-instance-area", type=int, default=fp_logic.MIN_INSTANCE_AREA)
    parser.add_argument("--min-fp-area", type=int, default=fp_logic.MIN_FP_AREA)
    parser.add_argument("--mask-overlap-threshold", type=float, default=fp_logic.MASK_OVERLAP_THRESHOLD)
    parser.add_argument("--mask-overlap-class-agnostic", action="store_true", default=fp_logic.MASK_OVERLAP_CLASS_AGNOSTIC)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        help="可选四联图输出目录；不传则不保存四联图",
    )
    parser.add_argument("--gt-overlap", choices=["error", "first", "last"], default="last")
    args = parser.parse_args()
    if args.min_instance_area < 0 or args.min_fp_area < 0:
        parser.error("面积阈值不能为负数")
    if args.folds < 1 or args.imgsz < 1 or args.max_det < 1:
        parser.error("folds/imgsz/max-det 必须为正数")
    if not all(0 <= v <= 1 for v in (args.conf, args.nms_iou, args.instance_iou, args.mask_overlap_threshold)):
        parser.error("置信度和交叠阈值必须在 0..1")
    return args


def print_table(headers, rows):
    """按实际内容计算列宽；首列左对齐，指标列右对齐。"""
    cells = [list(map(str, headers))] + [list(map(str, row)) for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    print()
    for row in cells:
        print("  ".join(
            value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
            for i, value in enumerate(row)
        ))


def main() -> None:
    args = parse_args()
    fold_results = [evaluate_fold(fold, args.fold_root.resolve(), args) for fold in range(1, args.folds + 1)]
    expected_classes = [(r['class_id'], r['name']) for r in fold_results[0]]
    if any([(r['class_id'], r['name']) for r in rows] != expected_classes for rows in fold_results):
        raise ValueError('各折类别定义不一致，不能合并统计')
    results = [row for rows in fold_results for row in rows]
    metric_names = ["Recall", "IoU", "Pixel_FPR", "FP", "Instance_Precision", "Instance_Recall","Instance_mIoU",]
    summaries = []
    for cid, class_name in expected_classes:
        class_rows = [row for row in results if row['class_id'] == cid]
        averages = {}
        for name in metric_names + COUNT_FIELDS:
            values = [row[name] for row in class_rows if row[name] is not None]
            averages[name] = float(np.mean(values)) if values else None
        totals = {name: sum(row[name] for row in class_rows) for name in COUNT_FIELDS}
        summaries.append(dict(class_id=cid, name=class_name, macro_average=averages, total_counts=totals))
    write_rows(args.output / "fold_metrics.csv", results)
    (args.output / "summary.json").write_text(json.dumps(
        {"folds": results, "per_class": summaries,
         **({"macro_average": summaries[0]['macro_average'], "total_counts": summaries[0]['total_counts']} if len(summaries) == 1 else {}),
         "definitions": {"Instance_mIoU": INSTANCE_MIOU_DEFINITION,
                         "Instance_Recall": "GT_TP/(GT_TP+Instance_FN)",
                         "FP": "FP_Count/(Prediction_TP+FP_Count); Fragment and small FP excluded"}},
        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print('Instance_mIoU: 匹配预测与其 GT 并集的平均 IoU；avg 为有效折的等权平均')
    for summary in summaries:
        print(f"\nclass {summary['class_id']}: {summary['name']}")
        class_rows = [(str(r['fold']), r) for r in results if r['class_id'] == summary['class_id']]
        print_table(['fold'] + metric_names, [
            [label] + ['N/A' if row[k] is None else f'{row[k]:.2%}' for k in metric_names]
            for label, row in class_rows + [('avg', summary['macro_average'])]
        ])
        print_table(['fold'] + COUNT_FIELDS, [
            [label] + [f'{row[k]:g}' for k in COUNT_FIELDS]
            for label, row in class_rows + [('avg', summary['macro_average']), ('total', summary['total_counts'])]
        ])


if __name__ == "__main__":
    main()
