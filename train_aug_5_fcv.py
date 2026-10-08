#!/usr/bin/env python3
"""Train five folds with synthetic images added only to each fold's train set."""

import argparse
import json
from pathlib import Path

import yaml

from train_5_fcv import IMAGE_EXTENSIONS, train_fold


def image_files(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in (root / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate(images: list[Path], description: str) -> None:
    missing = [str(label_for(image)) for image in images if not label_for(image).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{description} 存在缺少标签的图片：\n" + "\n".join(missing)
        )


def write_fold(output: Path, fold: int, base_train: list[Path], val: list[Path], synthetic: list[Path], dataset_root: Path) -> Path:
    fold_dir = output / f"fold{fold}"
    manifests = fold_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    train = base_train + synthetic
    (manifests / "train.txt").write_text("".join(f"{p}\n" for p in train), encoding="utf-8")
    (manifests / "val.txt").write_text("".join(f"{p}\n" for p in val), encoding="utf-8")
    data = {
        "path": str(dataset_root.resolve()),
        "train": str((manifests / "train.txt").resolve()),
        "val": str((manifests / "val.txt").resolve()),
        "nc": 1,
        "names": {0: "puddle"},
    }
    data_path = fold_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, default=project / "runs/data_795_fcv")
    parser.add_argument("--output", type=Path, default=project / "runs/data_aug_fcv_100")
    parser.add_argument("--dataset", type=Path, default=project / "datasets/data_795")
    parser.add_argument("--synthetic", type=Path, nargs="+", default=[project / "datasets/aug", project / "datasets/changcheng_aug"])
    parser.add_argument("--model", type=Path, default=project / "weights/yolo11n-seg.pt")
    parser.add_argument("--cfg", type=Path, default=project / "configs/train_cfg.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold is not None and not 1 <= args.fold <= 5:
        raise ValueError("--fold 必须在 1..5 范围内")
    synthetic = []
    for root in args.synthetic:
        files = image_files(root.resolve())
        validate(files, str(root))
        synthetic.extend(files)
    if len({str(p) for p in synthetic}) != len(synthetic):
        raise ValueError("合成数据中存在重复图片路径")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"synthetic_images": len(synthetic), "folds": []}
    folds = range(1, 6) if args.fold is None else [args.fold]
    for fold in folds:
        fold_dir = args.fold_root / f"fold{fold}"
        train_manifest = fold_dir / "manifests/train.txt"
        val_manifest = fold_dir / "manifests/val.txt"
        if not train_manifest.is_file() or not val_manifest.is_file():
            raise FileNotFoundError(f"原五折清单不完整：{fold_dir}")
        base_train = [Path(x) for x in train_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        val = [Path(x) for x in val_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        validate(base_train + val, f"fold{fold} 原始数据")
        data_path = write_fold(args.output, fold, base_train, val, synthetic, args.dataset)
        summary["folds"].append({"fold": fold, "base_train": len(base_train), "synthetic_train": len(synthetic), "train": len(base_train) + len(synthetic), "val": len(val), "data": str(data_path.resolve())})
        print(f"fold{fold}: train={len(base_train) + len(synthetic)} (base={len(base_train)}, synthetic={len(synthetic)}), val={len(val)}", flush=True)
        if not args.prepare_only:
            train_fold(args, fold, data_path, output_root=args.output, run_name=f"fold{fold}")
    (args.output / "folds.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
