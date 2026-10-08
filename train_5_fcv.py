#!/usr/bin/env python3
"""Run deterministic five-fold cross-validation for the data_795 dataset."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(dataset_root: Path) -> list[Path]:
    images = []
    for split in ("train", "val"):
        image_root = dataset_root / "images" / split
        images.extend(
            path.resolve()
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    images = sorted(images)
    if not images:
        raise ValueError(f"没有找到图片：{dataset_root / 'images'}")
    return images


def label_path_for(image: Path) -> Path:
    parts = list(image.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"图片路径不包含 images 目录：{image}") from error
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_dataset(images: list[Path]) -> list[str]:
    missing = [str(label_path_for(image)) for image in images if not label_path_for(image).is_file()]
    if missing:
        raise FileNotFoundError(
            "存在没有标签文件的图片；请先补齐空标签或真实标签：\n"
            + "\n".join(missing)
        )
    return missing


def make_folds(images: list[Path], folds: int, seed: int) -> list[tuple[list[Path], list[Path]]]:
    if folds < 2 or folds > len(images):
        raise ValueError(f"folds 必须在 2..{len(images)} 范围内")
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    chunks = [shuffled[index::folds] for index in range(folds)]
    return [
        (
            [image for index, chunk in enumerate(chunks) if index != fold for image in chunk],
            list(chunks[fold]),
        )
        for fold in range(folds)
    ]


def write_fold_files(
    output_root: Path,
    fold_index: int,
    train_images: list[Path],
    val_images: list[Path],
    dataset_root: Path,
) -> Path:
    fold_root = output_root / f"fold{fold_index}"
    manifest_root = fold_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "train.txt").write_text(
        "".join(f"{image}\n" for image in train_images), encoding="utf-8"
    )
    (manifest_root / "val.txt").write_text(
        "".join(f"{image}\n" for image in val_images), encoding="utf-8"
    )
    data = {
        "path": str(dataset_root.resolve()),
        "train": str((manifest_root / "train.txt").resolve()),
        "val": str((manifest_root / "val.txt").resolve()),
        "nc": 1,
        "names": {0: "puddle"},
    }
    data_path = fold_root / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path


def train_fold(
    args: argparse.Namespace,
    fold_index: int,
    data_path: Path,
    output_root: Path | None = None,
    run_name: str | None = None,
) -> None:
    from ultralytics import YOLO

    project_root = (output_root or args.output).resolve()
    selected_name = run_name or f"fold{fold_index}"
    model_path = args.model.resolve()
    resume_path = project_root / selected_name / "weights" / "last.pt"
    if args.resume and resume_path.is_file():
        model_path = resume_path
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        cfg=str(args.cfg.resolve()),
        pretrained=True,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        rect=True,
        workers=args.workers,
        close_mosaic=0,
        project=str(project_root),
        name=selected_name,
        exist_ok=False,
        resume=args.resume and model_path.name == "last.pt",
    )


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=project / "datasets/data_795")
    parser.add_argument("--model", type=Path, default=project / "weights/yolo11n-seg.pt")
    parser.add_argument("--cfg", type=Path, default=project / "configs/train_cfg.yaml")
    parser.add_argument("--output", type=Path, default=project / "runs/data_795_fcv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fold", type=int, help="只运行指定折，例如 5")
    parser.add_argument(
        "--fold-path",
        type=Path,
        help="直接使用已有折目录，例如 runs/data_795_fcv/fold5",
    )
    parser.add_argument("--resume", action="store_true", help="从指定折目录中的 weights/last.pt 继续训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fold_path:
        fold_dir = args.fold_path.resolve()
        match = re.fullmatch(r"fold(\d+)", fold_dir.name)
        if not match:
            raise ValueError(f"--fold-path 目录名必须是 foldN：{fold_dir}")
        selected_fold = int(match.group(1))
        if args.fold and args.fold != selected_fold:
            raise ValueError("--fold 与 --fold-path 指定的折不一致")
        data_path = fold_dir / "data.yaml"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        train_fold(
            args,
            selected_fold,
            data_path,
            output_root=fold_dir.parent,
            run_name=fold_dir.name,
        )
        print(f"已运行 {fold_dir}，权重来源：{args.model if not args.resume else fold_dir / 'weights/last.pt'}")
        return

    images = collect_images(args.dataset.resolve())
    validate_dataset(images)
    folds = make_folds(images, args.folds, args.seed)
    if args.fold is not None and not 1 <= args.fold <= args.folds:
        raise ValueError(f"--fold 必须在 1..{args.folds} 范围内")
    summary = {"dataset": str(args.dataset.resolve()), "folds": args.folds, "seed": args.seed, "total_images": len(images), "results": []}

    for fold_index, (train_images, val_images) in enumerate(folds, 1):
        if args.fold is not None and fold_index != args.fold:
            continue
        data_path = write_fold_files(args.output, fold_index, train_images, val_images, args.dataset)
        summary["results"].append({"fold": fold_index, "train_images": len(train_images), "val_images": len(val_images), "data": str(data_path.resolve())})
        print(f"fold{fold_index}: train={len(train_images)} val={len(val_images)} data={data_path}", flush=True)
        if not args.prepare_only:
            train_fold(args, fold_index, data_path)

    (args.output / "folds.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "folds.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["fold", "train_images", "val_images", "data"])
        writer.writeheader()
        writer.writerows(summary["results"])


if __name__ == "__main__":
    main()
