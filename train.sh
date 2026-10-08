#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) 
cd "$PROJECT_DIR"

exec "${YOLO_BIN:-yolo}" segment train \
  model="$PROJECT_DIR/weights/yolo11n-seg.pt" \
  data="$PROJECT_DIR/configs/mud_mingxian.yaml" \
  cfg="$PROJECT_DIR/configs/train_cfg.yaml" \
  epochs=100 patience=100 batch=16 imgsz=640 \
  rect=True \
  workers=8 close_mosaic=0 \
  project="$PROJECT_DIR/runs" \
  name=mud_mingxian_2cls "$@"
