#!/usr/bin/env bash

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR" || exit 1

exec "${PYTHON:-python3}" eval.py \
  --data "${EVAL_DATA:-configs/mud_mingxian.yaml}" \
  --weights "${EVAL_WEIGHTS:-runs/mud_mingxian_2cls-2/weights/best.pt}" \
  --split val \
  --device 0 \
  --imgsz 640 \
  --instance-iou 0.5 \
  --conf 0.25 \
  --min-instance-area 800 \
  --min-fp-area 1200 \
  --gt-overlap last \
  --output "${EVAL_OUTPUT:-runs/mud_mingxian_2cls-2/eval/pixel_metrics.csv}" \
  --visualize-dir "${EVAL_VISUALIZE_DIR:-./runs/mud_mingxian_2cls-2/eval}" "$@"
