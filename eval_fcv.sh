#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

if [ -n "${FCV_VISUALIZE_DIR:-}" ]; then
  set -- "$@" --visualize-dir "$FCV_VISUALIZE_DIR"
fi

# 与 eval.sh 同步；可通过环境变量或后面的命令行参数覆盖。
exec "${PYTHON:-python3}" "$PROJECT_DIR/eval_fcv.py" \
  --fold-root "${FCV_FOLD_ROOT:-runs/data_aug_fcv}" \
  --output "${FCV_OUTPUT:-runs/data_aug_fcv/eval}" \
  --min-instance-area "${FCV_MIN_INSTANCE_AREA:-800}" \
  --min-fp-area "${FCV_MIN_FP_AREA:-1200}" \
  --mask-overlap-threshold "${FCV_MASK_OVERLAP_THRESHOLD:-0.5}" "$@"
