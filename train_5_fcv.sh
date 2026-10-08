#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

FCV_FOLD_PATH="${FCV_FOLD_PATH:-}"

if [ -n "$FCV_FOLD_PATH" ]; then
  exec "${PYTHON:-python3}" "$PROJECT_DIR/train_5_fcv.py" \
    --fold-path "$FCV_FOLD_PATH" "$@"
fi

exec "${PYTHON:-python3}" "$PROJECT_DIR/train_5_fcv.py" "$@"
