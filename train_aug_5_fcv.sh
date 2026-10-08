#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

exec "${PYTHON:-python3}" "$PROJECT_DIR/train_aug_5_fcv.py" "$@"
