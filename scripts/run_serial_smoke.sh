#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_CSV="${ROOT_DIR}/results/smoke_metrics.csv"
SERIAL_BIN="${SERIAL_BIN:-${BUILD_DIR}/serial_train}"

SMOKE_EPOCHS="${SMOKE_EPOCHS:-3}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-2048}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-256}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-64}"
SMOKE_LEARNING_RATE="${SMOKE_LEARNING_RATE:-0.03}"
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_HIDDEN="${SMOKE_HIDDEN:-128,64}"

mkdir -p "${ROOT_DIR}/results"

bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

if [[ ! -x "${SERIAL_BIN}" ]]; then
  echo "Missing serial binary: ${SERIAL_BIN}" >&2
  echo "Build first with: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

"${SERIAL_BIN}" \
  --epochs "${SMOKE_EPOCHS}" \
  --batch "${SMOKE_BATCH_SIZE}" \
  --train-samples "${SMOKE_TRAIN_SAMPLES}" \
  --val-samples "${SMOKE_VAL_SAMPLES}" \
  --lr "${SMOKE_LEARNING_RATE}" \
  --seed "${SMOKE_SEED}" \
  --hidden "${SMOKE_HIDDEN}" \
  --data-dir "${ROOT_DIR}/data/mnist" \
  --output "${OUT_CSV}"

echo "Smoke run complete: ${OUT_CSV}"
