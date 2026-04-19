#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results"
OUT_CSV="${OUT_CSV:-${RESULT_DIR}/perf_metrics.csv}"
SERIAL_BIN="${SERIAL_BIN:-${BUILD_DIR}/serial_train}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-12000}"
VAL_SAMPLES="${VAL_SAMPLES:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-128,64}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULT_DIR}"

bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

if [[ ! -x "${SERIAL_BIN}" ]]; then
  echo "Missing serial binary: ${SERIAL_BIN}" >&2
  echo "Build first with: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

"${SERIAL_BIN}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH_SIZE}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --lr "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --hidden "${HIDDEN}" \
  --data-dir "${DATA_DIR}" \
  --output "${OUT_CSV}"

python3 - "${OUT_CSV}" "${TRAIN_SAMPLES}" <<'PY'
import csv
import sys
from statistics import mean

csv_path = sys.argv[1]
train_samples = float(sys.argv[2])

with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit(f"No rows found in {csv_path}")

epoch_ms = [float(r["epoch_time_ms"]) for r in rows]
final = rows[-1]
avg_epoch_ms = mean(epoch_ms)
avg_epoch_s = avg_epoch_ms / 1000.0
samples_per_sec = train_samples / avg_epoch_s if avg_epoch_s > 0 else 0.0

print(f"Performance run complete: {csv_path}")
print(f"Epochs: {len(rows)}")
print(f"Average epoch time: {avg_epoch_ms:.2f} ms")
print(f"Min/Max epoch time: {min(epoch_ms):.2f}/{max(epoch_ms):.2f} ms")
print(f"Approx throughput: {samples_per_sec:.2f} samples/sec")
print(
    "Final metrics: "
    f"train_loss={float(final['train_loss']):.6f}, "
    f"train_acc={float(final['train_acc']):.6f}, "
    f"val_loss={float(final['val_loss']):.6f}, "
    f"val_acc={float(final['val_acc']):.6f}"
)
PY
