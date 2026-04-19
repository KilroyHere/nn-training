#!/bin/bash
#SBATCH --job-name=nn-serial
#SBATCH --output=results/serial-%j.out
#SBATCH --error=results/serial-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --constraint=cpu

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${ROOT_DIR}/build"
SERIAL_BIN="${SERIAL_BIN:-${BUILD_DIR}/serial_train}"
SERIAL_EPOCHS="${SERIAL_EPOCHS:-10}"
SERIAL_BATCH_SIZE="${SERIAL_BATCH_SIZE:-64}"
SERIAL_TRAIN_SAMPLES="${SERIAL_TRAIN_SAMPLES:-4096}"
SERIAL_VAL_SAMPLES="${SERIAL_VAL_SAMPLES:-512}"
SERIAL_LEARNING_RATE="${SERIAL_LEARNING_RATE:-0.03}"
SERIAL_SEED="${SERIAL_SEED:-42}"
SERIAL_HIDDEN="${SERIAL_HIDDEN:-128,64}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${ROOT_DIR}/results"
bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

if [[ ! -x "${SERIAL_BIN}" ]]; then
  echo "Missing serial binary: ${SERIAL_BIN}" >&2
  echo "Build first with: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

srun "${SERIAL_BIN}" \
  --epochs "${SERIAL_EPOCHS}" \
  --batch "${SERIAL_BATCH_SIZE}" \
  --train-samples "${SERIAL_TRAIN_SAMPLES}" \
  --val-samples "${SERIAL_VAL_SAMPLES}" \
  --lr "${SERIAL_LEARNING_RATE}" \
  --seed "${SERIAL_SEED}" \
  --hidden "${SERIAL_HIDDEN}" \
  --data-dir "${ROOT_DIR}/data/mnist" \
  --output "${ROOT_DIR}/results/serial_metrics.csv"
