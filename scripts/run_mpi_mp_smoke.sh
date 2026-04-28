#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_CSV="${ROOT_DIR}/results/mpi_mp_smoke_metrics.csv"
MPI_BIN="${MPI_BIN:-${BUILD_DIR}/mpi_mp_train}"
MP_NODES="${MP_NODES:-1}"
MP_TASKS_PER_NODE="${MP_TASKS_PER_NODE:-2}"
MP_CPUS_PER_TASK="${MP_CPUS_PER_TASK:-1}"

SMOKE_EPOCHS="${SMOKE_EPOCHS:-3}"
SMOKE_GLOBAL_BATCH="${SMOKE_GLOBAL_BATCH:-${GLOBAL_BATCH:-128}}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-2048}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-256}"
SMOKE_LEARNING_RATE="${SMOKE_LEARNING_RATE:-0.03}"
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_HIDDEN="${SMOKE_HIDDEN:-128,64}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${ROOT_DIR}/results"
bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

if [[ ! -x "${MPI_BIN}" ]]; then
  echo "Missing MPI binary: ${MPI_BIN}" >&2
  echo "Build first with: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "srun not found in PATH. Run inside a Slurm environment." >&2
  exit 1
fi

if [[ "${MP_TASKS_PER_NODE}" == *"("* ]]; then
  MP_TASKS_PER_NODE="${MP_TASKS_PER_NODE%%(*}"
fi

MP_NTASKS=$((MP_NODES * MP_TASKS_PER_NODE))

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ALLOC_NODES="${SLURM_JOB_NUM_NODES:-${MP_NODES}}"
  ALLOC_TASKS="${SLURM_NTASKS:-${MP_NTASKS}}"
  if (( MP_NODES > ALLOC_NODES )) || (( MP_NTASKS > ALLOC_TASKS )); then
    echo "Requested srun resources exceed current allocation." >&2
    echo "  requested: nodes=${MP_NODES}, ntasks=${MP_NTASKS}" >&2
    echo "  allocated: nodes=${ALLOC_NODES}, ntasks=${ALLOC_TASKS}" >&2
    exit 1
  fi
fi

srun --nodes="${MP_NODES}" \
     --ntasks="${MP_NTASKS}" \
     --ntasks-per-node="${MP_TASKS_PER_NODE}" \
     --cpus-per-task="${MP_CPUS_PER_TASK}" \
     "${MPI_BIN}" \
     --epochs "${SMOKE_EPOCHS}" \
     --batch "${SMOKE_GLOBAL_BATCH}" \
     --train-samples "${SMOKE_TRAIN_SAMPLES}" \
     --val-samples "${SMOKE_VAL_SAMPLES}" \
     --lr "${SMOKE_LEARNING_RATE}" \
     --seed "${SMOKE_SEED}" \
     --hidden "${SMOKE_HIDDEN}" \
     --data-dir "${ROOT_DIR}/data/mnist" \
     --output "${OUT_CSV}"

echo "MPI Model-Parallel smoke run complete: ${OUT_CSV}"