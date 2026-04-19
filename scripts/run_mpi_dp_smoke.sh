#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_CSV="${ROOT_DIR}/results/mpi_dp_smoke_metrics.csv"
MPI_BIN="${MPI_BIN:-${BUILD_DIR}/mpi_dp_train}"
DP_NODES="${DP_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
DP_TASKS_PER_NODE="${DP_TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-1}}"
DP_CPUS_PER_TASK="${DP_CPUS_PER_TASK:-1}"

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
  echo "Missing MPI DP binary: ${MPI_BIN}" >&2
  echo "Build first with: bash ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "srun not found in PATH. Run inside a Slurm environment." >&2
  exit 1
fi

if [[ "${DP_TASKS_PER_NODE}" == *"("* ]]; then
  DP_TASKS_PER_NODE="${DP_TASKS_PER_NODE%%(*}"
fi

DP_NTASKS=$((DP_NODES * DP_TASKS_PER_NODE))

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ALLOC_NODES="${SLURM_JOB_NUM_NODES:-${DP_NODES}}"
  ALLOC_TASKS="${SLURM_NTASKS:-${DP_NTASKS}}"
  if (( DP_NODES > ALLOC_NODES )) || (( DP_NTASKS > ALLOC_TASKS )); then
    echo "Requested srun resources exceed current allocation." >&2
    echo "  requested: nodes=${DP_NODES}, ntasks=${DP_NTASKS}" >&2
    echo "  allocated: nodes=${ALLOC_NODES}, ntasks=${ALLOC_TASKS}" >&2
    echo "Reduce DP_NODES/DP_TASKS_PER_NODE, or request a larger salloc." >&2
    exit 1
  fi
fi

echo "MPI DP smoke launch config:"
echo "  allocation: job_id=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-unknown} ntasks=${SLURM_NTASKS:-unknown}"
echo "  srun args:  nodes=${DP_NODES} ntasks=${DP_NTASKS} tasks_per_node=${DP_TASKS_PER_NODE} cpus_per_task=${DP_CPUS_PER_TASK}"
echo "  train args: epochs=${SMOKE_EPOCHS} global_batch=${SMOKE_GLOBAL_BATCH} train_samples=${SMOKE_TRAIN_SAMPLES} val_samples=${SMOKE_VAL_SAMPLES}"

srun \
  --nodes "${DP_NODES}" \
  --ntasks "${DP_NTASKS}" \
  --cpus-per-task "${DP_CPUS_PER_TASK}" \
  --cpu-bind=cores \
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

echo "MPI DP smoke run complete: ${OUT_CSV} (nodes=${DP_NODES}, ntasks=${DP_NTASKS})"
