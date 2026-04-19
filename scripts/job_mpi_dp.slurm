#!/bin/bash
#SBATCH --job-name=nn-mpi-dp
#SBATCH --output=results/mpi-dp-%j.out
#SBATCH --error=results/mpi-dp-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${ROOT_DIR}/build"
MPI_BIN="${MPI_BIN:-${BUILD_DIR}/mpi_dp_train}"
DP_NODES="${DP_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
DP_TASKS_PER_NODE="${DP_TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-64}}"
DP_EPOCHS="${DP_EPOCHS:-10}"
DP_GLOBAL_BATCH="${DP_GLOBAL_BATCH:-256}"
DP_TRAIN_SAMPLES="${DP_TRAIN_SAMPLES:-12000}"
DP_VAL_SAMPLES="${DP_VAL_SAMPLES:-2000}"
DP_LEARNING_RATE="${DP_LEARNING_RATE:-0.03}"
DP_SEED="${DP_SEED:-42}"
DP_HIDDEN="${DP_HIDDEN:-128,64}"

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

if [[ "${DP_TASKS_PER_NODE}" == *"("* ]]; then
  DP_TASKS_PER_NODE="${DP_TASKS_PER_NODE%%(*}"
fi

DP_NTASKS=$((DP_NODES * DP_TASKS_PER_NODE))

echo "MPI DP batch launch config:"
echo "  allocation: job_id=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-unknown} ntasks=${SLURM_NTASKS:-unknown}"
echo "  srun args:  nodes=${DP_NODES} ntasks=${DP_NTASKS} tasks_per_node=${DP_TASKS_PER_NODE} cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
echo "  train args: epochs=${DP_EPOCHS} global_batch=${DP_GLOBAL_BATCH} train_samples=${DP_TRAIN_SAMPLES} val_samples=${DP_VAL_SAMPLES}"

srun \
  --nodes "${DP_NODES}" \
  --ntasks "${DP_NTASKS}" \
  --cpus-per-task "${SLURM_CPUS_PER_TASK:-1}" \
  --cpu-bind=cores \
  "${MPI_BIN}" \
  --epochs "${DP_EPOCHS}" \
  --batch "${DP_GLOBAL_BATCH}" \
  --train-samples "${DP_TRAIN_SAMPLES}" \
  --val-samples "${DP_VAL_SAMPLES}" \
  --lr "${DP_LEARNING_RATE}" \
  --seed "${DP_SEED}" \
  --hidden "${DP_HIDDEN}" \
  --data-dir "${ROOT_DIR}/data/mnist" \
  --output "${ROOT_DIR}/results/mpi_dp_metrics.csv"
