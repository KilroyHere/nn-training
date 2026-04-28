#!/bin/bash
#SBATCH --job-name=nn-mpi-mp
#SBATCH --output=results/mpi-mp-%j.out
#SBATCH --error=results/mpi-mp-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${ROOT_DIR}/build"
MPI_BIN="${MPI_BIN:-${BUILD_DIR}/mpi_mp_train}"
MP_NODES="${MP_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
MP_TASKS_PER_NODE="${MP_TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-2}}"
MP_CPUS_PER_TASK="${MP_CPUS_PER_TASK:-1}"
MP_EPOCHS="${MP_EPOCHS:-3}"
MP_GLOBAL_BATCH="${MP_GLOBAL_BATCH:-64}"
MP_TRAIN_SAMPLES="${MP_TRAIN_SAMPLES:-2048}"
MP_VAL_SAMPLES="${MP_VAL_SAMPLES:-256}"
MP_LEARNING_RATE="${MP_LEARNING_RATE:-0.03}"
MP_SEED="${MP_SEED:-42}"
MP_HIDDEN="${MP_HIDDEN:-128,64}"

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
    echo "Reduce MP_NODES/MP_TASKS_PER_NODE, or request a larger allocation." >&2
    exit 1
  fi
fi

echo "MPI MP batch launch config:"
echo "  allocation: job_id=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-unknown} ntasks=${SLURM_NTASKS:-unknown}"
echo "  srun args:  nodes=${MP_NODES} ntasks=${MP_NTASKS} tasks_per_node=${MP_TASKS_PER_NODE} cpus_per_task=${MP_CPUS_PER_TASK}"
echo "  train args: epochs=${MP_EPOCHS} global_batch=${MP_GLOBAL_BATCH} train_samples=${MP_TRAIN_SAMPLES} val_samples=${MP_VAL_SAMPLES}"

srun \
  --nodes "${MP_NODES}" \
  --ntasks "${MP_NTASKS}" \
  --cpus-per-task "${MP_CPUS_PER_TASK}" \
  --cpu-bind=cores \
  "${MPI_BIN}" \
  --epochs "${MP_EPOCHS}" \
  --batch "${MP_GLOBAL_BATCH}" \
  --train-samples "${MP_TRAIN_SAMPLES}" \
  --val-samples "${MP_VAL_SAMPLES}" \
  --lr "${MP_LEARNING_RATE}" \
  --seed "${MP_SEED}" \
  --hidden "${MP_HIDDEN}" \
  --data-dir "${ROOT_DIR}/data/mnist" \
  --output "${ROOT_DIR}/results/mpi_mp_metrics.csv"