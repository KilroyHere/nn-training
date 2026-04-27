#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results"
OUT_CSV="${OUT_CSV:-${RESULT_DIR}/mpi_mp_perf_metrics.csv}"
MPI_BIN="${MPI_BIN:-${BUILD_DIR}/mpi_dp_train}"
MP_NODES="${MP_NODES:-4}"
MP_TASKS_PER_NODE="${MP_TASKS_PER_NODE:-1}"
MP_CPUS_PER_TASK="${MP_CPUS_PER_TASK:-1}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-128,64,32}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULT_DIR}"
bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

if [[ ! -x "${MPI_BIN}" ]]; then
  echo "Missing MPI MP binary: ${MPI_BIN}" >&2
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
    echo "Reduce MP_NODES/MP_TASKS_PER_NODE, or request a larger salloc." >&2
    exit 1
  fi
fi

echo "MPI MP perf launch config:"
echo "  allocation: job_id=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-unknown} ntasks=${SLURM_NTASKS:-unknown}"
echo "  srun args:  nodes=${MP_NODES} ntasks=${MP_NTASKS} tasks_per_node=${MP_TASKS_PER_NODE} cpus_per_task=${MP_CPUS_PER_TASK}"
echo "  train args: epochs=${EPOCHS} batch_size=${BATCH_SIZE} train_samples=${TRAIN_SAMPLES} val_samples=${VAL_SAMPLES}"

srun \
  --nodes "${MP_NODES}" \
  --ntasks "${MP_NTASKS}" \
  --cpus-per-task "${MP_CPUS_PER_TASK}" \
  --cpu-bind=cores \
  "${MPI_BIN}" \
  --mode mpi-mp \
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

print(f"MPI MP performance run complete: {csv_path}")
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