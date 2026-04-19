#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results"
OUT_CSV="${OUT_CSV:-${RESULT_DIR}/mpi_dp_perf_metrics.csv}"
MPI_BIN="${MPI_BIN:-${BUILD_DIR}/mpi_dp_train}"
DP_NODES="${DP_NODES:-${SLURM_JOB_NUM_NODES:-2}}"
DP_TASKS_PER_NODE="${DP_TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-1}}"
DP_CPUS_PER_TASK="${DP_CPUS_PER_TASK:-1}"

EPOCHS="${EPOCHS:-10}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-128,64}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULT_DIR}"
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

echo "MPI DP perf launch config:"
echo "  allocation: job_id=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-unknown} ntasks=${SLURM_NTASKS:-unknown}"
echo "  srun args:  nodes=${DP_NODES} ntasks=${DP_NTASKS} tasks_per_node=${DP_TASKS_PER_NODE} cpus_per_task=${DP_CPUS_PER_TASK}"
echo "  train args: epochs=${EPOCHS} global_batch=${GLOBAL_BATCH} train_samples=${TRAIN_SAMPLES} val_samples=${VAL_SAMPLES}"

srun \
  --nodes "${DP_NODES}" \
  --ntasks "${DP_NTASKS}" \
  --cpus-per-task "${DP_CPUS_PER_TASK}" \
  --cpu-bind=cores \
  "${MPI_BIN}" \
  --epochs "${EPOCHS}" \
  --batch "${GLOBAL_BATCH}" \
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

print(f"MPI DP performance run complete: {csv_path}")
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
