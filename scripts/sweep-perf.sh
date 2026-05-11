#!/usr/bin/env bash
# sweep-perf.sh
#
# Chart 2: Model-parallel vs pipeline
#          Fixed architecture, fixed ws=8 (2x4) — vary batch size
#
#   Batch sizes: 256, 512, 1024, 2048, 4096
#
# Total: 5 batch sizes x 2 modes = 10 runs x EPOCHS
# Estimated time at EPOCHS=10: ~1.5 hours
#
# Usage:
#   bash scripts/sweep-perf.sh
#   EPOCHS=5 bash scripts/sweep-perf.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results/chart2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"

MASTER_CSV="${RESULT_DIR}/chart2_results.csv"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"
MP_BIN="${BUILD_DIR}/mpi_mp_train"
PIP_BIN="${BUILD_DIR}/mpi_mp_pip_train"

EPOCHS="${EPOCHS:-10}"
TRAIN_SAMPLES=50000
VAL_SAMPLES=10000
SEED=42
BASE_LR=0.03
BASE_BATCH=64

HIDDEN="512,512,512,512,512,512,512"   # 8 layers

# Fixed layout: ws=8, 2 nodes x 4 tasks
WS=8
NODES=2
TASKS=4
LAYOUT="2x4_ws8"

# Batch sizes and matching microbatch counts.
# Microbatch count chosen so mb_size >= 32 rows and efficiency >= 80%.
#   batch  mb   mb_size  efficiency
#     256   8      32     80%(8/10)
#     512  16      32     84%(16/21) -- wait ws=8 so 16/(16+7)=70%
#   Let's recalculate properly: efficiency = mb/(mb+ws-1) = mb/(mb+7)
#   For 80%: mb >= 4*7 = 28, round up to nearest divisor
#     256: divisors>=28: 32,64,128,256 -> 32 (256/32=8 rows, small but ok)
#     512: 32 -> 512/32=16 rows
#    1024: 32 -> 1024/32=32 rows
#    2048: 32 -> 2048/32=64 rows
#    4096: 64 -> 4096/64=64 rows
BATCHES=(256 512 1024 2048 4096)
declare -A MB_MAP=([256]=32 [512]=32 [1024]=32 [2048]=32 [4096]=64)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ─── Validate ─────────────────────────────────────────────────────────────────
for bin in "${MP_BIN}" "${PIP_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "Missing binary: ${bin}" >&2; exit 1
    fi
done
if ! command -v srun &>/dev/null; then
    echo "srun not found." >&2; exit 1
fi
bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

# ─── CSV header ───────────────────────────────────────────────────────────────
cat > "${MASTER_CSV}" <<'HDR'
chart,layout,mode,seed,learning_rate,batch_size,microbatch_count,world_size,nodes,tasks_per_node,train_samples,val_samples,hidden_layers,epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms
HDR

append_csv() {
    local run_csv=$1 batch=$2 mb=$3
    local lr
    lr=$(python3 -c "print(round(${BASE_LR} * ${batch} / ${BASE_BATCH}, 6))")
    python3 - "${run_csv}" "${MASTER_CSV}" "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}" "${mb}" <<'PY'
import csv, sys
run_csv, master, layout, ws, nodes, tasks, mb = sys.argv[1:]
with open(run_csv) as f, open(master, "a", newline="") as out:
    w = csv.writer(out)
    for row in csv.DictReader(f):
        w.writerow([
            2, layout, row["mode"], row["seed"], row["learning_rate"],
            row["batch_size"], mb, ws, nodes, tasks,
            row["train_samples"], row["val_samples"], row["hidden_layers"],
            row["epoch"], row["train_loss"], row["train_acc"],
            row["val_loss"], row["val_acc"], row["epoch_time_ms"],
        ])
PY
}

total=$(( ${#BATCHES[@]} * 2 ))
run=0

echo "Architecture : ${HIDDEN}"
echo "Layout       : ${LAYOUT}  ws=${WS}"
echo "Results      : ${MASTER_CSV}"
echo "Total runs   : ${total} x ${EPOCHS} epochs"
echo ""
echo "Batch  MB   MB_size  Efficiency  LR"
for BATCH in "${BATCHES[@]}"; do
    MB="${MB_MAP[${BATCH}]}"
    MB_SIZE=$(( BATCH / MB ))
    EFF=$(python3 -c "print(f'{${MB}/(${MB}+${WS}-1)*100:.0f}%')")
    LR=$(python3 -c "print(round(${BASE_LR} * ${BATCH} / ${BASE_BATCH}, 4))")
    printf "%-6s %-4s %-8s %-10s %s\n" "${BATCH}" "${MB}" "${MB_SIZE}" "${EFF}" "${LR}"
done
echo ""

# ─── Sweep ────────────────────────────────────────────────────────────────────
for BATCH in "${BATCHES[@]}"; do
    MB="${MB_MAP[${BATCH}]}"
    LR=$(python3 -c "print(round(${BASE_LR} * ${BATCH} / ${BASE_BATCH}, 6))")
    MB_SIZE=$(( BATCH / MB ))

    echo "--- batch=${BATCH}  lr=${LR}  mb=${MB}  mb_size=${MB_SIZE} ---"

    run=$((run+1))
    OUT_MP="${RESULT_DIR}/mp_b${BATCH}.csv"
    echo "[${run}/${total}] model-parallel  batch=${BATCH}"
    srun --nodes "${NODES}" --ntasks "${WS}" --ntasks-per-node "${TASKS}" \
         --cpus-per-task 1 --cpu-bind=cores \
         "${MP_BIN}" \
             --epochs "${EPOCHS}" --batch "${BATCH}" \
             --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
             --lr "${LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
             --data-dir "${DATA_DIR}" --output "${OUT_MP}" \
    && append_csv "${OUT_MP}" "${BATCH}" 0 \
    || echo "  FAILED"

    run=$((run+1))
    OUT_PIP="${RESULT_DIR}/pip_b${BATCH}.csv"
    echo "[${run}/${total}] pipeline  batch=${BATCH}  mb=${MB}"
    srun --nodes "${NODES}" --ntasks "${WS}" --ntasks-per-node "${TASKS}" \
         --cpus-per-task 1 --cpu-bind=cores \
         "${PIP_BIN}" \
             --epochs "${EPOCHS}" --batch "${BATCH}" --microbatches "${MB}" \
             --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
             --lr "${LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
             --data-dir "${DATA_DIR}" --output "${OUT_PIP}" \
    && append_csv "${OUT_PIP}" "${BATCH}" "${MB}" \
    || echo "  FAILED"

    echo ""
done

echo "Done. CSV: ${MASTER_CSV}"
echo "Visualize: python3 ${ROOT_DIR}/src/visualize_sweep.py ${MASTER_CSV}"