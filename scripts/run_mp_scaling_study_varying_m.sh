#!/usr/bin/env bash
# sweep_layout_mb.sh
#
# Chart 5 — Layout × Microbatch comparison at ws=8
#
# Compares two layouts and two microbatch counts at a fixed world size of 8,
# isolating the effect of node topology and microbatch count independently:
#
#   layout       M=8              M=32
#   2x4_ws8    2 nodes x 4 tasks  2 nodes x 4 tasks
#   4x2_ws8    4 nodes x 2 tasks  4 nodes x 2 tasks
#
# Total: 1 serial + 4 pipeline runs = 5 runs x EPOCHS epochs
# Estimated wall time at EPOCHS=10: ~30 minutes
#
# Usage:
#   bash scripts/sweep_layout_mb.sh
#   EPOCHS=5 bash scripts/sweep_layout_mb.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results/chart5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"

CHART5_CSV="${RESULT_DIR}/chart5_layout_mb_comparison.csv"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"
SERIAL_BIN="${BUILD_DIR}/serial_train"
PIP_BIN="${BUILD_DIR}/mpi_mp_pip_train"

EPOCHS="${EPOCHS:-10}"
TRAIN_SAMPLES=50000
VAL_SAMPLES=10000
SEED=42
BASE_LR=0.03
BASE_BATCH=64

HIDDEN="512,512,256,256,128,64,32"
BATCH=4096
WS=8
MBS=(8 32)

# layout_label:nodes:tasks_per_node
LAYOUTS=("2x4_ws8:2:4" "4x2_ws8:4:2")

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Validate ──────────────────────────────────────────────────────────────────
for bin in "${SERIAL_BIN}" "${PIP_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "Missing binary: ${bin}" >&2
        echo "Build first with: bash ${ROOT_DIR}/scripts/build.sh" >&2
        exit 1
    fi
done
if ! command -v srun &>/dev/null; then
    echo "srun not found in PATH. Run inside a Slurm environment." >&2
    exit 1
fi
bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    ALLOC_NODES="${SLURM_JOB_NUM_NODES:-0}"
    # SLURM_NTASKS is not always set depending on how the allocation was made.
    # Fall back to the required value so the check is skipped when unset.
    ALLOC_TASKS="${SLURM_NTASKS:-${WS}}"
    if (( 4 > ALLOC_NODES )) || (( WS > ALLOC_TASKS )); then
        echo "Insufficient allocation for this sweep." >&2
        echo "  required: nodes=4, ntasks=${WS}" >&2
        echo "  allocated: nodes=${ALLOC_NODES}, ntasks=${ALLOC_TASKS}" >&2
        exit 1
    fi
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
LR=$(python3 -c "print(round(${BASE_LR} * ${BATCH} / ${BASE_BATCH}, 6))")

METRICS_COLS="chart,layout,mode,seed,learning_rate,batch_size,microbatch_count,\
mb_size,world_size,nodes,tasks_per_node,train_samples,val_samples,hidden_layers,\
epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms"
echo "${METRICS_COLS}" > "${CHART5_CSV}"

append_metrics() {
    local run_csv=$1 mb=$2 layout=$3 nodes=$4 tpn=$5
    local mb_size=0
    if [[ "${mb}" -gt 0 ]]; then
        mb_size=$(( BATCH / mb ))
    fi
    python3 - \
        "${run_csv}" "${CHART5_CSV}" \
        "${layout}" "${WS}" "${nodes}" "${tpn}" \
        "${mb}" "${mb_size}" \
        <<'PY'
import csv, sys

run_csv, dest_csv, layout, ws, nodes, tpn, mb, mb_size = sys.argv[1:]

with open(run_csv) as f_in, open(dest_csv, "a", newline="") as f_out:
    writer = csv.writer(f_out)
    for row in csv.DictReader(f_in):
        writer.writerow([
            5, layout,
            row["mode"], row["seed"], row["learning_rate"],
            row["batch_size"], mb, mb_size,
            ws, nodes, tpn,
            row["train_samples"], row["val_samples"], row["hidden_layers"],
            row["epoch"],
            row["train_loss"], row["train_acc"],
            row["val_loss"],   row["val_acc"],
            row["epoch_time_ms"],
        ])
PY
}

TOTAL_RUNS=5   # serial(1) + 2 layouts x 2 microbatch counts
run=0

# ── Plan ──────────────────────────────────────────────────────────────────────
echo "Chart 5 — Layout × Microbatch comparison"
echo "Architecture : ${HIDDEN}"
echo "Batch        : ${BATCH}  lr=${LR}  ws=${WS}  epochs=${EPOCHS}"
echo ""
printf "%-12s %-6s %-4s %-6s %-10s\n" "layout" "nodes" "tpn" "M" "mb_size"
printf "%-12s %-6s %-4s %-6s %-10s\n" "------" "-----" "---" "-" "-------"
for layout_cfg in "${LAYOUTS[@]}"; do
    IFS=':' read -r c_label c_nodes c_tpn <<< "${layout_cfg}"
    for c_mb in "${MBS[@]}"; do
        printf "%-12s %-6s %-4s %-6s %-10s\n" \
            "${c_label}" "${c_nodes}" "${c_tpn}" \
            "${c_mb}" "$(( BATCH / c_mb ))"
    done
done
echo ""

# ── Serial baseline ───────────────────────────────────────────────────────────
run=$(( run + 1 ))
OUT_S="${RESULT_DIR}/c5_serial.csv"
echo "[${run}/${TOTAL_RUNS}] serial  batch=${BATCH}"
if "${SERIAL_BIN}" \
        --epochs "${EPOCHS}" --batch "${BATCH}" \
        --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
        --lr "${LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
        --data-dir "${DATA_DIR}" --output "${OUT_S}"; then
    # Replicate serial row for each (layout, M) pair so the visualizer
    # can draw a flat reference line across the full grouped chart.
    for layout_cfg in "${LAYOUTS[@]}"; do
        IFS=':' read -r c_label c_nodes c_tpn <<< "${layout_cfg}"
        for c_mb in "${MBS[@]}"; do
            append_metrics "${OUT_S}" 0 "${c_label}" "${c_nodes}" "${c_tpn}"
        done
    done
else
    echo "  FAILED (serial)"
fi

echo ""

# ── Pipeline: 2x2 grid ────────────────────────────────────────────────────────
for layout_cfg in "${LAYOUTS[@]}"; do
    IFS=':' read -r c_label c_nodes c_tpn <<< "${layout_cfg}"
    c_ntasks=$(( c_nodes * c_tpn ))

    for c_mb in "${MBS[@]}"; do
        c_mb_size=$(( BATCH / c_mb ))
        run=$(( run + 1 ))
        OUT_PIP="${RESULT_DIR}/c5_pip_${c_label}_mb${c_mb}.csv"
        LOG_PIP="${RESULT_DIR}/c5_pip_${c_label}_mb${c_mb}.log"

        echo "[${run}/${TOTAL_RUNS}] pipeline  layout=${c_label}  M=${c_mb}  mb_size=${c_mb_size}"
        if srun \
                --nodes "${c_nodes}" --ntasks "${c_ntasks}" \
                --cpus-per-task 1 --cpu-bind=cores \
                "${PIP_BIN}" \
                    --epochs "${EPOCHS}" \
                    --batch "${BATCH}" --microbatches "${c_mb}" \
                    --train-samples "${TRAIN_SAMPLES}" \
                    --val-samples "${VAL_SAMPLES}" \
                    --lr "${LR}" --seed "${SEED}" \
                    --hidden "${HIDDEN}" \
                    --data-dir "${DATA_DIR}" \
                    --output "${OUT_PIP}" \
                    --load-balance-layers \
        2>&1 | tee "${LOG_PIP}"; then
            append_metrics "${OUT_PIP}" "${c_mb}" "${c_label}" "${c_nodes}" "${c_tpn}"
        else
            echo "  FAILED (pipeline ${c_label} M=${c_mb})"
        fi

        echo ""
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "Done."
echo "  Chart 5 CSV : ${CHART5_CSV}"
echo ""
echo "Visualize:"
echo "  python3 ${ROOT_DIR}/src/visualize_sweep.py ${RESULT_DIR}"
echo "════════════════════════════════════════════════════════"