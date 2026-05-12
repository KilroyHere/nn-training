#!/usr/bin/env bash
# sweep-perf.sh
#
# Produces data for three model-parallelism charts from the poster:
#
#   Chart 1 — Epoch Time vs Batch Size
#              Serial + model-parallel + pipeline-parallel
#              Fixed ws=8 (2x4), vary batch: 256 512 1024 2048 4096
#
#   Chart 2 — Epoch Time vs Microbatch Count
#              Serial + model-parallel + pipeline-parallel
#              Fixed batch=4096, vary microbatches: 8 32 64
#              (serial and model-parallel are batch-only; recorded once as mb=0)
#
#   Chart 3 — Epoch Time Split vs Microbatch Count
#              Pipeline-parallel only, same sweep as Chart 2
#              Per-category timing breakdown parsed from stdout
#
# Usage:
#   bash scripts/sweep-perf.sh
#   EPOCHS=5 bash scripts/sweep-perf.sh
#
# Total runs: 15 (chart 1) + 5 (chart 2) = 20 x EPOCHS epochs
# Estimated wall time at EPOCHS=10: ~3 hours
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results/charts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"

# Output CSVs — one per chart.
CHART1_CSV="${RESULT_DIR}/chart1_batch_sweep.csv"
CHART2_CSV="${RESULT_DIR}/chart2_microbatch_sweep.csv"
CHART3_CSV="${RESULT_DIR}/chart3_time_split.csv"
CHART4_CSV="${RESULT_DIR}/chart4_worldsize_sweep.csv"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"

# NOTE: adjust SERIAL_BIN to match your actual serial binary name.
SERIAL_BIN="${BUILD_DIR}/serial_train"
MP_BIN="${BUILD_DIR}/mpi_mp_train"
PIP_BIN="${BUILD_DIR}/mpi_mp_pip_train"

EPOCHS="${EPOCHS:-10}"
TRAIN_SAMPLES=50000
VAL_SAMPLES=10000
SEED=42
BASE_LR=0.03
BASE_BATCH=64

# Architecture matching the poster exactly.
HIDDEN="512,512,256,256,128,64,32"

# Fixed layout: ws=8, 2 nodes x 4 tasks per node.
WS=8
NODES=2
TASKS=4
LAYOUT="2x4_ws8"

# Slurm sometimes injects values like "4(x2)" into env-derived variables.
# Strip any parenthesised suffix before using TASKS in arithmetic.
if [[ "${TASKS}" == *"("* ]]; then
    TASKS="${TASKS%%(*}"
fi

# Chart 1: batch sizes and matching microbatch counts.
# Microbatch count chosen so efficiency = mb/(mb+ws-1) >= ~80%.
BATCHES=(256 512 1024 2048 4096)
declare -A MB_MAP=([256]=32 [512]=32 [1024]=32 [2048]=32 [4096]=64)

# Charts 2 & 3: fixed batch, sweep microbatch counts.
FIXED_BATCH=4096
MICROBATCHES=(8 32 64)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Validate ──────────────────────────────────────────────────────────────────
for bin in "${SERIAL_BIN}" "${MP_BIN}" "${PIP_BIN}"; do
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
    ALLOC_NODES="${SLURM_JOB_NUM_NODES:-${NODES}}"
    ALLOC_TASKS="${SLURM_NTASKS:-${WS}}"
    # Chart 4 world-size sweep uses up to 4 nodes x 2 tasks = 8 total.
    # Charts 1-3 use 2 nodes x 4 tasks = 8 total.
    # Max nodes needed across all charts is 4; max ntasks is 8.
    MAX_NODES=4
    MAX_WS=8
    if (( MAX_NODES > ALLOC_NODES )) || (( MAX_WS > ALLOC_TASKS )); then
        echo "Requested srun resources exceed current allocation." >&2
        echo "  required: nodes=${MAX_NODES}, ntasks=${MAX_WS}" >&2
        echo "  allocated: nodes=${ALLOC_NODES}, ntasks=${ALLOC_TASKS}" >&2
        echo "Request a larger salloc before running the full sweep." >&2
        exit 1
    fi
fi

# ── CSV headers ───────────────────────────────────────────────────────────────
METRICS_COLS="chart,layout,mode,seed,learning_rate,batch_size,microbatch_count,\
mb_size,world_size,nodes,tasks_per_node,train_samples,val_samples,hidden_layers,\
epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms"

echo "${METRICS_COLS}" > "${CHART1_CSV}"
echo "${METRICS_COLS}" > "${CHART2_CSV}"
echo "${METRICS_COLS}" > "${CHART4_CSV}"

# Chart 3 captures per-category mean timing (mean across ranks) per epoch.
echo "chart,layout,mode,batch_size,microbatch_count,mb_size,epoch,metric,mean_s" \
    > "${CHART3_CSV}"

# ── Helpers ───────────────────────────────────────────────────────────────────

lr_for_batch() {
    # Linear LR scaling from BASE_LR at BASE_BATCH.
    python3 -c "print(round(${BASE_LR} * $1 / ${BASE_BATCH}, 6))"
}

# Append rows from a run CSV into a chart CSV with extra context columns.
# Args: run_csv dest_csv chart mb layout ws nodes tpn
#
# mb=0 means the mode has no microbatch dimension (serial, model-parallel);
# mb_size is derived from batch_size in the run CSV when mb > 0.
append_metrics() {
    local run_csv=$1 dest_csv=$2 chart=$3 mb=$4
    local layout=$5 ws=$6 nodes=$7 tpn=$8
    local mb_size=0
    if [[ "${mb}" -gt 0 ]]; then
        mb_size=$(python3 -c "
import csv
rows = list(csv.DictReader(open('${run_csv}')))
print(int(rows[0]['batch_size']) // ${mb}) if rows else print(0)
")
    fi
    python3 - \
        "${run_csv}" "${dest_csv}" "${chart}" \
        "${layout}" "${ws}" "${nodes}" "${tpn}" \
        "${mb}" "${mb_size}" \
        <<'PY'
import csv, sys

run_csv, dest_csv, chart, layout, ws, nodes, tpn, mb, mb_size = sys.argv[1:]

with open(run_csv) as f_in, open(dest_csv, "a", newline="") as f_out:
    writer = csv.writer(f_out)
    for row in csv.DictReader(f_in):
        writer.writerow([
            chart, layout,
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

# Parse per-category timing lines from a captured pipeline log and append to
# CHART3_CSV.  The C++ code prints (from rank 0) per epoch:
#
#   [mpi-mp-pip] epoch N timings (s, epoch total) - min / mean / max across R ranks:
#     fwd_comm     X.XXXX / X.XXXX / X.XXXX
#     fwd_compute  X.XXXX / X.XXXX / X.XXXX
#     ...
#
# We extract the mean column (middle value) for each metric.
# Args: log_file batch mb
parse_timing_split() {
    local log_file=$1 batch=$2 mb=$3
    local mb_size=$(( batch / mb ))
    python3 - \
        "${log_file}" "${CHART3_CSV}" \
        "${LAYOUT}" "${batch}" "${mb}" "${mb_size}" \
        <<'PY'
import re, sys, csv

log_file, out_csv, layout, batch, mb, mb_size = sys.argv[1:]
chart   = 3
mode    = "mpi-mp-pip"

epoch_pat  = re.compile(r'\[mpi-mp-pip\] epoch (\d+) timings')
metric_pat = re.compile(
    r'^\s{2}(\S+)\s+[\d.]+\s*/\s*([\d.]+)\s*/\s*[\d.]+')

current_epoch = None
rows = []

with open(log_file) as f:
    for line in f:
        m = epoch_pat.search(line)
        if m:
            current_epoch = int(m.group(1))
            continue
        if current_epoch is None:
            continue
        m = metric_pat.match(line)
        if m:
            metric = m.group(1).rstrip()
            mean_s = m.group(2)
            rows.append([
                chart, layout, mode,
                batch, mb, mb_size,
                current_epoch, metric, mean_s,
            ])

with open(out_csv, "a", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"  parsed {len(rows)} timing rows from {log_file}")
PY
}

# Run a pipeline job, capturing stdout+stderr to a log file while also
# printing to the terminal, then append to both metrics and timing CSVs.
# Args: out_csv log_file batch mb nodes ntasks lr [--load-balance-layers]
run_pipeline() {
    local out_csv=$1 log_file=$2 batch=$3 mb=$4
    local nodes=$5 ntasks=$6 lr=$7
    local balance_flag="${8:-}"

    srun \
        --nodes "${nodes}" --ntasks "${ntasks}" \
        --cpus-per-task 1 --cpu-bind=cores \
        "${PIP_BIN}" \
            --epochs "${EPOCHS}" \
            --batch "${batch}" --microbatches "${mb}" \
            --train-samples "${TRAIN_SAMPLES}" \
            --val-samples "${VAL_SAMPLES}" \
            --lr "${lr}" --seed "${SEED}" \
            --hidden "${HIDDEN}" \
            --data-dir "${DATA_DIR}" \
            --output "${out_csv}" \
            ${balance_flag} \
    2>&1 | tee "${log_file}"
}

# ── Progress tracking ─────────────────────────────────────────────────────────
# Chart 1: 5 batches x 3 modes = 15
# Chart 2: serial(1) + mp(1) + pip(3) = 5
# Chart 4: serial(1) + 4 ws configs x 3 modes (mp + pip + pip-balanced) = 13
TOTAL_RUNS=33
run=0

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════"
    echo ""
}

# ════════════════════════════════════════════════════════════════════════════
# Chart 1: Epoch Time vs Batch Size
# ════════════════════════════════════════════════════════════════════════════
print_header "Chart 1 — Epoch Time vs Batch Size"

echo "Architecture : ${HIDDEN}"
echo "Layout       : ${LAYOUT}  ws=${WS}"
echo ""
printf "%-6s %-4s %-8s %-12s %s\n" "Batch" "MB" "MB_size" "Efficiency" "LR"
printf "%-6s %-4s %-8s %-12s %s\n" "-----" "--" "-------" "----------" "--"
for BATCH in "${BATCHES[@]}"; do
    MB="${MB_MAP[${BATCH}]}"
    MB_SIZE=$(( BATCH / MB ))
    EFF=$(python3 -c "print(f'{${MB}/(${MB}+${WS}-1)*100:.0f}%')")
    LR=$(lr_for_batch "${BATCH}")
    printf "%-6s %-4s %-8s %-12s %s\n" \
        "${BATCH}" "${MB}" "${MB_SIZE}" "${EFF}" "${LR}"
done
echo ""

for BATCH in "${BATCHES[@]}"; do
    MB="${MB_MAP[${BATCH}]}"
    LR=$(lr_for_batch "${BATCH}")
    MB_SIZE=$(( BATCH / MB ))
    echo "--- batch=${BATCH}  lr=${LR}  mb=${MB}  mb_size=${MB_SIZE} ---"

    # Serial baseline — no srun, single process.
    run=$(( run + 1 ))
    OUT_S="${RESULT_DIR}/c1_serial_b${BATCH}.csv"
    echo "[${run}/${TOTAL_RUNS}] serial  batch=${BATCH}"
    if "${SERIAL_BIN}" \
            --epochs "${EPOCHS}" --batch "${BATCH}" \
            --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
            --lr "${LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
            --data-dir "${DATA_DIR}" --output "${OUT_S}"; then
        append_metrics "${OUT_S}" "${CHART1_CSV}" 1 0 "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}"
    else
        echo "  FAILED (serial batch=${BATCH})"
    fi

    # Sequential model parallel.
    run=$(( run + 1 ))
    OUT_MP="${RESULT_DIR}/c1_mp_b${BATCH}.csv"
    echo "[${run}/${TOTAL_RUNS}] model-parallel  batch=${BATCH}"
    if srun \
            --nodes "${NODES}" --ntasks "${WS}" \
            --cpus-per-task 1 --cpu-bind=cores \
            "${MP_BIN}" \
                --epochs "${EPOCHS}" --batch "${BATCH}" \
                --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
                --lr "${LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
                --data-dir "${DATA_DIR}" --output "${OUT_MP}"; then
        append_metrics "${OUT_MP}" "${CHART1_CSV}" 1 0 "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}"
    else
        echo "  FAILED (model-parallel batch=${BATCH})"
    fi

    # Pipeline parallel — also save log for potential inspection.
    run=$(( run + 1 ))
    OUT_PIP="${RESULT_DIR}/c1_pip_b${BATCH}.csv"
    LOG_PIP="${RESULT_DIR}/c1_pip_b${BATCH}.log"
    echo "[${run}/${TOTAL_RUNS}] pipeline  batch=${BATCH}  mb=${MB}  mb_size=${MB_SIZE}"
    if run_pipeline \
            "${OUT_PIP}" "${LOG_PIP}" \
            "${BATCH}" "${MB}" \
            "${NODES}" "${WS}" "${LR}"; then
        append_metrics "${OUT_PIP}" "${CHART1_CSV}" 1 "${MB}" "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}"
    else
        echo "  FAILED (pipeline batch=${BATCH} mb=${MB})"
    fi

    echo ""
done

# ════════════════════════════════════════════════════════════════════════════
# Charts 2 & 3: Epoch Time vs Microbatch Count  (fixed batch=4096)
# ════════════════════════════════════════════════════════════════════════════
print_header "Charts 2 & 3 — Epoch Time vs Microbatch Count (batch=${FIXED_BATCH})"

FIXED_LR=$(lr_for_batch "${FIXED_BATCH}")
echo "Fixed batch=${FIXED_BATCH}  lr=${FIXED_LR}"
echo ""

# Serial: no microbatch dimension — run once, recorded as mb=0.
run=$(( run + 1 ))
OUT_S_C2="${RESULT_DIR}/c2_serial.csv"
echo "[${run}/${TOTAL_RUNS}] serial  batch=${FIXED_BATCH}"
if "${SERIAL_BIN}" \
        --epochs "${EPOCHS}" --batch "${FIXED_BATCH}" \
        --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
        --lr "${FIXED_LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
        --data-dir "${DATA_DIR}" --output "${OUT_S_C2}"; then
    append_metrics "${OUT_S_C2}" "${CHART2_CSV}" 2 0 "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}"
else
    echo "  FAILED (serial chart2)"
fi

# Model parallel: also no microbatch dimension — run once, recorded as mb=0.
# Visualize as a flat horizontal line across all microbatch values.
run=$(( run + 1 ))
OUT_MP_C2="${RESULT_DIR}/c2_mp.csv"
echo "[${run}/${TOTAL_RUNS}] model-parallel  batch=${FIXED_BATCH}"
if srun \
        --nodes "${NODES}" --ntasks "${WS}" \
        --cpus-per-task 1 --cpu-bind=cores \
        "${MP_BIN}" \
            --epochs "${EPOCHS}" --batch "${FIXED_BATCH}" \
            --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
            --lr "${FIXED_LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
            --data-dir "${DATA_DIR}" --output "${OUT_MP_C2}"; then
    append_metrics "${OUT_MP_C2}" "${CHART2_CSV}" 2 0 "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}"
else
    echo "  FAILED (model-parallel chart2)"
fi

echo ""

# Pipeline: sweep microbatch counts — feeds both Chart 2 and Chart 3.
for MB in "${MICROBATCHES[@]}"; do
    MB_SIZE=$(( FIXED_BATCH / MB ))
    run=$(( run + 1 ))
    OUT_PIP_C2="${RESULT_DIR}/c2_pip_mb${MB}.csv"
    LOG_PIP_C2="${RESULT_DIR}/c2_pip_mb${MB}.log"

    echo "[${run}/${TOTAL_RUNS}] pipeline  batch=${FIXED_BATCH}  mb=${MB}  mb_size=${MB_SIZE}"
    if run_pipeline \
            "${OUT_PIP_C2}" "${LOG_PIP_C2}" \
            "${FIXED_BATCH}" "${MB}" \
            "${NODES}" "${WS}" "${FIXED_LR}"; then
        append_metrics "${OUT_PIP_C2}" "${CHART2_CSV}" 2 "${MB}" "${LAYOUT}" "${WS}" "${NODES}" "${TASKS}"
        parse_timing_split "${LOG_PIP_C2}" "${FIXED_BATCH}" "${MB}"
    else
        echo "  FAILED (pipeline chart2/3 mb=${MB})"
    fi

    echo ""
done

# ════════════════════════════════════════════════════════════════════════════
# Chart 4: Epoch Time vs World Size  (fixed batch=4096, 2 tasks/node)
#
# Configs — nodes:ws:layout_label
#   1 node  x 2 tasks = ws=2  (intra-node only)
#   2 nodes x 2 tasks = ws=4  (crosses one node boundary)
#   3 nodes x 2 tasks = ws=6  (uneven: ranks 0-1 get 2 layers, 2-5 get 1)
#   4 nodes x 2 tasks = ws=8  (every rank owns exactly 1 layer — ceiling)
#
# Microbatch count fixed at 32 for pipeline, giving efficiency:
#   ws=2: 94%  ws=4: 91%  ws=6: 85%  ws=8: 82%
# ════════════════════════════════════════════════════════════════════════════
print_header "Chart 4 — Epoch Time vs World Size (batch=${FIXED_BATCH}, 2 tasks/node)"

C4_BATCH="${FIXED_BATCH}"
C4_LR=$(lr_for_batch "${C4_BATCH}")
C4_MB=32
C4_TASKS_PER_NODE=2
C4_MB_SIZE=$(( C4_BATCH / C4_MB ))

# Array of "nodes:ws:label" tuples.
C4_CONFIGS=("1:2:1x2_ws2" "2:4:2x2_ws4" "3:6:3x2_ws6" "4:8:4x2_ws8")

echo "Fixed: batch=${C4_BATCH}  mb=${C4_MB}  mb_size=${C4_MB_SIZE}  lr=${C4_LR}  tasks/node=${C4_TASKS_PER_NODE}"
echo ""
printf "%-12s %-6s %-6s %-12s %s\n" "layout" "nodes" "ws" "layers/rank" "efficiency"
printf "%-12s %-6s %-6s %-12s %s\n" "------" "-----" "--" "-----------" "----------"
for cfg in "${C4_CONFIGS[@]}"; do
    IFS=':' read -r c_nodes c_ws c_label <<< "${cfg}"
    EFF=$(python3 -c "print(f'{${C4_MB}/(${C4_MB}+${c_ws}-1)*100:.0f}%')")
    LAYERS_PER_RANK=$(python3 -c "
n, ws = 8, ${c_ws}
base, rem = n // ws, n % ws
lo, hi = base, base + (1 if rem > 0 else 0)
print(str(lo) if lo == hi else f'{lo}-{hi}')
")
    printf "%-12s %-6s %-6s %-12s %s\n" \
        "${c_label}" "${c_nodes}" "${c_ws}" "${LAYERS_PER_RANK}" "${EFF}"
done
echo ""

# Serial baseline — run once; the same regardless of world size.
run=$(( run + 1 ))
OUT_S_C4="${RESULT_DIR}/c4_serial.csv"
echo "[${run}/${TOTAL_RUNS}] serial  batch=${C4_BATCH} (chart 4 baseline)"
if "${SERIAL_BIN}" \
        --epochs "${EPOCHS}" --batch "${C4_BATCH}" \
        --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
        --lr "${C4_LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
        --data-dir "${DATA_DIR}" --output "${OUT_S_C4}"; then
    # Record once per ws config so the visualization can draw a flat baseline.
    for cfg in "${C4_CONFIGS[@]}"; do
        IFS=':' read -r _ c_ws c_label <<< "${cfg}"
        append_metrics "${OUT_S_C4}" "${CHART4_CSV}" 4 0 \
            "${c_label}" "${c_ws}" 1 "${C4_TASKS_PER_NODE}"
    done
else
    echo "  FAILED (serial chart4)"
fi

echo ""

for cfg in "${C4_CONFIGS[@]}"; do
    IFS=':' read -r c_nodes c_ws c_label <<< "${cfg}"
    c_ntasks=$(( c_nodes * C4_TASKS_PER_NODE ))
    echo "--- layout=${c_label}  nodes=${c_nodes}  ws=${c_ws}  ntasks=${c_ntasks} ---"

    # Model parallel.
    run=$(( run + 1 ))
    OUT_MP_C4="${RESULT_DIR}/c4_mp_${c_label}.csv"
    echo "[${run}/${TOTAL_RUNS}] model-parallel  ws=${c_ws}"
    if srun \
            --nodes "${c_nodes}" --ntasks "${c_ntasks}" \
            --cpus-per-task 1 --cpu-bind=cores \
            "${MP_BIN}" \
                --epochs "${EPOCHS}" --batch "${C4_BATCH}" \
                --train-samples "${TRAIN_SAMPLES}" --val-samples "${VAL_SAMPLES}" \
                --lr "${C4_LR}" --seed "${SEED}" --hidden "${HIDDEN}" \
                --data-dir "${DATA_DIR}" --output "${OUT_MP_C4}"; then
        append_metrics "${OUT_MP_C4}" "${CHART4_CSV}" 4 0 \
            "${c_label}" "${c_ws}" "${c_nodes}" "${C4_TASKS_PER_NODE}"
    else
        echo "  FAILED (model-parallel ${c_label})"
    fi

    # Pipeline parallel — unbalanced (naive round-robin layer assignment).
    run=$(( run + 1 ))
    OUT_PIP_C4="${RESULT_DIR}/c4_pip_${c_label}.csv"
    LOG_PIP_C4="${RESULT_DIR}/c4_pip_${c_label}.log"
    echo "[${run}/${TOTAL_RUNS}] pipeline  ws=${c_ws}  mb=${C4_MB}  mb_size=${C4_MB_SIZE}"
    if run_pipeline \
            "${OUT_PIP_C4}" "${LOG_PIP_C4}" \
            "${C4_BATCH}" "${C4_MB}" \
            "${c_nodes}" "${c_ntasks}" "${C4_LR}"; then
        append_metrics "${OUT_PIP_C4}" "${CHART4_CSV}" 4 "${C4_MB}" \
            "${c_label}" "${c_ws}" "${c_nodes}" "${C4_TASKS_PER_NODE}"
    else
        echo "  FAILED (pipeline ${c_label})"
    fi

    # Pipeline parallel — load balanced.
    # NOTE: at ws=8 (1 layer/rank) balanced == unbalanced; results will be
    # identical but we run it anyway for a consistent data series.
    run=$(( run + 1 ))
    OUT_PIP_BAL_C4="${RESULT_DIR}/c4_pip_balanced_${c_label}.csv"
    LOG_PIP_BAL_C4="${RESULT_DIR}/c4_pip_balanced_${c_label}.log"
    echo "[${run}/${TOTAL_RUNS}] pipeline-balanced  ws=${c_ws}  mb=${C4_MB}  mb_size=${C4_MB_SIZE}"
    if run_pipeline \
            "${OUT_PIP_BAL_C4}" "${LOG_PIP_BAL_C4}" \
            "${C4_BATCH}" "${C4_MB}" \
            "${c_nodes}" "${c_ntasks}" "${C4_LR}" \
            "--load-balance-layers"; then
        append_metrics "${OUT_PIP_BAL_C4}" "${CHART4_CSV}" 4 "${C4_MB}" \
            "${c_label}" "${c_ws}" "${c_nodes}" "${C4_TASKS_PER_NODE}"
    else
        echo "  FAILED (pipeline-balanced ${c_label})"
    fi

    echo ""
done


echo "════════════════════════════════════════════════════════"
echo "Done."
echo "  Chart 1 CSV : ${CHART1_CSV}"
echo "  Chart 2 CSV : ${CHART2_CSV}"
echo "  Chart 3 CSV : ${CHART3_CSV}"
echo "  Chart 4 CSV : ${CHART4_CSV}"
echo ""
echo "Visualize:"
echo "  python3 ${ROOT_DIR}/src/visualize_sweep.py ${RESULT_DIR}"
echo "════════════════════════════════════════════════════════"