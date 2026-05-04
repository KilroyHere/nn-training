#!/usr/bin/env bash
# Strong-scaling study: runs flat DP, hierarchical DP, and local SGD across a
# rank ladder at two model sizes, capturing per-epoch CSVs and stdout timing logs.
#
# Usage (inside a Slurm allocation with >= 4 nodes, >= 32 tasks/node):
#   bash scripts/run_scaling_study.sh
#
# Override any default via environment variable, e.g.:
#   EPOCHS=5 MODELS="256,128" bash scripts/run_scaling_study.sh
#
# Outputs land in results/scaling/:
#   flat_dp_<nodes>n<ranks>r_<model>.csv            - per-epoch metrics
#   flat_dp_<nodes>n<ranks>r_<model>.log            - stdout with timing lines
#   hier_dp_<nodes>n<ranks>r_<model>.csv
#   hier_dp_<nodes>n<ranks>r_<model>.log
#   local_sgd_K<K>_<nodes>n<ranks>r_<model>.csv
#   local_sgd_K<K>_<nodes>n<ranks>r_<model>.log

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULT_DIR="${ROOT_DIR}/results/scaling"
FLAT_BIN="${BUILD_DIR}/mpi_dp_train"
HIER_BIN="${BUILD_DIR}/mpi_dp_hierarchial_train"
LOCAL_SGD_BIN="${BUILD_DIR}/mpi_dp_local_sgd_train"

EPOCHS="${EPOCHS:-10}"
GLOBAL_BATCH="${GLOBAL_BATCH:-1024}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"

# Space-separated list of "hidden" specs to sweep.
# MODELS="${MODELS:-1024,512,256}"
MODELS="${MODELS:-256,128,64}"

# Rank ladder: each entry is "nodes:tasks_per_node".
RANK_CONFIGS="${RANK_CONFIGS:-1:4 1:16 1:32 2:32 4:32}"

# Space-separated list of sync_every values to sweep for local SGD.
LOCAL_SGD_SYNC_VALUES="${LOCAL_SGD_SYNC_VALUES:-10 50}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------------------------------------------------------------------

if ! command -v srun >/dev/null 2>&1; then
    echo "srun not found. Run inside a Slurm allocation." >&2
    exit 1
fi

for bin in "${FLAT_BIN}" "${HIER_BIN}" "${LOCAL_SGD_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "Missing binary: ${bin}" >&2
        echo "Build first: bash ${ROOT_DIR}/scripts/build.sh" >&2
        exit 1
    fi
done

mkdir -p "${RESULT_DIR}"
bash "${ROOT_DIR}/scripts/prepare_mnist.sh"

ALLOC_NODES="${SLURM_JOB_NUM_NODES:-0}"
ALLOC_TASKS="${SLURM_NTASKS:-0}"

# ---------------------------------------------------------------------------
# Helper: run one configuration, write CSV + log.
# Args: $1=binary_path $2=label $3=nodes $4=tasks_per_node $5=hidden $6=out_tag
#       $7=extra_flags (optional, e.g. "--sync-every 10")
run_one() {
    local bin="$1"
    local label="$2"
    local nodes="$3"
    local tpn="$4"
    local hidden="$5"
    local tag="$6"
    local extra_flags="${7:-}"
    local ntasks=$(( nodes * tpn ))
    local out_csv="${RESULT_DIR}/${tag}.csv"
    local out_log="${RESULT_DIR}/${tag}.log"

    # Validate against allocation if inside a job.
    if [[ "${ALLOC_NODES}" -gt 0 ]]; then
        if (( nodes > ALLOC_NODES )); then
            echo "  [SKIP] ${tag}: requested ${nodes} nodes but allocation only has ${ALLOC_NODES}" >&2
            return 0
        fi
    fi
    if [[ "${ALLOC_TASKS}" -gt 0 ]]; then
        if (( ntasks > ALLOC_TASKS )); then
            echo "  [SKIP] ${tag}: requested ${ntasks} tasks but allocation only has ${ALLOC_TASKS}" >&2
            return 0
        fi
    fi

    echo "  --> ${label} | nodes=${nodes} tasks/node=${tpn} total=${ntasks} hidden=${hidden}${extra_flags:+ flags=${extra_flags}}"
    echo "      csv=${out_csv}"

    # shellcheck disable=SC2086
    srun \
        --nodes "${nodes}" \
        --ntasks "${ntasks}" \
        --cpus-per-task "${CPUS_PER_TASK}" \
        --cpu-bind=cores \
        "${bin}" \
        --epochs "${EPOCHS}" \
        --batch "${GLOBAL_BATCH}" \
        --train-samples "${TRAIN_SAMPLES}" \
        --val-samples "${VAL_SAMPLES}" \
        --lr "${LEARNING_RATE}" \
        --seed "${SEED}" \
        --hidden "${hidden}" \
        --data-dir "${DATA_DIR}" \
        --output "${out_csv}" \
        ${extra_flags} \
        2>&1 | tee "${out_log}"

    echo "      done."
}

# ---------------------------------------------------------------------------

echo "============================================================"
echo "Scaling study"
echo "  epochs=${EPOCHS}  batch=${GLOBAL_BATCH}  train=${TRAIN_SAMPLES}"
echo "  models: ${MODELS}"
echo "  rank ladder: ${RANK_CONFIGS}"
echo "  local SGD sync values: ${LOCAL_SGD_SYNC_VALUES}"
echo "  output dir: ${RESULT_DIR}"
echo "  allocation: nodes=${ALLOC_NODES} tasks=${ALLOC_TASKS}"
echo "============================================================"

for hidden in ${MODELS}; do
    # Build a filesystem-safe tag from the hidden spec (commas → dashes).
    model_tag="${hidden//,/-}"

    echo ""
    echo "--- Model: ${hidden} (${model_tag}) ---"

    for rc in ${RANK_CONFIGS}; do
        nodes="${rc%%:*}"
        tpn="${rc##*:}"
        ntasks=$(( nodes * tpn ))
        config_tag="${nodes}n${ntasks}r"

        echo ""
        echo "  [config] ${config_tag}  (${nodes} node(s) x ${tpn} tasks)"

        run_one "${FLAT_BIN}" "flat-dp" \
            "${nodes}" "${tpn}" "${hidden}" \
            "flat_dp_${config_tag}_${model_tag}"

        run_one "${HIER_BIN}" "hier-dp" \
            "${nodes}" "${tpn}" "${hidden}" \
            "hier_dp_${config_tag}_${model_tag}"

        for K in ${LOCAL_SGD_SYNC_VALUES}; do
            run_one "${LOCAL_SGD_BIN}" "local-sgd-K${K}" \
                "${nodes}" "${tpn}" "${hidden}" \
                "local_sgd_K${K}_${config_tag}_${model_tag}" \
                "--sync-every ${K}"
        done
    done
done

echo ""
echo "============================================================"
echo "All runs complete (flat DP + hier DP + local SGD). Results in ${RESULT_DIR}"
echo ""
echo "CSV files:"
ls -1 "${RESULT_DIR}"/*.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "To summarize results run:"
echo "  python3 ${ROOT_DIR}/result_scripts/summarize_scaling.py ${RESULT_DIR}"
echo "============================================================"
