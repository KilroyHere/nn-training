# NEXT STEPS: MPI Handoff Guide

This document tracks remaining work after the MPI-DP baseline integration.
Read this together with `README.md`.

## Current state (what already works)

- Serial training pipeline is working end-to-end with MNIST.
- Shared setup/parity helpers are in `train_common`.
- Serial backend logic is in `train_serial`.
- MPI-DP backend logic is in `train_mpi_data_parallel`.
- Common CLI dispatch is in `main.cpp` (`--mode serial|mpi-dp`).
- MPI run scripts launch with `srun` and print launch config.
- Metrics comparison tooling already exists.

Key files:

- `src/train_serial.cpp` (serial backend implementation)
- `src/train_mpi_data_parallel.cpp`, `include/train_mpi_data_parallel.h` (MPI-DP backend)
- `src/main.cpp` (shared entrypoint and mode dispatch)
- `src/train_common.cpp`, `include/train_common.h` (shared setup/parity helpers)
- `src/mlp.cpp`, `src/tensor.cpp`, `src/data_mnist.cpp` (model + math + data)
- `docs/serial_mpi_parity.md` (parity contract)
- `scripts/run_mpi_dp_smoke.sh`, `scripts/run_mpi_dp_perf.sh`, `scripts/job_mpi_dp.slurm` (MPI launch flow)
- `scripts/compare_metrics.py` (serial vs MPI metrics checks)

## Goal for the next phase

Lock down serial-vs-MPI parity and performance measurement workflows so future backends (model parallel/FSDP) can reuse the same baseline methodology.

Keep this separation:

- **Serial/MPI files:** backend-specific training logic
- **Common helpers:** config validation, dataset prep, metadata formatting, output parity helpers

## Step-by-step completion plan

### 1) Freeze and document launch semantics

- Keep MPI script defaults explicit and conservative (`DP_NODES=1`, `DP_TASKS_PER_NODE=1`).
- Continue printing allocation + effective `srun` args before launch.
- Keep 1-thread-per-rank env vars in all MPI scripts and job scripts.

### 2) Keep parity contract intact

Follow `docs/serial_mpi_parity.md` exactly for:

- sample subset policy
- batch inclusion policy (drop incomplete tail batch)
- epoch timing semantics
- output schema fields and ordering

Do not silently change these in one backend only.

### 3) Add comparison workflow script wrapper (optional but recommended)

Add a convenience script (for example `scripts/run_serial_vs_mpi_compare.sh`) that:

1. runs serial baseline
2. runs MPI candidate with same config
3. calls `scripts/compare_metrics.py`

## Minimum acceptance milestones

### Milestone A: MPI DP baseline remains buildable

- `mpi_dp_train` compiles and runs with `srun -N 1 -n 1`.
- Produces CSV with required schema and `mode=mpi`.

### Milestone B: n=1 parity

- Run serial and MPI with identical config.
- `compare_metrics.py` passes within chosen tolerances.

### Milestone C: multi-rank correctness

- Run MPI DP with 2 and 4 ranks via `srun` (or script wrappers with `DP_NODES` / `DP_TASKS_PER_NODE`).
- Metrics remain within documented tolerances vs serial baseline.

### Milestone D: performance runs

- Produce timing/throughput outputs for 1, 2, 4 (and optionally 8) ranks and at least one multi-node setting.
- Keep settings fixed for apples-to-apples comparisons.

## Recommended first commands

From project root:

```bash
bash scripts/build.sh
bash scripts/run_serial_smoke.sh
bash scripts/run_serial_perf.sh
python3 scripts/compare_metrics.py results/smoke_metrics.csv results/smoke_metrics.csv
```

Then for MPI DP:

```bash
cpu 1
DP_NODES=1 DP_TASKS_PER_NODE=2 GLOBAL_BATCH=128 bash scripts/run_mpi_dp_smoke.sh
DP_NODES=1 DP_TASKS_PER_NODE=4 GLOBAL_BATCH=256 EPOCHS=10 bash scripts/run_mpi_dp_perf.sh
python3 scripts/compare_metrics.py results/smoke_metrics.csv results/mpi_dp_smoke_metrics.csv --loss-tol 0.05 --acc-tol 0.02
```

Example multi-node perf run:

```bash
cpu 2
DP_NODES=2 DP_TASKS_PER_NODE=64 GLOBAL_BATCH=256 EPOCHS=10 bash scripts/run_mpi_dp_perf.sh
```

## Common pitfalls to avoid

- Changing train/eval semantics in MPI but not serial.
- Changing CSV schema in one backend only.
- Using different defaults across scripts.
- Comparing runs with different seeds/sample counts and treating them as regressions.

## Definition of done for handoff

Handoff is complete when:

- MPI DP backend is buildable/runnable.
- Serial-vs-MPI parity checks are scripted and reproducible.
- Performance scripts produce consistent rank-scaling results.
- README + this guide reflect the final MPI workflow.
