# NEXT STEPS: MPI Handoff Guide

This document is the implementation handoff for the MPI contributor.
Read this together with `README.md`.

## Current state (what already works)

- Serial training pipeline is working end-to-end with MNIST.
- Shared setup/parity helpers are in `train_common`.
- Serial backend logic is in `train_serial`.
- Run scripts and benchmarking flow are in place.
- Metrics comparison tooling already exists.

Key files:

- `src/train_serial.cpp` (serial backend implementation)
- `src/train_common.cpp`, `include/train_common.h` (shared setup/parity helpers)
- `src/mlp.cpp`, `src/tensor.cpp`, `src/data_mnist.cpp` (model + math + data)
- `docs/serial_mpi_parity.md` (parity contract)
- `scripts/compare_metrics.py` (serial vs candidate metrics checks)

## Goal for MPI implementation

Add an MPI backend that can be compared fairly against serial under matched configuration and output schema.

Keep this separation:

- **Serial/MPI files:** backend-specific training logic
- **Common helpers:** config validation, dataset prep, metadata formatting, output parity helpers

## Step-by-step implementation plan

### 1) Add MPI training entrypoint

Create:

- `include/train_mpi.h`
- `src/train_mpi.cpp`

Implement:

- `int run_mpi_training(const TrainConfig& config, std::string* error_message);`

Expected behavior:

- Initialize/finalize MPI.
- Use rank-aware execution and communication logic.
- Write metrics CSV with same columns as serial, but with `mode=mpi`.

### 2) Add MPI executable target in CMake

Update `CMakeLists.txt`:

- Find/link MPI.
- Build `mpi_train` executable (or equivalent name).
- Link shared core (`nn_core`) plus MPI libs.

### 3) Keep parity contract intact

Follow `docs/serial_mpi_parity.md` exactly for:

- sample subset policy
- batch inclusion policy (drop incomplete tail batch)
- epoch timing semantics
- output schema fields and ordering

Do not silently change these in one backend only.

### 4) Add MPI run scripts

Add:

- `scripts/run_mpi_smoke.sh`
- `scripts/run_mpi_perf.sh`
- `scripts/job-mpi.slurm`

Use the same default configuration values/pattern as serial scripts.

### 5) Add comparison workflow script wrapper (optional but recommended)

Add a convenience script (for example `scripts/run_serial_vs_mpi_compare.sh`) that:

1. runs serial baseline
2. runs MPI candidate with same config
3. calls `scripts/compare_metrics.py`

## Minimum acceptance milestones

### Milestone A: MPI skeleton builds

- `mpi_train` compiles and runs with `mpirun -n 1`.
- Produces CSV with required schema and `mode=mpi`.

### Milestone B: n=1 parity

- Run serial and MPI with identical config.
- `compare_metrics.py` passes within chosen tolerances.

### Milestone C: multi-rank correctness

- Run `mpirun -n 2` and `mpirun -n 4`.
- Metrics remain within documented tolerances vs serial baseline.

### Milestone D: performance runs

- Produce timing/throughput outputs for 1, 2, 4 (and optionally 8) ranks.
- Keep settings fixed for apples-to-apples comparisons.

## Recommended first commands

From project root:

```bash
bash scripts/run_serial_smoke.sh
bash scripts/run_serial_perf.sh
python3 scripts/compare_metrics.py results/smoke_metrics.csv results/smoke_metrics.csv
```

Then after MPI binary exists:

```bash
mpirun -n 1 ./build/mpi_train --epochs 3 --batch 64 --train-samples 2048 --val-samples 256 --seed 42 --lr 0.03 --hidden 128,64 --data-dir data/mnist --output results/mpi_smoke.csv
python3 scripts/compare_metrics.py results/smoke_metrics.csv results/mpi_smoke.csv --loss-tol 0.05 --acc-tol 0.02
```

## Common pitfalls to avoid

- Changing train/eval semantics in MPI but not serial.
- Changing CSV schema in one backend only.
- Using different defaults across scripts.
- Comparing runs with different seeds/sample counts and treating them as regressions.

## Definition of done for handoff

Handoff is complete when:

- MPI backend is buildable/runnable.
- Serial-vs-MPI parity checks are scripted and reproducible.
- Performance scripts produce consistent rank-scaling results.
- README + this guide reflect the final MPI workflow.
