# nn_training

Serial-first C++ neural-network training project using real MNIST IDX files.

This repository currently supports both a serial training baseline and an MPI data-parallel baseline on MNIST, with shared parity rules and reproducible script workflows.

For MPI contributor handoff tasks and execution order, see `NEXT_STEPS.md`.

## What this project currently does

- Trains a simple MLP in serial (CPU) with SGD.
- Loads MNIST directly from raw IDX files (no PyTorch/TensorFlow dependency).
- Uses BLAS-backed linear algebra for core dense operations.
- Writes per-epoch metrics to CSV.
- Provides scripts for:
  - MNIST download/prep
  - local smoke runs
  - cluster batch launch for serial and MPI-DP runs

## Repository structure

```text
nn_training/
├── CMakeLists.txt
├── .gitignore
├── README.md
├── include/
│   ├── config.h
│   ├── tensor.h
│   ├── mlp.h
│   ├── data_mnist.h
│   ├── train_common.h
│   ├── train_serial.h
│   └── train_mpi_data_parallel.h
├── src/
│   ├── main.cpp
│   ├── tensor.cpp
│   ├── mlp.cpp
│   ├── data_mnist.cpp
│   ├── train_common.cpp
│   ├── train_serial.cpp
│   └── train_mpi_data_parallel.cpp
├── scripts/
│   ├── prepare_mnist.sh
│   ├── build.sh
│   ├── run_serial_smoke.sh
│   ├── run_serial_perf.sh
│   ├── run_mpi_dp_smoke.sh
│   ├── run_mpi_dp_perf.sh
│   ├── compare_metrics.py
│   ├── job-serial.slurm
│   └── job_mpi_dp.slurm
├── docs/
│   └── serial_mpi_parity.md
├── data/
│   └── mnist/        # created/populated by prepare_mnist.sh
└── results/          # generated CSV/log outputs
```

## File-by-file explanation

### Build and repo config

- `CMakeLists.txt`
  - Defines the C++17 build.
  - Discovers and links MPI (`find_package(MPI REQUIRED)`).
  - Discovers and links BLAS (`find_package(BLAS REQUIRED)`).
  - Builds `nn_core`, `serial_train`, and `mpi_dp_train`.
- `.gitignore`
  - Ignores build artifacts and generated results/data outputs.

### Public headers (`include/`)

- `include/config.h`
  - Central runtime configuration (`TrainConfig`).
  - Holds model sizes, training hyperparameters, MNIST file paths, and metrics output path.
- `include/tensor.h`
  - Lightweight matrix container (`Matrix`) and CPU math helpers.
  - Declares ops used by MLP implementation (`matmul`, `transpose`, `relu`, `softmax`).
- `include/mlp.h`
  - Declares MLP class and batch metric struct.
  - Exposes `compute_batch_gradients(...)`, `apply_gradients(...)`, `train_batch(...)`, and `evaluate_batch(...)`.
- `include/data_mnist.h`
  - Declares dataset structure and MNIST data-loading/subset APIs.
- `include/train_common.h`
  - Shared training utilities (config/dataset prep, metadata helpers, parity-safe helpers).
- `include/train_serial.h`
  - Declares serial training entrypoint used by `main.cpp`.
- `include/train_mpi_data_parallel.h`
  - Declares MPI data-parallel training entrypoint.

### Source implementation (`src/`)

- `src/main.cpp`
  - Shared CLI front-end for training binaries.
  - Parses options like `--epochs`, `--batch`, `--hidden`, `--data-dir`, `--output`.
  - Builds `TrainConfig`, dispatches by `--mode` (`serial|mpi-dp`).
  - `serial_train` defaults to serial mode; `mpi_dp_train` defaults to mpi-dp mode.
- `src/tensor.cpp`
  - Implements matrix storage and numeric primitives.
  - Uses BLAS `sgemm_` / `saxpy_` calls for core matrix multiply and row-bias accumulation.
  - Includes ReLU and row-wise softmax utilities needed by forward/backward passes.
- `src/mlp.cpp`
  - Implements MLP initialization, forward pass, loss/accuracy metrics, backprop, and SGD updates.
  - Uses BLAS-backed update kernels (`sgemv_`, `saxpy_`) for gradient reductions and parameter updates.
- `src/data_mnist.cpp`
  - Implements binary IDX parsing for MNIST image/label files.
  - Validates MNIST magic numbers and normalizes pixel values to `[0, 1]`.
  - Supports extracting selected subsets.
- `src/train_serial.cpp`
  - Serial backend implementation and runner:
    - owns per-epoch training/evaluation loop
    - uses shared setup/output helpers
    - writes serial metrics output
- `src/train_mpi_data_parallel.cpp`
  - MPI data-parallel backend runner:
    - fixed global batch, rank-local batch slicing
    - gradient averaging via `MPI_Allreduce`
    - rank-0 CSV output with parity-compatible schema
- `src/train_common.cpp`
  - Backend-neutral setup and parity helpers:
    - validates train config
    - builds reproducible train/val subsets
    - provides shared metadata formatting and output-dir utility

### Scripts (`scripts/`)

- `scripts/prepare_mnist.sh`
  - Downloads MNIST `.gz` files from the official mirror and extracts IDX files into `data/mnist/`.
  - Safe to rerun; skips files that already exist.
- `scripts/build.sh`
  - Shared build entrypoint:
    1. configures CMake only when needed
    2. builds with `make`
  - Supports `--clean` for a clean reconfigure/rebuild.
- `scripts/run_serial_smoke.sh`
  - One-command local smoke run (run-only):
    1. prepare MNIST
    2. run a short serial training job
    3. output metrics to `results/smoke_metrics.csv`
- `scripts/run_serial_perf.sh`
  - One-command serial performance run (run-only):
    1. prepare MNIST
    2. run a larger training configuration
    3. write metrics CSV and print timing/throughput summary
- `scripts/run_mpi_dp_smoke.sh`
  - MPI data-parallel smoke run with fixed global batch semantics.
  - Invokes `srun` internally using `DP_NODES` and `DP_TASKS_PER_NODE`.
  - Validates requested resources against current Slurm allocation.
- `scripts/run_mpi_dp_perf.sh`
  - MPI data-parallel performance run with fixed global batch semantics.
  - Invokes `srun` internally using `DP_NODES` and `DP_TASKS_PER_NODE`.
  - Prints allocation and launch configuration before starting.
- `scripts/compare_metrics.py`
  - Compares two metrics CSVs with tolerance checks (useful for future serial-vs-MPI correctness gates).
- `scripts/job-serial.slurm`
  - Slurm batch script for serial run on cluster resources.
  - Run-only job launcher via `srun` (expects prebuilt binary).
- `scripts/job_mpi_dp.slurm`
  - Slurm batch script for MPI data-parallel run.
  - Enforces 1 thread per rank and launches via `srun`.

### Docs (`docs/`)

- `docs/serial_mpi_parity.md`
  - Defines parity contract rules for future serial vs MPI comparisons.

### Runtime/generated directories

- `data/`
  - Dataset storage root (currently `data/mnist/`).
- `results/`
  - Metrics and run artifacts generated by scripts/training.
- `build/` (generated)
  - CMake build directory and compiled binaries.

## Typical usage

Compiler note: on this system, `/usr/bin/c++` points to GCC 7, while C++17 support is cleaner with newer GCC. We enforce this with `NN_FORCE_MODERN_GCC=ON` in CMake and default build script compiler `NN_CXX_COMPILER=/usr/bin/g++`. Override if needed:

```bash
NN_CXX_COMPILER=/path/to/g++ bash scripts/build.sh --clean
```

BLAS note: scripts pin BLAS/OpenMP threads to 1 (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) so serial and future MPI comparisons remain fair and reproducible.

MPI script note: `run_mpi_dp_smoke.sh` and `run_mpi_dp_perf.sh` default to a single-rank launch (`DP_NODES=1`, `DP_TASKS_PER_NODE=1`). Set `DP_NODES` and `DP_TASKS_PER_NODE` explicitly for multi-rank runs.

Prepare dataset + run serial smoke:

```bash
bash scripts/build.sh
bash scripts/run_serial_smoke.sh
```

Build once:

```bash
bash scripts/build.sh
```

Run performance benchmark (defaults):

```bash
bash scripts/run_serial_perf.sh
```

Run MPI DP smoke after obtaining an interactive Slurm allocation:

```bash
bash scripts/build.sh
cpu 1
DP_NODES=1 DP_TASKS_PER_NODE=2 bash scripts/run_mpi_dp_smoke.sh
```

Run MPI DP performance (example: 4 ranks):

```bash
DP_NODES=1 DP_TASKS_PER_NODE=4 GLOBAL_BATCH=256 EPOCHS=10 \
  OUT_CSV=results/mpi_dp_perf_metrics.csv bash scripts/run_mpi_dp_perf.sh
```

Run MPI DP performance (example: 2 nodes x 64 ranks):

```bash
DP_NODES=2 DP_TASKS_PER_NODE=64 GLOBAL_BATCH=256 EPOCHS=10 \
  OUT_CSV=results/mpi_dp_perf_2n64r.csv bash scripts/run_mpi_dp_perf.sh
```

Run performance benchmark (custom):

```bash
EPOCHS=5 TRAIN_SAMPLES=8000 VAL_SAMPLES=1000 BATCH_SIZE=128 HIDDEN=256,128 \
OUT_CSV=results/perf_custom.csv bash scripts/run_serial_perf.sh
```

Compare two runs with tolerances:

```bash
python3 scripts/compare_metrics.py results/serial_baseline.csv results/mpi_candidate.csv \
  --loss-tol 0.05 --acc-tol 0.02
```

Direct binary run after build:

```bash
./build/serial_train \
  --epochs 5 \
  --batch 64 \
  --train-samples 4096 \
  --val-samples 512 \
  --hidden 128,64 \
  --data-dir data/mnist \
  --output results/metrics.csv
```

Common-main override examples:

```bash
./build/serial_train --mode serial --epochs 2 --batch 64 --train-samples 2048 --val-samples 256 --data-dir data/mnist --output results/serial_mode_check.csv
```

Direct MPI DP binary run (after build; launch with `srun`):

```bash
srun -N 1 -n 4 ./build/mpi_dp_train \
  --epochs 5 \
  --batch 256 \
  --train-samples 4096 \
  --val-samples 512 \
  --hidden 128,64 \
  --data-dir data/mnist \
  --output results/mpi_dp_metrics.csv
```

## Current milestone status

- Serial skeleton: complete.
- Real MNIST integration: complete.
- BLAS-backed kernels for serial path: complete.
- MPI data-parallel baseline (`MPI_Allreduce`): implemented.
- Common main/CLI dispatch (`serial` and `mpi-dp`): implemented.
- Serial-vs-MPI parity validation sweeps: in progress.
