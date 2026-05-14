# nn_training

C++17 MNIST training project with serial and MPI backends, focused on data-parallel and model/pipeline-parallel scaling experiments.

## Current capabilities

- Trains MLPs on raw MNIST IDX files (no framework dependency).
- Uses BLAS-backed dense ops for core compute kernels.
- Supports these training modes:
  - `serial_train`
  - `mpi_dp_train` (flat data parallel, all-reduce gradients)
  - `mpi_dp_hierarchial_train` (hierarchical data parallel)
  - `mpi_dp_local_sgd_train` (periodic model averaging)
  - `mpi_mp_train` (sequential model parallel)
  - `mpi_mp_pip_train` (pipeline model parallel, optional load-balanced partitioning)
- Writes per-epoch metrics to CSV for all modes.

## Build requirements

- CMake >= 3.16
- C++17 compiler (GCC >= 9 recommended; enforced by `NN_FORCE_MODERN_GCC=ON`)
- BLAS
- MPI (for MPI binaries)
- Slurm `srun` for cluster launch scripts

## Build

```bash
bash scripts/build.sh
```

Clean rebuild:

```bash
bash scripts/build.sh --clean
```

Override compiler:

```bash
NN_CXX_COMPILER=/path/to/g++ bash scripts/build.sh --clean
```

## Data preparation

Download/extract MNIST into `data/mnist`:

```bash
bash scripts/prepare_mnist.sh
```

Most run scripts call this automatically.

## Main scripts

### Serial

- Smoke:
  - `bash scripts/run_serial_smoke.sh`
- Perf:
  - `bash scripts/run_serial_perf.sh`

Serial smoke overrides: `SMOKE_EPOCHS`, `SMOKE_BATCH_SIZE`, `SMOKE_TRAIN_SAMPLES`, `SMOKE_VAL_SAMPLES`, `SMOKE_LEARNING_RATE`, `SMOKE_HIDDEN`.

Serial perf overrides: `EPOCHS`, `BATCH_SIZE`, `TRAIN_SAMPLES`, `VAL_SAMPLES`, `LEARNING_RATE`, `HIDDEN`, `OUT_CSV`.

### Data parallel (MPI)

- Flat DP smoke:
  - `bash scripts/run_mpi_dp_smoke.sh`
- Flat DP perf:
  - `bash scripts/run_mpi_dp_perf.sh`
- Hierarchical DP smoke:
  - `bash scripts/run_mpi_dp_hierarchial_smoke.sh`
- Hierarchical DP perf:
  - `bash scripts/run_mpi_dp_hierarchial_perf.sh`
- Local-SGD DP perf:
  - `bash scripts/run_mpi_dp_local_sgd_perf.sh`

Common DP launcher overrides:

- `DP_NODES`
- `DP_TASKS_PER_NODE`
- `DP_CPUS_PER_TASK`
- `GLOBAL_BATCH` (or `SMOKE_GLOBAL_BATCH` in smoke scripts)
- `OUT_CSV` (all perf scripts; also hierarchical smoke)
- `SYNC_EVERY` (Local-SGD only)

### Model/pipeline parallel (MPI)

- Model-parallel smoke:
  - `bash scripts/run_mpi_mp_smoke.sh`
- Model-parallel perf:
  - `bash scripts/run_mpi_mp_perf.sh`
- Pipeline smoke:
  - `bash scripts/run_mpi_mp_pip_smoke.sh`
- Pipeline perf:
  - `bash scripts/run_mpi_mp_pip_perf.sh`

Common MP launcher overrides:

- `MP_NODES`
- `MP_TASKS_PER_NODE`
- `MP_CPUS_PER_TASK`
- `BATCH_SIZE` / `SMOKE_BATCH`
- `MICROBATCHES` / `SMOKE_MICROBATCHES` (pipeline)
- `BALANCE_LAYERS=true` (pipeline perf; passes `--load-balance-layers`)
- `OUT_CSV` (all perf scripts)

## Reproduce scaling studies

Data-parallel sweep (flat + hierarchical + local-SGD):

```bash
bash scripts/run_dp_scaling_study.sh
```

Outputs under `results/scaling/` and can be summarized with:

```bash
python3 results/scaling/summarize_dp_scaling.py results/scaling
```

Model/pipeline sweep (batch/microbatch/world-size charts):

```bash
bash scripts/run_mp_scaling_study.sh
```

Alternative fixed layout sweep:

```bash
bash scripts/run_mp_scaling_study_2x4_m_32.sh
```

Visualize sweep outputs:

```bash
python3 src/visualize_sweep.py <charts_output_dir>
```

## Direct binary usage

All binaries share common CLI flags:

- `--epochs`
- `--batch`
- `--lr`
- `--seed`
- `--train-samples`
- `--val-samples`
- `--hidden` (comma-separated)
- `--data-dir`
- `--output`

Mode-specific flags:

- `--sync-every` (local-SGD DP)
- `--microbatches` (pipeline MP)
- `--load-balance-layers` (pipeline MP)

Examples:

```bash
./build/serial_train --epochs 5 --batch 256 --hidden 256,128,64 --data-dir data/mnist --output results/serial.csv
```

```bash
srun -N 1 -n 4 ./build/mpi_dp_train --epochs 5 --batch 1024 --hidden 256,128,64 --data-dir data/mnist --output results/mpi_dp.csv
```

```bash
srun -N 2 -n 4 ./build/mpi_mp_pip_train --epochs 2 --batch 4096 --microbatches 32 --load-balance-layers --hidden 512,512,256,256,128,64,32 --data-dir data/mnist --output results/mpi_mp_pip.csv
```

## Report

The report sources are in `report/` (`main.tex`, generated `main.pdf`).

Build PDF:

```bash
cd report
latexmk -pdf main.tex
```

## Notes

- Threading is pinned to 1 in run scripts (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) to avoid oversubscription.
- Some filenames intentionally use `hierarchial` (spelling preserved for compatibility with existing scripts/binaries).
