# Serial vs MPI DP Parity Contract

This file defines the parity rules between `serial_train` and `mpi_dp_train`.

## Core parity rules

- Keep model architecture and optimizer identical.
- Keep dataset sampling deterministic via shared `seed`.
- Keep CSV schema identical:
  - `mode,seed,learning_rate,batch_size,train_samples,val_samples,hidden_layers,epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms`
- `mode` must be:
  - `serial` for serial runs
  - `mpi` for MPI data-parallel runs

## Batch semantics

- `batch_size` means **global batch** in both modes.
- For MPI DP:
  - `local_batch = global_batch / world_size`
  - require exact divisibility.
- Drop incomplete tail batches in both paths (same loop semantics).

## Metric semantics

- Train metrics:
  - serial: mean over serial train steps
  - mpi: mean over globally reduced per-rank step metrics
- Validation metrics:
  - computed on rank 0 and broadcast in MPI path
  - same validation dataset subset and epoch cadence as serial.

## Timing semantics

- `epoch_time_ms` includes:
  - shuffle
  - train-step loop
  - validation for that epoch
- MPI timing is barrier-aligned to represent global epoch wall time.

## Threading policy for MPI runs

Always pin to 1 thread per rank for fair process scaling:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
