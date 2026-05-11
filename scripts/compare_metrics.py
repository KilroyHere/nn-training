#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in metrics file: {path}")
    return rows


def as_float(row, key, path):
    try:
        return float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{key}' in {path}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Compare two training metrics CSV files with tolerances.")
    parser.add_argument("baseline", type=Path, help="Reference metrics CSV")
    parser.add_argument("candidate", type=Path, help="Candidate metrics CSV")
    parser.add_argument("--loss-tol", type=float, default=0.05, help="Allowed final val_loss delta")
    parser.add_argument("--acc-tol", type=float, default=0.02, help="Allowed final val_acc delta")
    args = parser.parse_args()

    baseline_rows = read_rows(args.baseline)
    candidate_rows = read_rows(args.candidate)

    if len(baseline_rows) != len(candidate_rows):
        raise SystemExit(
            f"Epoch count mismatch: baseline={len(baseline_rows)} candidate={len(candidate_rows)}")

    base_last = baseline_rows[-1]
    cand_last = candidate_rows[-1]

    base_loss = as_float(base_last, "val_loss", args.baseline)
    cand_loss = as_float(cand_last, "val_loss", args.candidate)
    base_acc = as_float(base_last, "val_acc", args.baseline)
    cand_acc = as_float(cand_last, "val_acc", args.candidate)

    loss_delta = abs(cand_loss - base_loss)
    acc_delta = abs(cand_acc - base_acc)

    print(
        "Comparison summary: "
        f"baseline_val_loss={base_loss:.6f}, candidate_val_loss={cand_loss:.6f}, "
        f"baseline_val_acc={base_acc:.6f}, candidate_val_acc={cand_acc:.6f}, "
        f"loss_delta={loss_delta:.6f}, acc_delta={acc_delta:.6f}")

    failed = []
    if loss_delta > args.loss_tol:
        failed.append(
            f"val_loss delta {loss_delta:.6f} exceeds tolerance {args.loss_tol:.6f}")
    if acc_delta > args.acc_tol:
        failed.append(
            f"val_acc delta {acc_delta:.6f} exceeds tolerance {args.acc_tol:.6f}")

    if failed:
        raise SystemExit("Comparison failed: " + "; ".join(failed))

    print("Comparison passed.")


if __name__ == "__main__":
    main()
