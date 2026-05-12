#!/usr/bin/env python3
"""
visualize_sweep.py

Usage:
    python3 src/visualize_sweep.py <result_dir>

Reads from result_dir:
    chart1_batch_sweep.csv       -- batch size sweep  (serial + mp + pip)
    chart2_microbatch_sweep.csv  -- microbatch sweep  (serial + mp + pip)
    chart3_time_split.csv        -- pipeline timing breakdown per epoch
    chart4_worldsize_sweep.csv   -- world size sweep  (serial + mp + pip)

Each chart CSV is optional; missing files are skipped with a warning.

Outputs (written to result_dir):
    chart1_epoch_time.png
    chart1_throughput.png
    chart2_epoch_time.png
    chart2_timing_split.png
    chart4_epoch_time.png
    chart4_throughput.png
    chart4_speedup.png
"""

import sys
import os
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required: pip install matplotlib numpy")
    sys.exit(1)

# ── Style constants ────────────────────────────────────────────────────────────

COLORS = {
    "serial":                "#4C72B0",
    "mpi-mp":                "#DD8452",
    "mpi-mp-pip":            "#55A868",
    "mpi-mp-pip-balanced":   "#8172B2",
}
LABELS = {
    "serial":                "Serial",
    "mpi-mp":                "Model parallel",
    "mpi-mp-pip":            "Pipeline parallel",
    "mpi-mp-pip-balanced":   "Pipeline parallel (balanced)",
}
MODES = ["serial", "mpi-mp", "mpi-mp-pip", "mpi-mp-pip-balanced"]

TIMING_ORDER = [
    "fwd_comm", "fwd_compute", "fwd_send",
    "bwd_comm", "bwd_compute",
    "grad_accum", "grad_apply", "bwd_send",
]
TIMING_COLORS = {
    "fwd_comm":    "#d62728",
    "fwd_compute": "#1f77b4",
    "fwd_send":    "#ff9896",
    "bwd_comm":    "#e377c2",
    "bwd_compute": "#2ca02c",
    "grad_accum":  "#bcbd22",
    "grad_apply":  "#17becf",
    "bwd_send":    "#f7b6d2",
}
TIMING_LABELS = {
    "fwd_comm":    "fwd comm (wait)",
    "fwd_compute": "fwd compute",
    "fwd_send":    "fwd send drain",
    "bwd_comm":    "bwd comm (wait)",
    "bwd_compute": "bwd compute",
    "grad_accum":  "grad accumulate",
    "grad_apply":  "grad apply",
    "bwd_send":    "bwd send drain",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_metrics(path):
    """
    Load a chart metrics CSV (chart1/2/4 format).
    Extra columns are ignored; rows with parse errors are skipped silently.
    """
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "mode":             row["mode"],
                    "batch_size":       int(row["batch_size"]),
                    "microbatch_count": int(row["microbatch_count"]),
                    "mb_size":          int(row["mb_size"]),
                    "world_size":       int(row["world_size"]),
                    "nodes":            int(row["nodes"]),
                    "layout":           row["layout"],
                    "epoch":            int(row["epoch"]),
                    "train_loss":       float(row["train_loss"]),
                    "train_acc":        float(row["train_acc"]),
                    "val_loss":         float(row["val_loss"]),
                    "val_acc":          float(row["val_acc"]),
                    "epoch_time_ms":    float(row["epoch_time_ms"]),
                    "train_samples":    int(row["train_samples"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_timings(path):
    """
    Load chart3_time_split.csv.
    Only mean_s is required; min_s and max_s are absent in the new format.
    """
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "mode":             row["mode"],
                    "batch_size":       int(row["batch_size"]),
                    "microbatch_count": int(row["microbatch_count"]),
                    "mb_size":          int(row["mb_size"]),
                    "epoch":            int(row["epoch"]),
                    "metric":           row["metric"].strip(),
                    "mean_s":           float(row["mean_s"]),
                })
            except (ValueError, KeyError):
                continue
    return rows

# ── Aggregation helpers ────────────────────────────────────────────────────────

def _matches(row, filters):
    return all(row.get(k) == v for k, v in filters.items())


def avg_last_half(rows, value_key, **filters):
    """
    Average value_key over the last half of epochs for rows matching filters.
    Returns None if no matching rows exist.
    """
    subset = [r[value_key] for r in rows if _matches(r, filters)]
    if not subset:
        return None
    half = max(1, len(subset) // 2)
    return sum(subset[-half:]) / half


def timing_means(timings, **filters):
    """
    Return {metric: mean_s} averaged over last half of epochs, for rows
    matching filters.  Safe to stack — uses the mean-across-ranks column.
    """
    by_metric = defaultdict(list)
    for row in timings:
        if _matches(row, filters):
            by_metric[row["metric"]].append((row["epoch"], row["mean_s"]))
    result = {}
    for metric, epoch_vals in by_metric.items():
        epoch_vals.sort()
        half = max(1, len(epoch_vals) // 2)
        result[metric] = sum(v for _, v in epoch_vals[-half:]) / half
    return result

# ── Derived metrics ────────────────────────────────────────────────────────────

def epoch_time_s(rows, **filters):
    v = avg_last_half(rows, "epoch_time_ms", **filters)
    return v / 1000.0 if v is not None else None


def throughput(rows, **filters):
    t = epoch_time_s(rows, **filters)
    samples = next(
        (r["train_samples"] for r in rows if _matches(r, filters)), None)
    return samples / t if (t and samples) else None

# ── Shared figure helpers ──────────────────────────────────────────────────────

def add_note(fig, text):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=8, color="#777", style="italic")


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _line_plot(ax, x_vals, y_vals, mode, linestyle="-"):
    valid = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
    if not valid:
        return
    xs, ys = zip(*valid)
    ax.plot(xs, ys, marker="o", label=LABELS[mode],
            color=COLORS[mode], lw=2, linestyle=linestyle)

# ── Chart 1: epoch time and throughput vs batch size ──────────────────────────

def chart1(metrics, out_dir):
    batches = sorted({r["batch_size"] for r in metrics})
    if not batches:
        print("  chart1: no data, skipping")
        return

    # Build a note from a non-serial row (serial has no layout/ws context).
    ref = next((r for r in metrics if r["mode"] != "serial"), None)
    note = (f"layout={ref['layout']}  ws={ref['world_size']}" if ref else "")

    # Epoch time.
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ys = [epoch_time_s(metrics, mode=mode, batch_size=b) for b in batches]
        _line_plot(ax, batches, ys, mode)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Epoch time vs batch size\n(mean over last half of epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart1_epoch_time.png"))

    # Throughput.
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ys = [throughput(metrics, mode=mode, batch_size=b) for b in batches]
        _line_plot(ax, batches, ys, mode)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput vs batch size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart1_throughput.png"))

# ── Chart 2: epoch time and timing split vs microbatch count ──────────────────

def chart2(metrics, timings, out_dir):
    # Distinct microbatch counts used by the pipeline runs (mb=0 means N/A).
    mb_counts = sorted({r["microbatch_count"] for r in metrics
                        if r["mode"] == "mpi-mp-pip" and r["microbatch_count"] > 0})
    if not mb_counts:
        print("  chart2: no pipeline microbatch data found, skipping")
        return

    batch = next((r["batch_size"] for r in metrics
                  if r["mode"] == "mpi-mp-pip"), None)
    ws    = next((r["world_size"] for r in metrics
                  if r["mode"] == "mpi-mp-pip"), "?")
    note = f"batch={batch}  ws={ws}"

    # Build mb_size lookup for axis labels.
    mb_size_of = {r["microbatch_count"]: r["mb_size"] for r in metrics
                  if r["mode"] == "mpi-mp-pip" and r["microbatch_count"] > 0}
    xlabels = [f"M={m}\n(mb_size={mb_size_of.get(m, '?')})" for m in mb_counts]
    x = np.arange(len(mb_counts))

    # ── Epoch time ──
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pipeline varies with mb.
    pip_ys = [epoch_time_s(metrics, mode="mpi-mp-pip", microbatch_count=m)
              for m in mb_counts]
    ax.plot(x, pip_ys, marker="o", label=LABELS["mpi-mp-pip"],
            color=COLORS["mpi-mp-pip"], lw=2)

    # Serial and mp are flat reference lines (recorded with mb=0).
    for mode in ("serial", "mpi-mp"):
        t = epoch_time_s(metrics, mode=mode, microbatch_count=0)
        if t is not None:
            ax.axhline(t, linestyle="--", color=COLORS[mode],
                       label=LABELS[mode], lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Microbatch count (M)")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Epoch time vs microbatch count\n(mean over last half of epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart2_epoch_time.png"))

    # ── Timing split stacked bar (pipeline only) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(mb_counts))

    for metric in TIMING_ORDER:
        heights = np.array([
            timing_means(timings, mode="mpi-mp-pip",
                         microbatch_count=m).get(metric, 0.0)
            for m in mb_counts
        ])
        if heights.sum() == 0:
            continue
        ax.bar(x, heights, 0.55, bottom=bottoms,
               label=TIMING_LABELS.get(metric, metric),
               color=TIMING_COLORS.get(metric, "#888"))
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Microbatch count (M)")
    ax.set_ylabel("Time (s, mean across ranks, epoch total)")
    ax.set_title("Pipeline timing breakdown vs microbatch count")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart2_timing_split.png"))

# ── Chart 4: epoch time, throughput, and speedup vs world size ────────────────

def chart4(metrics, out_dir):
    ws_values = sorted({r["world_size"] for r in metrics})
    if not ws_values:
        print("  chart4: no data, skipping")
        return

    batch = next((r["batch_size"] for r in metrics
                  if r["mode"] == "mpi-mp-pip"), None)
    mb    = next((r["microbatch_count"] for r in metrics
                  if r["mode"] == "mpi-mp-pip" and r["microbatch_count"] > 0), None)
    note = f"batch={batch}  M={mb}  2 tasks/node"

    x       = np.arange(len(ws_values))
    xlabels = [str(ws) for ws in ws_values]

    # ── Epoch time ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ys = [epoch_time_s(metrics, mode=mode, world_size=ws)
              for ws in ws_values]
        ls = "--" if mode == "serial" else "-"
        _line_plot(ax, x, ys, mode, linestyle=ls)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("World size (total ranks)")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Epoch time vs world size\n(mean over last half of epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart4_epoch_time.png"))

    # ── Throughput ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ys = [throughput(metrics, mode=mode, world_size=ws)
              for ws in ws_values]
        ls = "--" if mode == "serial" else "-"
        _line_plot(ax, x, ys, mode, linestyle=ls)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("World size (total ranks)")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput vs world size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart4_throughput.png"))

    # ── Balanced improvement over unbalanced pipeline ──
    # Shows (pip_time - pip_bal_time) / pip_time * 100 per world size.
    # Positive = balanced is faster. Zero at ws=8 is expected (1 layer/rank,
    # no skew to fix). The interesting signal is at ws=2 and ws=4.
    fig, ax = plt.subplots(figsize=(8, 5))

    improvements = []
    for ws in ws_values:
        t_pip = epoch_time_s(metrics, mode="mpi-mp-pip",          world_size=ws)
        t_bal = epoch_time_s(metrics, mode="mpi-mp-pip-balanced",  world_size=ws)
        if t_pip and t_bal:
            improvements.append((t_pip - t_bal) / t_pip * 100.0)
        else:
            improvements.append(None)

    valid_x = [xi for xi, v in zip(x, improvements) if v is not None]
    valid_y = [v  for v       in improvements         if v is not None]

    if valid_x:
        ax.bar(valid_x, valid_y, color=[
            COLORS["mpi-mp-pip-balanced"] if v >= 0 else "#d62728"
            for v in valid_y
        ], alpha=0.85, width=0.5)

        # Label each bar. For near-zero bars place the label above the
        # baseline; for normal bars place it just above the bar top.
        for xi, v in zip(valid_x, valid_y):
            label_y = max(v, 0) + 0.15
            ax.text(xi, label_y, f"{v:+.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("World size (total ranks)")
    ax.set_ylabel("Epoch time improvement (%)")
    ax.set_title(
        "Load-balanced pipeline improvement over unbalanced\n"
        "(pip_time − bal_time) / pip_time × 100  —  positive = balanced is faster")
    ax.grid(True, axis="y", alpha=0.3)

    # Add headroom above the tallest bar so its label is never clipped.
    if valid_y:
        y_max = max(valid_y)
        y_min = min(valid_y)
        padding = (y_max - min(y_min, 0)) * 0.18 + 0.5
        ax.set_ylim(bottom=min(y_min, 0) - 0.3, top=y_max + padding)
    add_note(fig, note)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart4_balanced_improvement.png"))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    result_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(result_dir):
        print(f"error: {result_dir} is not a directory")
        sys.exit(1)

    def try_load(filename, loader):
        path = os.path.join(result_dir, filename)
        if not os.path.exists(path):
            print(f"  warning: {filename} not found, skipping")
            return None
        data = loader(path)
        print(f"  loaded {len(data):>5} rows  <- {filename}")
        return data

    print(f"Result dir: {result_dir}\n")

    m1 = try_load("chart1_batch_sweep.csv",      load_metrics)
    m2 = try_load("chart2_microbatch_sweep.csv", load_metrics)
    t3 = try_load("chart3_time_split.csv",       load_timings)
    m4 = try_load("chart4_worldsize_sweep.csv",  load_metrics)

    print()

    if m1:
        print("── Chart 1: batch size sweep ──")
        chart1(m1, result_dir)

    if m2 and t3:
        print("── Chart 2/3: microbatch sweep ──")
        chart2(m2, t3, result_dir)
    elif m2:
        print("── Chart 2: microbatch sweep (no timing data for split chart) ──")
        chart2(m2, [], result_dir)

    if m4:
        print("── Chart 4: world size sweep ──")
        chart4(m4, result_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()