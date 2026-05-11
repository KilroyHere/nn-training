#!/usr/bin/env python3
"""
visualize_sweep.py  --  Chart 2 visualisation

Usage:
    python3 scripts/visualize_sweep.py <metrics.csv> <timings.csv>

Produces (saved next to metrics.csv):
    chart2_epoch_time.png     -- epoch time vs batch size, all three modes
    chart2_throughput.png     -- throughput vs batch size, all three modes
    chart2_timing_mp.png      -- timing breakdown stacked bar, model-parallel
    chart2_timing_pip.png     -- timing breakdown stacked bar, pipeline
    chart2_timing_compare.png -- compute vs comm side-by-side, both MPI modes
    chart2_time_budget.png    -- horizontal time budget, poster-ready
"""

import sys
import os
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required: pip install matplotlib numpy")
    sys.exit(1)

# ─── Run configuration ────────────────────────────────────────────────────────
# Edit these to match your experiment before running.
MICROBATCH_COUNT = 32
NODES            = 4
TASKS_PER_NODE   = 2

def run_config_note():
    """Single annotation line shown on every chart."""
    W = NODES * TASKS_PER_NODE
    return (f"M={MICROBATCH_COUNT} microbatches  ·  "
            f"{NODES} nodes × {TASKS_PER_NODE} proc/node  (W={W})")


# ─── Load data ────────────────────────────────────────────────────────────────

def load_metrics(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "mode":          row["mode"],
                    "batch_size":    int(row["batch_size"]),
                    "epoch":         int(row["epoch"]),
                    "train_loss":    float(row["train_loss"]),
                    "train_acc":     float(row["train_acc"]),
                    "val_loss":      float(row["val_loss"]),
                    "val_acc":       float(row["val_acc"]),
                    "epoch_time_ms": float(row["epoch_time_ms"]),
                    "train_samples": int(row["train_samples"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_timings(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "mode":       row["mode"],
                "batch_size": int(row["batch_size"]),
                "epoch":      int(row["epoch"]),
                "metric":     row["metric"],
                "min_s":      float(row["min_s"]),
                "mean_s":     float(row["mean_s"]),
                "max_s":      float(row["max_s"]),
            })
    return rows


def avg_last_half(rows, key, mode, batch):
    """Average a metric over the last half of epochs to reduce noise."""
    subset = [r[key] for r in rows
              if r["mode"] == mode and r["batch_size"] == batch]
    if not subset:
        return None
    half = max(1, len(subset) // 2)
    return sum(subset[-half:]) / half


def _timing_by_metric(timings, mode, batch, field):
    by_metric_epoch = defaultdict(list)
    for row in timings:
        if row["mode"] == mode and row["batch_size"] == batch:
            by_metric_epoch[row["metric"]].append((row["epoch"], row[field]))
    result = {}
    for metric, epoch_vals in by_metric_epoch.items():
        epoch_vals.sort()
        half = max(1, len(epoch_vals) // 2)
        result[metric] = sum(v for _, v in epoch_vals[-half:]) / half
    return result


def timing_mean_by_metric(timings, mode, batch):
    """Mean across ranks — safe to stack. Sum ≈ mean total rank time."""
    return _timing_by_metric(timings, mode, batch, "mean_s")


def timing_max_by_metric(timings, mode, batch):
    """Max across ranks — NOT safe to stack. Use for % budget charts only."""
    return _timing_by_metric(timings, mode, batch, "max_s")


# ─── Shared style constants ───────────────────────────────────────────────────

COLORS = {
    "serial":     "#4C72B0",
    "mpi-mp":     "#DD8452",
    "mpi-mp-pip": "#55A868",
}
LABELS = {
    "serial":     "Serial",
    "mpi-mp":     "Model parallel",
    "mpi-mp-pip": "Pipeline parallel",
}

TIMING_METRICS_MP = [
    "fwd_comm", "fwd_compute", "fwd_send",
    "bwd_comm", "bwd_compute", "grad_apply", "bwd_send",
]
TIMING_METRICS_PIP = [
    "fwd_comm", "fwd_compute", "fwd_send",
    "bwd_comm", "bwd_compute", "grad_accum", "grad_apply", "bwd_send",
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
    "fwd_comm":    "fwd comm (MPI wait)",
    "fwd_compute": "fwd compute",
    "fwd_send":    "fwd send drain",
    "bwd_comm":    "bwd comm (MPI wait)",
    "bwd_compute": "bwd compute",
    "grad_accum":  "grad accumulate",
    "grad_apply":  "grad apply",
    "bwd_send":    "bwd send drain",
}


def add_config_note(fig):
    """Stamp the run config as a small italic note at the bottom of the figure."""
    fig.text(0.99, 0.01, run_config_note(),
             ha="right", va="bottom", fontsize=8,
             color="#777", style="italic",
             transform=fig.transFigure)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ─── Chart 1: epoch time ──────────────────────────────────────────────────────

def plot_epoch_time(metrics, batches, modes, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes:
        times = [avg_last_half(metrics, "epoch_time_ms", mode, b) for b in batches]
        valid = [(b, t) for b, t in zip(batches, times) if t is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, [y / 1000 for y in ys],
                marker="o", label=LABELS[mode], color=COLORS[mode], linewidth=2)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Epoch time vs batch size\n(avg over last half of epochs)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_config_note(fig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart2_epoch_time.png"))


# ─── Chart 2: throughput ──────────────────────────────────────────────────────

def plot_throughput(metrics, batches, modes, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes:
        vals = []
        for b in batches:
            t = avg_last_half(metrics, "epoch_time_ms", mode, b)
            samples = next(
                (r["train_samples"] for r in metrics
                 if r["mode"] == mode and r["batch_size"] == b), None)
            vals.append((b, samples / (t / 1000)) if t and samples else (b, None))
        valid = [(b, v) for b, v in vals if v is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o", label=LABELS[mode],
                color=COLORS[mode], linewidth=2)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput vs batch size")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_config_note(fig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart2_throughput.png"))


# ─── Chart 3 & 4: timing breakdown stacked bar ────────────────────────────────

def plot_timing_breakdown(timings, batches, mode, metric_order, out_dir, tag,
                          metrics=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(batches))
    width = 0.55
    bottoms = np.zeros(len(batches))

    for metric in metric_order:
        heights = np.array([
            timing_mean_by_metric(timings, mode, b).get(metric, 0.0)
            for b in batches
        ])
        if heights.sum() == 0:
            continue
        ax.bar(x, heights, width, bottom=bottoms,
               label=TIMING_LABELS.get(metric, metric),
               color=TIMING_COLORS.get(metric, "#888888"))
        bottoms += heights

    # Wall-clock epoch time as a reference diamond — bars should track this
    # closely since we now use mean_s (safe to sum) rather than max_s.
    if metrics is not None:
        wall_s = [avg_last_half(metrics, "epoch_time_ms", mode, b) for b in batches]
        wall_s = [w / 1000 if w is not None else None for w in wall_s]
        vx = [xi for xi, w in zip(x, wall_s) if w is not None]
        vw = [w for w in wall_s if w is not None]
        if vx:
            ax.plot(vx, vw, marker="D", color="black", linewidth=1.5,
                    markersize=5, label="wall-clock epoch time", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time (s, mean across ranks, epoch total)")
    ax.set_title(f"Timing breakdown — {LABELS[mode]}\n"
                 f"(mean across ranks — safe to stack)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    add_config_note(fig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"chart2_timing_{tag}.png"))


# ─── Chart 5: time budget (poster-ready) ─────────────────────────────────────

def plot_time_budget(timings, batches, out_dir):
    COMPUTE  = {"fwd_compute", "bwd_compute"}
    FWD_WAIT = {"fwd_comm"}
    BWD_WAIT = {"bwd_comm"}

    mpi_modes   = ["mpi-mp", "mpi-mp-pip"]
    mode_labels = {"mpi-mp": "Model parallel", "mpi-mp-pip": "Pipeline parallel"}
    BUD_COLORS  = {
        "compute": "#6b1515",
        "fwd":     "#c0392b",
        "bwd":     "#922b21",
        "other":   "#e8a0a0",
    }

    batch = batches[-1]
    rows  = [("Serial", None)] + [(mode_labels[m], m) for m in mpi_modes]

    fig, ax = plt.subplots(figsize=(10, 0.9 + len(rows) * 0.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    yticks, ylabels = [], []
    for idx, (label, mode) in enumerate(rows):
        y = len(rows) - 1 - idx
        yticks.append(y)
        ylabels.append(label)

        if mode is None:
            segs = {"compute": 100.0, "fwd": 0.0, "bwd": 0.0, "other": 0.0}
        else:
            t     = timing_mean_by_metric(timings, mode, batch)
            total = sum(t.values()) or 1.0
            segs  = {
                "compute": sum(t.get(m, 0) for m in COMPUTE)  / total * 100,
                "fwd":     sum(t.get(m, 0) for m in FWD_WAIT) / total * 100,
                "bwd":     sum(t.get(m, 0) for m in BWD_WAIT) / total * 100,
            }
            segs["other"] = 100 - sum(segs.values())

        left = 0.0
        for key, color in BUD_COLORS.items():
            w = segs.get(key, 0.0)
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.55, color=color)
            if w >= 7:
                ax.text(left + w / 2, y,
                        f"{'compute' if key == 'compute' else key} ({w:.0f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if key != "other" else "#7b1a1a",
                        fontweight="bold")
            left += w

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total timing budget (mean rank)", fontsize=10)
    ax.set_title(f"Time budget — batch {batch}, {NODES * TASKS_PER_NODE} ranks\n"
                 f"compute | P2P fwd wait | P2P bwd wait | other",
                 fontsize=11, pad=10)
    ax.tick_params(left=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.legend(handles=[
        mpatches.Patch(facecolor=BUD_COLORS["compute"], label="Compute (fwd + bwd)"),
        mpatches.Patch(facecolor=BUD_COLORS["fwd"],     label="P2P wait — fwd  (≈ AR fwd phase)"),
        mpatches.Patch(facecolor=BUD_COLORS["bwd"],     label="P2P wait — bwd  (≈ AR bwd phase)"),
        mpatches.Patch(facecolor=BUD_COLORS["other"],   label="Send drain + grad overhead"),
    ], loc="lower right", fontsize=9, framealpha=0.0)

    add_config_note(fig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart2_time_budget.png"))


# ─── Chart 6: compute vs communication comparison ─────────────────────────────

def plot_compute_vs_comm(timings, batches, out_dir):
    COMPUTE = {"fwd_compute", "bwd_compute"}
    COMM    = {"fwd_comm", "bwd_comm", "fwd_send", "bwd_send",
               "grad_accum", "grad_apply"}

    mpi_modes = ["mpi-mp", "mpi-mp-pip"]
    x = np.arange(len(batches))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, mode in zip(axes, mpi_modes):
        compute_vals = [sum(timing_mean_by_metric(timings, mode, b).get(m, 0) for m in COMPUTE)
                        for b in batches]
        comm_vals    = [sum(timing_mean_by_metric(timings, mode, b).get(m, 0) for m in COMM)
                        for b in batches]

        ax.bar(x - width / 2, compute_vals, width,
               label="Compute", color="#1f77b4", alpha=0.85)
        ax.bar(x + width / 2, comm_vals, width,
               label="MPI overhead", color="#d62728", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches])
        ax.set_xlabel("Batch size")
        ax.set_title(LABELS[mode])
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Time (s, mean across ranks, epoch total)")
    fig.suptitle("Compute vs MPI overhead\n(mean across ranks)", y=1.02)
    add_config_note(fig)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "chart2_timing_compare.png"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    metrics_path = sys.argv[1]
    timings_path = sys.argv[2]
    out_dir      = os.path.dirname(os.path.abspath(metrics_path))

    metrics = load_metrics(metrics_path)
    timings = load_timings(timings_path)

    batches = sorted({r["batch_size"] for r in metrics})
    modes   = ["serial", "mpi-mp", "mpi-mp-pip"]

    print(f"Loaded {len(metrics)} metric rows, {len(timings)} timing rows")
    print(f"Batch sizes : {batches}")
    print(f"Modes found : {sorted({r['mode'] for r in metrics})}")
    print(f"Output dir  : {out_dir}")
    print(f"Config note : {run_config_note()}")
    print()

    plot_epoch_time(metrics, batches, modes, out_dir)
    plot_throughput(metrics, batches, modes, out_dir)
    plot_timing_breakdown(
        timings, batches, "mpi-mp",     TIMING_METRICS_MP,  out_dir, "mp",
        metrics=metrics)
    plot_timing_breakdown(
        timings, batches, "mpi-mp-pip", TIMING_METRICS_PIP, out_dir, "pip",
        metrics=metrics)
    plot_time_budget(timings, batches, out_dir)
    plot_compute_vs_comm(timings, batches, out_dir)

    print("All charts written.")


if __name__ == "__main__":
    main()