#!/usr/bin/env python3
"""
visualize_sweep.py

Chart 1: Serial vs model-parallel vs pipeline
         x-axis = node layout, y-axis = epoch time (ms)

Chart 2: Model-parallel vs pipeline
         x-axis = batch size, y-axis = epoch time (ms)

Usage:
    python3 scripts/visualize_sweep.py results/chart1_*/chart1_results.csv
    python3 scripts/visualize_sweep.py results/chart2_*/chart2_results.csv
"""

import sys
import csv
import os
from collections import defaultdict

PALETTE = {
    "Serial":         "#8D8C87",
    "Model parallel": "#2E86C1",
    "Pipeline":       "#1A9E75",
}

MODE_LABEL = {
    "serial":       "Serial",
    "mpi-mp":       "Model parallel",
    "mpi-mp-pip":   "Pipeline",
}

LAYOUT_ORDER = [
    "1x1_ws1",
    "1x2__ws2",
    "1x4__ws4",
    "1x8__ws8",
    "2x4__ws8",
    "4x2__ws8",
]

LAYOUT_TICK = {
    "1x1_ws1":  "1×1\n(serial)",
    "1x2__ws2": "1×2\nws=2",
    "1x4__ws4": "1×4\nws=4",
    "1x8__ws8": "1×8\nws=8",
    "2x4__ws8": "2×4\nws=8\n(inter-node)",
    "4x2__ws8": "4×2\nws=8\n(inter-node)",
}

INTER_NODE_LAYOUTS = {"2x4__ws8", "4x2__ws8"}
MODES_C1 = ["serial", "mpi-mp", "mpi-mp-pip"]
MODES_C2 = ["mpi-mp", "mpi-mp-pip"]


# ─── Load ─────────────────────────────────────────────────────────────────────

def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "chart":    int(row["chart"]),
                "layout":   row["layout"],
                "mode":     row["mode"],
                "batch":    int(row["batch_size"]),
                "ws":       int(row["world_size"]),
                "epoch":    int(row["epoch"]),
                "val_acc":  float(row["val_acc"]),
                "epoch_ms": float(row["epoch_time_ms"]),
                "n_train":  int(row["train_samples"]),
                "hidden":   row["hidden_layers"],
            })
    return rows


def group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def steady_ms(epoch_rows):
    times = [r["epoch_ms"] for r in epoch_rows if r["epoch"] > 1]
    return sum(times) / len(times) if times else epoch_rows[0]["epoch_ms"]


def arch_summary(hidden_str):
    sep = "-" if "-" in hidden_str else ","
    layers = [x.strip() for x in hidden_str.split(sep) if x.strip()]
    sizes = " → ".join(layers)
    return f"784 → {sizes} → 10   ({len(layers)} hidden layers)"


def text_summary(rows, chart_num):
    by = group(rows, ["mode", "layout"] if chart_num == 1 else ["mode", "batch"])
    print(f"\n=== Chart {chart_num} epoch times (ms) ===")
    for k, er in sorted(by.items()):
        print(f"  {k}: {steady_ms(er):.0f} ms")


# ─── Chart 1 ──────────────────────────────────────────────────────────────────

def plot_chart1(rows, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    by = group(rows, ["mode", "layout"])
    layouts  = [l for l in LAYOUT_ORDER if any(k[1] == l for k in by)]
    x_labels = [LAYOUT_TICK.get(l, l) for l in layouts]
    x_pos    = list(range(len(layouts)))

    hidden_str = rows[0]["hidden"]
    batch      = rows[0]["batch"]

    serial_er = next((by[k] for k in by if k[0] == "serial"), None)
    serial_ms = steady_ms(serial_er) if serial_er else None

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        "Serial vs Model Parallel vs Pipeline — Epoch time by node layout\n"
        f"Architecture: {arch_summary(hidden_str)}   |   Batch size: {batch}",
        fontsize=11
    )

    for mode in MODES_C1:
        label = MODE_LABEL[mode]
        ls = "--" if mode == "serial" else "-"

        if mode == "serial":
            if serial_ms is None:
                continue
            xs = x_pos
            ys = [serial_ms] * len(layouts)
        else:
            xs, ys = [], []
            for i, layout in enumerate(layouts):
                k = (mode, layout)
                if k in by:
                    xs.append(i)
                    ys.append(steady_ms(by[k]))

        if not xs:
            continue

        ax.plot(xs, ys,
                color=PALETTE[label], marker="o", markersize=8,
                linewidth=2.2, linestyle=ls, label=label)

        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi/1000:.1f}s",
                        xy=(xi, yi), xytext=(0, 9),
                        textcoords="offset points",
                        ha="center", fontsize=8,
                        color=PALETTE[label])

    inter_idx = [i for i, l in enumerate(layouts) if l in INTER_NODE_LAYOUTS]
    if inter_idx:
        lo, hi = min(inter_idx) - 0.5, max(inter_idx) + 0.5
        ax.axvspan(lo, hi, alpha=0.07, color="#F5A623", zorder=0)
        ylim = ax.get_ylim()
        ax.text((lo + hi) / 2, ylim[1] * 0.99,
                "inter-node", ha="center", va="top",
                fontsize=9, color="#B87A00", style="italic")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Node layout  (nodes × tasks/node)", fontsize=10)
    ax.set_ylabel("Epoch time (ms)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle=":", axis="y")
    sns.despine(ax=ax)

    plt.tight_layout()
    out = os.path.join(out_dir, "chart1_epoch_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ─── Chart 2 ──────────────────────────────────────────────────────────────────

def plot_chart2(rows, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    by      = group(rows, ["mode", "batch"])
    batches = sorted(set(r["batch"] for r in rows))
    ws      = rows[0]["ws"]
    layout  = rows[0]["layout"]
    hidden_str = rows[0]["hidden"]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Model Parallel vs Pipeline — Epoch time vs batch size\n"
        f"Architecture: {arch_summary(hidden_str)}   |   Layout: {layout} (ws={ws})",
        fontsize=11
    )

    for mode in MODES_C2:
        label = MODE_LABEL[mode]
        xs, ys = [], []
        for batch in batches:
            k = (mode, batch)
            if k in by:
                xs.append(batch)
                ys.append(steady_ms(by[k]))
        if not xs:
            continue

        ax.plot(xs, ys,
                color=PALETTE[label], marker="o", markersize=8,
                linewidth=2.2, label=label)

        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi/1000:.1f}s",
                        xy=(xi, yi), xytext=(0, 9),
                        textcoords="offset points",
                        ha="center", fontsize=8,
                        color=PALETTE[label])

    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches], fontsize=9)
    ax.set_xlabel("Batch size", fontsize=10)
    ax.set_ylabel("Epoch time (ms)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle=":")
    sns.despine(ax=ax)

    plt.tight_layout()
    out = os.path.join(out_dir, "chart2_epoch_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: visualize_sweep.py <chart1_results.csv or chart2_results.csv>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    rows = load(path)
    if not rows:
        print("CSV is empty.")
        sys.exit(1)

    chart_num = rows[0]["chart"]
    out_dir   = os.path.dirname(os.path.abspath(path))

    print(f"Loaded {len(rows)} rows  (chart={chart_num})")
    print(f"Architecture: {rows[0]['hidden']}")
    text_summary(rows, chart_num)

    try:
        import seaborn  # noqa
    except ImportError:
        print("seaborn not found: pip install seaborn --break-system-packages")
        sys.exit(1)

    if chart_num == 1:
        plot_chart1(rows, out_dir)
    else:
        plot_chart2(rows, out_dir)


if __name__ == "__main__":
    main()