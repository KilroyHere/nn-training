#!/usr/bin/env python3
"""
Poster-ready scaling visualizations using seaborn + matplotlib.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    "Flat DP":        "#2563EB",
    "Hier DP":        "#DC2626",
    "Local SGD K=10": "#16A34A",
    "Local SGD K=50": "#D97706",
    "Ideal":          "#9CA3AF",
    "serial":         "#374151",
    "flat-dp":        "#2563EB",
    "local-sgd K=10": "#16A34A",
    "local-sgd K=50": "#D97706",
}

# ── axis labels ─────────────────────────────────────────────────────────────
# Two-line labels for grouped bars / scatter
CONFIG_LABELS = [
    "1 node\n4 ranks",
    "1 node\n16 ranks",
    "1 node\n32 ranks",
    "2 nodes\n64 ranks",
    "4 nodes\n128 ranks",
]
# Single-line for tight axes
CONFIG_LABELS_SHORT = ["1N·4R", "1N·16R", "1N·32R", "2N·64R", "4N·128R"]
MODEL_NAME = "MLP  256 → 128 → 64  on MNIST"

# ── raw data ────────────────────────────────────────────────────────────────
SPEEDUP = {
    "ranks":    [4,     16,    32,    64,    128],
    "flat_su":  [1.00,  3.22,  5.00,  6.71,  8.40],
    "hier_su":  [0.98,  2.92,  4.01,  5.31,  6.14],
    "k10_su":   [1.05,  3.93,  6.97, 12.41, 18.92],
    "k50_su":   [1.06,  4.01,  7.33, 13.01, 21.46],
    "ideal":    [1.0,   4.0,   8.0,  16.0,  32.0],
    "flat_eff": [100.0, 80.4,  62.5,  41.9,  26.3],
    "hier_eff": [97.9,  73.0,  50.1,  33.2,  19.2],
    "k10_eff":  [105.1, 98.3,  87.1,  77.5,  59.1],
    "k50_eff":  [105.8,100.3,  91.7,  81.3,  67.1],
}
EPOCH_MS = {
    "flat": [5003, 1555, 1001,  746,  595],
    "hier": [5111, 1713, 1247,  943,  815],
    "k10":  [4761, 1273,  718,  403,  264],
    "k50":  [4730, 1248,  682,  384,  233],
}
TIMING = {
    "flat_compute":   [4614, 1198, 639, 354, 210],
    "flat_allreduce": [  39,  107, 128, 164, 164],
    "flat_other":     [ 350,  250, 234, 227, 221],
    "k10_compute":    [4585, 1202, 661, 359, 213],
    "k10_sync":       [   4,   11,  14,  13,  25],
    "k10_other":      [ 171,   60,  42,  31,  26],
}
TRADEOFF = [
    # label,          epoch_ms, val_acc
    ("Serial\n(1 process)", 9696, 0.9226),
    ("Flat DP\n(128 ranks)",  595, 0.9225),
    ("Local SGD K=10\n(128 ranks)", 264, 0.9191),
    ("Local SGD K=50\n(128 ranks)", 233, 0.9173),
]
SPEEDUP_VS_SERIAL = {
    "flat_su": [1.94,  6.24,  9.69, 13.00, 16.28],
    "hier_su": [1.90,  5.66,  7.78, 10.28, 11.89],
    "k10_su":  [2.04,  7.62, 13.51, 24.05, 36.67],
    "k50_su":  [2.05,  7.77, 14.21, 25.22, 41.59],
}


# ── theme ──────────────────────────────────────────────────────────────────
def set_theme():
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "axes.linewidth": 1.1,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#D1D5DB",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.8,
    })


# ══════════════════════════════════════════════════════════════════════════
# Fig 0 – Model / Dataflow Architecture
# ══════════════════════════════════════════════════════════════════════════
def plot_architecture(out_dir):
    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.28)
    ax_table = fig.add_subplot(gs[0])
    ax_timeline = fig.add_subplot(gs[1])

    fig.text(0.055, 0.955, "Training Architecture Comparison", fontsize=22, fontweight="bold", va="top")
    fig.text(
        0.055,
        0.922,
        "Serial, Flat DP, Hierarchical DP, and Local SGD shown as communication patterns per training step.",
        fontsize=11.5,
        color="#4B5563",
        va="top",
    )

    # -------- top table --------
    ax_table.axis("off")
    columns = ["Method", "Serial", "Flat DP", "Hierarchical DP", "Local SGD"]
    rows = [
        ["Who computes?", "1 rank updates full model", "All ranks compute local grads", "All ranks compute local grads", "All ranks do local SGD"],
        ["Sync cadence", "No MPI sync", "Every step", "Every step (3 phases)", "Every K steps"],
        ["MPI ops per sync", "0 collectives / step", "1 x MPI_Allreduce", "Reduce + Allreduce + Bcast", "1 x MPI_Allreduce"],
        ["What is synchronized?", "N/A", "Gradients", "Gradients", "Weights"],
    ]

    tbl = ax_table.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.35)

    header_colors = ["#F3F4F6", "#2D63D8", "#1D4ED8", "#DC2626", "#059669"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(header_colors[c])
            if c > 0:
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.get_text().set_fontweight("bold")
        elif c == 0:
            cell.set_facecolor("#F9FAFB")
            cell.get_text().set_ha("left")
            cell.PAD = 0.06

    # -------- timeline --------
    methods = ["Serial", "Flat DP", "Hier DP", "Local SGD"]
    y_pos = [3, 2, 1, 0]

    for y, m in zip(y_pos, methods):
        ax_timeline.text(-2.5, y, m, ha="right", va="center", fontsize=11)

    # Each tuple: (start, width, color, label)
    bars = {
        "Serial": [(0, 100, "#2D63D8", "compute")],
        "Flat DP": [(0, 72, "#1D4ED8", "compute"), (72, 28, "#60A5FA", "allreduce")],
        "Hier DP": [(0, 62, "#DC2626", "compute"), (62, 14, "#FCA5A5", "intra"), (76, 10, "#EF4444", "inter"), (86, 14, "#F87171", "bcast")],
        "Local SGD": [(0, 90, "#059669", "compute"), (90, 10, "#34D399", "sync every K")],
    }

    for y, m in zip(y_pos, methods):
        for start, width, color, label in bars[m]:
            ax_timeline.barh(y, width, left=start, height=0.44, color=color, edgecolor=color, zorder=3)
            txt_color = "white" if color not in ("#FCA5A5", "#34D399") else "#111827"
            ax_timeline.text(start + width / 2.0, y, label, ha="center", va="center", fontsize=9, color=txt_color)

    ax_timeline.set_xlim(-22, 102)
    ax_timeline.set_ylim(-0.8, 3.8)
    ax_timeline.set_yticks([])
    ax_timeline.set_xticks([])
    for spine in ax_timeline.spines.values():
        spine.set_visible(False)
    ax_timeline.set_title("Communication Timeline (one training step)", loc="left", fontsize=14, fontweight="bold", pad=6)

    fig.text(
        0.055,
        0.06,
        "Interpretation: Hierarchical DP has the most sync phases per update; "
        "Local SGD reduces sync frequency by communicating once every K steps.",
        fontsize=10.5,
        color="#4B5563",
    )

    p = os.path.join(out_dir, "00_architecture.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 1 – Strong Scaling Speedup (log-log)
# ══════════════════════════════════════════════════════════════════════════
def plot_speedup(out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ranks = np.array(SPEEDUP["ranks"])

    series = [
        ("Flat DP",        "flat_su", "o", "-"),
        ("Hier DP",        "hier_su", "s", "-"),
        ("Local SGD K=10", "k10_su",  "^", "-"),
        ("Local SGD K=50", "k50_su",  "D", "-"),
        ("Ideal linear",   "ideal",   None,"--"),
    ]
    for name, key, marker, ls in series:
        color = PALETTE.get(name, PALETTE["Ideal"])
        kw = dict(color=color, linewidth=2, linestyle=ls, zorder=3)
        if marker:
            kw.update(marker=marker, markersize=7, markerfacecolor="white",
                      markeredgewidth=2, markeredgecolor=color)
        ax.plot(ranks, SPEEDUP[key], **kw, label=name)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(ranks)
    ax.set_xticklabels(CONFIG_LABELS_SHORT)
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1×", "2×", "4×", "8×", "16×", "32×"])
    ax.set_xlabel("Cluster size  (total MPI ranks, both axes log₂ scale)")
    ax.set_ylabel("Speedup  (vs 4-rank Flat DP baseline)")
    ax.set_title(f"Strong Scaling Speedup\n{MODEL_NAME}", pad=10)
    ax.legend(loc="upper left", frameon=True)
    ax.fill_between(ranks, SPEEDUP["ideal"], alpha=0.04, color=PALETTE["Ideal"])

    # annotation: best at 128 ranks
    ax.annotate("21.5× at 128 ranks",
                xy=(128, 21.46), xytext=(60, 24),
                fontsize=9, color=PALETTE["Local SGD K=50"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["Local SGD K=50"], lw=1))

    fig.text(0.01, -0.02, "N = nodes,  R = MPI ranks  (e.g. 2N·64R = 2 nodes × 32 ranks each)", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    p = os.path.join(out_dir, "01_speedup.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 2 – Parallel Efficiency
# ══════════════════════════════════════════════════════════════════════════
def plot_efficiency(out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ranks = SPEEDUP["ranks"]

    eff_series = [
        ("Flat DP",        "flat_eff", "o"),
        ("Hier DP",        "hier_eff", "s"),
        ("Local SGD K=10", "k10_eff",  "^"),
        ("Local SGD K=50", "k50_eff",  "D"),
    ]
    for name, key, marker in eff_series:
        color = PALETTE[name]
        ax.plot(ranks, SPEEDUP[key], marker=marker, markersize=7, linewidth=2,
                color=color, label=name, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=color, zorder=3)

    ax.axhline(100, color="#9CA3AF", linestyle="--", linewidth=1.2, label="Perfect efficiency")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(ranks)
    ax.set_xticklabels(CONFIG_LABELS_SHORT)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xlabel("Cluster size  (total MPI ranks, log₂ scale)")
    ax.set_ylabel("Parallel efficiency  (% of ideal speedup achieved)")
    ax.set_title(f"Parallel Efficiency\n{MODEL_NAME}", pad=10)
    ax.legend(loc="upper right", frameon=True)

    fig.text(0.01, -0.02, "N = nodes,  R = MPI ranks  (e.g. 2N·64R = 2 nodes × 32 ranks each)", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    p = os.path.join(out_dir, "02_efficiency.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 3 – Epoch Time Grouped Bar
# ══════════════════════════════════════════════════════════════════════════
def plot_epoch_time(out_dir):
    n = len(CONFIG_LABELS)
    keys   = ["flat",      "hier",    "k10",           "k50"]
    labels = ["Flat DP", "Hierarchical DP", "Local SGD  K=10\n(sync every 10 steps)",
              "Local SGD  K=50\n(sync every 50 steps)"]
    display = ["Flat DP", "Hier DP", "Local SGD K=10", "Local SGD K=50"]
    colors = [PALETTE[l] for l in display]

    x = np.arange(n)
    group_w, bar_w = 0.76, 0.76 / len(keys)
    offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, len(keys))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, label, color, off in zip(keys, labels, colors, offsets):
        vals = EPOCH_MS[key]
        bars = ax.bar(x + off, vals, width=bar_w * 0.88, color=color,
                      label=label, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                    f"{v:,}", ha="center", va="bottom", fontsize=7.5,
                    color=color, fontweight="semibold")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks([200, 500, 1000, 2500, 5000])
    ax.set_yticklabels(["200 ms", "500 ms", "1 s", "2.5 s", "5 s"])
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS)
    ax.set_xlabel("Cluster configuration")
    ax.set_ylabel("Average time per epoch  (log scale)")
    ax.set_title(f"Epoch Time Comparison\n{MODEL_NAME}", pad=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_ylim(100, 8000)

    fig.tight_layout()
    p = os.path.join(out_dir, "03_epoch_time.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 4 – Timing Breakdown Stacked Bars
# ══════════════════════════════════════════════════════════════════════════
def plot_timing_breakdown(out_dir):
    n = len(CONFIG_LABELS)
    x = np.arange(n)
    bar_w, gap = 0.30, 0.06

    fig, ax = plt.subplots(figsize=(10, 5.5))

    flat_stacks = [
        ("Compute",     TIMING["flat_compute"],   "#93C5FD"),
        ("All-Reduce",  TIMING["flat_allreduce"],  "#2563EB"),
        ("Other",       TIMING["flat_other"],      "#1E3A8A"),
    ]
    k10_stacks = [
        ("Compute",     TIMING["k10_compute"],    "#86EFAC"),
        ("Sync",        TIMING["k10_sync"],       "#16A34A"),
        ("Other",       TIMING["k10_other"],      "#14532D"),
    ]

    def draw_stack(stacks, xpos, prefix):
        bottom = np.zeros(n)
        for seg, vals, color in stacks:
            arr = np.array(vals, dtype=float)
            ax.bar(xpos, arr, bottom=bottom, width=bar_w, color=color,
                   label=f"{prefix}  –  {seg}", zorder=3)
            bottom += arr

    draw_stack(flat_stacks, x - bar_w/2 - gap/2, "Flat DP")
    draw_stack(k10_stacks,  x + bar_w/2 + gap/2, "Local SGD K=10")

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS)
    ax.set_xlabel("Cluster configuration")
    ax.set_ylabel("Time per epoch  (ms)")
    ax.set_title(f"Timing Breakdown: Flat DP vs Local SGD K=10\n{MODEL_NAME}", pad=10)

    legend_elems = [
        Patch(color="#93C5FD", label="Flat DP – Compute"),
        Patch(color="#2563EB", label="Flat DP – All-Reduce comm."),
        Patch(color="#1E3A8A", label="Flat DP – Other overhead"),
        Patch(color="#86EFAC", label="Local SGD K=10 – Compute"),
        Patch(color="#16A34A", label="Local SGD K=10 – Sync comm."),
        Patch(color="#14532D", label="Local SGD K=10 – Other overhead"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    p = os.path.join(out_dir, "04_timing_breakdown.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 5 – Trade-off: epoch time vs val accuracy (bubble chart)
# ══════════════════════════════════════════════════════════════════════════
def plot_tradeoff(out_dir):
    # label, epoch_ms, val_acc, speedup_vs_flat
    rows = [
        ("Serial\n(1 process,\nno parallelism)", 9696, 0.9226, 1/16.28, "#374151"),
        ("Flat DP\n(128 ranks,\nsync every step)",  595, 0.9225, 1.0,    "#2563EB"),
        ("Local SGD K=10\n(128 ranks,\nsync every 10 steps)", 264, 0.9191, 595/264, "#16A34A"),
        ("Local SGD K=50\n(128 ranks,\nsync every 50 steps)", 233, 0.9173, 595/233, "#D97706"),
    ]
    labels   = [r[0] for r in rows]
    epoch_ms = np.array([r[1] for r in rows], dtype=float)
    val_acc  = np.array([r[2] for r in rows])
    colors   = [r[4] for r in rows]
    # bubble size encodes epoch_ms (inverted: faster = bigger bubble)
    max_ms = max(epoch_ms)
    sizes  = [(1 - ms/max_ms) * 1200 + 120 for ms in epoch_ms]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(epoch_ms, val_acc, s=sizes, c=colors, zorder=5,
               edgecolors="white", linewidths=2, alpha=0.88)

    # Label each point
    label_offsets = {
        "Serial\n(1 process,\nno parallelism)":         (-1400, -0.0008),
        "Flat DP\n(128 ranks,\nsync every step)":        ( 100, -0.0010),
        "Local SGD K=10\n(128 ranks,\nsync every 10 steps)": ( 80,  0.0005),
        "Local SGD K=50\n(128 ranks,\nsync every 50 steps)": ( 80, -0.0013),
    }
    for lbl, ms, acc, _, color in rows:
        dx, dy = label_offsets[lbl]
        ax.annotate(
            lbl,
            xy=(ms, acc), xytext=(ms + dx, acc + dy),
            fontsize=9, color=color, fontweight="semibold",
            multialignment="center",
            arrowprops=dict(arrowstyle="-", color="#CBD5E1", lw=0.9),
        )

    # Epoch-time annotations inside bubbles
    for ms, acc in zip(epoch_ms, val_acc):
        ax.text(ms, acc, f"{int(ms):,} ms", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # accuracy loss callout
    flat_acc  = rows[1][2]
    k50_acc   = rows[3][2]
    ax.annotate("",
                xy=(233, k50_acc), xytext=(233, flat_acc),
                arrowprops=dict(arrowstyle="<->", color="#9CA3AF", lw=1.2))
    ax.text(190, (flat_acc + k50_acc)/2,
            f"−{flat_acc - k50_acc:.4f}\nacc", ha="right", va="center",
            fontsize=8.5, color="#6B7280")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,} ms" if v >= 1000 else f"{int(v)} ms"))
    ax.set_xticks([233, 264, 595, 9696])
    ax.set_xlabel("Time per epoch  (log scale  —  lower is faster ←)", labelpad=8)
    ax.set_ylabel("Validation accuracy on MNIST test set\n(higher is better ↑)", labelpad=8)
    ax.set_title(
        f"Speed vs Accuracy Trade-off  —  all runs at 4 nodes / 128 ranks\n{MODEL_NAME}",
        pad=10)
    ax.set_ylim(0.9155, 0.9240)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # "sweet spot" annotation
    ax.annotate("Sweet spot:\n2.6× faster than Flat DP,\nonly 0.005 accuracy drop",
                xy=(264, 0.9191), xytext=(400, 0.9205),
                fontsize=9, color="#16A34A",
                bbox=dict(boxstyle="round,pad=0.35", fc="#F0FDF4", ec="#16A34A", lw=0.8),
                arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.2))

    # legend for bubble size
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
               markersize=11, label=l.replace("\n", " "))
        for l, _, _, _, c in rows
    ]
    ax.legend(handles=legend_elems, loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    p = os.path.join(out_dir, "05_tradeoff.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 6 – Speedup vs Serial Baseline
# ══════════════════════════════════════════════════════════════════════════
def plot_speedup_vs_serial(out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ranks = SPEEDUP["ranks"]

    series = [
        ("Flat DP",        "flat_su", "o"),
        ("Hier DP",        "hier_su", "s"),
        ("Local SGD K=10", "k10_su",  "^"),
        ("Local SGD K=50", "k50_su",  "D"),
    ]
    for name, key, marker in series:
        color = PALETTE[name]
        ax.plot(ranks, SPEEDUP_VS_SERIAL[key], marker=marker, markersize=7,
                linewidth=2, color=color, label=name, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=color, zorder=3)

    # ideal line anchored at 4-rank flat (1.94×)
    ideal_at_4 = SPEEDUP_VS_SERIAL["flat_su"][0]
    ideal = [ideal_at_4 * (r / 4) for r in ranks]
    ax.plot(ranks, ideal, "--", color=PALETTE["Ideal"], linewidth=1.5,
            label=f"Ideal linear (from {ideal_at_4:.1f}× at 4 ranks)", zorder=2)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(ranks)
    ax.set_xticklabels(CONFIG_LABELS_SHORT)
    ax.set_xlabel("Cluster size  (total MPI ranks, log₂ scale)")
    ax.set_ylabel("Speedup  (vs single-process serial, 9696 ms/epoch)")
    ax.set_title(f"Speedup vs Single-Process Serial Baseline\n{MODEL_NAME}", pad=10)
    ax.legend(loc="upper left", frameon=True)

    ax.annotate("41.6× faster than\nrunning serial",
                xy=(128, 41.59), xytext=(55, 36),
                fontsize=9, color=PALETTE["Local SGD K=50"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["Local SGD K=50"], lw=1))

    fig.text(0.01, -0.02, "N = nodes,  R = MPI ranks  (e.g. 2N·64R = 2 nodes × 32 ranks each)", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    p = os.path.join(out_dir, "06_speedup_vs_serial.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓  {p}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_theme()
    print(f"Writing plots to {args.out_dir}/")
    plot_architecture(args.out_dir)
    plot_speedup(args.out_dir)
    plot_efficiency(args.out_dir)
    plot_epoch_time(args.out_dir)
    plot_timing_breakdown(args.out_dir)
    plot_tradeoff(args.out_dir)
    plot_speedup_vs_serial(args.out_dir)
    print("Done — 7 figures written.")


if __name__ == "__main__":
    main()