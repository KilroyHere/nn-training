#!/usr/bin/env python3
"""
Summarize strong-scaling results from run_scaling_study.sh outputs.

Usage:
    python3 result_scripts/summarize_scaling.py results/scaling/

Reads every flat_dp_*.csv and hier_dp_*.csv in the directory,
parses the matching *.log files for timing breakdown, and prints:
  - Epoch-time table
  - Speedup and efficiency table (relative to single-node, fewest ranks)
  - Per-phase timing breakdown table
  - Amdahl's Law analysis (serial fraction estimate)
"""

import csv
import os
import re
import sys
from collections import defaultdict
from statistics import mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def avg_epoch_ms(csv_path: str, skip_first: bool = True):
    """Return mean epoch_time_ms, or None if the file has no data rows."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    times = [float(r["epoch_time_ms"]) for r in rows]
    if skip_first and len(times) > 1:
        times = times[1:]
    return mean(times)


def parse_timing_log(log_path: str, mode: str) -> dict:
    """
    Parse timing lines from a .log file.
    Returns dict with mean ms values for each timing field.
    mode: 'flat' or 'hier'
    """
    patterns = {
        "flat": re.compile(
            r"compute_ms=(?P<compute>[0-9.]+).*"
            r"allreduce_ms=(?P<allreduce>[0-9.]+).*"
            r"other_ms=(?P<other>[0-9.-]+)"
        ),
        "hier": re.compile(
            r"compute_ms=(?P<compute>[0-9.]+).*"
            r"intra_reduce_ms=(?P<intra_reduce>[0-9.]+).*"
            r"inter_allreduce_ms=(?P<inter_allreduce>[0-9.]+).*"
            r"intra_bcast_ms=(?P<intra_bcast>[0-9.]+).*"
            r"other_ms=(?P<other>[0-9.-]+)"
        ),
    }
    pat = patterns[mode]
    records = defaultdict(list)
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                for k, v in m.groupdict().items():
                    records[k].append(float(v))

    if not records:
        return {}
    # Skip first epoch (warmup) if more than one epoch recorded
    result = {}
    for k, vals in records.items():
        result[k] = mean(vals[1:] if len(vals) > 1 else vals)
    return result


def config_sort_key(cfg: str) -> tuple:
    """Sort '1n4r' < '1n16r' < '1n32r' < '2n64r' correctly."""
    m = re.match(r"(\d+)n(\d+)r", cfg)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def col(s, w):
    return str(s)[:w].ljust(w)


def rcol(s, w):
    return str(s)[:w].rjust(w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(scaling_dir: str) -> None:
    if not os.path.isdir(scaling_dir):
        print(f"Directory not found: {scaling_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover all CSV files.
    flat_csvs = {}
    hier_csvs = {}
    for fname in os.listdir(scaling_dir):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(scaling_dir, fname)
        m = re.match(r"(flat|hier)_dp_(\d+n\d+r)_(.+)\.csv", fname)
        if not m:
            continue
        kind, config, model = m.group(1), m.group(2), m.group(3)
        if kind == "flat":
            flat_csvs.setdefault(model, {})[config] = path
        else:
            hier_csvs.setdefault(model, {})[config] = path

    all_models = sorted(set(list(flat_csvs.keys()) + list(hier_csvs.keys())))
    if not all_models:
        print("No scaling CSVs found.", file=sys.stderr)
        sys.exit(1)

    for model in all_models:
        flat_by_cfg = flat_csvs.get(model, {})
        hier_by_cfg = hier_csvs.get(model, {})
        all_cfgs = sorted(
            set(list(flat_by_cfg.keys()) + list(hier_by_cfg.keys())),
            key=config_sort_key,
        )

        print("=" * 72)
        print(f"  Model: {model}")
        print("=" * 72)

        # ---- Epoch time table ----
        flat_times = {}
        hier_times = {}
        for cfg in all_cfgs:
            if cfg in flat_by_cfg:
                val = avg_epoch_ms(flat_by_cfg[cfg])
                if val is not None:
                    flat_times[cfg] = val
            if cfg in hier_by_cfg:
                val = avg_epoch_ms(hier_by_cfg[cfg])
                if val is not None:
                    hier_times[cfg] = val

        # Drop configs where neither implementation produced data.
        all_cfgs = [c for c in all_cfgs if c in flat_times or c in hier_times]
        if not all_cfgs:
            print("  (no data)\n")
            continue

        print("\n[1] Avg epoch time (ms), epochs 2-end:")
        hdr = col("Config", 10) + rcol("flat_dp", 12) + rcol("hier_dp", 12) + rcol("hier_overhead", 16)
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for cfg in all_cfgs:
            ft = flat_times.get(cfg)
            ht = hier_times.get(cfg)
            ft_s = f"{ft:.0f}" if ft else "—"
            ht_s = f"{ht:.0f}" if ht else "—"
            if ft and ht:
                diff = ht - ft
                pct = 100.0 * diff / ft
                oh_s = f"{diff:+.0f}ms ({pct:+.1f}%)"
            else:
                oh_s = "—"
            print("  " + col(cfg, 10) + rcol(ft_s, 12) + rcol(ht_s, 12) + rcol(oh_s, 16))

        # ---- Speedup table (relative to slowest config = fewest ranks) ----
        base_cfg = all_cfgs[0]
        base_flat = flat_times.get(base_cfg)
        base_hier = hier_times.get(base_cfg)

        print(f"\n[2] Strong-scaling speedup (base = {base_cfg}):")
        hdr2 = col("Config", 10) + rcol("ranks", 8) + rcol("flat_speedup", 14) + rcol("flat_eff%", 11) + rcol("hier_speedup", 14) + rcol("hier_eff%", 11) + rcol("ideal", 8)
        print("  " + hdr2)
        print("  " + "-" * len(hdr2))
        base_ranks = config_sort_key(base_cfg)[1]
        for cfg in all_cfgs:
            nodes, ranks = config_sort_key(cfg)
            ideal_speedup = ranks / base_ranks
            ft = flat_times.get(cfg)
            ht = hier_times.get(cfg)
            flat_su = f"{base_flat/ft:.2f}x" if (base_flat and ft) else "—"
            hier_su = f"{base_hier/ht:.2f}x" if (base_hier and ht) else "—"
            flat_eff = f"{100.0*base_flat/ft/ideal_speedup:.1f}%" if (base_flat and ft) else "—"
            hier_eff = f"{100.0*base_hier/ht/ideal_speedup:.1f}%" if (base_hier and ht) else "—"
            print("  " + col(cfg, 10) + rcol(str(ranks), 8) + rcol(flat_su, 14) + rcol(flat_eff, 11) + rcol(hier_su, 14) + rcol(hier_eff, 11) + rcol(f"{ideal_speedup:.1f}x", 8))

        # ---- Timing breakdown from logs ----
        print("\n[3] Timing breakdown (avg ms, epochs 2-end):")

        # Flat
        print("\n  Flat DP:  (epoch_time_ms = training only, val eval excluded)")
        flat_hdr = col("Config", 10) + rcol("compute", 12) + rcol("allreduce", 12) + rcol("other", 12) + rcol("comm%", 8) + rcol("other%", 8)
        print("  " + flat_hdr)
        print("  " + "-" * len(flat_hdr))
        for cfg in all_cfgs:
            log_path = os.path.join(scaling_dir, f"flat_dp_{cfg}_{model}.log")
            if not os.path.exists(log_path):
                continue
            t = parse_timing_log(log_path, "flat")
            if not t:
                continue
            epoch_ms = flat_times.get(cfg, 1)
            comm_pct = f"{100.0*t.get('allreduce',0)/epoch_ms:.1f}%"
            other_pct = f"{100.0*t.get('other',0)/epoch_ms:.1f}%"
            print("  " + col(cfg, 10)
                  + rcol(f"{t.get('compute',0):.0f}", 12)
                  + rcol(f"{t.get('allreduce',0):.0f}", 12)
                  + rcol(f"{t.get('other',0):.0f}", 12)
                  + rcol(comm_pct, 8)
                  + rcol(other_pct, 8))

        # Hier
        print("\n  Hierarchical DP:  (epoch_time_ms = training only, val eval excluded)")
        hier_hdr = col("Config", 10) + rcol("compute", 12) + rcol("intra_red", 12) + rcol("inter_ar", 10) + rcol("intra_bc", 12) + rcol("other", 12) + rcol("comm%", 8)
        print("  " + hier_hdr)
        print("  " + "-" * len(hier_hdr))
        for cfg in all_cfgs:
            log_path = os.path.join(scaling_dir, f"hier_dp_{cfg}_{model}.log")
            if not os.path.exists(log_path):
                continue
            t = parse_timing_log(log_path, "hier")
            if not t:
                continue
            epoch_ms = hier_times.get(cfg, 1)
            total_comm = t.get("intra_reduce", 0) + t.get("inter_allreduce", 0) + t.get("intra_bcast", 0)
            comm_pct = f"{100.0*total_comm/epoch_ms:.1f}%"
            print("  " + col(cfg, 10)
                  + rcol(f"{t.get('compute',0):.0f}", 12)
                  + rcol(f"{t.get('intra_reduce',0):.0f}", 12)
                  + rcol(f"{t.get('inter_allreduce',0):.0f}", 10)
                  + rcol(f"{t.get('intra_bcast',0):.0f}", 12)
                  + rcol(f"{t.get('other',0):.0f}", 12)
                  + rcol(comm_pct, 8))

        # ---- Amdahl analysis ----
        print("\n[4] Amdahl's Law analysis (flat DP):")
        print("  epoch_time_ms is now pure training time (val eval excluded).")
        print("  Serial fraction f ≈ other_ms / epoch_ms  (other = shuffle + data prep + barriers)")

        largest_cfg = max((c for c in all_cfgs if c in flat_times), key=config_sort_key)
        log_path = os.path.join(scaling_dir, f"flat_dp_{largest_cfg}_{model}.log")
        if os.path.exists(log_path):
            t = parse_timing_log(log_path, "flat")
            epoch_ms = flat_times[largest_cfg]
            other = t.get("other", 0) or 0
            serial_frac = other / epoch_ms if epoch_ms > 0 else 0
            max_speedup = 1.0 / serial_frac if serial_frac > 0 else float("inf")
            print(f"  At {largest_cfg}: other={other:.0f}ms / train_epoch={epoch_ms:.0f}ms")
            print(f"  Estimated serial fraction f ≈ {serial_frac:.3f} ({100*serial_frac:.1f}%)")
            print(f"  Amdahl max speedup = 1/f ≈ {max_speedup:.1f}x  (regardless of rank count)")

        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <scaling_results_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
