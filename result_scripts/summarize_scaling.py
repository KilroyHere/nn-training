#!/usr/bin/env python3
"""
Summarize strong-scaling results from run_scaling_study.sh outputs.

Usage:
    python3 result_scripts/summarize_scaling.py results/scaling/

Reads flat_dp_*.csv, hier_dp_*.csv, and local_sgd_K*_*.csv in the directory,
parses the matching *.log files for timing breakdown, and prints:
  [1] Epoch-time table (flat, hier, local-SGD per K)
  [2] Speedup and efficiency (flat and local-SGD vs serial baseline)
  [3] Per-phase timing breakdown
  [4] Amdahl's Law analysis
  [5] Local SGD: time vs accuracy trade-off table
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

def avg_epoch_ms(csv_path, skip_first=True):
    """Return mean epoch_time_ms, or None if the file has no data rows."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    times = [float(r["epoch_time_ms"]) for r in rows]
    if skip_first and len(times) > 1:
        times = times[1:]
    return mean(times)


def final_val_acc(csv_path):
    """Return val_acc of the last epoch, or None."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["val_acc"])


def parse_timing_log(log_path, mode):
    """
    Parse timing lines from a .log file.
    mode: 'flat' | 'hier' | 'local_sgd'
    Returns dict with mean ms values for each timing field.
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
        "local_sgd": re.compile(
            r"compute_ms=(?P<compute>[0-9.]+).*"
            r"sync_ms=(?P<sync>[0-9.]+).*"
            r"syncs_per_epoch=(?P<syncs>[0-9]+).*"
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
    result = {}
    for k, vals in records.items():
        result[k] = mean(vals[1:] if len(vals) > 1 else vals)
    return result


def config_sort_key(cfg):
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

def main(scaling_dir):
    if not os.path.isdir(scaling_dir):
        print(f"Directory not found: {scaling_dir}", file=sys.stderr)
        sys.exit(1)

    flat_csvs  = {}   # model -> config -> path
    hier_csvs  = {}
    lsgd_csvs  = {}   # model -> K -> config -> path

    for fname in sorted(os.listdir(scaling_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(scaling_dir, fname)

        m = re.match(r"(flat|hier)_dp_(\d+n\d+r)_(.+)\.csv", fname)
        if m:
            kind, config, model = m.group(1), m.group(2), m.group(3)
            if kind == "flat":
                flat_csvs.setdefault(model, {})[config] = path
            else:
                hier_csvs.setdefault(model, {})[config] = path
            continue

        m = re.match(r"local_sgd_K(\d+)_(\d+n\d+r)_(.+)\.csv", fname)
        if m:
            K, config, model = m.group(1), m.group(2), m.group(3)
            lsgd_csvs.setdefault(model, {}).setdefault(K, {})[config] = path

    all_models = sorted(set(
        list(flat_csvs.keys()) + list(hier_csvs.keys()) + list(lsgd_csvs.keys())
    ))
    if not all_models:
        print("No scaling CSVs found.", file=sys.stderr)
        sys.exit(1)

    for model in all_models:
        flat_by_cfg = flat_csvs.get(model, {})
        hier_by_cfg = hier_csvs.get(model, {})
        lsgd_by_K   = lsgd_csvs.get(model, {})   # K -> config -> path
        all_K = sorted(lsgd_by_K.keys(), key=lambda k: int(k))

        all_cfgs = sorted(
            set(list(flat_by_cfg.keys())
                + list(hier_by_cfg.keys())
                + [c for kd in lsgd_by_K.values() for c in kd]),
            key=config_sort_key,
        )

        print("=" * 72)
        print(f"  Model: {model}")
        print("=" * 72)

        # --- Collect epoch times ---
        flat_times = {}
        hier_times = {}
        lsgd_times = {}   # K -> config -> ms

        for cfg in all_cfgs:
            if cfg in flat_by_cfg:
                v = avg_epoch_ms(flat_by_cfg[cfg])
                if v is not None:
                    flat_times[cfg] = v
            if cfg in hier_by_cfg:
                v = avg_epoch_ms(hier_by_cfg[cfg])
                if v is not None:
                    hier_times[cfg] = v

        for K in all_K:
            kd = lsgd_by_K[K]
            for cfg, path in kd.items():
                v = avg_epoch_ms(path)
                if v is not None:
                    lsgd_times.setdefault(K, {})[cfg] = v

        # Drop configs with no data at all
        all_cfgs = [
            c for c in all_cfgs
            if c in flat_times or c in hier_times
               or any(c in lsgd_times.get(K, {}) for K in all_K)
        ]
        if not all_cfgs:
            print("  (no data)\n")
            continue

        # ---- [1] Epoch time table ----
        lsgd_cols = [f"K={K}" for K in all_K]
        hdr_parts = [col("Config", 10), rcol("flat_dp", 10), rcol("hier_dp", 10)]
        for lbl in lsgd_cols:
            hdr_parts.append(rcol(lbl, 10))
        hdr = "".join(hdr_parts)

        print("\n[1] Avg epoch time (ms), epochs 2-end:")
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for cfg in all_cfgs:
            ft = flat_times.get(cfg)
            ht = hier_times.get(cfg)
            parts = [
                col(cfg, 10),
                rcol(f"{ft:.0f}" if ft else "—", 10),
                rcol(f"{ht:.0f}" if ht else "—", 10),
            ]
            for K in all_K:
                lt = lsgd_times.get(K, {}).get(cfg)
                parts.append(rcol(f"{lt:.0f}" if lt else "—", 10))
            print("  " + "".join(parts))

        # ---- [2] Speedup table (flat DP and local SGD vs flat DP baseline) ----
        base_cfg = all_cfgs[0]
        base_flat = flat_times.get(base_cfg)
        base_ranks = config_sort_key(base_cfg)[1]

        print(f"\n[2] Strong-scaling speedup (base = {base_cfg} flat DP):")
        hdr2_parts = [col("Config", 10), rcol("ranks", 7),
                      rcol("flat_su", 9), rcol("flat_eff%", 10),
                      rcol("hier_su", 9), rcol("hier_eff%", 10)]
        for K in all_K:
            hdr2_parts += [rcol(f"K{K}_su", 9), rcol(f"K{K}_eff%", 10)]
        hdr2_parts.append(rcol("ideal", 7))
        hdr2 = "".join(hdr2_parts)
        print("  " + hdr2)
        print("  " + "-" * len(hdr2))
        for cfg in all_cfgs:
            _, ranks = config_sort_key(cfg)
            ideal = ranks / base_ranks
            ft = flat_times.get(cfg)
            ht = hier_times.get(cfg)
            flat_su  = f"{base_flat/ft:.2f}x"  if (base_flat and ft) else "—"
            flat_eff = f"{100*base_flat/ft/ideal:.1f}%" if (base_flat and ft) else "—"
            hier_su  = f"{base_flat/ht:.2f}x"  if (base_flat and ht) else "—"
            hier_eff = f"{100*base_flat/ht/ideal:.1f}%" if (base_flat and ht) else "—"
            parts = [col(cfg, 10), rcol(str(ranks), 7),
                     rcol(flat_su, 9), rcol(flat_eff, 10),
                     rcol(hier_su, 9), rcol(hier_eff, 10)]
            for K in all_K:
                lt = lsgd_times.get(K, {}).get(cfg)
                lsu  = f"{base_flat/lt:.2f}x"  if (base_flat and lt) else "—"
                leff = f"{100*base_flat/lt/ideal:.1f}%" if (base_flat and lt) else "—"
                parts += [rcol(lsu, 9), rcol(leff, 10)]
            parts.append(rcol(f"{ideal:.1f}x", 7))
            print("  " + "".join(parts))

        # ---- [3] Timing breakdown from logs ----
        print("\n[3] Timing breakdown (avg ms, epochs 2-end):")

        # Flat
        print("\n  Flat DP:  (epoch_time_ms = training only)")
        fhdr = col("Config", 10) + rcol("compute", 12) + rcol("allreduce", 12) + rcol("other", 10) + rcol("comm%", 8) + rcol("other%", 8)
        print("  " + fhdr)
        print("  " + "-" * len(fhdr))
        for cfg in all_cfgs:
            log_path = os.path.join(scaling_dir, f"flat_dp_{cfg}_{model}.log")
            if not os.path.exists(log_path):
                continue
            t = parse_timing_log(log_path, "flat")
            if not t:
                continue
            ep = flat_times.get(cfg, 1)
            comm_pct  = f"{100*t.get('allreduce',0)/ep:.1f}%"
            other_pct = f"{100*t.get('other',0)/ep:.1f}%"
            print("  " + col(cfg, 10)
                  + rcol(f"{t.get('compute',0):.0f}", 12)
                  + rcol(f"{t.get('allreduce',0):.0f}", 12)
                  + rcol(f"{t.get('other',0):.0f}", 10)
                  + rcol(comm_pct, 8) + rcol(other_pct, 8))

        # Hier
        print("\n  Hierarchical DP:  (epoch_time_ms = training only)")
        hhdr = col("Config", 10) + rcol("compute", 12) + rcol("intra_red", 11) + rcol("inter_ar", 10) + rcol("intra_bc", 10) + rcol("other", 10) + rcol("comm%", 8)
        print("  " + hhdr)
        print("  " + "-" * len(hhdr))
        for cfg in all_cfgs:
            log_path = os.path.join(scaling_dir, f"hier_dp_{cfg}_{model}.log")
            if not os.path.exists(log_path):
                continue
            t = parse_timing_log(log_path, "hier")
            if not t:
                continue
            ep = hier_times.get(cfg, 1)
            total_comm = t.get("intra_reduce",0) + t.get("inter_allreduce",0) + t.get("intra_bcast",0)
            comm_pct = f"{100*total_comm/ep:.1f}%"
            print("  " + col(cfg, 10)
                  + rcol(f"{t.get('compute',0):.0f}", 12)
                  + rcol(f"{t.get('intra_reduce',0):.0f}", 11)
                  + rcol(f"{t.get('inter_allreduce',0):.0f}", 10)
                  + rcol(f"{t.get('intra_bcast',0):.0f}", 10)
                  + rcol(f"{t.get('other',0):.0f}", 10)
                  + rcol(comm_pct, 8))

        # Local SGD
        for K in all_K:
            print(f"\n  Local SGD K={K}:  (epoch_time_ms = training only)")
            lhdr = col("Config", 10) + rcol("compute", 12) + rcol("sync_ms", 12) + rcol("syncs", 8) + rcol("other", 10) + rcol("sync%", 8)
            print("  " + lhdr)
            print("  " + "-" * len(lhdr))
            for cfg in all_cfgs:
                log_path = os.path.join(scaling_dir, f"local_sgd_K{K}_{cfg}_{model}.log")
                if not os.path.exists(log_path):
                    continue
                t = parse_timing_log(log_path, "local_sgd")
                if not t:
                    continue
                ep = lsgd_times.get(K, {}).get(cfg, 1)
                sync_pct = f"{100*t.get('sync',0)/ep:.1f}%"
                print("  " + col(cfg, 10)
                      + rcol(f"{t.get('compute',0):.0f}", 12)
                      + rcol(f"{t.get('sync',0):.0f}", 12)
                      + rcol(f"{t.get('syncs',0):.0f}", 8)
                      + rcol(f"{t.get('other',0):.0f}", 10)
                      + rcol(sync_pct, 8))

        # ---- [4] Amdahl analysis ----
        print("\n[4] Amdahl's Law analysis:")
        print("  Serial fraction f ≈ other_ms / epoch_ms  (shuffle + data prep + barriers)")
        largest_cfg = max((c for c in all_cfgs if c in flat_times), key=config_sort_key)
        log_path = os.path.join(scaling_dir, f"flat_dp_{largest_cfg}_{model}.log")
        if os.path.exists(log_path):
            t = parse_timing_log(log_path, "flat")
            ep = flat_times[largest_cfg]
            other = t.get("other", 0) or 0
            f = other / ep if ep > 0 else 0
            max_su = 1.0 / f if f > 0 else float("inf")
            print(f"  Flat DP at {largest_cfg}: other={other:.0f}ms / epoch={ep:.0f}ms")
            print(f"  Serial fraction f ≈ {f:.3f} ({100*f:.1f}%)")
            print(f"  Amdahl max speedup = 1/f ≈ {max_su:.1f}x")

        # ---- [5] Local SGD: time vs accuracy table ----
        if all_K:
            print("\n[5] Local SGD: epoch time vs final val_acc (largest rank config):")
            lhdr5 = col("impl", 18) + rcol("epoch_ms", 12) + rcol("val_acc", 10) + rcol("time_vs_flat%", 15)
            print("  " + lhdr5)
            print("  " + "-" * len(lhdr5))

            ref_cfg = largest_cfg  # compare at the same rank count
            ft_ref = flat_times.get(ref_cfg)
            ft_acc = final_val_acc(flat_by_cfg[ref_cfg]) if ref_cfg in flat_by_cfg else None
            if ft_ref:
                acc_s = f"{ft_acc:.4f}" if ft_acc else "—"
                print("  " + col("flat-dp", 18)
                      + rcol(f"{ft_ref:.0f}", 12)
                      + rcol(acc_s, 10)
                      + rcol("1.00x (ref)", 15))
            for K in all_K:
                lt = lsgd_times.get(K, {}).get(ref_cfg)
                lpath = lsgd_by_K.get(K, {}).get(ref_cfg)
                lt_acc = final_val_acc(lpath) if lpath else None
                time_vs = f"{lt/ft_ref:.2f}x" if (lt and ft_ref) else "—"
                acc_s = f"{lt_acc:.4f}" if lt_acc else "—"
                print("  " + col(f"local-sgd K={K}", 18)
                      + rcol(f"{lt:.0f}" if lt else "—", 12)
                      + rcol(acc_s, 10)
                      + rcol(time_vs, 15))

        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <scaling_results_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
