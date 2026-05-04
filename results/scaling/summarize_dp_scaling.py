#!/usr/bin/env python3
"""
Summarize strong-scaling outputs from scripts/run_scaling_study.sh.

Usage:
    python3 results/scaling/summarize_scaling.py results/scaling
"""

import csv
import os
import re
import sys
from collections import defaultdict
from statistics import mean


def avg_epoch_ms(csv_path, skip_first=True):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    vals = [float(r["epoch_time_ms"]) for r in rows]
    if skip_first and len(vals) > 1:
        vals = vals[1:]
    return mean(vals)


def final_val_acc(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["val_acc"])


def parse_timing_log(log_path, mode):
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
        "local": re.compile(
            r"compute_ms=(?P<compute>[0-9.]+).*"
            r"sync_ms=(?P<sync>[0-9.]+).*"
            r"syncs_per_epoch=(?P<syncs>[0-9]+).*"
            r"other_ms=(?P<other>[0-9.-]+)"
        ),
    }
    pat = patterns[mode]
    rec = defaultdict(list)
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            for k, v in m.groupdict().items():
                rec[k].append(float(v))
    if not rec:
        return {}
    out = {}
    for k, vals in rec.items():
        out[k] = mean(vals[1:] if len(vals) > 1 else vals)
    return out


def parse_serial_epoch_log(log_path):
    """Parse [serial] epoch lines and return avg time_ms (epochs 2-end)."""
    pat = re.compile(r"\[serial\].*time_ms=(?P<epoch_ms>[0-9.]+)")
    vals = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                vals.append(float(m.group("epoch_ms")))
    if not vals:
        return None
    vals = vals[1:] if len(vals) > 1 else vals
    return mean(vals)


def cfg_key(cfg):
    m = re.match(r"(\d+)n(\d+)r", cfg)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def col(v, w):
    return str(v)[:w].ljust(w)


def rcol(v, w):
    return str(v)[:w].rjust(w)


def main(scaling_dir):
    if not os.path.isdir(scaling_dir):
        print(f"Directory not found: {scaling_dir}", file=sys.stderr)
        sys.exit(1)

    flat_csvs = {}
    hier_csvs = {}
    lsgd_csvs = {}  # model -> K -> cfg -> path
    serial_csvs = {}  # model -> path

    for fname in sorted(os.listdir(scaling_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(scaling_dir, fname)
        m = re.match(r"(flat|hier)_dp_(\d+n\d+r)_(.+)\.csv", fname)
        if m:
            kind, cfg, model = m.group(1), m.group(2), m.group(3)
            if kind == "flat":
                flat_csvs.setdefault(model, {})[cfg] = path
            else:
                hier_csvs.setdefault(model, {})[cfg] = path
            continue
        m = re.match(r"local_sgd_K(\d+)_(\d+n\d+r)_(.+)\.csv", fname)
        if m:
            k, cfg, model = m.group(1), m.group(2), m.group(3)
            lsgd_csvs.setdefault(model, {}).setdefault(k, {})[cfg] = path
            continue
        m = re.match(r"serial_(.+)\.csv", fname)
        if m:
            model = m.group(1)
            serial_csvs[model] = path

    models = sorted(set(flat_csvs) | set(hier_csvs) | set(lsgd_csvs) | set(serial_csvs))
    if not models:
        print("No scaling CSVs found.", file=sys.stderr)
        sys.exit(1)

    for model in models:
        flat_by_cfg = flat_csvs.get(model, {})
        hier_by_cfg = hier_csvs.get(model, {})
        lsgd_by_k = lsgd_csvs.get(model, {})
        ks = sorted(lsgd_by_k.keys(), key=lambda x: int(x))
        serial_csv = serial_csvs.get(model)
        serial_epoch_ms = avg_epoch_ms(serial_csv) if serial_csv else None

        cfgs = sorted(
            set(flat_by_cfg) | set(hier_by_cfg) | {c for d in lsgd_by_k.values() for c in d},
            key=cfg_key,
        )

        flat_t = {c: avg_epoch_ms(p) for c, p in flat_by_cfg.items() if avg_epoch_ms(p) is not None}
        hier_t = {c: avg_epoch_ms(p) for c, p in hier_by_cfg.items() if avg_epoch_ms(p) is not None}
        lsgd_t = {}
        for k in ks:
            for c, p in lsgd_by_k[k].items():
                v = avg_epoch_ms(p)
                if v is not None:
                    lsgd_t.setdefault(k, {})[c] = v

        cfgs = [c for c in cfgs if c in flat_t or c in hier_t or any(c in lsgd_t.get(k, {}) for k in ks)]
        if not cfgs:
            continue

        print("=" * 72)
        print(f"Model: {model}")
        print("=" * 72)
        if serial_epoch_ms is not None:
            print(f"Serial baseline epoch_ms (epochs 2-end): {serial_epoch_ms:.0f}")

        # [1] epoch-time table
        print("\n[1] Avg epoch time (ms), epochs 2-end")
        hdr = col("Config", 10) + rcol("serial", 10) + rcol("flat", 10) + rcol("hier", 10)
        for k in ks:
            hdr += rcol(f"K={k}", 10)
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for c in cfgs:
            row = col(c, 10)
            row += rcol(f"{serial_epoch_ms:.0f}" if serial_epoch_ms is not None else "—", 10)
            row += rcol(f"{flat_t.get(c):.0f}" if c in flat_t else "—", 10)
            row += rcol(f"{hier_t.get(c):.0f}" if c in hier_t else "—", 10)
            for k in ks:
                row += rcol(f"{lsgd_t.get(k, {}).get(c):.0f}" if c in lsgd_t.get(k, {}) else "—", 10)
            print("  " + row)

        # [2] speedup/efficiency vs base flat config
        base_cfg = cfgs[0]
        base_flat = flat_t.get(base_cfg)
        base_ranks = cfg_key(base_cfg)[1]
        print(f"\n[2] Strong scaling (base={base_cfg}, flat)")
        hdr2 = col("Config", 10) + rcol("ranks", 7)
        hdr2 += rcol("flat_su", 9) + rcol("flat_eff%", 10)
        hdr2 += rcol("hier_su", 9) + rcol("hier_eff%", 10)
        for k in ks:
            hdr2 += rcol(f"K{k}_su", 9) + rcol(f"K{k}_eff%", 10)
        hdr2 += rcol("ideal", 7)
        print("  " + hdr2)
        print("  " + "-" * len(hdr2))
        for c in cfgs:
            ranks = cfg_key(c)[1]
            ideal = ranks / base_ranks
            ft = flat_t.get(c)
            ht = hier_t.get(c)
            row = col(c, 10) + rcol(str(ranks), 7)
            row += rcol(f"{base_flat/ft:.2f}x" if (base_flat and ft) else "—", 9)
            row += rcol(f"{100.0*base_flat/ft/ideal:.1f}%" if (base_flat and ft) else "—", 10)
            row += rcol(f"{base_flat/ht:.2f}x" if (base_flat and ht) else "—", 9)
            row += rcol(f"{100.0*base_flat/ht/ideal:.1f}%" if (base_flat and ht) else "—", 10)
            for k in ks:
                lt = lsgd_t.get(k, {}).get(c)
                row += rcol(f"{base_flat/lt:.2f}x" if (base_flat and lt) else "—", 9)
                row += rcol(f"{100.0*base_flat/lt/ideal:.1f}%" if (base_flat and lt) else "—", 10)
            row += rcol(f"{ideal:.1f}x", 7)
            print("  " + row)

        # [2b] speedup/efficiency vs serial baseline
        if serial_epoch_ms is not None:
            print("\n[2b] Speedup vs serial baseline")
            hdr2b = col("Config", 10) + rcol("ranks", 7)
            hdr2b += rcol("flat_su", 9) + rcol("flat_eff%", 10)
            hdr2b += rcol("hier_su", 9) + rcol("hier_eff%", 10)
            for k in ks:
                hdr2b += rcol(f"K{k}_su", 9) + rcol(f"K{k}_eff%", 10)
            print("  " + hdr2b)
            print("  " + "-" * len(hdr2b))
            for c in cfgs:
                ranks = cfg_key(c)[1]
                ft = flat_t.get(c)
                ht = hier_t.get(c)
                row = col(c, 10) + rcol(str(ranks), 7)
                row += rcol(f"{serial_epoch_ms/ft:.2f}x" if ft else "—", 9)
                row += rcol(f"{100.0*serial_epoch_ms/ft/ranks:.1f}%" if ft else "—", 10)
                row += rcol(f"{serial_epoch_ms/ht:.2f}x" if ht else "—", 9)
                row += rcol(f"{100.0*serial_epoch_ms/ht/ranks:.1f}%" if ht else "—", 10)
                for k in ks:
                    lt = lsgd_t.get(k, {}).get(c)
                    row += rcol(f"{serial_epoch_ms/lt:.2f}x" if lt else "—", 9)
                    row += rcol(f"{100.0*serial_epoch_ms/lt/ranks:.1f}%" if lt else "—", 10)
                print("  " + row)

        # [3] timing breakdowns
        print("\n[3] Timing breakdown (avg ms, epochs 2-end)")
        serial_log = os.path.join(scaling_dir, f"serial_{model}.log")
        serial_log_ms = parse_serial_epoch_log(serial_log) if os.path.exists(serial_log) else None
        print("\n  Serial")
        print("  " + col("metric", 16) + rcol("value", 12))
        print("  " + col("epoch_ms(csv)", 16) + rcol(f"{serial_epoch_ms:.0f}" if serial_epoch_ms is not None else "—", 12))
        print("  " + col("epoch_ms(log)", 16) + rcol(f"{serial_log_ms:.0f}" if serial_log_ms is not None else "—", 12))

        print("\n  Flat DP")
        print("  " + col("Config", 10) + rcol("compute", 12) + rcol("allreduce", 12) + rcol("other", 10))
        for c in cfgs:
            p = os.path.join(scaling_dir, f"flat_dp_{c}_{model}.log")
            if not os.path.exists(p):
                continue
            t = parse_timing_log(p, "flat")
            if not t:
                continue
            print("  " + col(c, 10) + rcol(f"{t.get('compute', 0):.0f}", 12) + rcol(f"{t.get('allreduce', 0):.0f}", 12) + rcol(f"{t.get('other', 0):.0f}", 10))

        print("\n  Hierarchical DP")
        print("  " + col("Config", 10) + rcol("compute", 12) + rcol("intra_red", 11) + rcol("inter_ar", 10) + rcol("intra_bc", 10) + rcol("other", 10))
        for c in cfgs:
            p = os.path.join(scaling_dir, f"hier_dp_{c}_{model}.log")
            if not os.path.exists(p):
                continue
            t = parse_timing_log(p, "hier")
            if not t:
                continue
            print("  " + col(c, 10) + rcol(f"{t.get('compute', 0):.0f}", 12) + rcol(f"{t.get('intra_reduce', 0):.0f}", 11) + rcol(f"{t.get('inter_allreduce', 0):.0f}", 10) + rcol(f"{t.get('intra_bcast', 0):.0f}", 10) + rcol(f"{t.get('other', 0):.0f}", 10))

        for k in ks:
            print(f"\n  Local SGD K={k}")
            print("  " + col("Config", 10) + rcol("compute", 12) + rcol("sync_ms", 12) + rcol("syncs", 8) + rcol("other", 10))
            for c in cfgs:
                p = os.path.join(scaling_dir, f"local_sgd_K{k}_{c}_{model}.log")
                if not os.path.exists(p):
                    continue
                t = parse_timing_log(p, "local")
                if not t:
                    continue
                print("  " + col(c, 10) + rcol(f"{t.get('compute', 0):.0f}", 12) + rcol(f"{t.get('sync', 0):.0f}", 12) + rcol(f"{t.get('syncs', 0):.0f}", 8) + rcol(f"{t.get('other', 0):.0f}", 10))

        # [4] amdahl on largest flat config
        flat_cfgs = [c for c in cfgs if c in flat_t]
        if flat_cfgs:
            largest = max(flat_cfgs, key=cfg_key)
            p = os.path.join(scaling_dir, f"flat_dp_{largest}_{model}.log")
            if os.path.exists(p):
                t = parse_timing_log(p, "flat")
                other = t.get("other", 0.0)
                epoch = flat_t[largest]
                f = (other / epoch) if epoch > 0 else 0.0
                cap = (1.0 / f) if f > 0 else float("inf")
                print("\n[4] Amdahl estimate (flat)")
                print(f"  At {largest}: other={other:.0f}ms / epoch={epoch:.0f}ms")
                print(f"  Serial fraction f ~ {f:.3f} ({100.0*f:.1f}%)")
                print(f"  Max speedup ~ {cap:.1f}x")

        # [5] local sgd trade-off at largest flat config
        if ks and flat_cfgs:
            ref = max(flat_cfgs, key=cfg_key)
            ft = flat_t.get(ref)
            facc = final_val_acc(flat_by_cfg[ref]) if ref in flat_by_cfg else None
            print(f"\n[5] Local SGD trade-off at {ref}")
            print("  " + col("impl", 18) + rcol("epoch_ms", 12) + rcol("val_acc", 10) + rcol("time_vs_flat", 14))
            if serial_epoch_ms is not None and serial_csv:
                sacc = final_val_acc(serial_csv)
                print("  " + col("serial", 18) + rcol(f"{serial_epoch_ms:.0f}", 12) + rcol(f"{sacc:.4f}" if sacc is not None else "—", 10) + rcol(f"{serial_epoch_ms/ft:.2f}x" if ft else "—", 14))
            if ft:
                print("  " + col("flat-dp", 18) + rcol(f"{ft:.0f}", 12) + rcol(f"{facc:.4f}" if facc is not None else "—", 10) + rcol("1.00x", 14))
            for k in ks:
                lt = lsgd_t.get(k, {}).get(ref)
                path = lsgd_by_k.get(k, {}).get(ref)
                acc = final_val_acc(path) if path else None
                ratio = f"{lt/ft:.2f}x" if (lt and ft) else "—"
                print("  " + col(f"local-sgd K={k}", 18) + rcol(f"{lt:.0f}" if lt else "—", 12) + rcol(f"{acc:.4f}" if acc is not None else "—", 10) + rcol(ratio, 14))

        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <scaling_results_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
