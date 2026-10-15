#!/usr/bin/env python3
"""Turn per-file scoring CSVs (from score.py) into one summary table with
95% bootstrap confidence intervals over recordings.

Bootstrap: for each (system, dataset) group, resample recordings with
replacement 2000 times and take the mean each time; the CI is the 2.5th and
97.5th percentile of those means. This is a percentile bootstrap over files,
which is the right unit here since DER is reported per recording and errors
are correlated within a recording, not across recordings.

Usage: scripts/summarise.py results/raw/**/*.csv --out results/summary.csv
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = [
    "der_no_collar", "miss_no_collar", "fa_no_collar", "conf_no_collar",
    "der_collar_0.25", "miss_collar_0.25", "fa_collar_0.25", "conf_collar_0.25",
    "jer",
]
N_BOOTSTRAP = 2000
SEED = 0


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return values.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="per-file scoring CSVs (glob patterns ok)")
    ap.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = ap.parse_args()

    paths = []
    for pattern in args.csvs:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    if not paths:
        print("no input CSVs matched")
        return

    groups = defaultdict(lambda: defaultdict(list))
    n_files = defaultdict(int)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["system"], row["dataset"])
                n_files[key] += 1
                for metric in METRICS:
                    groups[key][metric].append(float(row[metric]))

    rows = []
    for key in sorted(groups):
        system, dataset = key
        row = {"system": system, "dataset": dataset, "n_recordings": n_files[key]}
        for metric in METRICS:
            mean, lo, hi = bootstrap_ci(groups[key][metric])
            row[f"{metric}_mean"] = f"{mean:.4f}"
            row[f"{metric}_ci95_lo"] = f"{lo:.4f}"
            row[f"{metric}_ci95_hi"] = f"{hi:.4f}"
        rows.append(row)

    fieldnames = ["system", "dataset", "n_recordings"]
    for metric in METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_ci95_lo", f"{metric}_ci95_hi"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['system']} / {row['dataset']} (n={row['n_recordings']}): "
            f"DER(no collar)={row['der_no_collar_mean']} "
            f"[{row['der_no_collar_ci95_lo']}, {row['der_no_collar_ci95_hi']}] "
            f"DER(0.25s collar)={row['der_collar_0.25_mean']} "
            f"[{row['der_collar_0.25_ci95_lo']}, {row['der_collar_0.25_ci95_hi']}] "
            f"JER={row['jer_mean']} [{row['jer_ci95_lo']}, {row['jer_ci95_hi']}]"
        )
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
