#!/usr/bin/env python3
"""Score the community-1 runs the same way scripts/score_ami_test.sh scores the others.

Per file: scripts/score.py's score_file (no collar and collar 0.25, overlap scored, JER, UEM).
Summary: scripts/summarise.py's bootstrap_ci, so results/summary_community1.csv has the same
columns and the same 2,000-resample intervals as results/summary.csv. The text report also
gives 10,000-resample intervals, pooled DER, speaker-count accuracy and, on AMI, paired
per-meeting bootstraps against System 3 and pyannote 3.1.

Per-file CSVs go to results/raw/<dataset>/scores_community1*.csv, not results/raw/scores_*.csv,
so the other scoring scripts' globs never pick them up.

usage: ./.venv/bin/python scripts/run_community1.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from score import load_annotation, load_uem, score_file
from summarise import METRICS, bootstrap_ci

D = ROOT / "data"
DATASETS = {
    "ami_test": ("ami", ROOT / "ref/ami/only_words/rttms/test", ROOT / "ref/ami/uems/test", ROOT / "ref/ami/test.meetings.txt"),
    "indic_diarbench_ne": ("indic_diarbench_ne", D / "nepali_real/diarbench/rttm", D / "nepali_real/diarbench/uem", D / "nepali_real/diarbench/clips.txt"),
    "indicvoices_ne_calls": ("indicvoices_ne_calls", D / "nepali_real/indicvoices/rttm", D / "nepali_real/indicvoices/uem", D / "nepali_real/indicvoices/calls.txt"),
    "nepali_sim": ("nepali_sim", D / "nepali_sim/rttm", D / "nepali_sim/uem", D / "nepali_sim/conversations.txt"),
    "nepali_sim_rn": ("nepali_sim_rn", D / "nepali_sim/rttm", D / "nepali_sim/uem", D / "nepali_sim/conversations.txt"),
}
SYSTEMS = ["community1", "community1_oracle"]
FIELDS = ["system", "dataset", "uri", "ref_speakers", "hyp_speakers",
          "der_no_collar", "miss_no_collar", "fa_no_collar", "conf_no_collar",
          "der_collar_0.25", "miss_collar_0.25", "fa_collar_0.25", "conf_collar_0.25", "jer"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def paired(a, b, metric, n=10000, seed=0):
    a = {r["uri"]: float(r[metric]) for r in a}
    b = {r["uri"]: float(r[metric]) for r in b}
    uris = sorted(a)
    assert sorted(b) == uris, "paired runs cover different files"
    d = np.array([a[u] - b[u] for u in uris])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi, int((d < 0).sum()), len(d)


def main():
    summary, report = [], []
    for ds, (sub, ref, uem, lst) in DATASETS.items():
        if not lst.exists():
            report.append(f"{ds}: no file list yet, skipped")
            continue
        uris = [u.strip() for u in lst.read_text().splitlines() if u.strip()]
        for s in SYSTEMS:
            hyp = ROOT / "results/raw" / sub / f"{s}_rttm"
            have = sum((hyp / f"{u}.rttm").exists() for u in uris)
            if have != len(uris):
                report.append(f"{ds} {s}: {have}/{len(uris)} rttms, not scored")
                continue
            rows = []
            m0, m25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
            for u in uris:
                r = score_file(ref / f"{u}.rttm", hyp / f"{u}.rttm", u, uem / f"{u}.uem")
                rows.append({"system": s, "dataset": ds, **r})
                rf, hf, uu = load_annotation(ref / f"{u}.rttm", u), load_annotation(hyp / f"{u}.rttm", u), load_uem(uem / f"{u}.uem", u)
                m0(rf, hf, uem=uu)
                m25(rf, hf, uem=uu)
            out = ROOT / "results/raw" / sub / f"scores_{s}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

            row = {"system": s, "dataset": ds, "n_recordings": len(rows)}
            for m in METRICS:
                mean, lo, hi = bootstrap_ci([float(r[m]) for r in rows])
                row[f"{m}_mean"], row[f"{m}_ci95_lo"], row[f"{m}_ci95_hi"] = f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}"
            summary.append(row)

            d = m0[:]
            t = d["total"]
            right = sum(int(r["hyp_speakers"]) == int(r["ref_speakers"]) for r in rows)
            off = np.mean([abs(int(r["hyp_speakers"]) - int(r["ref_speakers"])) for r in rows])
            report.append(f"\n{ds} {s}, {len(rows)} files")
            report.append(f"  pooled: DER0 {abs(m0):.4f} miss {d['missed detection'] / t:.4f} fa {d['false alarm'] / t:.4f} "
                          f"conf {d['confusion'] / t:.4f} DER25 {abs(m25):.4f}")
            for m in ("der_no_collar", "der_collar_0.25", "jer"):
                vals = [float(r[m]) for r in rows]
                _, lo2, hi2 = bootstrap_ci(vals)
                mean, lo10, hi10 = bootstrap_ci(vals, n=10000, seed=0)
                report.append(f"  mean {m}: {mean:.4f}  95% CI 2k [{lo2:.4f}, {hi2:.4f}]  10k [{lo10:.4f}, {hi10:.4f}]")
            report.append(f"  speaker count right on {right}/{len(rows)}, mean |hyp - ref| {off:.2f}")

    ami = ROOT / "results/raw/ami/scores_community1.csv"
    if ami.exists():
        c1 = read_csv(ami)
        report.append("\nAMI paired per meeting, 10,000 resamples, seed 0 (negative = community-1 better)")
        pairs = [("community-1 minus System 3", "scores_pyannote_style_ami.csv"),
                 ("community-1 minus pyannote 3.1", "scores_pyannote31_ami.csv")]
        for name, other in pairs:
            o = read_csv(ROOT / "results/raw" / other)
            for m in ("der_no_collar", "der_collar_0.25"):
                mean, lo, hi, better, n = paired(c1, o, m)
                report.append(f"  {name}, {m}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}], community-1 better on {better}/{n}")
        s3, p31 = read_csv(ROOT / "results/raw/scores_pyannote_style_ami.csv"), read_csv(ROOT / "results/raw/scores_pyannote31_ami.csv")
        for m in ("der_no_collar", "der_collar_0.25"):
            mean, lo, hi, better, n = paired(s3, p31, m)
            report.append(f"  method check, System 3 minus 3.1 (NOTES has -0.010 [-0.021, -0.0001] / -0.009 [-0.020, +0.001]), "
                          f"{m}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}], better on {better}/{n}")

    fieldnames = ["system", "dataset", "n_recordings"] + [f"{m}_{k}" for m in METRICS for k in ("mean", "ci95_lo", "ci95_hi")]
    with open(ROOT / "results/summary_community1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary)
    text = "\n".join(report) + "\n"
    (ROOT / "results/corpus_der_community1.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
