"""Paired bootstrap of per-meeting DER, S3 (pyannote_style) minus S4 (pyannote31), AMI test.

Uses the repo's own paired() from scripts/run_community1.py (meetings sorted, 10,000 resamples, seed 0).
Writes paper/analysis/ami_paired_s3_s4.csv.
"""
import csv
import importlib.util
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts"))
spec = importlib.util.spec_from_file_location("rc1", R / "scripts/run_community1.py")
rc1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc1)

a = rc1.read_csv(R / "results/raw/scores_pyannote_style_ami.csv")
b = rc1.read_csv(R / "results/raw/scores_pyannote31_ami.csv")
out = Path(__file__).with_suffix(".csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "mean_diff_pts", "ci_lo_pts", "ci_hi_pts", "s3_better", "n", "resamples", "seed"])
    for m in ("der_no_collar", "der_collar_0.25"):
        d, lo, hi, k, n = rc1.paired(a, b, m)
        w.writerow([m, f"{100 * d:.4f}", f"{100 * lo:.5f}", f"{100 * hi:.5f}", k, n, 10000, 0])
print(out.read_text())
