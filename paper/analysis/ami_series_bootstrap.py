# S3 minus S4, S5 minus S4 and S5 minus S3 per-meeting DER on AMI test, resampled by meeting series (IS1009, ES2004, TS3003, EN2002)
# because the four meetings of a series share speakers. read only. run: python3 -I ami_series_bootstrap.py
import csv, random
from pathlib import Path
from statistics import mean

RAW = Path(__file__).resolve().parents[2] / "results/raw"
N, SEED = 10000, 0


def der(s, col):
    p = RAW / "ami/scores_community1.csv" if s == "community1" else RAW / f"scores_{s}_ami.csv"
    return {r["uri"]: float(r[col]) for r in csv.DictReader(open(p))}


PAIRS = (("S3-S4", "pyannote_style", "pyannote31"), ("S5-S4", "community1", "pyannote31"), ("S5-S3", "community1", "pyannote_style"))
rows = []
for pair, sa, sb in PAIRS:
    for col in ("der_no_collar", "der_collar_0.25"):
        a, b = der(sa, col), der(sb, col)
        assert a.keys() == b.keys() and len(a) == 16
        series = {}
        for m in a:
            series.setdefault(m[:6], []).append(a[m] - b[m])
        assert len(series) == 4 and all(len(v) == 4 for v in series.values())
        rng = random.Random(SEED)
        keys = sorted(series)
        boots = sorted(mean(d for k in rng.choices(keys, k=4) for d in series[k]) for _ in range(N))
        est = mean(d for v in series.values() for d in v)
        row = {"pair": pair, "collar": col, "mean_diff": est, "ci95_lo": boots[int(0.025 * N)], "ci95_hi": boots[int(0.975 * N) - 1]}
        for k in keys:
            row[f"{k}_mean_diff"] = mean(series[k])
            row[f"{k}_first_better"] = sum(d < 0 for d in series[k])
        rows.append(row)

with open(Path(__file__).with_suffix(".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in r.items()})
