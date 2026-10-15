#!/usr/bin/env python3
"""Paired Nepali minus English DER test on NepDiar-Sim and its English twin, as fixed in NOTES.md
("Analysis plan fixed before the Nepali tables"). Per configuration and condition: d_i = DER no collar
of nepsim conversation i minus that of engsim conversation i; mean d, 95% percentile bootstrap CI
(10,000 resamples of the 30 pairs, default_rng(0) per test), two-sided p = 2 min(P(mean* <= 0), P(mean* >= 0)),
Holm across the 6 configurations inside each condition. Writes results/sim_pair_test.csv."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

SYSTEMS = ["sherpa_onnx", "wespeaker_spectral", "wespeaker_spectral_silero", "pyannote_style", "pyannote31", "community1"]
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
CONDITIONS = {"clean": ("nepali_sim", "english_sim", "primary"), "reverb_noise": ("nepali_sim_rn", "english_sim_rn", "secondary")}
N_BOOT = 10000


def der(sys_, ds):
    p = ROOT / "results/raw" / f"scores_{sys_}_{ds}.csv"
    if not p.exists():
        return None
    return {r["uri"]: float(r["der_no_collar"]) for r in csv.DictReader(open(p))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


uris = [u.strip() for u in (ROOT / "data/nepali_sim/conversations.txt").read_text().splitlines() if u.strip()]
rows = []
for cond, (nds, eds, family) in CONDITIONS.items():
    fam = []
    for s in SYSTEMS:
        n, e = der(s, nds), der(s, eds)
        if n is None or e is None:
            print(f"{s} {cond}: scores missing, family incomplete", file=sys.stderr)
            continue
        twins = [(u, u.replace("nepsim_", "engsim_")) for u in uris]
        missing = [u for u, t in twins if u not in n or t not in e]
        if missing:
            sys.exit(f"{s} {cond}: no score for {missing[:3]}")
        d = np.array([n[u] - e[t] for u, t in twins])
        idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
        fam.append({"system": s, "config": LABEL[s], "condition": cond, "family": family, "n_pairs": len(d),
                    "mean_der_nepali": round(float(np.mean([n[u] for u, _ in twins])), 4),
                    "mean_der_english": round(float(np.mean([e[t] for _, t in twins])), 4),
                    "mean_diff": round(float(d.mean()), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "p_raw": round(float(p), 4)})
    for r, a in zip(fam, holm([r["p_raw"] for r in fam])):
        r["p_holm"] = round(a, 4)
        r["holm_family_size"] = len(fam)
    rows += fam
if not rows:
    sys.exit("no paired scores yet")
with open(ROOT / "results/sim_pair_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['condition']:12} {r['config']:4} nep {r['mean_der_nepali']:.4f} eng {r['mean_der_english']:.4f} "
          f"diff {r['mean_diff']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p {r['p_raw']:.4f} holm {r['p_holm']:.4f} (m={r['holm_family_size']})")
