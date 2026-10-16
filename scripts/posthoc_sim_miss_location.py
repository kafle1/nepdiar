#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: where the simulated missed speech sits.
Part 1: each config's mean missed-speech share (over reference speech, estimated count) with no collar and with
the 0.25 s collar, Nepali against English, read from results/summary_{nepali,english}_sim.csv, both conditions.
Part 2: S3 and S4, clean, both twins. On a 1 ms grid, missed time is max(0, reference speakers - hypothesis
speakers) inside the UEM, as pyannote.metrics counts it. Each missed ms goes to "edge" when it is under 300 ms
from the start or end of a reference segment active there (nearest one if several), else to "deep". The
reference trim bridges gaps under 300 ms, so the deep group holds the merged pauses. The same split of the
reference speech itself is the base rate. The 0.25 s collar is centred, so it only drops 125 ms each side;
miss_edge125 gives that band on the same grid for comparison. Writes results/posthoc_sim_miss_location.csv."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2a", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
EDGE_MS = 300


def rttm(p):
    segs = defaultdict(list)
    for line in open(p):
        f = line.split()
        if f and f[0] == "SPEAKER":
            s = int(round(float(f[3]) * 1000))
            segs[f[7]].append((s, s + int(round(float(f[4]) * 1000))))
    return segs


def count(segs, n):
    c = np.zeros(n, np.int16)
    for spans in segs.values():
        on = np.zeros(n, bool)
        for s, e in spans:
            on[max(s, 0):max(min(e, n), 0)] = True
        c += on
    return c


def split(ref_p, hyp_p, uem_p):
    f = uem_p.read_text().split()
    a, b = int(round(float(f[2]) * 1000)), int(round(float(f[3]) * 1000))
    ref, hyp = rttm(ref_p), rttm(hyp_p)
    n = b
    nr, nh = count(ref, n)[a:b], count(hyp, n)[a:b]
    dist = np.full(n, 10 ** 9, np.int64)
    t = np.arange(n)
    for spans in ref.values():
        for s, e in spans:
            s, e = max(s, 0), min(e, n)
            dist[s:e] = np.minimum(dist[s:e], np.minimum(t[s:e] - s, e - 1 - t[s:e]))
    edge = dist[a:b] < EDGE_MS
    miss = np.maximum(nr - nh, 0).astype(np.int64)
    return miss[edge].sum(), miss[~edge].sum(), nr[edge].sum(), nr[~edge].sum(), miss[dist[a:b] < 125].sum()


rows = []
for cond, sfx in (("clean", ""), ("reverb_noise", "_rn")):
    s = {lang: {r["system"]: r for r in csv.DictReader(open(ROOT / f"results/summary_{lang}_sim.csv")) if r["dataset"] == f"{lang}_sim{sfx}"}
         for lang in ("nepali", "english")}
    for sysname, lab in LABEL.items():
        row = {"part": "collar", "condition": cond, "config": lab, "system": sysname}
        for lang in ("nepali", "english"):
            m0, m25 = float(s[lang][sysname]["miss_no_collar_mean"]), float(s[lang][sysname]["miss_collar_0.25_mean"])
            row.update({f"{lang}_miss_no_collar_pct": round(100 * m0, 2), f"{lang}_miss_collar_pct": round(100 * m25, 2),
                        f"{lang}_miss_kept_by_collar_share": round(m25 / m0, 3)})
        rows.append(row)

for sysname in ("pyannote_style", "pyannote31"):
    for lang, ds in (("nepali", "nepali_sim"), ("english", "english_sim")):
        d = ROOT / "data" / ds
        tot = np.zeros(5, np.int64)
        for u in (d / "conversations.txt").read_text().split():
            tot += split(d / "rttm" / f"{u}.rttm", ROOT / "results/raw" / ds / f"{sysname}_rttm" / f"{u}.rttm", d / "uem" / f"{u}.uem")
        me, md, re_, rd, m125 = (int(x) for x in tot)
        rows.append({"part": "location", "condition": "clean", "config": LABEL[sysname], "system": sysname, "lang": lang,
                     "miss_edge_s": round(me / 1000, 1), "miss_deep_s": round(md / 1000, 1),
                     "miss_edge_share": round(me / (me + md), 3), "miss_deep_share": round(md / (me + md), 3), "miss_edge125_share": round(m125 / (me + md), 3),
                     "ref_edge_share": round(re_ / (re_ + rd), 3), "ref_speech_s": round((re_ + rd) / 1000, 1),
                     "pooled_miss_pct": round(100 * (me + md) / (re_ + rd), 2),
                     "miss_rate_edge_pct": round(100 * me / re_, 2), "miss_rate_deep_pct": round(100 * md / rd, 2)})

with open(ROOT / "results/posthoc_sim_miss_location.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), restval="")
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: v for k, v in r.items() if k not in ("part", "system")})
