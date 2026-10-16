#!/usr/bin/env python3
"""How close the English twin is to NepDiar-Sim, from the reference RTTMs: speech share of the
conversation, overlap share of speech, speaker changes and segments per minute of audio (time-sorted segments
whose speaker differs from the one before), mean segment length, plus the paper's pooled per-speech-minute table numbers. AMI test is printed with the same code for scale.
The reverb + noise condition reuses the clean references, so its row is the same by construction.
Writes results/sim_twin_stats.csv."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT


def rttm(p):
    segs = []
    for line in open(p):
        f = line.split()
        if f and f[0] == "SPEAKER":
            s = int(round(float(f[3]) * 1000))
            segs.append((s, s + int(round(float(f[4]) * 1000)), f[7]))
    return sorted(segs)


def stats(paths, uem_of):
    out = []
    for p in paths:
        segs = rttm(p)
        a, b = uem_of(p, segs)
        grid = np.zeros(b - a, np.int8)
        for s, e, _ in segs:
            grid[max(s - a, 0):max(min(e - a, b - a), 0)] += 1
        speech, ov = int((grid > 0).sum()), int((grid > 1).sum())
        changes = sum(segs[i][2] != segs[i - 1][2] for i in range(1, len(segs)))
        out.append((speech / (b - a), ov / speech, changes / ((b - a) / 60000), len(segs) / ((b - a) / 60000), np.mean([e - s for s, e, _ in segs]) / 1000))
    return np.array(out)


def uem(u):
    f = u.read_text().split()
    return int(round(float(f[2]) * 1000)), int(round(float(f[3]) * 1000))


def sim_uem(p, _):
    return uem(p.parent.parent / "uem" / f"{p.stem}.uem")


rows = []
order = [u.strip() for u in (ROOT / "data/nepali_sim/conversations.txt").read_text().splitlines() if u.strip()]
sets = {"nepali": [ROOT / "data/nepali_sim/rttm" / f"{u}.rttm" for u in order],
        "english": [ROOT / "data/english_sim/rttm" / f"{u.replace('nepsim_', 'engsim_')}.rttm" for u in order]}
res = {k: stats(v, sim_uem) for k, v in sets.items()}
ami_dir = ROOT / "ref/ami/only_words/rttms/test"
ami = stats(sorted(ami_dir.glob("*.rttm")), lambda p, _: uem(ROOT / "ref/ami/uems/test" / f"{p.stem}.uem"))
names = ["speech_share", "overlap_share_of_speech", "speaker_changes_per_audio_min", "segments_per_audio_min", "mean_segment_s"]
print(f"{'set':10} {'cond':13} " + " ".join(f"{n:>24}" for n in names))
for cond in ("clean", "reverb_noise"):
    for k in ("nepali", "english"):
        m = res[k].mean(axis=0)
        rows.append({"set": k, "condition": cond, "n": len(res[k]), **{n: round(float(v), 4) for n, v in zip(names, m)}})
        print(f"{k:10} {cond:13} " + " ".join(f"{v:24.4f}" for v in m))
d = res["nepali"] - res["english"]
print("per-pair nepali minus english, mean [min, max]: " + "; ".join(
    f"{n} {d[:, i].mean():+.4f} [{d[:, i].min():+.4f}, {d[:, i].max():+.4f}]" for i, n in enumerate(names)))
m = ami.mean(axis=0)
rows.append({"set": "ami_test_ref", "condition": "real", "n": len(ami), **{n: round(float(v), 4) for n, v in zip(names, m)}})
print(f"{'ami test':10} {'uem':13} " + " ".join(f"{v:24.4f}" for v in m))
# the paper's table definition (paper/analysis/turn_stats_table.py, imported read-only): pooled, per minute of speech
sys.path.insert(0, str(ROOT / "paper/analysis"))
from turn_stats_table import measure
for k, d in (("nepali", "nepali_sim"), ("english", "english_sim")):
    m = measure(k, k, ROOT / f"data/{d}/rttm", ROOT / f"data/{d}/uem", ROOT / f"data/{d}/conversations.txt", "")
    print(f"{k}: changes per speech min {m['changes_per_speech_min']}, speech {m['speech_pct']}%, overlap {m['overlap_pct']}%, "
          f"median segment {m['median_segment_s']} s, under 1 s {m['segments_under_1s_pct']}%, median turn {m['median_turn_s']} s")
    for r in rows:
        if r["set"] == k:
            r.update({x: m[x] for x in ("changes_per_speech_min", "speech_pct", "overlap_pct", "median_segment_s", "segments_under_1s_pct", "median_turn_s")})
with open(ROOT / "results/sim_twin_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]), restval="")
    w.writeheader()
    w.writerows(rows)
