#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: is the Indic DiarBench undercount partly speakers who barely talk?
Per clip: reference speakers with under 5 s and under 10 s of speech inside the UEM (each speaker's own segments
merged, 1 ms grid). Per config and its oracle run: clips with the count right and mean hypothesis minus reference
speakers, against all reference speakers and against only those with 5 s or more, and 10 s or more. Hypothesis
speakers are the labels in results/raw/indic_diarbench_ne/<system>_rttm/ with any speech inside the UEM.
Source recordings come from stats.csv. Labels are read straight from the RTTMs (SPEAKER_0n as released).
Writes results/posthoc_diarbench_count.csv."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

D = ROOT / "data/nepali_real/diarbench"
RAW = ROOT / "results/raw/indic_diarbench_ne"
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2a", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}


def speech_per_label(p, a, b):
    segs = defaultdict(list)
    if p.exists():
        for line in open(p):
            f = line.split()
            if f and f[0] == "SPEAKER":
                s = int(round(float(f[3]) * 1000))
                segs[f[7]].append((s, s + int(round(float(f[4]) * 1000))))
    out = {}
    for k, spans in segs.items():
        on = np.zeros(b - a, bool)
        for s, e in spans:
            on[max(s - a, 0):max(min(e - a, b - a), 0)] = True
        out[k] = on.sum() / 1000
    return out


clips = (D / "clips.txt").read_text().split()
stats = {r["uri"]: r for r in csv.DictReader(open(D / "stats.csv"))}
uem = {}
for u in clips:
    f = (D / "uem" / f"{u}.uem").read_text().split()
    uem[u] = (int(round(float(f[2]) * 1000)), int(round(float(f[3]) * 1000)))
ref = {u: speech_per_label(D / "rttm" / f"{u}.rttm", *uem[u]) for u in clips}

rows = []
by_rec = defaultdict(set)
for u in clips:
    by_rec[stats[u]["recording_id"]].update(ref[u])
    rows.append({"part": "clip", "uri": u, "recording_id": stats[u]["recording_id"], "ref_speakers": len(ref[u]),
                 "ref_under_5s": sum(v < 5 for v in ref[u].values()), "ref_under_10s": sum(v < 10 for v in ref[u].values()),
                 "ref_labels": " ".join(sorted(ref[u])),
                 "ref_speech_s_per_speaker": " ".join(f"{ref[u][k]:.1f}" for k in sorted(ref[u]))})
print(f"{len(clips)} clips from {len(by_rec)} recordings: " + "; ".join(
    f"{r}: {sum(stats[u]['recording_id'] == r for u in clips)} clips, labels {sorted(v)}" for r, v in sorted(by_rec.items())))
gaps = [u for u in clips if sorted(ref[u]) != [f"SPEAKER_{i:02d}" for i in range(1, len(ref[u]) + 1)]]
print(f"clips whose labels skip a number (so not renumbered per clip): {gaps}")
print(f"speakers under 5 s: {sum(r['ref_under_5s'] for r in rows)} of {sum(r['ref_speakers'] for r in rows)} "
      f"in {sum(r['ref_under_5s'] > 0 for r in rows)} clips; under 10 s: {sum(r['ref_under_10s'] for r in rows)} "
      f"in {sum(r['ref_under_10s'] > 0 for r in rows)} clips")

for sysname, lab in LABEL.items():
    for run in (sysname, f"{sysname}_oracle"):
        if not (RAW / f"{run}_rttm").is_dir():
            print(f"skip {run}: no rttm dir")
            continue
        hyp = np.array([sum(v > 0 for v in speech_per_label(RAW / f"{run}_rttm" / f"{u}.rttm", *uem[u]).values()) for u in clips])
        row = {"part": "system", "config": lab, "system": run, "n_clips": len(clips)}
        for tag, floor in (("all", 0), ("5s", 5), ("10s", 10)):
            k = np.array([sum(v >= floor for v in ref[u].values()) for u in clips])
            row[f"count_right_{tag}"] = int((hyp == k).sum())
            row[f"mean_hyp_minus_ref_{tag}"] = round(float((hyp - k).mean()), 2)
            row[f"clips_under_{tag}"] = int((hyp < k).sum())
            row[f"clips_over_{tag}"] = int((hyp > k).sum())
        rows.append(row)
        print({k: v for k, v in row.items() if k != "part"})

with open(ROOT / "results/posthoc_diarbench_count.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), restval="")
    w.writeheader()
    w.writerows(rows)
