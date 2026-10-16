#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: under the true speaker count on AMI test, do two real speakers end up in
one cluster? Per meeting and oracle-count run: the reference-by-hypothesis overlap matrix in seconds inside the
UEM (1 ms grid, each label's own segments merged), the optimal one-to-one mapping (scipy linear_sum_assignment,
maximising overlap, as DER's mapping does), and a merge wherever one hypothesis cluster holds the majority
(over half) of the speech of two or more reference speakers. Also each reference speaker's share of time in its
mapped cluster. The estimated-count runs get the same check for comparison.
Writes results/posthoc_ami_oracle_merge.csv."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

REF, UEM, RAW = ROOT / "ref/ami/only_words/rttms/test", ROOT / "ref/ami/uems/test", ROOT / "results/raw/ami"
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2a", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}


def tracks(p, a, b):
    segs = defaultdict(list)
    for line in open(p):
        f = line.split()
        if f and f[0] == "SPEAKER":
            s = int(round(float(f[3]) * 1000))
            segs[f[7]].append((s, s + int(round(float(f[4]) * 1000))))
    out = {}
    for k, spans in sorted(segs.items()):
        on = np.zeros(b - a, bool)
        for s, e in spans:
            on[max(s - a, 0):max(min(e - a, b - a), 0)] = True
        out[k] = on
    return out


meetings = sorted(p.stem for p in REF.glob("*.rttm"))
rows = []
for sysname, lab in LABEL.items():
    for run, kind in ((f"{sysname}_oracle", "oracle"), (sysname, "estimated")):
        for m in meetings:
            f = (UEM / f"{m}.uem").read_text().split()
            a, b = int(round(float(f[2]) * 1000)), int(round(float(f[3]) * 1000))
            ref, hyp = tracks(REF / f"{m}.rttm", a, b), tracks(RAW / f"{run}_rttm" / f"{m}.rttm", a, b)
            hyp = {k: v for k, v in hyp.items() if v.any()}
            rk, hk = list(ref), list(hyp)
            ov = np.array([[(ref[r] & hyp[h]).sum() / 1000 for h in hk] for r in rk]).reshape(len(rk), len(hk))
            ri, ci = linear_sum_assignment(-ov) if hk else ([], [])
            mapped = {rk[i]: hk[j] for i, j in zip(ri, ci)}
            rs = {r: ref[r].sum() / 1000 for r in rk}
            major = defaultdict(list)
            for i, r in enumerate(rk):
                if len(hk) and ov[i].max() > rs[r] / 2:
                    major[hk[int(ov[i].argmax())]].append(r)
            merges = {h: rr for h, rr in major.items() if len(rr) > 1}
            rows.append({"config": lab, "system": run, "count": kind, "meeting": m, "ref_speakers": len(rk),
                         "hyp_speakers": len(hk), "n_merges": len(merges),
                         "merged": "; ".join(f"{h}<-{'+'.join(rr)}" for h, rr in merges.items()),
                         "ref_unmapped": " ".join(r for r in rk if r not in mapped),
                         "share_in_mapped": " ".join(f"{r}:{ov[i, hk.index(mapped[r])] / rs[r]:.2f}" if r in mapped else f"{r}:0.00"
                                                     for i, r in enumerate(rk)),
                         "min_share_in_mapped": round(min((ov[i, hk.index(mapped[r])] / rs[r]) if r in mapped else 0.0
                                                          for i, r in enumerate(rk)), 3),
                         "overlap_matrix_s": " | ".join(f"{r}: " + " ".join(f"{x:.0f}" for x in ov[i]) for i, r in enumerate(rk))})

with open(ROOT / "results/posthoc_ami_oracle_merge.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for lab in LABEL.values():
    for kind in ("oracle", "estimated"):
        rr = [r for r in rows if r["config"] == lab and r["count"] == kind]
        print(f"{lab:4} {kind:9} merges {sum(r['n_merges'] for r in rr)} in {sum(r['n_merges'] > 0 for r in rr)}/{len(rr)} meetings; "
              f"hyp != ref count in {sum(r['hyp_speakers'] != r['ref_speakers'] for r in rr)}; "
              + "; ".join(f"{r['meeting']} {r['merged']} (min share {r['min_share_in_mapped']})" for r in rr if r["n_merges"]))
