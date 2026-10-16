#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: paired comparisons of pooled DER (no collar) on the IndicVoices calls,
default and tight reference. Speakers are disjoint across calls, so each call is its own cluster: resample calls
(10,000, default_rng(0) per comparison), pool sum(error) / sum(reference speech) per system on the same resample,
take the difference, percentile 95% CI. Every pair of estimated-count configs (A minus B), and each config's
oracle minus estimated. Error = der_no_collar x the call's reference speech, which comes from the reference
RTTM and UEM as pyannote.metrics counts it. A system whose scores file is missing or has fewer than 92 calls
is skipped and named. Writes results/posthoc_iv_paired.csv."""
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from score import load_annotation, load_uem

D = ROOT / "data/nepali_real/indicvoices"
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2a", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
N_CALLS, N_BOOT = 92, 10000

calls = (D / "calls.txt").read_text().split()
assert len(calls) == N_CALLS, len(calls)
rows, skipped = [], set()
for ds, rdir in (("indicvoices_ne_calls", "rttm"), ("indicvoices_ne_calls_tight", "rttm_tight")):
    m = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    tot = np.array([m(r := load_annotation(D / rdir / f"{u}.rttm", u), r, uem=load_uem(D / "uem" / f"{u}.uem", u),
                      detailed=True)["total"] for u in calls])
    err = {}
    for s in LABEL:
        for run in (s, f"{s}_oracle"):
            p = ROOT / "results/raw" / f"scores_{run}_{ds}.csv"
            got = {r["uri"]: float(r["der_no_collar"]) for r in csv.DictReader(open(p))} if p.exists() else {}
            if any(u not in got for u in calls):
                skipped.add(f"{run} ({'missing' if not p.exists() else f'{len(got)} calls'}, {ds})")
                continue
            err[run] = np.array([got[u] for u in calls]) * tot
    pairs = [(a, b, "pair") for a, b in combinations(LABEL, 2) if a in err and b in err]
    pairs += [(f"{s}_oracle", s, "oracle_minus_estimated") for s in LABEL if s in err and f"{s}_oracle" in err]
    for a, b, kind in pairs:
        idx = np.random.default_rng(0).integers(0, N_CALLS, size=(N_BOOT, N_CALLS))
        t = tot[idx].sum(1)
        diff = 100 * (err[a][idx].sum(1) - err[b][idx].sum(1)) / t
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pa, pb = 100 * err[a].sum() / tot.sum(), 100 * err[b].sum() / tot.sum()
        la = LABEL[a.removesuffix("_oracle")] + ("-oracle" if a.endswith("_oracle") else "")
        rows.append({"reference": "tight" if ds.endswith("_tight") else "default", "kind": kind, "a": la, "b": LABEL[b],
                     "system_a": a, "system_b": b, "n_calls": N_CALLS, "pooled_der_a": round(pa, 2), "pooled_der_b": round(pb, 2),
                     "diff_a_minus_b_pts": round(pa - pb, 2), "ci95_lo": round(float(lo), 2), "ci95_hi": round(float(hi), 2),
                     "excludes_zero": int(lo > 0 or hi < 0)})
for s in sorted(skipped):
    print(f"skipped {s}")
with open(ROOT / "results/posthoc_iv_paired.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['reference']:8} {r['a']:9} - {r['b']:4} {r['pooled_der_a']:6.2f} - {r['pooled_der_b']:6.2f} = "
          f"{r['diff_a_minus_b_pts']:+6.2f} [{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]{' *' if r['excludes_zero'] else ''}")
