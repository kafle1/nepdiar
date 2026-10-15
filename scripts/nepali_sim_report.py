#!/usr/bin/env python3
"""Pooled DER (same as corpus_der.py) on the simulated Nepali set, overall and per speaker count,
plus the per-recording mean with the same bootstrap as summarise.py. Reads the score.py CSVs for the
means, recomputes pooled DER from the RTTMs. Writes results/summary_nepali_sim_by_count.csv.
Pass `english` to do the same for the English twin (data/english_sim, results/summary_english_sim_by_count.csv)."""
import csv
import sys
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from score import load_annotation, load_uem
from summarise import bootstrap_ci

NAME = sys.argv[1] if len(sys.argv) > 1 else "nepali"
SIM = ROOT / f"data/{NAME}_sim"
SYSTEMS = ["pyannote_style", "pyannote_style_oracle", "pyannote31", "pyannote31_oracle", "sherpa_onnx",
           "sherpa_onnx_oracle", "wespeaker_spectral", "wespeaker_spectral_oracle", "wespeaker_spectral_silero",
           "wespeaker_spectral_silero_oracle", "community1", "community1_oracle"]
uris = [u.strip() for u in (SIM / "conversations.txt").read_text().splitlines() if u.strip()]
nspk = {u: int(u.split("_")[1][0]) for u in uris}
groups = {"all": uris, **{f"{n}spk": [u for u in uris if nspk[u] == n] for n in (2, 3, 4)}}
out_rows = []
for ds in (f"{NAME}_sim", f"{NAME}_sim_rn"):
    for g, gu in groups.items():
        print(f"\n[{ds}] {g}: pooled over {len(gu)} conversations; mean of per-conversation DER with 95% bootstrap CI")
        print(f"{'system':34} {'DER0':>6} {'miss':>6} {'fa':>6} {'conf':>6} {'DER25':>6}  {'mean DER0 [CI]':>24} {'mean DER25 [CI]':>24} {'JER':>6} {'k ok':>5}")
        for s in SYSTEMS:
            hyp_dir = ROOT / "results/raw" / ds / f"{s}_rttm"
            sc = ROOT / "results/raw" / f"scores_{s}_{ds}.csv"
            if not sc.exists() or not all((hyp_dir / f"{u}.rttm").exists() for u in uris):
                continue
            per = {r["uri"]: r for r in csv.DictReader(open(sc))}
            m0, m25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
            for u in gu:
                ref, uem = load_annotation(SIM / "rttm" / f"{u}.rttm", u), load_uem(SIM / "uem" / f"{u}.uem", u)
                hyp = load_annotation(hyp_dir / f"{u}.rttm", u)
                m0(ref, hyp, uem=uem)
                m25(ref, hyp, uem=uem)
            d = m0[:]
            t = d["total"]
            a0 = bootstrap_ci([float(per[u]["der_no_collar"]) for u in gu])
            a25 = bootstrap_ci([float(per[u]["der_collar_0.25"]) for u in gu])
            jer = bootstrap_ci([float(per[u]["jer"]) for u in gu])
            kok = sum(per[u]["ref_speakers"] == per[u]["hyp_speakers"] for u in gu)
            print(f"{s:34} {abs(m0):6.4f} {d['missed detection'] / t:6.4f} {d['false alarm'] / t:6.4f} "
                  f"{d['confusion'] / t:6.4f} {abs(m25):6.4f}  {a0[0]:.4f} [{a0[1]:.4f},{a0[2]:.4f}] "
                  f"{a25[0]:.4f} [{a25[1]:.4f},{a25[2]:.4f}] {jer[0]:6.4f} {kok:>2}/{len(gu)}")
            out_rows.append({"dataset": ds, "group": g, "system": s, "n": len(gu),
                             "pooled_der_no_collar": f"{abs(m0):.4f}", "pooled_miss": f"{d['missed detection'] / t:.4f}",
                             "pooled_fa": f"{d['false alarm'] / t:.4f}", "pooled_conf": f"{d['confusion'] / t:.4f}",
                             "pooled_der_collar_0.25": f"{abs(m25):.4f}",
                             "mean_der_no_collar": f"{a0[0]:.4f}", "ci_lo": f"{a0[1]:.4f}", "ci_hi": f"{a0[2]:.4f}",
                             "mean_der_collar_0.25": f"{a25[0]:.4f}", "ci25_lo": f"{a25[1]:.4f}", "ci25_hi": f"{a25[2]:.4f}",
                             "mean_jer": f"{jer[0]:.4f}", "jer_ci_lo": f"{jer[1]:.4f}", "jer_ci_hi": f"{jer[2]:.4f}",
                             "speaker_count_right": kok})
if out_rows:
    with open(ROOT / f"results/summary_{NAME}_sim_by_count.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
