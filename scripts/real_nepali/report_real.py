#!/usr/bin/env python3
"""Pooled DER (as corpus_der.py), per-recording means with a 10,000-resample bootstrap (seed 0),
JER, speaker-count accuracy and RTF for the real Nepali sets, plus DiarBench by speaker count and
AMI test next to them (AMI read only, from its existing score CSVs and RTTMs).
Writes results/summary_nepali_real.csv; the printed text goes to results/corpus_der_nepali_real.txt."""
import csv
import sys
from pathlib import Path

import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402
from score import load_annotation, load_uem  # noqa: E402
from summarise import bootstrap_ci  # noqa: E402

NB = 10000
SYSTEMS = ["pyannote_style", "pyannote_style_oracle", "pyannote31", "pyannote31_oracle", "sherpa_onnx",
           "sherpa_onnx_oracle", "wespeaker_spectral_silero", "wespeaker_spectral_silero_oracle",
           "wespeaker_spectral", "wespeaker_spectral_oracle", "community1", "community1_oracle"]
SETS = {
    "indic_diarbench_ne": (ROOT / "data/nepali_real/diarbench", "clips.txt"),
    "indicvoices_ne_calls": (ROOT / "data/nepali_real/indicvoices", "calls.txt"),
    "ami_test": (ROOT / "ref/ami", "test.meetings.txt"),
}


def paths(ds, s, u):
    if ds == "ami_test":
        return (ROOT / "ref/ami/only_words/rttms/test" / f"{u}.rttm", ROOT / "ref/ami/uems/test" / f"{u}.uem",
                ROOT / "results/raw/ami" / f"{s}_rttm" / f"{u}.rttm")
    d = SETS[ds][0]
    return d / "rttm" / f"{u}.rttm", d / "uem" / f"{u}.uem", ROOT / "results/raw" / ds / f"{s}_rttm" / f"{u}.rttm"


def rtf(ds, s):
    """audio-weighted RTF from the run's timing csv (System 3 adds its extract stage; cluster is tiny)."""
    base = ROOT / "results/raw" / ("ami" if ds == "ami_test" else ds)
    f = base / ("pyannote_style_extract_timing.csv" if s.startswith("pyannote_style") else f"{s}_timing.csv")
    if not f.exists():
        return "", ""
    rows = list(csv.DictReader(open(f)))
    dur, proc = sum(float(r["duration_s"]) for r in rows), sum(float(r["proc_time_s"]) for r in rows)
    return f"{proc / dur:.3f}", rows[0].get("timing_note", "")


def block(ds, s, uris, per, label):
    m0, m25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
    for u in uris:
        r, e, h = paths(ds, s, u)
        ref, uem = load_annotation(r, u), load_uem(e, u)
        hyp = load_annotation(h, u)
        m0(ref, hyp, uem=uem)
        m25(ref, hyp, uem=uem)
    d0, d25 = m0[:], m25[:]
    out = {"dataset": ds, "group": label, "system": s, "n": len(uris)}
    for tag, d, m in (("no_collar", d0, m0), ("collar_0.25", d25, m25)):
        t = d["total"]
        out[f"pooled_der_{tag}"] = abs(m)
        out[f"pooled_miss_{tag}"] = d["missed detection"] / t
        out[f"pooled_fa_{tag}"] = d["false alarm"] / t
        out[f"pooled_conf_{tag}"] = d["confusion"] / t
    for k in ("der_no_collar", "der_collar_0.25", "jer"):
        mean, lo, hi = bootstrap_ci([float(per[u][k]) for u in uris], n=NB, seed=0)
        out[f"mean_{k}"], out[f"{k}_ci95_lo"], out[f"{k}_ci95_hi"] = mean, lo, hi
    ref_k = np.array([int(per[u]["ref_speakers"]) for u in uris])
    hyp_k = np.array([int(per[u]["hyp_speakers"]) for u in uris])
    out["speaker_count_right"] = int((ref_k == hyp_k).sum())
    out["speaker_count_acc"] = float((ref_k == hyp_k).mean())
    out["mean_hyp_minus_ref_speakers"] = float((hyp_k - ref_k).mean())
    out["rtf"], out["timing_note"] = rtf(ds, s) if label == "all" else ("", "")
    return out


def main():
    rows = []
    for ds, (d, lst) in SETS.items():
        uris = [u.strip() for u in (d / lst).read_text().splitlines() if u.strip()] if (d / lst).exists() else []
        if not uris:
            continue
        groups = {"all": uris}
        if ds == "indic_diarbench_ne":
            nspk = {u: len({l.split()[7] for l in open(d / "rttm" / f"{u}.rttm")}) for u in uris}
            groups.update({f"{k}spk": [u for u in uris if nspk[u] == k] for k in sorted(set(nspk.values()))})
        if ds == "ami_test":
            groups = {"all": uris}
        for s in SYSTEMS:
            sc = ROOT / "results/raw" / (f"scores_{s}_ami.csv" if ds == "ami_test" else f"scores_{s}_{ds}.csv")
            if not sc.exists() or not all(paths(ds, s, u)[2].exists() for u in uris):
                continue
            per = {r["uri"]: r for r in csv.DictReader(open(sc))}
            if not all(u in per for u in uris):
                continue
            for g, gu in groups.items():
                rows.append(block(ds, s, gu, per, g))
    if not rows:
        print("nothing scored yet")
        return
    hdr = (f"{'system':34} {'n':>3} {'DER0':>6} {'miss':>6} {'fa':>6} {'conf':>6} {'DER25':>6} "
           f"{'mean DER0 [95% CI]':>24} {'mean DER25 [95% CI]':>24} {'JER':>6} {'k ok':>6} {'k err':>6} {'RTF':>6}")
    print("Pooled DER (total error / total speech), miss/fa/conf at no collar; per-recording means with a "
          f"{NB}-resample bootstrap, seed 0. k ok = speaker count right; k err = mean (found - true).")
    last = None
    for r in rows:
        if (r["dataset"], r["group"]) != last:
            last = (r["dataset"], r["group"])
            print(f"\n[{r['dataset']}] {r['group']}, {r['n']} recordings")
            print(hdr)
        print(f"{r['system']:34} {r['n']:>3} {r['pooled_der_no_collar']:6.4f} {r['pooled_miss_no_collar']:6.4f} "
              f"{r['pooled_fa_no_collar']:6.4f} {r['pooled_conf_no_collar']:6.4f} {r['pooled_der_collar_0.25']:6.4f} "
              f"{r['mean_der_no_collar']:.4f} [{r['der_no_collar_ci95_lo']:.4f},{r['der_no_collar_ci95_hi']:.4f}] "
              f"{r['mean_der_collar_0.25']:.4f} [{r['der_collar_0.25_ci95_lo']:.4f},{r['der_collar_0.25_ci95_hi']:.4f}] "
              f"{r['mean_jer']:6.4f} {r['speaker_count_right']:>3}/{r['n']:<2} {r['mean_hyp_minus_ref_speakers']:+6.2f} "
              f"{r['rtf']:>6}")
    fmt = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in rows[0].items()}
    with open(ROOT / "results/summary_nepali_real.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(fmt))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


if __name__ == "__main__":
    main()
