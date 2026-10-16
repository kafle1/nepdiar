#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: is the Nepali minus English sim gap a reference artefact?
Reviewer worry: OpenSLR 54 has a quieter floor, so the Nepali reference may count more quiet stretches
as speech. Both references come from the same trim() in check_nepali_speakers.py.

Views, all six configs (estimated k, no collar, overlap scored), clean and reverb+noise:
- as_scored: the official reference and UEM (repeats sim_pause_diagnostic.py).
- quiet_removed: sim_pause_diagnostic.py's quiet UEM (cells where a reference speaker is under their
  utterance's trim threshold dropped), now for every config. The UEM is built from the clean tracks;
  the reference is the same in both conditions, so the rn rows use it too.
- strict_+N: the reference rebuilt with trim()'s threshold raised by N dB on each utterance's own
  source signal, everything else in trim() unchanged (25 ms / 10 ms frames, gaps under 300 ms bridged,
  pieces under 200 ms dropped, 20 ms pad), then cut to the region the manifest placed and shifted to its
  placed start. No audio is re-rendered. N = 0 must give back the official RTTM byte for byte.
d is the mean per-conversation DER difference (points), percentile bootstrap CI, default_rng(0),
10,000 resamples of the 30 pairs, as in sim_pair_test.py. Writes results/sim_reference_sensitivity.csv.
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
import make_nepali_sim as nep
import make_english_sim as eng
from score import load_annotation
from sim_pause_diagnostic import LANG, QUIET_UEM, SYSTEMS, boot
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

SR = 16000
DELTAS = (10, 20)
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2a", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
STRICT = ROOT / "results/cache/sim_strict_ref"
OUT = ROOT / "results/sim_reference_sensitivity.csv"


def trim_delta(x, delta):
    """trim() from check_nepali_speakers.py line for line, threshold raised by delta dB."""
    hop, win = SR // 100, SR // 40
    n = max(1, 1 + (len(x) - win) // hop)
    db = np.array([10 * np.log10(np.mean(x[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)])
    thr = max(db.max() - 35, np.percentile(db, 10) + 10) + delta
    on = db > thr
    regions, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            regions.append([i * 10, j * 10 + 25])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in regions:
        if merged and a - merged[-1][1] < 300:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    total_ms = len(x) * 1000 // SR
    return [[max(0, a - 20), min(total_ms, b + 20)] for a, b in merged if b - a >= 200]


def build(lang):
    data = ROOT / "data" / LANG[lang][0]
    reader = nep.ZipAudio(nep.SLR / "zips") if lang == "nepali" else eng.LibriAudio(ROOT / "data/librispeech")
    rows = nep.read_tsv(data / "placement.tsv")
    regs = {}
    for r in rows:
        fid = r["slr54_file_id"] if "slr54_file_id" in r else r["librispeech_id"]
        if fid not in regs:
            x = reader.read(fid)
            regs[fid] = {d: trim_delta(x, d) for d in (0,) + DELTAS}
    stats = []
    for d in (0,) + DELTAS:
        out = STRICT / f"{lang}_{d}"
        out.mkdir(parents=True, exist_ok=True)
        refs, dropped = {}, 0
        for r in rows:
            fid = r["slr54_file_id"] if "slr54_file_id" in r else r["librispeech_id"]
            a, b, s = int(r["src_start_ms"]), int(r["src_end_ms"]), int(r["placed_start_ms"])
            pieces = [(max(a, p), min(b, q)) for p, q in regs[fid][d] if min(b, q) > max(a, p)]
            dropped += not pieces
            refs.setdefault(r["uri"], []).extend((s + p - a, s + q - a, r["speaker"]) for p, q in pieces)
        ms = 0
        for uri, ref in refs.items():
            with open(out / f"{uri}.rttm", "w") as f:
                for p, q, k in sorted(ref):
                    f.write(f"SPEAKER {uri} 1 {p / 1000:.3f} {(q - p) / 1000:.3f} <NA> <NA> {k} <NA> <NA>\n")
            ms += sum(q - p for p, q, _ in ref)
        if d == 0:  # the rebuild must give back the official reference exactly
            for uri in refs:
                assert (out / f"{uri}.rttm").read_text() == (data / "rttm" / f"{uri}.rttm").read_text(), uri
        stats.append({"view": f"ref_strict_+{d}", "lang": lang, "ref_segment_min": round(ms / 60000, 2),
                      "segments_emptied": dropped, "segments": len(rows)})
    return stats


def score(job):
    lang, ds, system, view = job
    data = ROOT / "data" / LANG[lang][0]
    uem_dir = QUIET_UEM / lang if view == "quiet_removed" else data / "uem"
    ref_dir = STRICT / f"{lang}_{view.split('+')[1]}" if view.startswith("strict") else data / "rttm"
    out = {}
    for uri in (data / "conversations.txt").read_text().split():
        ref = load_annotation(ref_dir / f"{uri}.rttm", uri)
        hyp = load_annotation(ROOT / "results/raw" / ds / f"{system}_rttm" / f"{uri}.rttm", uri)
        uem = Timeline([Segment(float(p[2]), float(p[3])) for p in (ln.split() for ln in open(uem_dir / f"{uri}.uem"))], uri=uri)
        d = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
        out[uri.split("_", 1)[1]] = (d["total"], d["missed detection"], d["false alarm"], d["confusion"])
    return job, out


def main():
    assert all(len(list((QUIET_UEM / l).glob("*.uem"))) == 30 for l in LANG), "run sim_pause_diagnostic.py first"
    with Pool(2) as p:
        rows = [s for st in p.map(build, list(LANG)) for s in st]
    for r in rows:
        print(r, flush=True)
    views = ("as_scored", "quiet_removed") + tuple(f"strict_+{d}" for d in DELTAS)
    jobs = [(lang, LANG[lang][0] + sfx, system, view) for sfx in ("", "_rn") for system in SYSTEMS
            for view in views for lang in LANG]
    with Pool(3) as p:
        res = dict(p.map(score, jobs))
    for cond, sfx in (("clean", ""), ("reverb_noise", "_rn")):
        for system in SYSTEMS:
            base = None
            for view in views:
                row = {"view": view, "condition": cond, "system": LABEL[system]}
                per = {}
                for lang in LANG:
                    r = res[(lang, LANG[lang][0] + sfx, system, view)]
                    T, M, F, C = (sum(v[i] for v in r.values()) for i in range(4))
                    row.update({f"{lang}_der": round(100 * (M + F + C) / T, 2), f"{lang}_miss": round(100 * M / T, 2),
                                f"{lang}_fa": round(100 * F / T, 2), f"{lang}_conf": round(100 * C / T, 2)})
                    per[lang] = {k: (v[1] + v[2] + v[3]) / v[0] for k, v in r.items()}
                d = np.array([per["nepali"][k] - per["english"][k] for k in sorted(per["nepali"])]) * 100
                lo, hi = boot(d)
                base = float(d.mean()) if view == "as_scored" else base
                row.update({"mean_d": round(float(d.mean()), 2), "d_ci_lo": round(lo, 2), "d_ci_hi": round(hi, 2),
                            "share_of_as_scored_d_pct": round(100 * float(d.mean()) / base, 1)})
                # same paired d on one component at a time (component / that conversation's reference time)
                for i, name in ((1, "miss"), (2, "fa"), (3, "conf")):
                    c = {lang: {k: v[i] / v[0] for k, v in res[(lang, LANG[lang][0] + sfx, system, view)].items()} for lang in LANG}
                    dc = np.array([c["nepali"][k] - c["english"][k] for k in sorted(c["nepali"])]) * 100
                    lo, hi = boot(dc)
                    row.update({f"d_{name}": round(float(dc.mean()), 2), f"d_{name}_ci_lo": round(lo, 2), f"d_{name}_ci_hi": round(hi, 2)})
                rows.append(row)
                print(row, flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
