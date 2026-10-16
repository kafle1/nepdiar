#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: the sex mix of NepDiar-Sim conversations, from median F0.

1. F0 per 10 ms frame with YIN (25 ms integration window, lags up to 16.6 ms, cumulative mean normalised
   difference, absolute threshold 0.15, search 60 to 400 Hz) on the clean source segments each
   conversation places (placement.tsv), never the mix. A frame counts as voiced when YIN finds a dip
   under the threshold and the frame is within 30 dB of the segment's loudest frame. A speaker's F0 is
   the median over all their voiced frames. Below 160 Hz reads as male, above as female; 150 to 170 Hz
   is "uncertain".
2. The same estimate on the twin's LibriSpeech speakers, scored against SPEAKERS.TXT.
3. Per conversation: Nepali mix vs the twin's mix (exact / mixed_diff / same_sex), and the paired mean
   d (Nepali minus English DER, no collar) with sim_pair_test.py's bootstrap (10,000, default_rng(0))
   on the full 30, the exact-match subset and the mixed-sex subset. Descriptive only, no Holm.

usage: nice -n 10 .venv/bin/python scripts/sim_sex_mix.py
Writes results/sim_sex_f0_speakers.csv, results/sim_sex_mix_conversations.csv, results/sim_sex_mix_pair.csv.
"""
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

SR = 16000
HOP, W = 160, 400  # 10 ms hop, 25 ms integration window
TAU_MIN, TAU_MAX = SR // 400, SR // 60  # 40 .. 266 samples
FRAME = W + TAU_MAX
YIN_THR = 0.15
FLOOR_DB = 30.0
CUT, UNC_LO, UNC_HI = 160.0, 150.0, 170.0
N_BOOT = 10000
SYSTEMS = ["sherpa_onnx", "wespeaker_spectral", "wespeaker_spectral_silero", "pyannote_style", "pyannote31", "community1"]
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
CONDITIONS = {"clean": ("nepali_sim", "english_sim"), "reverb_noise": ("nepali_sim_rn", "english_sim_rn")}


def yin_f0(x):
    """F0 (Hz) of every voiced 10 ms frame of x, YIN with FFT cross-correlation."""
    x = x.astype(np.float64)
    if len(x) < FRAME:
        return np.zeros(0)
    n = 1 + (len(x) - FRAME) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP][:n]
    nfft = 1024
    head = np.zeros_like(fr)
    head[:, :W] = fr[:, :W]
    r = np.fft.irfft(np.conj(np.fft.rfft(head, nfft)) * np.fft.rfft(fr, nfft), nfft)[:, :TAU_MAX + 1]
    c = np.concatenate([np.zeros((n, 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    taus = np.arange(TAU_MAX + 1)
    e0 = c[:, W][:, None]
    et = c[:, taus + W] - c[:, taus]
    d = np.maximum(e0 + et - 2 * r, 0)
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cm, 1e-12)
    energy_db = 10 * np.log10(e0[:, 0] / W + 1e-12)
    loud = energy_db > energy_db.max() - FLOOR_DB
    out = []
    for i in np.where(loud)[0]:
        row = dn[i]
        below = np.where(row[TAU_MIN:TAU_MAX] < YIN_THR)[0]
        if not len(below):
            continue
        t = TAU_MIN + below[0]
        while t + 1 < TAU_MAX and row[t + 1] < row[t]:
            t += 1
        a, b, cc = row[t - 1], row[t], row[t + 1]  # parabolic refinement
        den = a - 2 * b + cc
        tt = t + (0.5 * (a - cc) / den if den > 0 else 0.0)
        out.append(SR / tt)
    return np.array(out)


def call(f0):
    return "F" if f0 > CUT else "M"


def nepali_f0():
    where = {}
    for z in sorted((ROOT / "data/slr54/zips").glob("asr_nepali_*.zip")):
        for nm in zipfile.ZipFile(z).namelist():
            if nm.endswith(".flac"):
                where[Path(nm).stem] = (z, nm)
    zips, frames, cache = {}, defaultdict(list), {}
    for r in csv.DictReader(open(ROOT / "data/nepali_sim/placement.tsv"), delimiter="\t"):
        fid = r["slr54_file_id"]
        if fid not in cache:
            z, nm = where[fid]
            zf = zips.setdefault(z, zipfile.ZipFile(z))
            x, sr = sf.read(io.BytesIO(zf.read(nm)), dtype="float32", always_2d=True)
            assert sr == SR, (fid, sr)
            cache = {fid: x[:, 0]}  # placement rows of one file sit together often; keep only the last
        x = cache[fid]
        frames[r["speaker"]].append(yin_f0(x[int(r["src_start_ms"]) * 16:int(r["src_end_ms"]) * 16]))
    return {s: np.concatenate(v) for s, v in frames.items()}


def english_f0():
    root = ROOT / "data/librispeech/LibriSpeech"
    frames = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "data/english_sim/placement.tsv"), delimiter="\t"):
        spk, chap, _ = r["librispeech_id"].split("-")
        p = next(q for q in (root / sub / spk / chap / f"{r['librispeech_id']}.flac" for sub in ("dev-clean", "test-clean")) if q.exists())
        x, sr = sf.read(p, dtype="float32", start=int(r["src_start_ms"]) * 16, stop=int(r["src_end_ms"]) * 16)
        frames[spk].append(yin_f0(x if x.ndim == 1 else x[:, 0]))
    sex = {}
    for line in (root / "SPEAKERS.TXT").read_text().splitlines():
        if not line.startswith(";"):
            p = [t.strip() for t in line.split("|")]
            if len(p) >= 3:
                sex[p[0]] = p[1]
    return {s: np.concatenate(v) for s, v in frames.items()}, sex


def hist(vals):
    bins = np.arange(70, 311, 10)
    h, _ = np.histogram(np.clip(vals, 70, 309.9), bins)
    return "  ".join(f"{int(a)}:{c}" for a, c in zip(bins[:-1], h) if c)


def main():
    rows = []
    ef, esex = english_f0()
    hit = hit_sure = n_sure = 0
    for s, f in sorted(ef.items()):
        med = float(np.median(f))
        unc = UNC_LO <= med <= UNC_HI
        rows.append({"set": "english_twin", "speaker": s, "voiced_frames": len(f), "median_f0": round(med, 1),
                     "call": call(med), "uncertain": int(unc), "true_sex": esex[s]})
        hit += call(med) == esex[s]
        if not unc:
            n_sure += 1
            hit_sure += call(med) == esex[s]
    emed = np.array([r["median_f0"] for r in rows])
    print(f"English twin: {len(ef)} speakers, F0 call right on {hit}/{len(ef)}; outside 150-170 Hz {hit_sure}/{n_sure}, "
          f"{len(ef) - n_sure} uncertain")
    wrong = [(r["speaker"], r["median_f0"], r["true_sex"]) for r in rows if r["call"] != r["true_sex"]]
    print(f"  wrong: {wrong}")
    for g in "FM":
        v = [r["median_f0"] for r in rows if r["true_sex"] == g]
        print(f"  true {g}: n={len(v)} median F0 {np.median(v):.0f}, range {min(v):.0f}-{max(v):.0f}")
    nf = nepali_f0()
    nrows = []
    for s, f in sorted(nf.items()):
        med = float(np.median(f))
        nrows.append({"set": "nepsim", "speaker": s, "voiced_frames": len(f), "median_f0": round(med, 1),
                      "call": call(med), "uncertain": int(UNC_LO <= med <= UNC_HI), "true_sex": ""})
    nmed = np.array([r["median_f0"] for r in nrows])
    print(f"NepDiar-Sim: {len(nrows)} speakers, F {sum(r['call'] == 'F' for r in nrows)}, M {sum(r['call'] == 'M' for r in nrows)}, "
          f"uncertain (150-170 Hz) {sum(r['uncertain'] for r in nrows)}; frames per speaker min {min(r['voiced_frames'] for r in nrows)}")
    print(f"  Nepali F0 histogram (10 Hz bins, bin start:count): {hist(nmed)}")
    print(f"  English F0 histogram: {hist(emed)}")
    print(f"  Nepali uncertain: {[(r['speaker'], r['median_f0']) for r in nrows if r['uncertain']]}")
    with open(ROOT / "results/sim_sex_f0_speakers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows + nrows)

    ncall = {r["speaker"]: r for r in nrows}
    conv = []
    for r in csv.DictReader(open(ROOT / "data/english_sim/conversations.tsv"), delimiter="\t"):
        nsp = r["nepali_speakers"].split()
        calls = [ncall[s]["call"] for s in nsp]
        mix = f"{calls.count('F')}F{calls.count('M')}M"
        twin = f"{r['sexes'].count('F')}F{r['sexes'].count('M')}M"
        kind = "exact" if mix == twin else ("same_sex" if len(set(calls)) == 1 else "mixed_diff")
        unc = [s for s in nsp if ncall[s]["uncertain"]]
        # does the category survive flipping any uncertain speaker?
        kinds = set()
        for flip in range(2 ** len(unc)):
            c2 = [("M" if c == "F" else "F") if s in unc and (flip >> unc.index(s)) & 1 else c for s, c in zip(nsp, calls)]
            m2 = f"{c2.count('F')}F{c2.count('M')}M"
            kinds.add("exact" if m2 == twin else ("same_sex" if len(set(c2)) == 1 else "mixed_diff"))
        pos_agree = sum(a == b for a, b in zip(calls, r["sexes"]))
        conv.append({"uri": r["nepali_twin"], "n_speakers": len(nsp), "nepali_speakers": " ".join(nsp),
                     "nepali_f0": " ".join(f"{ncall[s]['median_f0']:.0f}" for s in nsp), "nepali_mix": mix,
                     "twin_mix": twin, "match": kind, "uncertain_speakers": len(unc),
                     "match_robust_to_uncertain": int(len(kinds) == 1), "same_sex_per_slot": pos_agree})
    with open(ROOT / "results/sim_sex_mix_conversations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(conv[0]))
        w.writeheader()
        w.writerows(conv)
    for c in conv:
        print(f"  {c['uri']:16} nep {c['nepali_mix']:5} ({c['nepali_f0']:15}) twin {c['twin_mix']:5} {c['match']:10} "
              f"uncertain {c['uncertain_speakers']} robust {c['match_robust_to_uncertain']}")
    for k in ("exact", "mixed_diff", "same_sex"):
        print(f"{k}: {sum(c['match'] == k for c in conv)} conversations "
              f"(by size {[sum(c['match'] == k and c['n_speakers'] == n for c in conv) for n in (2, 3, 4)]})")
    print(f"conversations with an uncertain speaker: {sum(c['uncertain_speakers'] > 0 for c in conv)}, "
          f"category changes if an uncertain speaker flips: {sum(not c['match_robust_to_uncertain'] for c in conv)}")

    uris = [c["uri"] for c in conv]
    subsets = {"full": [True] * len(conv), "exact_match": [c["match"] == "exact" for c in conv],
               "mixed_sex": [c["match"] != "same_sex" for c in conv],
               "exact_match_robust": [c["match"] == "exact" and bool(c["match_robust_to_uncertain"]) for c in conv],
               "mixed_sex_robust": [c["match"] != "same_sex" and bool(c["match_robust_to_uncertain"]) for c in conv]}
    out = []
    for cond, (nds, eds) in CONDITIONS.items():
        for s in SYSTEMS:
            n = {r["uri"]: float(r["der_no_collar"]) for r in csv.DictReader(open(ROOT / f"results/raw/scores_{s}_{nds}.csv"))}
            e = {r["uri"]: float(r["der_no_collar"]) for r in csv.DictReader(open(ROOT / f"results/raw/scores_{s}_{eds}.csv"))}
            dall = np.array([n[u] - e[u.replace("nepsim_", "engsim_")] for u in uris])
            for name, mask in subsets.items():
                d = dall[np.array(mask, dtype=bool)]
                idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
                lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
                out.append({"condition": cond, "config": LABEL[s], "system": s, "subset": name, "n": len(d),
                            "mean_diff": round(float(d.mean()), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
    with open(ROOT / "results/sim_sex_mix_pair.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)
    for cond in CONDITIONS:
        print(f"\n{cond}: mean d [95% CI], DER points")
        for s in SYSTEMS:
            cells = [r for r in out if r["condition"] == cond and r["system"] == s]
            print(f"  {LABEL[s]:4} " + " | ".join(f"{r['subset']} n={r['n']} {100 * r['mean_diff']:+.1f} "
                                                 f"[{100 * r['ci_lo']:+.1f}, {100 * r['ci_hi']:+.1f}]" for r in cells))


if __name__ == "__main__":
    main()
