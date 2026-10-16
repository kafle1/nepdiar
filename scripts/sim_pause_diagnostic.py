#!/usr/bin/env python3
"""Post hoc, descriptive: is the Nepali minus English gap on the sim set made of quiet stretches inside
reference regions? No tests, sim_pair_test.csv is untouched. Writes results/sim_pause_diagnostic.csv.

1. pauses: per language, share of reference speech frames that are near-silent, on each speaker's clean
   track as mixed into wav/ (same gain and peak scale, no other speakers): under the utterance's own trim()
   threshold, and under -45 dBFS. Frames are trim()'s 25 ms windows every 10 ms, counted when the whole
   window sits in the placed region. Internal gaps: runs of sub-threshold frames inside a region, length
   10*L - 15 ms as trim() measures them, counted when 50 to 340 ms, per minute of reference speech.
2. pause_merged: every hypothesis gets same-speaker gaps under 0.3 s merged (the reference's own rule),
   est k, no collar, all six configs, both conditions.
3. quiet_removed: a UEM per conversation that drops every 10 ms cell where some reference speaker is
   under their utterance's trim threshold; S4 and S5 clean.
DER is pooled (summed components); d is the mean per-conversation DER difference with a percentile
bootstrap CI (default_rng(0), 10,000 resamples of the 30 pairs)."""
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
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

SR = 16000
HOP, WIN = SR // 100, SR // 40
SYSTEMS = ["sherpa_onnx", "wespeaker_spectral", "wespeaker_spectral_silero", "pyannote_style", "pyannote31", "community1"]
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
LANG = {"nepali": ("nepali_sim", "nepsim"), "english": ("english_sim", "engsim")}
OUT = ROOT / "results/sim_pause_diagnostic.csv"
QUIET_UEM = ROOT / "results/cache/sim_quiet_uem"


def frame_db(x):
    n = max(1, 1 + (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)


def trim_thr(x):
    db = frame_db(x)
    return max(db.max() - 35, np.percentile(db, 10) + 10)


def pauses(lang):
    data = ROOT / "data" / LANG[lang][0]
    reader = nep.ZipAudio(nep.SLR / "zips") if lang == "nepali" else eng.LibriAudio(ROOT / "data/librispeech")
    conv = nep.conversations(data)
    rows = nep.read_tsv(data / "placement.tsv")
    tot = q_thr = q_45 = gaps = 0
    thr_cache = {}
    quiet_cells = {}
    QUIET_UEM.joinpath(lang).mkdir(parents=True, exist_ok=True)
    for uri, (spks, total_ms) in conv.items():
        rs = [r for r in rows if r["uri"] == uri]
        tracks = nep.speaker_tracks(uri, spks, total_ms, rs, reader)
        mix = sum(tracks[k] for k in spks)
        peak = float(np.abs(mix).max())
        scale = 0.99 / peak if peak > 0.99 else 1.0
        dbs = {k: frame_db(tracks[k] * scale) for k in spks}
        quiet = np.zeros(len(next(iter(dbs.values()))), bool)
        ref = load_annotation(data / "rttm" / f"{uri}.rttm", uri)
        ref_starts = {round(s.start * 1000) for s, _ in ref.itertracks()}
        for r in rs:
            fid = r["slr54_file_id"] if "slr54_file_id" in r else r["librispeech_id"]
            if fid not in thr_cache:
                thr_cache[fid] = trim_thr(reader.read(fid).astype(np.float64))
            a, b, s, g = int(r["src_start_ms"]), int(r["src_end_ms"]), int(r["placed_start_ms"]), float(r["gain"])
            assert s in ref_starts, (uri, s)
            thr = thr_cache[fid] + 20 * np.log10(g * scale)
            i0, i1 = -(-s // 10), (s + b - a - 25) // 10  # frames whose 25 ms window is inside the region
            if i1 < i0:
                continue
            d = dbs[r["speaker"]][i0:i1 + 1]
            under = d < thr
            tot += len(d)
            q_thr += int(under.sum())
            q_45 += int((d < -45).sum())
            quiet[i0:i1 + 1] |= under
            # internal runs: sub-threshold runs that do not touch either end of the region
            edges = np.diff(np.r_[0, under.astype(int), 0])
            for st, en in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                if st > 0 and en < len(under) and 50 <= 10 * (en - st) - 15 <= 340:
                    gaps += 1
        # uem: drop the 10 ms cell centred on each quiet frame (frame i centre is 10 i + 12.5 ms)
        keep, t = [], 0.0
        for i in np.where(quiet)[0]:
            c0 = (10 * i + 7.5) / 1000
            if c0 > t:
                keep.append((t, c0))
            t = max(t, c0 + 0.01)
        end = total_ms / 1000
        if end > t:
            keep.append((t, end))
        with open(QUIET_UEM / lang / f"{uri}.uem", "w") as f:
            f.writelines(f"{uri} 1 {a:.4f} {b:.4f}\n" for a, b in keep)
    speech_min = tot * 10 / 60000
    return {"view": "pauses", "lang": lang, "ref_speech_min": round(speech_min, 2),
            "near_silent_under_trim_thr_pct": round(100 * q_thr / tot, 2),
            "near_silent_under_-45dBFS_pct": round(100 * q_45 / tot, 2),
            "internal_gaps_50_340ms_per_speech_min": round(gaps / speech_min, 2)}


def score(job):
    lang, ds, system, view = job
    data = ROOT / "data" / LANG[lang][0]
    uem_dir = QUIET_UEM / lang if view == "quiet_removed" else data / "uem"
    out = {}
    for uri in (data / "conversations.txt").read_text().split():
        ref = load_annotation(data / "rttm" / f"{uri}.rttm", uri)
        hyp = load_annotation(ROOT / "results/raw" / ds / f"{system}_rttm" / f"{uri}.rttm", uri)
        if view == "pause_merged":
            hyp = hyp.support(collar=0.3)
        uem = Timeline([Segment(float(p[2]), float(p[3])) for p in (ln.split() for ln in open(uem_dir / f"{uri}.uem"))], uri=uri)
        d = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=uem, detailed=True)
        out[uri.split("_", 1)[1]] = (d["total"], d["missed detection"], d["false alarm"], d["confusion"])
    return job, out


def boot(d):
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    with Pool(2) as p:
        prow = p.map(pauses, ["nepali", "english"])
    for r in prow:
        print(r, flush=True)
    jobs = []
    for cond, sfx in (("clean", ""), ("reverb_noise", "_rn")):
        for system in SYSTEMS:
            for view in ("as_scored", "pause_merged") + (("quiet_removed",) if cond == "clean" and system in ("pyannote31", "community1") else ()):
                jobs += [(lang, LANG[lang][0] + sfx, system, view) for lang in LANG]
    with Pool(3) as p:
        res = dict(p.map(score, jobs))
    rows = list(prow)
    for cond, sfx in (("clean", ""), ("reverb_noise", "_rn")):
        for system in SYSTEMS:
            for view in ("as_scored", "pause_merged", "quiet_removed"):
                key = {lang: (lang, LANG[lang][0] + sfx, system, view) for lang in LANG}
                if key["nepali"] not in res:
                    continue
                row = {"view": view, "condition": cond, "system": LABEL[system]}
                per = {}
                for lang in LANG:
                    r = res[key[lang]]
                    T, M, F, C = (sum(v[i] for v in r.values()) for i in range(4))
                    row.update({f"{lang}_der": round(100 * (M + F + C) / T, 2), f"{lang}_miss": round(100 * M / T, 2),
                                f"{lang}_fa": round(100 * F / T, 2), f"{lang}_conf": round(100 * C / T, 2)})
                    per[lang] = {k: (v[1] + v[2] + v[3]) / v[0] for k, v in r.items()}
                d = np.array([per["nepali"][k] - per["english"][k] for k in sorted(per["nepali"])]) * 100
                lo, hi = boot(d)
                row.update({"mean_d": round(float(d.mean()), 2), "d_ci_lo": round(lo, 2), "d_ci_hi": round(hi, 2)})
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
