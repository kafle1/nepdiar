#!/usr/bin/env python3
"""IndicVoices Nepali "Conversation" calls to two-speaker diarization data.

Each call is two files, <uuid>_0 and <uuid>_1, one per phone side. Keeps calls that have both
sides, sums the two sides into one track at the native rate, resamples to 16 kHz mono for the
systems, and merges each side's timed transcript segments into one reference (speaker = side).
Checks per call: the two sides have the same length, each side's segments sit on that side's
speech energy (lag of best match between segment mask and energy, in-vs-out energy gap), and how
loud the other side leaks in.

run: ./.venv/bin/python -I scripts/real_nepali/prep_indicvoices.py RAW_DIR [RAW_DIR ...]
Writes data/nepali_real/indicvoices/{wav,rttm,uem}, calls.txt, stats.csv, check.txt.
"""
import ast
import csv
import os
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/nepali_real/indicvoices"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")  # results were made with ffmpeg 4, ffprobe must sit beside it
HOP = 0.01


def segs_of(d):
    v = d["verbatim"]
    v = ast.literal_eval(v) if isinstance(v, str) else v
    return [(float(s["start"]), float(s["end"]), s.get("speaker_id")) for s in v]


def merge(segs):
    out = []
    for st, en in sorted(segs):
        if out and st <= out[-1][1]:
            out[-1][1] = max(out[-1][1], en)
        else:
            out.append([st, en])
    return out


def frame_db(x, sr):
    hop, win = int(sr * HOP), int(sr * 0.025)
    n = max(1, (len(x) - win) // hop + 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = (x[np.minimum(idx, len(x) - 1)] ** 2).mean(1)
    return 10 * np.log10(e + 1e-10)


def mask_of(segs, n):
    m = np.zeros(n, bool)
    for st, en in segs:
        m[int(st / HOP):int(en / HOP)] = True
    return m


def best_lag(mask, db, max_lag=200):
    """lag (frames) that best lines up a side's segment mask with its own energy; 0 means aligned."""
    z = (db - db.mean()) / (db.std() + 1e-9)
    m = mask.astype(float) - mask.mean()
    best, arg = -9, 0
    for lag in range(-max_lag, max_lag + 1):
        c = np.dot(np.roll(m, lag), z) / len(z)
        if c > best:
            best, arg = c, lag
    return arg * HOP, float(np.dot(m, z) / len(z))


def main():
    raws = [Path(p) for p in sys.argv[1:]]
    for sub in ("wav", "rttm", "uem", "wav8k"):
        (D / sub).mkdir(parents=True, exist_ok=True)
    files = {}
    for r in raws:
        for j in r.glob("*.json"):
            if j.name.startswith("_"):
                continue
            m = re.match(r"(.+)_([01])$", j.stem)
            if m and (j.with_suffix(".wav")).exists():
                files.setdefault(m.group(1), {})[m.group(2)] = j
    singles = sorted(k for k, v in files.items() if len(v) == 1)
    calls = sorted(k for k, v in files.items() if len(v) == 2)
    stats, problems, kept = [], [], []
    for uri in calls:
        meta = {s: json.loads(files[uri][s].read_text()) for s in "01"}
        aud, sr = {}, {}
        for s in "01":
            aud[s], sr[s] = sf.read(str(files[uri][s].with_suffix(".wav")), dtype="float32", always_2d=True)
            if aud[s].shape[1] != 1:
                problems.append(f"{uri}_{s}: {aud[s].shape[1]} channels, using the mean")
            aud[s] = aud[s].mean(1)
        if sr["0"] != sr["1"]:
            problems.append(f"{uri}: EXCLUDED, sample rates differ {sr}")
            continue
        fs = sr["0"]
        len_diff = (len(aud["0"]) - len(aud["1"])) / fs
        n = max(len(aud["0"]), len(aud["1"]))
        # offset rule: both sides start at t=0 (the call start), a shorter side is padded with silence at the end
        x = {s: np.pad(aud[s], (0, n - len(aud[s]))) for s in "01"}
        dur = n / fs
        ref, row = {}, {"uri": uri, "sr": fs, "duration_s": round(dur, 3), "side_len_diff_s": round(len_diff, 3)}
        bad = False
        for s in "01":
            raw = segs_of(meta[s])
            ids = {i for _, _, i in raw}
            if len(ids) > 1:
                problems.append(f"{uri}_{s}: segment speaker ids {sorted(map(str, ids))} on one side")
            badseg = [(a, b) for a, b, _ in raw if a < 0 or b <= a or b > dur + 0.05]
            if badseg:
                problems.append(f"{uri}_{s}: {len(badseg)} segments negative, empty or past the end, e.g. {badseg[:2]}")
            ref[s] = merge([(max(0.0, a), min(b, dur)) for a, b, _ in raw if min(b, dur) > max(0.0, a)])
            if not ref[s]:
                problems.append(f"{uri}: EXCLUDED, side {s} has no segments")
                bad = True
                continue
            db = frame_db(x[s], fs)
            mk = mask_of(ref[s], len(db))
            lag, corr = best_lag(mk, db)
            gap = float(np.median(db[mk]) - np.median(db[~mk])) if (~mk).any() else float("nan")
            row[f"side{s}_segments"] = len(ref[s])
            row[f"side{s}_speech_s"] = round(sum(b - a for a, b in ref[s]), 3)
            row[f"side{s}_lag_s"] = round(lag, 2)
            row[f"side{s}_corr"] = round(corr, 3)
            row[f"side{s}_in_out_db"] = round(gap, 1)
            sm = np.repeat(mk, int(fs * HOP))[:n]
            sm = np.pad(sm, (0, n - len(sm)))
            row[f"side{s}_zero_in"] = round(float((x[s][sm] == 0).mean()), 3)
            row[f"side{s}_zero_out"] = round(float((x[s][~sm] == 0).mean()), 3) if (~sm).any() else float("nan")
            row[f"side{s}_decision"] = (ast.literal_eval(meta[s]["verification_report"])
                                        if isinstance(meta[s]["verification_report"], str)
                                        else meta[s]["verification_report"]).get("decision")
            row[f"side{s}_speaker_id"] = meta[s].get("speaker_id")
            row[f"side{s}_gender"] = meta[s].get("gender")
            if abs(lag) > 0.3:
                problems.append(f"{uri}_{s}: segments line up best with energy at a {lag:+.2f} s shift")
            if gap < 6:
                problems.append(f"{uri}_{s}: speech inside segments only {gap:.1f} dB above outside")
        if bad:
            continue
        # leak: on side s, energy during the other side's solo speech vs during its own solo speech
        for s, o in (("0", "1"), ("1", "0")):
            db = frame_db(x[s], fs)
            ms, mo = mask_of(ref[s], len(db)), mask_of(ref[o], len(db))
            row[f"side{s}_leak_db"] = (round(float(np.median(db[mo & ~ms]) - np.median(db[ms & ~mo])), 1)
                                       if (mo & ~ms).any() and (ms & ~mo).any() else float("nan"))
        if abs(len_diff) > 0.05:
            problems.append(f"{uri}: sides differ in length by {len_diff:+.3f} s, padded at the end")
        mix = x["0"] + x["1"]
        peak = float(np.abs(mix).max())
        if peak > 0.999:
            mix *= 0.999 / peak
        w8 = D / "wav8k" / f"{uri}.wav"
        sf.write(str(w8), mix, fs, subtype="PCM_16")
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(w8), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(D / "wav" / f"{uri}.wav")], check=True)
        with open(D / "rttm" / f"{uri}.rttm", "w") as f:
            for st, en, s in sorted((a, b, s) for s in "01" for a, b in ref[s]):
                f.write(f"SPEAKER {uri} 1 {st:.3f} {en - st:.3f} <NA> <NA> side{s} <NA> <NA>\n")
        (D / "uem" / f"{uri}.uem").write_text(f"{uri} 1 0.000 {dur:.3f}\n")
        both = mask_of(ref["0"], int(dur / HOP) + 1) & mask_of(ref["1"], int(dur / HOP) + 1)
        anyv = mask_of(ref["0"], int(dur / HOP) + 1) | mask_of(ref["1"], int(dur / HOP) + 1)
        row["speech_s"] = round(anyv.sum() * HOP, 2)
        row["overlap_s"] = round(both.sum() * HOP, 2)
        row["overlap_ratio"] = round(both.sum() / max(1, anyv.sum()), 4)
        row["source"] = files[uri]["0"].parent.name
        stats.append(row)
        kept.append(uri)
    (D / "calls.txt").write_text("\n".join(kept) + "\n")
    keys = list(dict.fromkeys(k for r in stats for k in r))
    with open(D / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(stats)
    tot = sum(r["duration_s"] for r in stats)
    sp, ov = sum(r["speech_s"] for r in stats), sum(r["overlap_s"] for r in stats)
    med = lambda k: float(np.nanmedian([r[f"side{s}_{k}"] for r in stats for s in "01"]))  # noqa: E731
    lines = [f"conversation files with a wav: {sum(len(v) for v in files.values())}, calls {len(files)}, "
             f"both sides {len(calls)}, one side only {len(singles)}",
             f"kept calls {len(kept)}, hours {tot / 3600:.3f}, by source {dict(Counter(r['source'] for r in stats))}",
             f"sample rates {dict(Counter(r['sr'] for r in stats))}",
             f"side length difference: max |diff| {max(abs(r['side_len_diff_s']) for r in stats):.3f} s",
             f"speech {sp / 3600:.3f} h ({sp / tot:.3f} of audio), overlap {ov / 60:.2f} min, overlap ratio {ov / sp:.4f}",
             f"energy check, median over sides: in-vs-out {med('in_out_db'):.1f} dB, best lag {med('lag_s'):.2f} s, "
             f"other-side leak {med('leak_db'):.1f} dB (own solo speech = 0)",
             f"exact-zero samples, median over sides: inside segments {med('zero_in'):.3f}, outside {med('zero_out'):.3f}",
             f"lag histogram: {dict(Counter(r[f'side{s}_lag_s'] for r in stats for s in '01').most_common(8))}",
             f"verification decisions: {dict(Counter(r[f'side{s}_decision'] for r in stats for s in '01'))}",
             f"distinct speaker ids {len({r[f'side{s}_speaker_id'] for r in stats for s in '01'})} over {2 * len(stats)} sides",
             f"problems ({len(problems)}):", *problems]
    (D / "check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
