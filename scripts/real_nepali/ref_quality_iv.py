#!/usr/bin/env python3
"""IndicVoices reference quality: per call side, the share of reference speech time that a frame energy
detector calls silence. Run with -I. Reads each side's own 8 kHz channel (raw json+wav pairs).
Detector: 25 ms frames, 10 ms hop, frame RMS in dBFS; a frame is silence if it is below an absolute
threshold (-50 dBFS, also -40 and -60 shown) or more than 30 dB under that side's 95th-percentile frame.
A frame counts as reference speech when its centre falls inside the side's merged reference segments.
Writes data/nepali_real/indicvoices/ref_quality.csv and prints a summary."""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

D = Path(__file__).resolve().parents[2] / "data/nepali_real/indicvoices"
ABS = (-40.0, -50.0, -60.0)
REL = 30.0


def main():
    uris = [u.strip() for u in (D / "calls.txt").read_text().splitlines() if u.strip()]
    src = {r["uri"]: r["source"] for r in csv.DictReader(open(D / "stats.csv"))}
    rows = []
    for u in uris:
        segs = {}
        for line in open(D / "rttm" / f"{u}.rttm"):
            f = line.split()
            segs.setdefault(f[7], []).append((float(f[3]), float(f[3]) + float(f[4])))
        for side in ("side0", "side1"):
            x, sr = sf.read(str(D / src[u] / f"{u}_{side[-1]}.wav"), dtype="float64", always_2d=True)
            x = x[:, 0]
            win, hop = int(0.025 * sr), int(0.010 * sr)
            n = max(0, 1 + (len(x) - win) // hop)
            fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
            rms = np.sqrt((fr ** 2).mean(1))
            db = 20 * np.log10(rms + 1e-10)
            t = (np.arange(n) * hop + win / 2) / sr
            sp = np.zeros(n, bool)
            for a, b in segs.get(side, []):
                sp |= (t >= a) & (t < b)
            if not sp.any():
                print(f"{u} {side}: no reference speech frames", file=sys.stderr)
                continue
            p95 = np.percentile(db[sp], 95)
            r = {"uri": u, "side": side, "speech_s": round(sp.sum() * hop / sr, 2), "p95_speech_dbfs": round(p95, 1)}
            for a in ABS:
                r[f"sil_share_abs{int(a)}"] = round(float((db[sp] < a).mean()), 4)
            r[f"sil_share_rel{int(REL)}"] = round(float((db[sp] < p95 - REL).mean()), 4)
            r["digital_zero_share"] = round(float((rms[sp] == 0).mean()), 4)
            rows.append(r)
    with open(D / "ref_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = sum(r["speech_s"] for r in rows)
    print(f"{len(rows)} sides of {len(uris)} calls, {tot / 3600:.3f} h reference speech")
    for k in [c for c in rows[0] if c.startswith(("sil_", "digital"))]:
        v = np.array([r[k] for r in rows])
        w8 = sum(r[k] * r["speech_s"] for r in rows) / tot
        print(f"{k:22} speech-weighted {w8:.4f}  per side median {np.median(v):.4f}  "
              f"p90 {np.percentile(v, 90):.4f}  max {v.max():.4f}  sides over 0.20: {(v > 0.2).sum()}")


if __name__ == "__main__":
    main()
