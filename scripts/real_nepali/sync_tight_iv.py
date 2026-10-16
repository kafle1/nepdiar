#!/usr/bin/env python3
"""IndicVoices calls (calls.txt): A-to-B sync check and a tighter second reference. Run with -I.
Energy detector on each side's own 8 kHz channel: 25 ms frames, 10 ms hop, a frame is speech when it is
at or above -50 dBFS and within 30 dB of that side's 95th-percentile frame (same rule as ref_quality_iv.py).
Sync: (1) waveform cross-correlation of A and B within +-1 s (a crosstalk or echo peak would pin the offset);
(2) lag scan of +-2 s for the overlap between A's and B's speech, with energy masks and with reference
masks; turn-taking talk should overlap least near the true offset. (3) floor-transfer offsets at lag 0.
Tight reference: each side's reference segments intersected with its energy mask, gaps under 0.25 s
bridged, pieces under 0.1 s dropped. Writes rttm_tight/, sync.csv and tight.csv next to calls.txt."""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

D = Path(__file__).resolve().parents[2] / "data/nepali_real/indicvoices"
HOP, WIN, MAXLAG = 0.010, 0.025, 200


def frames_db(x, sr):
    w, h = int(WIN * sr), int(HOP * sr)
    n = 1 + (len(x) - w) // h
    fr = np.lib.stride_tricks.sliding_window_view(x, w)[::h][:n]
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-10)


def mask_from(segs, n):
    t = np.arange(n) * HOP + WIN / 2
    m = np.zeros(n, bool)
    for a, b in segs:
        m |= (t >= a) & (t < b)
    return m


def segs_from(m, gap=0.25, minlen=0.10):
    d = np.diff(np.r_[0, m.astype(int), 0])
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    out = []
    for a, b in zip(on * HOP, off * HOP):
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a >= minlen]


def lag_overlap(a, b):
    """overlap of a with b shifted by k frames (b delayed by k), over min speech, for k in -MAXLAG..MAXLAG."""
    ks = np.arange(-MAXLAG, MAXLAG + 1)
    n = len(a)
    o = []
    for k in ks:
        if k >= 0:
            v = (a[k:] & b[:n - k]).sum()
        else:
            v = (a[:n + k] & b[-k:]).sum()
        o.append(v)
    return ks * HOP, np.array(o) / max(1, min(a.sum(), b.sum()))


def fto(a_segs, b_segs):
    """floor-transfer offsets: each turn start minus the other side's last end (negative = starts in overlap)."""
    out = []
    for mine, other in ((a_segs, b_segs), (b_segs, a_segs)):
        for s, _ in mine:
            prev = [e for st, e in other if st < s]
            if prev:
                out.append(s - max(prev))
    return out


def main():
    uris = [u.strip() for u in (D / "calls.txt").read_text().splitlines() if u.strip()]
    src = {r["uri"]: r["source"] for r in csv.DictReader(open(D / "stats.csv"))}
    (D / "rttm_tight").mkdir(exist_ok=True)
    srows, trows, ftos = [], [], []
    for u in uris:
        ref = {"side0": [], "side1": []}
        for line in open(D / "rttm" / f"{u}.rttm"):
            f = line.split()
            ref[f[7]].append((float(f[3]), float(f[3]) + float(f[4])))
        x = {}
        for s in ref:
            y, sr = sf.read(str(D / src[u] / f"{u}_{s[-1]}.wav"), dtype="float64", always_2d=True)
            x[s] = y[:, 0]
        n = min(len(x["side0"]), len(x["side1"]))
        a, b = x["side0"][:n], x["side1"][:n]
        # (1) waveform cross-correlation within +-1 s
        L = 1 << int(np.ceil(np.log2(2 * n)))
        r = np.fft.irfft(np.fft.rfft(a, L) * np.conj(np.fft.rfft(b, L)), L)
        r = np.r_[r[-sr:], r[:sr + 1]] / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12)
        lags = np.arange(-sr, sr + 1) / sr
        pk = int(np.abs(r).argmax())
        away = np.abs(lags - lags[pk]) > 0.05
        prom = float(np.abs(r[pk]) / (np.percentile(np.abs(r[away]), 99) + 1e-12))
        # energy masks and tight reference
        em, rm, tight = {}, {}, {}
        for s in ref:
            db = frames_db(x[s], sr)
            p95 = np.percentile(db, 95)
            em[s] = (db >= -50) & (db >= p95 - 30)
            rm[s] = mask_from(ref[s], len(db))
            tight[s] = segs_from(em[s] & rm[s])
        m = min(len(em["side0"]), len(em["side1"]))
        lg, oe = lag_overlap(em["side0"][:m], em["side1"][:m])
        _, orf = lag_overlap(rm["side0"][:m], rm["side1"][:m])
        z = MAXLAG
        srows.append({"uri": u, "xcorr_peak": round(float(np.abs(r[pk])), 4), "xcorr_lag_s": round(float(lags[pk]), 4),
                      "xcorr_prominence": round(prom, 2),
                      "energy_ovl_lag0": round(float(oe[z]), 4), "energy_ovl_min": round(float(oe.min()), 4),
                      "energy_best_lag_s": round(float(lg[oe.argmin()]), 2),
                      "ref_ovl_lag0": round(float(orf[z]), 4), "ref_ovl_min": round(float(orf.min()), 4),
                      "ref_best_lag_s": round(float(lg[orf.argmin()]), 2)})
        ftos += fto(tight["side0"], tight["side1"])
        with open(D / "rttm_tight" / f"{u}.rttm", "w") as f:
            for s in ("side0", "side1"):
                for st, en in tight[s]:
                    f.write(f"SPEAKER {u} 1 {st:.3f} {en - st:.3f} <NA> <NA> {s} <NA> <NA>\n")

        def so(segs):
            ma, mb = mask_from(segs["side0"], m), mask_from(segs["side1"], m)
            return (ma | mb).sum() * HOP, (ma & mb).sum() * HOP
        sp0, ov0 = so(ref)
        sp1, ov1 = so(tight)
        trows.append({"uri": u, "speech_s": round(sp0, 2), "overlap_s": round(ov0, 2),
                      "tight_speech_s": round(sp1, 2), "tight_overlap_s": round(ov1, 2)})
    for name, rows in (("sync.csv", srows), ("tight.csv", trows)):
        with open(D / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def q(v):
        v = np.asarray(v, float)
        return f"median {np.median(v):.3f}, p10 {np.percentile(v, 10):.3f}, p90 {np.percentile(v, 90):.3f}, min {v.min():.3f}, max {v.max():.3f}"
    c = lambda k: [r[k] for r in srows]  # noqa: E731
    print(f"{len(uris)} calls")
    print("waveform xcorr |peak|:", q(c("xcorr_peak")), "| prominence:", q(c("xcorr_prominence")))
    strong = [r for r in srows if r["xcorr_prominence"] >= 5]
    print(f"calls with a crosstalk peak (prominence >= 5): {len(strong)}, their lags: {[r['xcorr_lag_s'] for r in strong]}")
    for k in ("energy", "ref"):
        bl = np.array(c(f"{k}_best_lag_s"))
        gap = np.array(c(f"{k}_ovl_lag0")) - np.array(c(f"{k}_ovl_min"))
        print(f"{k} masks: overlap at lag 0 {q(c(k + '_ovl_lag0'))}; best lag {q(bl)}; "
              f"|best lag| <= 0.1 s: {(np.abs(bl) <= 0.1).sum()}; lag-0 overlap within 0.02 of the minimum: {(gap <= 0.02).sum()}; "
              f"flagged (|best lag| > 0.1 s and lag 0 more than 0.02 above minimum): {((np.abs(bl) > 0.1) & (gap > 0.02)).sum()}")
    f = np.array(ftos)
    print(f"floor-transfer offsets at lag 0 (tight ref, {len(f)} turn starts): median {np.median(f):.3f} s, "
          f"share starting inside the other's speech {np.mean(f < 0):.3f}, share inside by more than 1 s {np.mean(f < -1):.3f}")
    t = {k: np.array([r[k] for r in trows]) for k in trows[0] if k != "uri"}
    print(f"current ref: speech {t['speech_s'].sum() / 3600:.3f} h, overlap {t['overlap_s'].sum() / 60:.2f} min, "
          f"overlap/speech {t['overlap_s'].sum() / t['speech_s'].sum():.4f}")
    print(f"tight ref:   speech {t['tight_speech_s'].sum() / 3600:.3f} h, overlap {t['tight_overlap_s'].sum() / 60:.2f} min, "
          f"overlap/speech {t['tight_overlap_s'].sum() / t['tight_speech_s'].sum():.4f}")
    print(f"pooled shrink: speech {1 - t['tight_speech_s'].sum() / t['speech_s'].sum():.4f}, "
          f"overlap {1 - t['tight_overlap_s'].sum() / t['overlap_s'].sum():.4f}; per call median shrink: speech "
          f"{np.median(1 - t['tight_speech_s'] / t['speech_s']):.4f}, overlap "
          f"{np.median(1 - t['tight_overlap_s'] / np.maximum(t['overlap_s'], 1e-9)):.4f}; per call median overlap/speech "
          f"{np.median(t['overlap_s'] / t['speech_s']):.4f} -> {np.median(t['tight_overlap_s'] / t['tight_speech_s']):.4f}")


if __name__ == "__main__":
    main()
