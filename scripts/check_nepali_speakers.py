#!/usr/bin/env python3
"""Trim and speaker-check OpenSLR 54 utterances before they go into simulated conversations.

1. Every utterance of a candidate speaker (enough files in the downloaded zip parts) gets an energy
   trim: 25 ms frames every 10 ms, speech where a frame is within 35 dB of the loudest frame and
   10 dB above the quietest tenth, gaps under 0.3 s bridged, bits under 0.2 s dropped, 20 ms pad.
   Energy, not Silero or WebRTC, so the reference isn't built by a VAD one of the tested systems uses.
2. A WeSpeaker embedding (the System 2 embedder) of each utterance's trimmed speech.
3. Checks: an utterance is an outlier when it sits closer to another speaker's centroid than to its
   own (own centroid without it). A speaker ID is doubtful when too many of its utterances are
   outliers, when its utterances split into two sizeable groups that look like two people, or when its
   centroid looks like another ID's centroid. Thresholds are printed and written to summary.json.

usage: nice -n 10 .venv/bin/python scripts/check_nepali_speakers.py [--workers 6]
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from run_wespeaker_spectral import EMB_MODEL, Embedder

SR = 16000
SLR = ROOT / "data/slr54"
OUT = ROOT / "data/nepali_sim/speaker_check"
MIN_FILES = 50  # files a speaker needs in the downloaded parts to be a candidate
MIN_UTT_SPEECH_MS = 1000
MIN_SPEAKER_SPEECH_S = 150  # a 2-speaker conversation needs about 130 s from each
LEVEL_DBFS = -26.0
OUTLIER_FRAC_MAX = 0.10
SPLIT_MIN_SHARE = 0.25


def trim(x):
    hop, win = SR // 100, SR // 40
    n = max(1, 1 + (len(x) - win) // hop)
    db = np.array([10 * np.log10(np.mean(x[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)])
    thr = max(db.max() - 35, np.percentile(db, 10) + 10)
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


def work(job):
    spk, items = job  # items: (file_id, zip name, member)
    emb = Embedder(EMB_MODEL)
    zips, rows, vecs = {}, [], []
    for fid, zname, member in items:
        zf = zips.setdefault(zname, zipfile.ZipFile(SLR / "zips" / zname))
        x, sr = sf.read(io.BytesIO(zf.read(member)), dtype="float32", always_2d=True)
        x = x[:, 0]
        regs = trim(x) if sr == SR else []
        speech = np.concatenate([x[a * 16:b * 16] for a, b in regs]) if regs else np.zeros(0, np.float32)
        sp_ms = sum(b - a for a, b in regs)
        rms = float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))) if len(speech) else 0.0
        clipped = float(np.mean(np.abs(x) >= 0.999))
        v = emb(speech) if sp_ms >= MIN_UTT_SPEECH_MS else None
        rows.append({"file_id": fid, "speaker": spk, "sr": sr, "file_ms": len(x) * 1000 // sr, "speech_ms": sp_ms,
                     "regions_ms": json.dumps(regs), "gain": 10 ** (LEVEL_DBFS / 20) / rms if rms > 0 else 0.0,
                     "clipped_frac": clipped, "has_emb": int(v is not None)})
        vecs.append(v if v is not None else np.zeros(256, np.float32))
    return rows, np.stack(vecs)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    where = {}
    for z in sorted((SLR / "zips").glob("asr_nepali_*.zip")):
        for n in zipfile.ZipFile(z).namelist():
            if n.endswith(".flac"):
                where[Path(n).stem] = (z.name, n)
    by_spk = defaultdict(list)
    for line in (SLR / "utt_spk_text.tsv").read_text().splitlines():
        fid, spk = line.split("\t")[:2]
        if fid in where:
            by_spk[spk].append((fid, *where[fid]))
    cands = sorted(s for s, v in by_spk.items() if len(v) >= MIN_FILES)
    print(f"{len(where)} files in {len(by_spk)} speakers in the downloaded parts, {len(cands)} speakers with >= {MIN_FILES} files")

    cache = OUT / "embeddings.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        E = d["emb"]
        rows = list(csv.DictReader(open(OUT / "utterances_raw.tsv"), delimiter="\t"))
    else:
        rows, embs = [], []
        with Pool(args.workers) as pool:
            for i, (r, e) in enumerate(pool.imap(work, [(s, sorted(by_spk[s])) for s in cands])):
                rows += r
                embs.append(e)
                if i % 25 == 0:
                    print(f"  {i + 1}/{len(cands)} speakers embedded", flush=True)
        E = np.concatenate(embs)
        np.savez(cache, emb=E)
        with open(OUT / "utterances_raw.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    spk_of = np.array([r["speaker"] for r in rows])
    ok = np.array([r["has_emb"] == "1" or r["has_emb"] == 1 for r in rows]) & \
        np.array([float(r["clipped_frac"]) < 0.001 for r in rows]) & np.array([int(r["sr"]) == SR for r in rows])
    E = unit(E)
    spks = sorted(set(spk_of[ok]))
    idx = {s: np.where(ok & (spk_of == s))[0] for s in spks}
    C = unit(np.stack([E[idx[s]].sum(0) for s in spks]))
    S_of = {s: i for i, s in enumerate(spks)}

    # utterance outliers: own leave-one-out centroid vs the nearest other centroid
    outlier = np.zeros(len(rows), bool)
    own_cos = np.full(len(rows), np.nan)
    for s in spks:
        ii = idx[s]
        tot = E[ii].sum(0)
        loo = unit(tot[None, :] - E[ii])
        own = (E[ii] * loo).sum(1)
        other = E[ii] @ C.T
        other[:, S_of[s]] = -1
        own_cos[ii] = own
        outlier[ii] = own < other.max(1)

    # same-person reference: random split halves of each speaker; cross reference: centroid pairs
    rng = np.random.default_rng(0)
    halves = []
    for s in spks:
        ii = rng.permutation(idx[s])
        h = len(ii) // 2
        halves.append(float(unit(E[ii[:h]].sum(0)) @ unit(E[ii[h:]].sum(0))))
    halves = np.array(halves)
    cross = C @ C.T
    iu = np.triu_indices(len(spks), 1)
    same_lo = float(np.percentile(halves, 5))  # below this two groups of one ID look like different people
    cross_hi = float(np.percentile(cross[iu], 99.9))
    print(f"split-half same-ID cosine: median {np.median(halves):.3f}, 5th pct {same_lo:.3f}, min {halves.min():.3f}")
    print(f"different-ID centroid cosine: median {np.median(cross[iu]):.3f}, 99.9th pct {cross_hi:.3f}, max {cross[iu].max():.3f}")

    reasons = defaultdict(list)
    stats = {}
    for s in spks:
        ii = idx[s]
        frac = float(outlier[ii].mean())
        if frac > OUTLIER_FRAC_MAX:
            reasons[s].append(f"outlier_frac={frac:.2f}")
        lab = fcluster(linkage(E[ii], "average", metric="cosine"), 2, "maxclust")
        share = min((lab == 1).mean(), (lab == 2).mean())
        if share >= SPLIT_MIN_SHARE:
            c12 = float(unit(E[ii][lab == 1].sum(0)) @ unit(E[ii][lab == 2].sum(0)))
            if c12 < same_lo:
                reasons[s].append(f"two_groups share={share:.2f} cos={c12:.3f}")
        stats[s] = (frac, len(ii))
    # two IDs that look like one person: closer than the weakest 5% of same-ID halves
    dup_pairs = [(spks[a], spks[b], float(cross[a, b])) for a, b in zip(*iu) if cross[a, b] >= same_lo]
    for a, b, c in dup_pairs:
        drop = a if len(idx[a]) < len(idx[b]) else b
        reasons[drop].append(f"looks_like={b if drop == a else a} cos={c:.3f}")

    use = ok & ~outlier
    speech_s = {s: sum(int(rows[i]["speech_ms"]) for i in idx[s] if use[i]) / 1000 for s in spks}
    kept, little = [], []
    for s in spks:
        if reasons[s]:
            continue
        (kept if speech_s[s] >= MIN_SPEAKER_SPEECH_S else little).append(s)

    with open(OUT / "utterances.tsv", "w", newline="") as f:
        fields = list(rows[0]) + ["own_cos", "outlier", "use"]
        w = csv.DictWriter(f, fields, delimiter="\t")
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "own_cos": f"{own_cos[i]:.4f}", "outlier": int(outlier[i]),
                        "use": int(bool(use[i]) and r["speaker"] in kept)})
    (OUT / "kept_speakers.txt").write_text("".join(f"{s}\t{speech_s[s]:.1f}\n" for s in kept))
    with open(OUT / "dropped_speakers.tsv", "w") as f:
        f.write("speaker\treason\n")
        for s in spks:
            if reasons[s]:
                f.write(f"{s}\t{'; '.join(reasons[s])}\n")
            elif s in little:
                f.write(f"{s}\tunder {MIN_SPEAKER_SPEECH_S} s usable speech ({speech_s[s]:.0f} s)\n")
    summ = {
        "candidates": len(cands), "checked": len(spks), "utterances": len(rows),
        "utterances_unusable_short_clipped_or_wrong_rate": int((~ok).sum()),
        "utterance_outliers": int(outlier[ok].sum()), "utterance_outlier_rate": float(outlier[ok].mean()),
        "split_half_same_cos_median": float(np.median(halves)), "split_half_same_cos_p5": same_lo,
        "cross_centroid_cos_median": float(np.median(cross[iu])), "cross_centroid_cos_p999": cross_hi,
        "cross_centroid_cos_max": float(cross[iu].max()),
        "dropped_outlier_frac": sum(any(r.startswith("outlier") for r in v) for v in reasons.values()),
        "dropped_two_groups": sum(any(r.startswith("two_groups") for r in v) for v in reasons.values()),
        "dropped_looks_like_other_id": sum(any(r.startswith("looks_like") for r in v) for v in reasons.values()),
        "dropped_doubtful_total": sum(1 for v in reasons.values() if v),
        "dropped_too_little_speech": len(little), "kept": len(kept),
        "thresholds": {"outlier_frac_max": OUTLIER_FRAC_MAX, "split_min_share": SPLIT_MIN_SHARE,
                       "same_person_cos": same_lo, "min_speaker_speech_s": MIN_SPEAKER_SPEECH_S,
                       "min_utt_speech_ms": MIN_UTT_SPEECH_MS, "min_files": MIN_FILES},
        "outlier_frac_per_speaker_median": float(np.median([v[0] for v in stats.values()])),
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
