#!/usr/bin/env python3
"""Post hoc: the check_nepali_speakers.py speaker check run on the LibriSpeech speakers of the English twin.

Same energy trim, same WeSpeaker ResNet34-LM embedding of each utterance's trimmed speech, same rule
(outlier fraction > 0.10, two groups with share >= 0.25 below the same-person cosine, two IDs closer than
it, under 150 s usable speech). Nothing here changes which speakers the twin uses.

The same-person cosine is the 5th percentile of the pool's split-half cosines, so it depends on the pool.
Three readings are written:
  all80_own       the 80 dev-clean + test-clean speakers (the twin's candidate pool), own threshold
  used70_own      only the 70 speakers the twin uses, own threshold
  all80_nepthr    the 80, with the threshold fixed at the Nepali run's value (summary.json)

usage: nice -n 10 .venv/bin/python scripts/check_english_speakers.py [--workers 3]
Writes data/english_sim/speaker_check/ and results/english_speaker_check.csv.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from check_nepali_speakers import (MIN_SPEAKER_SPEECH_S, MIN_UTT_SPEECH_MS, OUTLIER_FRAC_MAX, SPLIT_MIN_SHARE,
                                   SR, trim, unit)
from run_wespeaker_spectral import EMB_MODEL, Embedder

LIBRI = ROOT / "data/librispeech/LibriSpeech"
OUT = ROOT / "data/english_sim/speaker_check"
NEP_SUMMARY = ROOT / "data/nepali_sim/speaker_check/summary.json"


def work(job):
    spk, paths = job
    emb = Embedder(EMB_MODEL)
    rows, vecs = [], []
    for p in paths:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x[:, 0]
        regs = trim(x) if sr == SR else []
        speech = np.concatenate([x[a * 16:b * 16] for a, b in regs]) if regs else np.zeros(0, np.float32)
        sp_ms = sum(b - a for a, b in regs)
        v = emb(speech) if sp_ms >= MIN_UTT_SPEECH_MS else None
        rows.append({"file_id": p.stem, "speaker": spk, "sr": sr, "speech_ms": sp_ms,
                     "clipped_frac": float(np.mean(np.abs(x) >= 0.999)), "has_emb": int(v is not None)})
        vecs.append(v if v is not None else np.zeros(256, np.float32))
    return rows, np.stack(vecs)


def check(rows, E, pool, same_fixed=None):
    """check_nepali_speakers.py's decision rule on the speakers in `pool`. Returns (reasons, kept, little, info)."""
    spk_of = np.array([r["speaker"] for r in rows])
    ok = np.array([int(r["has_emb"]) == 1 for r in rows]) & np.array([float(r["clipped_frac"]) < 0.001 for r in rows]) \
        & np.array([int(r["sr"]) == SR for r in rows]) & np.isin(spk_of, list(pool))
    E = unit(E)
    spks = sorted(set(spk_of[ok]))
    idx = {s: np.where(ok & (spk_of == s))[0] for s in spks}
    C = unit(np.stack([E[idx[s]].sum(0) for s in spks]))
    S_of = {s: i for i, s in enumerate(spks)}
    outlier = np.zeros(len(rows), bool)
    for s in spks:
        ii = idx[s]
        loo = unit(E[ii].sum(0)[None, :] - E[ii])
        own = (E[ii] * loo).sum(1)
        other = E[ii] @ C.T
        other[:, S_of[s]] = -1
        outlier[ii] = own < other.max(1)
    rng = np.random.default_rng(0)
    halves = []
    for s in spks:
        ii = rng.permutation(idx[s])
        h = len(ii) // 2
        halves.append(float(unit(E[ii[:h]].sum(0)) @ unit(E[ii[h:]].sum(0))))
    halves = np.array(halves)
    cross = C @ C.T
    iu = np.triu_indices(len(spks), 1)
    same_lo = float(np.percentile(halves, 5)) if same_fixed is None else same_fixed
    reasons = defaultdict(list)
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
    for a, b in zip(*iu):
        if cross[a, b] >= same_lo:
            drop = spks[a] if len(idx[spks[a]]) < len(idx[spks[b]]) else spks[b]
            reasons[drop].append(f"looks_like={spks[b] if drop == spks[a] else spks[a]} cos={cross[a, b]:.3f}")
    use = ok & ~outlier
    speech_s = {s: sum(int(rows[i]["speech_ms"]) for i in idx[s] if use[i]) / 1000 for s in spks}
    kept = [s for s in spks if not reasons[s] and speech_s[s] >= MIN_SPEAKER_SPEECH_S]
    little = [s for s in spks if not reasons[s] and speech_s[s] < MIN_SPEAKER_SPEECH_S]
    info = {"checked": len(spks), "utterances": int(ok.sum()), "utterance_outliers": int(outlier[ok].sum()),
            "same_person_cos": round(same_lo, 4), "split_half_median": round(float(np.median(halves)), 4),
            "cross_median": round(float(np.median(cross[iu])), 4), "cross_max": round(float(cross[iu].max()), 4),
            "dropped_outlier_frac": sum(any(r.startswith("outlier") for r in reasons[s]) for s in spks),
            "dropped_two_groups": sum(any(r.startswith("two_groups") for r in reasons[s]) for s in spks),
            "dropped_looks_like": sum(any(r.startswith("looks_like") for r in reasons[s]) for s in spks),
            "dropped_doubtful": sum(1 for s in spks if reasons[s]), "dropped_little_speech": len(little),
            "kept": len(kept)}
    return reasons, kept, little, speech_s, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    by_spk = defaultdict(list)
    for sub in ("dev-clean", "test-clean"):
        for p in sorted((LIBRI / sub).glob("*/*/*.flac")):
            by_spk[p.stem.split("-")[0]].append(p)
    cands = sorted(by_spk)
    used = sorted({s for r in csv.DictReader(open(ROOT / "data/english_sim/conversations.tsv"), delimiter="\t")
                   for s in r["speakers"].split()})
    print(f"{len(cands)} LibriSpeech candidates, {len(used)} used in the twin, files per speaker min "
          f"{min(len(v) for v in by_spk.values())}")
    cache = OUT / "embeddings.npz"
    if cache.exists():
        E = np.load(cache)["emb"]
        rows = list(csv.DictReader(open(OUT / "utterances_raw.tsv"), delimiter="\t"))
    else:
        rows, embs = [], []
        with Pool(args.workers) as pool:
            for i, (r, e) in enumerate(pool.imap(work, [(s, by_spk[s]) for s in cands])):
                rows += r
                embs.append(e)
                if i % 10 == 0:
                    print(f"  {i + 1}/{len(cands)} speakers embedded", flush=True)
        E = np.concatenate(embs)
        np.savez(cache, emb=E)
        with open(OUT / "utterances_raw.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    nep_thr = json.loads(NEP_SUMMARY.read_text())["split_half_same_cos_p5"]
    out_rows, per_spk = [], {}
    for name, pool_, fixed in (("all80_own", cands, None), ("used70_own", used, None), ("all80_nepthr", cands, nep_thr)):
        reasons, kept, little, speech_s, info = check(rows, E, pool_, fixed)
        dropped_used = [s for s in used if (reasons.get(s) or s in little)]
        out_rows.append({"reading": name, "pool": len(pool_), **info, "used_speakers_dropped": len(dropped_used),
                         "used_dropped_ids": " ".join(dropped_used)})
        for s in pool_:
            why = "; ".join(reasons.get(s, [])) or (f"under {MIN_SPEAKER_SPEECH_S} s ({speech_s.get(s, 0):.0f} s)" if s in little else "")
            per_spk.setdefault(s, {"speaker": s, "used_in_twin": int(s in used)})[name] = why or "kept"
        print(name, json.dumps(out_rows[-1]))
    with open(ROOT / "results/english_speaker_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    with open(OUT / "per_speaker.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, ["speaker", "used_in_twin", "all80_own", "used70_own", "all80_nepthr"], delimiter="\t",
                           restval="")
        w.writeheader()
        w.writerows(per_spk[s] for s in sorted(per_spk))


if __name__ == "__main__":
    main()
