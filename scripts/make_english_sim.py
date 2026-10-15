#!/usr/bin/env python3
"""English twin of NepDiar-Sim: the same 30 conversation plans with LibriSpeech voices.

Kept from NepDiar-Sim (data/nepali_sim): speaker count, turn order, every gap and overlap draw,
the room, room responses, noise clip, noise offset and SNR. Only the voices change.
- Each Nepali speaker in a conversation maps to one distinct LibriSpeech dev-clean/test-clean
  speaker. OpenSLR 54 has no sex labels, so each conversation is sex-balanced instead
  (2 spk 1F+1M, 3 spk 2+1 alternating, 4 spk 2F+2M). Busiest Nepali speakers pick first, taking
  the least-used English speaker of that sex with enough short regions for its slots.
- Each Nepali segment slot gets a distinct English speech region of that speaker (same energy trim
  as the Nepali set), assigned to minimise the total length mismatch per speaker.
- The Nepali gap draws are replayed exactly (same seed, same rng calls; checked against
  placement.tsv) and applied to the English lengths with the same rule:
  start = max(last end + gap, this speaker's last end, 0).
- Loudness, fades, dither, reverb, noise and the RTTM/UEM come from make_nepali_sim.py unchanged.

Stages, from the study root:
  .venv/bin/python scripts/make_english_sim.py prep      trim + gain for every LibriSpeech utterance
  .venv/bin/python scripts/make_english_sim.py plan      placement.tsv, conversations.tsv, rttm, uem, rirs, manifest_rn
  .venv/bin/python scripts/make_english_sim.py render_clean | render_rn   audio from the manifests
--libri points at the folder holding LibriSpeech/{dev-clean,test-clean}.
"""
import argparse
import csv
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_nepali_sim as nep

try:
    from common import ROOT
except ImportError:
    ROOT = Path.cwd()

SR = nep.SR
NEP = ROOT / "data/nepali_sim"
OUT = ROOT / "data/english_sim"
SUBSETS = ("dev-clean", "test-clean")
LEVEL_DBFS = -26.0
SHORT_MS = 2500  # about the nepali p90 segment length (2.52 s)


class LibriAudio:
    def __init__(self, libri):
        self.root = Path(libri) / "LibriSpeech"

    def path(self, uid):
        spk, chap, _ = uid.split("-")
        for sub in SUBSETS:
            p = self.root / sub / spk / chap / f"{uid}.flac"
            if p.exists():
                return p
        sys.exit(f"{uid}.flac not found under {self.root}")

    def read(self, uid):
        x, sr = sf.read(self.path(uid), dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{uid} is {sr} Hz")
        return x[:, 0]


def prep_one(p):
    from check_nepali_speakers import trim  # only prep needs it, so the render stages run from the release folder
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x[:, 0]
    regs = trim(x) if sr == SR else []
    speech = np.concatenate([x[a * 16:b * 16] for a, b in regs]) if regs else np.zeros(0, np.float32)
    rms = float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))) if len(speech) else 0.0
    return {"librispeech_id": p.stem, "speaker": p.stem.split("-")[0], "subset": p.parts[-4], "sr": sr,
            "file_ms": len(x) * 1000 // sr, "speech_ms": sum(b - a for a, b in regs), "regions_ms": json.dumps(regs),
            "gain": repr(10 ** (LEVEL_DBFS / 20) / rms) if rms > 0 else "0.0",
            "clipped_frac": float(np.mean(np.abs(x) >= 0.999))}


def prep(libri, out, workers):
    files = sorted(p for sub in SUBSETS for p in (Path(libri) / "LibriSpeech" / sub).rglob("*.flac"))
    with Pool(workers) as pool:
        rows = pool.map(prep_one, files, chunksize=20)
    out.mkdir(parents=True, exist_ok=True)
    nep.write_tsv(out / "librispeech_utterances.tsv", rows)
    d = np.array([b - a for r in rows for a, b in json.loads(r["regions_ms"])]) / 1000
    print(f"{len(rows)} utterances, {len({r['speaker'] for r in rows})} speakers, {len(d)} regions, "
          f"region s mean {d.mean():.2f} median {np.median(d):.2f} p10 {np.percentile(d, 10):.2f} p90 {np.percentile(d, 90):.2f}")


def speaker_sex(libri):
    sex = {}
    for line in (Path(libri) / "LibriSpeech/SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 3 and p[2] in SUBSETS:
            sex[p[0]] = p[1]
    return sex


def replay_gaps(uri, ci, spks, rows, st):
    """The gap (ms) the Nepali builder drew before each turn, replayed from its seed and checked."""
    rng = np.random.default_rng([nep.SEED, ci])
    rng.shuffle(np.zeros(len(rows), dtype=np.int64))  # build_turns' one shuffle of the turn order
    same, pause, ovl = (np.array(st[k]) for k in ("same_spk_pause", "diff_spk_pause", "diff_spk_overlap"))
    gaps, spk_last, last_end = [], {}, 0
    for i, r in enumerate(rows):
        k = r["speaker"]
        if i == 0 or k == rows[i - 1]["speaker"]:
            g = rng.choice(same)
        elif rng.random() < st["p_pause_on_speaker_change"]:
            g = rng.choice(pause)
        else:
            g = -rng.choice(ovl)
        g = int(round(g * 1000))
        s = max(last_end + g, spk_last.get(k, 0), 0)
        if s != int(r["placed_start_ms"]):
            sys.exit(f"{uri} turn {i}: replay gives {s} ms, placement.tsv has {r['placed_start_ms']}; replay is wrong")
        d = int(r["src_end_ms"]) - int(r["src_start_ms"])
        spk_last[k] = s + d
        last_end = max(last_end, s + d)
        gaps.append(g)
    return gaps


def plan(libri, out):
    st = json.loads((NEP / "turn_stats.json").read_text())
    nconv = nep.read_tsv(NEP / "conversations.tsv")
    nplace = nep.read_tsv(NEP / "placement.tsv")
    nman = {r["uri"]: r for r in nep.read_tsv(NEP / "manifest_rn.tsv")}
    nrirs = np.load(NEP / "rirs.npz")
    sex = speaker_sex(libri)
    regs = {}
    for r in nep.read_tsv(out / "librispeech_utterances.tsv"):
        if int(r["sr"]) != SR or float(r["clipped_frac"]) >= 0.001 or float(r["gain"]) <= 0:
            continue
        for a, b in json.loads(r["regions_ms"]):
            regs.setdefault(r["speaker"], []).append((b - a, r["librispeech_id"], a, b, r["gain"]))
    pools = {g: sorted(s for s in regs if sex.get(s) == g) for g in ("F", "M")}
    used = {s: 0 for s in regs}
    # speaker choice: busiest nepali speakers first, each gets an english speaker of the planned sex with at
    # least as many short (<= SHORT_MS) regions as it has slots, least used first, then most short regions
    supply = {s: sum(d <= SHORT_MS for d, *_ in regs[s]) for s in regs}
    demand = {}
    for r in nplace:
        demand[(r["uri"], r["speaker"])] = demand.get((r["uri"], r["speaker"]), 0) + 1
    want = []
    for ci, c in enumerate(nconv):
        nspks = c["speakers"].split()
        n = len(nspks)
        sexes = ["F", "M"] * (n // 2) + ([("F", "M")[ci % 2]] if n % 2 else [])
        np.random.default_rng([nep.SEED, ci, 4]).shuffle(sexes)
        want += [(demand[(c["uri"], k)], ci, pos, g) for pos, (k, g) in enumerate(zip(nspks, sexes))]
    assign = {}
    for dem, ci, pos, g in sorted(want, key=lambda w: (-w[0], w[1], w[2])):
        taken = {v for (c2, _), v in assign.items() if c2 == ci}
        cand = [s for s in pools[g] if s not in taken]
        pick = min([s for s in cand if supply[s] >= dem] or cand, key=lambda s: (used[s], -supply[s], s))
        used[pick] += 1
        assign[(ci, pos)] = pick
    for d in ("rttm", "uem"):
        (out / d).mkdir(parents=True, exist_ok=True)
    conv_rows, place_rows, man_rows, rirs, match = [], [], [], {}, []
    for ci, c in enumerate(nconv):
        uri_n = c["uri"]
        uri = uri_n.replace("nepsim_", "engsim_")
        nspks = c["speakers"].split()
        n = len(nspks)
        espk = [assign[(ci, pos)] for pos in range(n)]
        emap = dict(zip(nspks, espk))
        rows = [r for r in nplace if r["uri"] == uri_n]
        gaps = replay_gaps(uri_n, ci, nspks, rows, st)
        # each nepali slot gets a distinct english region of its speaker, minimising the total length mismatch
        # (a speaker in two conversations may repeat a region across them, never inside one)
        chosen = {}
        for k in nspks:
            free = regs[emap[k]]
            slots = [i for i, r in enumerate(rows) if r["speaker"] == k]
            if len(free) < len(slots):
                sys.exit(f"{uri}: {emap[k]} has {len(free)} regions for {len(slots)} slots")
            want = np.array([int(rows[i]["src_end_ms"]) - int(rows[i]["src_start_ms"]) for i in slots])
            si, rj = linear_sum_assignment(np.abs(want[:, None] - np.array([f[0] for f in free])[None, :]))
            for a, j in zip(si, rj):
                chosen[slots[a]] = free[j]
                match.append((int(want[a]), free[j][0]))
        spk_last, last_end, ref = {}, 0, []
        for i, r in enumerate(rows):
            k = emap[r["speaker"]]
            d, uid, a, b, gain = chosen[i]
            s = max(last_end + gaps[i], spk_last.get(k, 0), 0)
            spk_last[k] = s + d
            last_end = max(last_end, s + d)
            ref.append((s, s + d, k))
            place_rows.append({"uri": uri, "speaker": k, "librispeech_id": uid, "src_start_ms": a, "src_end_ms": b,
                               "placed_start_ms": s, "gain": gain, "nepali_speaker": r["speaker"], "gap_ms": gaps[i]})
        total = last_end
        with open(out / "rttm" / f"{uri}.rttm", "w") as f:
            for s, e, k in sorted(ref):
                f.write(f"SPEAKER {uri} 1 {s / 1000:.3f} {(e - s) / 1000:.3f} <NA> <NA> {k} <NA> <NA>\n")
        (out / "uem" / f"{uri}.uem").write_text(f"{uri} 1 0.000 {total / 1000:.3f}\n")
        speech, ov, per = nep.frame_stats(total, ref)
        times = [per[k] / 1000 for k in espk]
        conv_rows.append({"uri": uri, "n_speakers": n, "speakers": " ".join(espk), "duration_s": total / 1000,
                          "speech_s": speech / 1000, "overlap_s": ov / 1000, "overlap_ratio": round(ov / speech, 4),
                          "silence_ratio": round(1 - speech / total, 4), "segments": len(ref),
                          "spk_time_s": " ".join(f"{t:.1f}" for t in times),
                          "spk_share_min": round(min(times) / sum(times), 4), "spk_share_max": round(max(times) / sum(times), 4),
                          "sexes": "".join(sex[s] for s in espk), "nepali_twin": uri_n,
                          "nepali_speakers": " ".join(nspks)})
        m = dict(nman[uri_n])
        m["uri"] = uri
        m["speaker_pos_m"] = "; ".join(f"{emap[p.split(':')[0]]}:{p.split(':')[1]}" for p in m["speaker_pos_m"].split("; "))
        man_rows.append(m)
        for k in nspks:
            rirs[f"{uri}__{emap[k]}"] = nrirs[f"{uri_n}__{k}"]
        print(f"{uri}: {total / 1000:.1f}s (nepali {float(c['duration_s']):.1f}s) overlap={ov / speech:.3f} "
              f"(nepali {float(c['overlap_ratio']):.3f}) {conv_rows[-1]['sexes']}", flush=True)
    (out / "conversations.txt").write_text("".join(r["uri"] + "\n" for r in conv_rows))
    for n in nep.SPEAKER_COUNTS:
        (out / f"conversations_{n}spk.txt").write_text("".join(r["uri"] + "\n" for r in conv_rows if r["n_speakers"] == n))
    nep.write_tsv(out / "conversations.tsv", conv_rows)
    nep.write_tsv(out / "placement.tsv", place_rows)
    nep.write_tsv(out / "manifest_rn.tsv", man_rows)
    np.savez_compressed(out / "rirs.npz", **rirs)
    shutil.copy(NEP / "turn_stats.json", out / "turn_stats.json")
    w = np.array([a for a, _ in match]) / 1000
    g = np.array([b for _, b in match]) / 1000
    err = np.abs(g - w)
    print(f"\nslot length match: |english - nepali| median {np.median(err):.3f}s, p90 {np.percentile(err, 90):.3f}s, "
          f"p99 {np.percentile(err, 99):.3f}s, max {err.max():.3f}s; within 0.1 s {np.mean(err <= 0.1):.3f}, "
          f"within 0.25 s {np.mean(err <= 0.25):.3f}; mean nepali {w.mean():.3f}s english {g.mean():.3f}s")
    print(f"speakers used: {sum(v > 0 for v in used.values())} of {len(used)}, max uses {max(used.values())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "plan", "render_clean", "render_rn"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--libri", type=Path, default=ROOT / "data/librispeech")
    ap.add_argument("--demand-zips", type=Path, default=ROOT / "data/demand/zips")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "prep":
        prep(a.libri, a.out, a.workers)
    elif a.stage == "plan":
        plan(a.libri, a.out)
    elif a.stage == "render_clean":
        nep.render_clean(a.out, LibriAudio(a.libri))
    else:
        nep.render_rn(a.out, LibriAudio(a.libri), a.demand_zips)


if __name__ == "__main__":
    main()
