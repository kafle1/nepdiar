#!/usr/bin/env python3
"""Simulated Nepali conversations for a zero-shot diarization test, built from OpenSLR 54 read speech.

Method: Landini, Lozano-Diez, Diez, Burget, "From Simulated Mixtures to Simulated Conversations as
Training Data for End-to-End Neural Diarization", Interspeech 2022, pp. 5095-5099,
doi 10.21437/Interspeech.2022-10451. The placement loop follows their released generator
(github.com/BUTSpeechFIT/EEND_dataprep, v1/conv_generator.py, Apache 2.0): every speaker's speech
segments are shuffled into one turn order and placed one at a time. Same speaker as the segment
before: a pause from the same-speaker pause distribution. New speaker: a pause or an overlap, chosen
by the measured pause/overlap ratio, each drawn from its own distribution. Values of 7 s or more
are dropped, as in their code. The distributions here come from the AMI dev only_words references,
not from their telephone stats.

Stages (run from the study root):
  .venv/bin/python scripts/make_nepali_sim.py stats         AMI dev turn stats -> turn_stats.json
  .venv/bin/python scripts/make_nepali_sim.py plan          30 conversations -> rttm/, uem/, conversations.tsv,
                                                           placement.tsv (needs check_nepali_speakers.py output)
  .venv/bin/python scripts/make_nepali_sim.py render_clean  wav/ from placement.tsv and the OpenSLR 54 zips
  .venv-sim/bin/python scripts/make_nepali_sim.py rooms     manifest_rn.tsv + rirs.npz (needs pyroomacoustics)
  .venv/bin/python scripts/make_nepali_sim.py render_rn     wav_rn/ from placement.tsv, rirs.npz, manifest_rn.tsv, DEMAND
A rebuild from the release only needs render_clean and render_rn (numpy, scipy, soundfile).
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common import ROOT
except ImportError:  # running from the release folder, pass --out and the zip folders
    ROOT = Path.cwd()

SR = 16000
SEED = 2083
OUT = ROOT / "data/nepali_sim"
SLR = ROOT / "data/slr54"
AMI_DEV = ROOT / "ref/ami/only_words/rttms/dev"
MAX_GAP_S = 7.0  # landini's cap on pauses and overlaps
TARGET_S = 300.0
TOL_S = 3.0
N_PER_COUNT = 10
SPEAKER_COUNTS = (2, 3, 4)
LEVEL_DBFS = -26.0  # every utterance's speech brought to the same rms, so no speaker is buried
FADE_S = 0.01
DITHER = 1e-4  # about -80 dBFS so silence isn't digital zero
# DEMAND indoor scenes without background talkers (PCAFETER, PRESTO, OMEETING left out for babble)
DEMAND_ENVS = ("DKITCHEN", "DLIVING", "DWASHING", "OOFFICE", "OHALLWAY")


def read_rttm(path):
    segs = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(segs)


def ami_stats():
    """Gaps measured the way the generator uses them: start of each segment minus the latest end so far."""
    same, pause, overlap, same_neg = [], [], [], 0
    files = sorted(AMI_DEV.glob("*.rttm"))
    for f in files:
        segs = read_rttm(f)
        last_end, prev = segs[0][1], segs[0][2]
        for s, e, k in segs[1:]:
            g = s - last_end
            if k == prev:
                if g >= 0:
                    same.append(g)
                else:
                    same_neg += 1  # same speaker restarting inside someone else's turn, not a pause
            elif g >= 0:
                pause.append(g)
            else:
                overlap.append(-g)
            last_end, prev = max(last_end, e), k
    keep = lambda v: [round(x, 3) for x in v if x < MAX_GAP_S]
    st = {
        "source": f"AMI dev only_words reference RTTMs, {len(files)} meetings ({AMI_DEV.relative_to(ROOT)})",
        "same_spk_pause": keep(same), "diff_spk_pause": keep(pause), "diff_spk_overlap": keep(overlap),
        "dropped_same_spk_inside_other_turn": same_neg,
        "dropped_7s_or_more": len(same) + len(pause) + len(overlap) - len(keep(same)) - len(keep(pause)) - len(keep(overlap)),
    }
    # landini takes the ratio from the counts before the 7 s cut
    st["p_pause_on_speaker_change"] = len(pause) / (len(pause) + len(overlap))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "turn_stats.json").write_text(json.dumps(st))
    for k in ("same_spk_pause", "diff_spk_pause", "diff_spk_overlap"):
        v = np.array(st[k])
        print(f"{k}: n={len(v)} mean={v.mean():.3f} median={np.median(v):.3f} p90={np.percentile(v, 90):.3f}")
    print(f"p(pause | speaker change)={st['p_pause_on_speaker_change']:.3f}, same-speaker restarts inside another turn dropped: {same_neg}")


def place(turns, rng, st):
    """turns: list of (speaker index, duration in ms). Returns start times in ms and total length (ms)."""
    same = np.array(st["same_spk_pause"])
    pause = np.array(st["diff_spk_pause"])
    ovl = np.array(st["diff_spk_overlap"])
    p_pause = st["p_pause_on_speaker_change"]
    starts, spk_last, last_end = [], {}, 0
    for i, (k, d) in enumerate(turns):
        if i == 0 or k == turns[i - 1][0]:
            gap = rng.choice(same)
        elif rng.random() < p_pause:
            gap = rng.choice(pause)
        else:
            gap = -rng.choice(ovl)
        s = max(last_end + int(round(gap * 1000)), spk_last.get(k, 0), 0)  # a speaker never overlaps itself
        starts.append(s)
        spk_last[k] = s + d
        last_end = max(last_end, s + d)
    return starts, last_end


def build_turns(seg_lists, budget_ms, rng):
    """Each speaker's segments in order up to budget_ms of speech (at least one), shuffled into one turn order."""
    chosen = []
    for k, segs in enumerate(seg_lists):
        tot, take = 0, []
        for seg in segs:
            if take and tot + seg["dur_ms"] > budget_ms:
                break
            take.append(seg)
            tot += seg["dur_ms"]
        chosen.append(take)
    order = np.repeat(np.arange(len(chosen)), [len(c) for c in chosen])
    rng.shuffle(order)
    pos = [0] * len(chosen)
    turns = []
    for k in order:
        turns.append((int(k), chosen[k][pos[k]]))
        pos[k] += 1
    return turns


def simulate(seg_lists, st, conv_seed):
    """Bisect the per-speaker speech budget until the conversation lands within TOL_S of TARGET_S.
    Every trial reuses the same seed, so the result only depends on conv_seed."""
    lo, hi = 1000, max(sum(s["dur_ms"] for s in segs) for segs in seg_lists)
    best = None
    for _ in range(40):
        mid = (lo + hi) // 2
        rng = np.random.default_rng(conv_seed)
        turns = build_turns(seg_lists, mid, rng)
        starts, total = place([(k, s["dur_ms"]) for k, s in turns], rng, st)
        if best is None or abs(total - TARGET_S * 1000) < abs(best[2] - TARGET_S * 1000):
            best = (turns, starts, total, mid)
        if abs(total - TARGET_S * 1000) <= TOL_S * 1000:
            break
        if total < TARGET_S * 1000:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    return best



class ZipAudio:
    def __init__(self, zip_dir):
        self.zips, self.where = {}, {}
        for z in sorted(Path(zip_dir).glob("asr_nepali_*.zip")):
            zf = zipfile.ZipFile(z)
            self.zips[z.name] = zf
            for n in zf.namelist():
                if n.endswith(".flac"):
                    self.where[Path(n).stem] = (z.name, n)
        if not self.where:
            sys.exit(f"no asr_nepali_*.zip in {zip_dir}")

    def read(self, fid):
        if fid not in self.where:
            sys.exit(f"{fid} not in the zips; it is in asr_nepali_{fid[0]}.zip")
        zname, n = self.where[fid]
        audio, sr = sf.read(io.BytesIO(self.zips[zname].read(n)), dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{fid} is {sr} Hz, expected {SR}")
        return audio[:, 0]


def frame_stats(ms_grid_len, segs):
    """segs: (start_ms, end_ms, spk). Returns speech, overlap, per-speaker ms on a 1 ms grid."""
    count = np.zeros(ms_grid_len, dtype=np.int8)
    per = {}
    for s, e, k in segs:
        count[s:e] += 1
        per[k] = per.get(k, 0) + (e - s)
    return int((count > 0).sum()), int((count > 1).sum()), per


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def conversations(out):
    """uri -> (speakers in order, length in ms), from conversations.tsv."""
    return {r["uri"]: (r["speakers"].split(), int(round(float(r["duration_s"]) * 1000))) for r in read_tsv(out / "conversations.tsv")}


def speaker_tracks(uri, spks, total_ms, rows, za):
    """One dry track per speaker: each placed segment with its gain and 10 ms fades."""
    tracks = {k: np.zeros(total_ms * SR // 1000, dtype=np.float64) for k in spks}
    fade = int(FADE_S * SR)
    ramp = np.linspace(0, 1, fade)
    cache = {}
    for r in rows:
        fid, a, b, s = r["slr54_file_id"], int(r["src_start_ms"]), int(r["src_end_ms"]), int(r["placed_start_ms"])
        if fid not in cache:
            cache[fid] = za.read(fid)
        x = cache[fid][a * SR // 1000: b * SR // 1000].astype(np.float64) * float(r["gain"])
        if len(x) > 2 * fade:
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        p = s * SR // 1000
        tracks[r["speaker"]][p:p + len(x)] += x
    return tracks


def write_wav(path, mix):
    peak = float(np.abs(mix).max())
    scale = 0.99 / peak if peak > 0.99 else 1.0
    sf.write(path, (mix * scale).astype(np.float32), SR, subtype="PCM_16")
    return scale


def sha_list(d, uris):
    lines = []
    for u in uris:
        lines.append(f"{hashlib.sha256((d / f'{u}.wav').read_bytes()).hexdigest()}  {u}.wav\n")
    (d / "SHA256SUMS").write_text("".join(lines))


def plan(out):
    """Speakers per conversation, turn order and placement. Writes rttm, uem, conversations.tsv and
    placement.tsv (the manifest the audio is rendered from). No audio is read here."""
    st = json.loads((out / "turn_stats.json").read_text())
    utts = {}
    for r in read_tsv(out / "speaker_check/utterances.tsv"):
        if r["use"] == "1":
            utts.setdefault(r["speaker"], []).append(r)
    kept = [l.split()[0] for l in (out / "speaker_check/kept_speakers.txt").read_text().splitlines() if l.strip()]
    need = sum(SPEAKER_COUNTS) * N_PER_COUNT
    rng = np.random.default_rng(SEED)
    pool = sorted(kept)
    rng.shuffle(pool)
    if len(pool) < need:
        sys.exit(f"only {len(pool)} usable speakers, need {need}")
    # each speaker appears in exactly one conversation
    convs, i = [], 0
    for n in SPEAKER_COUNTS:
        for c in range(N_PER_COUNT):
            convs.append((f"nepsim_{n}spk_{c + 1:02d}", pool[i:i + n]))
            i += n
    for d in ("rttm", "uem"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, used = [], []
    for ci, (uri, spks) in enumerate(convs):
        seg_lists = []
        for k in spks:
            u = sorted(utts[k], key=lambda r: r["file_id"])
            np.random.default_rng([SEED, ci, int(k, 16)]).shuffle(u)
            segs = []
            for r in u:
                for a, b in json.loads(r["regions_ms"]):
                    segs.append({"fid": r["file_id"], "a": a, "b": b, "dur_ms": b - a, "gain": r["gain"]})
            seg_lists.append(segs)
        turns, starts, total, budget = simulate(seg_lists, st, [SEED, ci])
        ref = []
        for (k, seg), s in zip(turns, starts):
            ref.append((s, s + seg["dur_ms"], spks[k]))
            used.append({"uri": uri, "speaker": spks[k], "slr54_file_id": seg["fid"], "src_start_ms": seg["a"],
                         "src_end_ms": seg["b"], "placed_start_ms": s, "gain": seg["gain"]})
        with open(out / "rttm" / f"{uri}.rttm", "w") as f:
            for s, e, k in sorted(ref):
                f.write(f"SPEAKER {uri} 1 {s / 1000:.3f} {(e - s) / 1000:.3f} <NA> <NA> {k} <NA> <NA>\n")
        (out / "uem" / f"{uri}.uem").write_text(f"{uri} 1 0.000 {total / 1000:.3f}\n")
        speech, ov, per = frame_stats(total, ref)
        times = [per[k] / 1000 for k in spks]
        rows.append({"uri": uri, "n_speakers": len(spks), "speakers": " ".join(spks), "duration_s": total / 1000,
                     "speech_s": speech / 1000, "overlap_s": ov / 1000, "overlap_ratio": round(ov / speech, 4),
                     "silence_ratio": round(1 - speech / total, 4), "segments": len(ref),
                     "spk_time_s": " ".join(f"{t:.1f}" for t in times),
                     "spk_share_min": round(min(times) / sum(times), 4), "spk_share_max": round(max(times) / sum(times), 4),
                     "budget_s": budget / 1000})
        print(f"{uri}: {total / 1000:.1f}s overlap={ov / speech:.3f} speakers={len(spks)} segs={len(ref)}", flush=True)
    (out / "conversations.txt").write_text("".join(r["uri"] + "\n" for r in rows))
    for n in SPEAKER_COUNTS:
        (out / f"conversations_{n}spk.txt").write_text("".join(r["uri"] + "\n" for r in rows if r["n_speakers"] == n))
    write_tsv(out / "conversations.tsv", rows)
    write_tsv(out / "placement.tsv", used)
    tot = sum(r["duration_s"] for r in rows)
    sp = sum(r["speech_s"] for r in rows)
    ov = sum(r["overlap_s"] for r in rows)
    print(f"\n{len(rows)} conversations, {tot / 3600:.2f} h, {len({k for _, s in convs for k in s})} speakers, "
          f"overlap/speech {ov / sp:.3f}, silence {1 - sp / tot:.3f}")


def render_clean(out, slr_zips):
    """Clean condition from placement.tsv: sum of the dry speaker tracks plus -80 dBFS dither."""
    za = ZipAudio(slr_zips)
    conv = conversations(out)
    place_rows = read_tsv(out / "placement.tsv")
    (out / "wav").mkdir(parents=True, exist_ok=True)
    for ci, (uri, (spks, total)) in enumerate(conv.items()):
        tr = speaker_tracks(uri, spks, total, [r for r in place_rows if r["uri"] == uri], za)
        mix = sum(tr[k] for k in spks)
        mix += np.random.default_rng([SEED, ci, 1]).normal(0, DITHER, len(mix))
        scale = write_wav(out / "wav" / f"{uri}.wav", mix)
        print(f"clean {uri}: scaled by {scale:.3f}", flush=True)
    sha_list(out / "wav", list(conv))


def rooms(out):
    """Reverb + noise plan: one random shoebox room per conversation (pyroomacoustics image method),
    a phone-height mic, one position per speaker, a DEMAND indoor noise and an SNR. Writes
    manifest_rn.tsv and rirs.npz, so rendering needs no pyroomacoustics. Run with .venv-sim."""
    import pyroomacoustics as pra
    conv = conversations(out)
    rirs, rows = {}, []
    for ci, (uri, (spks, _)) in enumerate(conv.items()):
        rng = np.random.default_rng([SEED, ci, 2])
        while True:
            dim = [rng.uniform(3, 8), rng.uniform(3, 6), rng.uniform(2.5, 3.5)]
            rt60 = rng.uniform(0.2, 0.7)
            try:
                absorb, order = pra.inverse_sabine(rt60, dim)
                break
            except ValueError:
                continue  # room too small for that rt60
        mic = [rng.uniform(1.0, dim[0] - 1.0), rng.uniform(1.0, dim[1] - 1.0), rng.uniform(0.7, 1.0)]
        pos = []
        for _ in spks:
            for _try in range(10000):
                d, az = rng.uniform(0.5, 2.5), rng.uniform(0, 2 * np.pi)
                p = [mic[0] + d * np.cos(az), mic[1] + d * np.sin(az), rng.uniform(1.1, 1.3)]
                inside = 0.5 <= p[0] <= dim[0] - 0.5 and 0.5 <= p[1] <= dim[1] - 0.5
                if inside and all(np.hypot(p[0] - q[0], p[1] - q[1]) >= 0.5 for q in pos):
                    break
            else:
                sys.exit(f"{uri}: could not place speakers")
            pos.append(p)
        # sabine overshoots in the image method, so rescale the design rt60 until the measured mean is within 5%
        design = rt60
        for _cal in range(4):
            room = pra.ShoeBox(dim, fs=SR, materials=pra.Material(absorb), max_order=order)
            for p in pos:
                room.add_source(p)
            room.add_microphone(mic)
            room.compute_rir()
            measured = room.measure_rt60()[0]
            if abs(np.mean(measured) / rt60 - 1) <= 0.05:
                break
            design *= rt60 / float(np.mean(measured))
            try:
                absorb, order = pra.inverse_sabine(design, dim)
            except ValueError:
                break
        env = DEMAND_ENVS[int(rng.integers(len(DEMAND_ENVS)))]
        offset = float(rng.uniform(0, 290))
        snr = float(rng.uniform(5, 20))
        for j, k in enumerate(spks):
            h = np.asarray(room.rir[0][j], dtype=np.float64)
            pk = int(np.argmax(np.abs(h)))
            rirs[f"{uri}__{k}"] = (h[max(0, pk - 20):] * 4 * np.pi).astype(np.float32)  # direct path at 1 m ~ 1
        rows.append({"uri": uri, "room_m": " ".join(f"{v:.3f}" for v in dim), "rt60_target_s": round(rt60, 3), "rt60_design_s": round(design, 3),
                     "rt60_measured_s": " ".join(f"{v:.3f}" for v in measured), "max_order": order,
                     "mic_m": " ".join(f"{v:.3f}" for v in mic),
                     "speaker_pos_m": "; ".join(f"{k}:" + ",".join(f"{v:.3f}" for v in p) for k, p in zip(spks, pos)),
                     "speaker_dist_m": " ".join(f"{np.linalg.norm(np.subtract(p, mic)):.3f}" for p in pos),
                     "noise": f"DEMAND {env}/ch01.wav", "noise_offset_s": round(offset, 3), "snr_db": round(snr, 3)})
        print(f"room {uri}: {dim[0]:.1f}x{dim[1]:.1f}x{dim[2]:.1f} rt60 {rt60:.2f} (measured {np.mean(measured):.2f}) {env} snr {snr:.1f}", flush=True)
    np.savez_compressed(out / "rirs.npz", **rirs)
    write_tsv(out / "manifest_rn.tsv", rows)


def render_rn(out, slr_zips, demand_zips):
    """Reverb + noise condition: each dry speaker track convolved with its room response, summed,
    then DEMAND noise scaled so speech power (over reference speech) / noise power = the SNR."""
    from scipy.signal import fftconvolve
    za = ZipAudio(slr_zips)
    conv = conversations(out)
    place_rows = read_tsv(out / "placement.tsv")
    man = {r["uri"]: r for r in read_tsv(out / "manifest_rn.tsv")}
    rirs = np.load(out / "rirs.npz")
    noises = {}
    (out / "wav_rn").mkdir(parents=True, exist_ok=True)
    for uri, (spks, total) in conv.items():
        m = man[uri]
        rows = [r for r in place_rows if r["uri"] == uri]
        tr = speaker_tracks(uri, spks, total, rows, za)
        n = total * SR // 1000
        rev = np.zeros(n)
        for k in spks:
            rev += fftconvolve(tr[k], rirs[f"{uri}__{k}"].astype(np.float64))[:n]
        speech = np.zeros(n, bool)
        for r in rows:
            p = int(r["placed_start_ms"]) * SR // 1000
            speech[p:p + (int(r["src_end_ms"]) - int(r["src_start_ms"])) * SR // 1000] = True
        env = m["noise"].split()[1].split("/")[0]
        if env not in noises:
            with zipfile.ZipFile(Path(demand_zips) / f"{env}_16k.zip") as z:
                x, sr = sf.read(io.BytesIO(z.read(f"{env}/ch01.wav")), dtype="float64")
            if sr != SR:
                sys.exit(f"{env} is {sr} Hz")
            noises[env] = x
        x = noises[env]
        noise = x[(int(round(float(m["noise_offset_s"]) * SR)) + np.arange(n)) % len(x)]  # wraps past the end
        ps, pn = float(np.mean(rev[speech] ** 2)), float(np.mean(noise ** 2))
        noise *= np.sqrt(ps / (pn * 10 ** (float(m["snr_db"]) / 10)))
        scale = write_wav(out / "wav_rn" / f"{uri}.wav", rev + noise)
        print(f"rn {uri}: snr {float(m['snr_db']):.1f} dB, scaled by {scale:.3f}", flush=True)
    sha_list(out / "wav_rn", list(conv))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["stats", "plan", "render_clean", "rooms", "render_rn"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--slr-zips", type=Path, default=SLR / "zips", help="folder with asr_nepali_0..3.zip")
    ap.add_argument("--demand-zips", type=Path, default=ROOT / "data/demand/zips", help="folder with the DEMAND *_16k.zip files")
    a = ap.parse_args()
    if a.stage == "stats":
        ami_stats()
    elif a.stage == "plan":
        plan(a.out)
    elif a.stage == "render_clean":
        render_clean(a.out, a.slr_zips)
    elif a.stage == "rooms":
        rooms(a.out)
    else:
        render_rn(a.out, a.slr_zips, a.demand_zips)


if __name__ == "__main__":
    main()
