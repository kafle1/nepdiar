#!/usr/bin/env python3
"""Indic DiarBench, Nepali config, to wav (16 kHz mono) + RTTM + UEM (whole clip) + a sanity report.

run: uv run --no-project --with pyarrow --with numpy --with soundfile python -I scripts/real_nepali/prep_diarbench.py
uri = sample_id. Writes data/nepali_real/diarbench/{wav,rttm,uem}, clips.txt, stats.csv, check.txt.
"""
import csv
import os
import io
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/nepali_real/diarbench"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")  # results were made with ffmpeg 4, ffprobe must sit beside it


def overlap_stats(segs, dur, step=0.001):
    """speech time, time with 2+ speakers, and total speaker time, on a 1 ms grid."""
    n = int(np.ceil(dur / step)) + 1
    cnt = np.zeros(n, dtype=np.int16)
    for _, s, e in segs:
        a, b = max(0, int(round(s / step))), min(n, int(round(e / step)))
        cnt[a:b] += 1
    return (cnt > 0).sum() * step, (cnt > 1).sum() * step, cnt.sum() * step


def merge_per_speaker(segs):
    out = []
    for spk in sorted({s[0] for s in segs}):
        cur = None
        for _, st, en in sorted(s for s in segs if s[0] == spk):
            if cur and st <= cur[1]:
                cur[1] = max(cur[1], en)
            else:
                cur and out.append((spk, cur[0], cur[1]))
                cur = [st, en]
        out.append((spk, cur[0], cur[1]))
    return out


def main():
    for sub in ("wav", "rttm", "uem"):
        (D / sub).mkdir(parents=True, exist_ok=True)
    t = pq.read_table(D / "Nepali_test-00000-of-00001.parquet")
    rows = t.to_pylist()
    problems, stats, uris = [], [], []
    for r in rows:
        uri = r["sample_id"]
        uris.append(uri)
        wav = D / "wav" / f"{uri}.wav"
        if not wav.exists():
            subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(wav)], input=r["audio"]["bytes"], check=True)
        info = subprocess.run([FFMPEG[:-6] + "ffprobe", "-v", "error", "-show_entries",
                               "stream=sample_rate,channels:format=duration,format_name", "-of", "csv=p=0",
                               "pipe:0"], input=r["audio"]["bytes"], capture_output=True).stdout.decode().split()
        audio_dur = sf.info(str(wav)).frames / 16000
        segs = []
        for s in r["annotated_transcript"]:
            st, en, spk = s["start_time"], s["end_time"], s["speaker_id"]
            if st is None or en is None or st < 0 or en <= st or en > audio_dur + 0.05:
                problems.append(f"{uri}: segment {spk} {st}-{en} (audio {audio_dur:.3f} s)")
            st, en = max(0.0, st), min(en, audio_dur)
            if en > st:
                segs.append((spk, st, en))
        # pyannote.metrics counts a speaker's self-overlap twice, so merge each speaker's own segments
        n_raw, segs = len(segs), merge_per_speaker(segs)
        if n_raw != len(segs):
            problems.append(f"{uri}: {n_raw - len(segs)} same-speaker overlapping segments merged")
        with open(D / "rttm" / f"{uri}.rttm", "w") as f:
            for spk, st, en in sorted(segs, key=lambda x: x[1]):
                f.write(f"SPEAKER {uri} 1 {st:.3f} {en - st:.3f} <NA> <NA> {spk} <NA> <NA>\n")
        (D / "uem" / f"{uri}.uem").write_text(f"{uri} 1 0.000 {audio_dur:.3f}\n")
        speech, ovl, spk_time = overlap_stats(segs, audio_dur)
        nspk = len({s[0] for s in segs})
        if nspk != int(r["num_speakers"]):
            problems.append(f"{uri}: num_speakers field {r['num_speakers']} but {nspk} speakers in the transcript")
        if abs(audio_dur - float(r["duration_seconds"])) > 0.5:
            problems.append(f"{uri}: duration field {r['duration_seconds']} vs audio {audio_dur:.2f}")
        stats.append({"uri": uri, "recording_id": r["recording_id"], "dataset_type": r["dataset_type"],
                      "source_format": " ".join(info), "duration_s": round(audio_dur, 3), "speakers": nspk,
                      "num_speakers_field": r["num_speakers"], "segments": len(segs),
                      "speech_s": round(speech, 3), "overlap_s": round(ovl, 3),
                      "overlap_ratio": round(ovl / speech, 4) if speech else 0,
                      "speech_ratio": round(speech / audio_dur, 4)})
    (D / "clips.txt").write_text("\n".join(uris) + "\n")
    with open(D / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    tot = sum(s["duration_s"] for s in stats)
    sp, ov = sum(s["speech_s"] for s in stats), sum(s["overlap_s"] for s in stats)
    lines = [f"clips {len(stats)}, hours {tot / 3600:.3f}, recordings {len({s['recording_id'] for s in stats})}",
             f"speakers per clip: {dict(sorted(Counter(s['speakers'] for s in stats).items()))}",
             f"dataset_type: {dict(Counter(s['dataset_type'] for s in stats))}",
             f"source formats: {dict(Counter(s['source_format'] for s in stats))}",
             f"speech {sp / 3600:.3f} h ({sp / tot:.3f} of audio), overlap {ov / 60:.2f} min, "
             f"overlap ratio (2+ speakers / speech) {ov / sp:.4f}",
             f"per-clip overlap ratio min {min(s['overlap_ratio'] for s in stats)} max {max(s['overlap_ratio'] for s in stats)}",
             f"problems ({len(problems)}):", *problems]
    (D / "check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
