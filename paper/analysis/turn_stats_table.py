"""Reference statistics for tab:chars, measured from each set's reference RTTM and UEM files.

Writes paper/analysis/turn_stats_table.csv (one row per set). Run from anywhere:
    python3 paper/analysis/turn_stats_table.py
"""
import csv
import glob
import os
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".csv")

# key, column label, rttm dir, uem dir, file list (None = every rttm), channel (descriptive, not measured)
SETS = [
    ("ami", "AMI test", R / "ref/ami/only_words/rttms/test", R / "ref/ami/uems/test", None, "headset mix, 16 kHz"),
    ("diarbench", "Indic DiarBench, Nepali", R / "data/nepali_real/diarbench/rttm", R / "data/nepali_real/diarbench/uem",
     R / "data/nepali_real/diarbench/clips.txt", "meeting stream, 16 kHz mono"),
    ("indicvoices", "IndicVoices calls", R / "data/nepali_real/indicvoices/rttm", R / "data/nepali_real/indicvoices/uem",
     R / "data/nepali_real/indicvoices/calls.txt", "phone, 8 kHz"),
    ("nepdiar_sim", "NepDiar-Sim", R / "data/nepali_sim/rttm", R / "data/nepali_sim/uem",
     R / "data/nepali_sim/conversations.txt", "read speech, 16 kHz"),
    ("english_twin", "English twin", R / "data/english_sim/rttm", R / "data/english_sim/uem",
     R / "data/english_sim/conversations.txt", "read speech, 16 kHz"),
]


def union(iv):
    t, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if cs is None or s > ce:
            if cs is not None:
                t += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return t + (ce - cs if cs is not None else 0.0)


def overlap(S):
    ev = sorted([(s, 1) for s, e, _ in S] + [(e, -1) for s, e, _ in S])
    ov, c, last = 0.0, 0, None
    for t, d in ev:
        if last is not None and c >= 2:
            ov += t - last
        c += d
        last = t
    return ov


def distinct_speakers(key, uris, labels):
    # AMI and the simulated set use corpus-wide speaker labels; IndicVoices labels are per side, so use its speaker IDs
    if key in ("ami", "nepdiar_sim", "english_twin"):
        return str(len(labels))
    if key == "indicvoices":
        ids = set()
        for r in csv.DictReader(open(R / "data/nepali_real/indicvoices/stats.csv")):
            if r["uri"] in uris:
                ids.update([r["side0_speaker_id"], r["side1_speaker_id"]])
        return str(len(ids))
    return "not given"  # DiarBench labels are per clip


def measure(key, label, rd, ud, lst, channel):
    files = sorted(glob.glob(str(rd / "*.rttm")))
    if lst is not None:
        keep = {u.strip() for u in open(lst) if u.strip()}
        files = [f for f in files if os.path.basename(f)[:-5] in keep]
        assert len(files) == len(keep), f"{key}: {len(files)} rttms for {len(keep)} listed files"
    segs, turns, changes, speech, ov, audio, labels, counts = [], [], 0, 0.0, 0.0, 0.0, set(), []
    uris = set()
    for f in files:
        u = os.path.basename(f)[:-5]
        uris.add(u)
        S = []
        for line in open(f):
            p = line.split()
            if p and p[0] == "SPEAKER":
                S.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
        for line in open(ud / f"{u}.uem"):
            p = line.split()
            if p:
                audio += float(p[3]) - float(p[2])
        S.sort(key=lambda x: (x[0], x[1]))
        segs += [e - s for s, e, _ in S]
        labels.update(x for _, _, x in S)
        counts.append(len({x for _, _, x in S}))
        cur = None
        for s, e, x in S:
            if cur and cur[2] == x:
                cur[1] = max(cur[1], e)
            else:
                if cur:
                    turns.append(cur[1] - cur[0])
                    changes += 1
                cur = [s, e, x]
        if cur:
            turns.append(cur[1] - cur[0])
        speech += union([(s, e) for s, e, _ in S])
        ov += overlap(S)
    lo, hi = min(counts), max(counts)
    return {
        "set": key, "label": label, "recordings": len(files), "audio_h": f"{audio / 3600:.2f}",
        "channel": channel, "speakers_min": lo, "speakers_max": hi,
        "distinct_speakers": distinct_speakers(key, uris, labels),
        "speech_pct": f"{100 * speech / audio:.1f}", "overlap_pct": f"{100 * ov / speech:.1f}",
        "median_segment_s": f"{st.median(segs):.2f}", "segments_under_1s_pct": f"{100 * sum(d < 1 for d in segs) / len(segs):.1f}",
        "median_turn_s": f"{st.median(turns):.2f}", "changes_per_speech_min": f"{changes / (speech / 60):.1f}",
    }


if __name__ == "__main__":
    rows = [measure(*s) for s in SETS]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(OUT.read_text())
