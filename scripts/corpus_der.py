#!/usr/bin/env python3
"""Corpus-level (pooled) DER per system: total error time over total speech
time across all test meetings. This is how pyannote reports its AMI numbers,
so it's the figure to set next to a published one. summarise.py gives the
per-meeting mean with bootstrap intervals instead.

usage: ./.venv/bin/python scripts/corpus_der.py [system ...]   (default: every complete run)
"""
import sys
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from score import load_annotation, load_uem

REF, UEM, HYP = ROOT / "ref/ami/only_words/rttms/test", ROOT / "ref/ami/uems/test", ROOT / "results/raw/ami"
uris = [u.strip() for u in (ROOT / "ref/ami/test.meetings.txt").read_text().splitlines() if u.strip()]
systems = sys.argv[1:] or sorted(p.name[:-5] for p in HYP.glob("*_rttm")
                                 if all((p / f"{u}.rttm").exists() for u in uris))

print(f"{'system':34} {'DER0':>6} {'miss':>6} {'fa':>6} {'conf':>6} {'DER25':>6}  (pooled over {len(uris)} meetings)")
for s in systems:
    m0, m25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
    for u in uris:
        ref, uem = load_annotation(REF / f"{u}.rttm", u), load_uem(UEM / f"{u}.uem", u)
        hyp = load_annotation(HYP / f"{s}_rttm" / f"{u}.rttm", u)
        m0(ref, hyp, uem=uem)
        m25(ref, hyp, uem=uem)
    d = m0[:]
    t = d["total"]
    print(f"{s:34} {abs(m0):6.4f} {d['missed detection'] / t:6.4f} {d['false alarm'] / t:6.4f} "
          f"{d['confusion'] / t:6.4f} {abs(m25):6.4f}")
