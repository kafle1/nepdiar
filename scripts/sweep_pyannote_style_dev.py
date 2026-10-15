#!/usr/bin/env python3
"""Grid-search system 3's clustering settings on the AMI dev caches only.

Scores in-process (same metric as score.py) and prints corpus-level DER, so a
full grid costs minutes. Test meetings are never read here.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.cluster.hierarchy import linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, ref_num_speakers
from run_pyannote_style import diarize, train_set
from score import load_annotation, load_uem


def floats(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "results/cache/pyannote_style")
    ap.add_argument("--meetings-file", type=Path, default=ROOT / "ref/ami/dev.meetings.txt")
    ap.add_argument("--thresholds", type=floats, default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--min-cluster-sizes", type=floats, default="6,12,24")
    ap.add_argument("--min-clean-ratios", type=floats, default="0.0,0.2")
    ap.add_argument("--min-offs", type=floats, default="0.0")
    ap.add_argument("--oracle", action="store_true", help="true speaker count from the dev references")
    args = ap.parse_args()

    ref_dir, uem_dir = ROOT / "ref/ami/only_words/rttms/dev", ROOT / "ref/ami/uems/dev"
    uris = [u for u in (l.strip() for l in args.meetings_file.read_text().splitlines())
            if u and (args.cache_dir / f"{u}.npz").exists()]
    if "test" in str(args.meetings_file):
        sys.exit("dev only: never tune on test")
    print(f"{len(uris)} dev meetings cached: {' '.join(uris)}")
    data = {u: dict(np.load(args.cache_dir / f"{u}.npz")) for u in uris}
    refs = {u: load_annotation(ref_dir / f"{u}.rttm", u) for u in uris}
    uems = {u: load_uem(uem_dir / f"{u}.uem", u) for u in uris}
    true_k = {u: ref_num_speakers(ref_dir, u) for u in uris}
    Zs = {}

    rows = []
    for mcr, thr, mcs, moff in itertools.product(args.min_clean_ratios, args.thresholds,
                                                 args.min_cluster_sizes, args.min_offs):
        m0, m25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
        kdiff = []
        for u in uris:
            d = data[u]
            if (u, mcr) not in Zs:
                ci, si = train_set(d["seg"], d["emb"], mcr)
                Zs[u, mcr] = linkage(d["emb"][ci, si], method="centroid", metric="euclidean") if len(ci) > 1 else None
            segs, k = diarize(d, thr, int(mcs), mcr, true_k[u] if args.oracle else None, None, moff, Zs[u, mcr])
            hyp = Annotation(uri=u)
            for a, b, lab in segs:
                hyp[Segment(a, b)] = lab
            m0(refs[u], hyp, uem=uems[u])
            m25(refs[u], hyp, uem=uems[u])
            kdiff.append(k - true_k[u])
        r25 = m25[:]
        tot = r25["total"]
        rows.append((abs(m25), abs(m0), r25["missed detection"] / tot, r25["false alarm"] / tot,
                     r25["confusion"] / tot, mcr, thr, int(mcs), moff, np.mean(np.abs(kdiff)), kdiff))
        print(f"mcr={mcr} thr={thr} mcs={int(mcs)} off={moff}: DER25={abs(m25):.4f} DER0={abs(m0):.4f} "
              f"k-err={np.mean(np.abs(kdiff)):.2f}", flush=True)

    print("\nsorted by DER at collar 0.25 (corpus level)")
    print("DER25   DER0    miss   fa     conf   mcr  thr   mcs off  |k err| per-meeting k - true")
    for r in sorted(rows)[:15]:
        print(f"{r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:<4} {r[6]:<5} {r[7]:<3} {r[8]:<4} "
              f"{r[9]:.2f}  {r[10]}")


if __name__ == "__main__":
    main()
