#!/usr/bin/env python3
"""Score hypothesis RTTMs against reference RTTMs with pyannote.metrics.

Reports, per file: DER with no collar (overlap scored), DER with a 0.25s
collar (overlap scored), and JER. Uses the dataset's UEM file to restrict
scoring to the annotated region when one is provided.

Writes one row per file to a CSV under results/raw/, so paired bootstrap
and Wilcoxon tests can run on it later (see scripts/summarise.py).
"""
import argparse
import csv
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

COLLAR_SETTINGS = {"no_collar": 0.0, "collar_0.25": 0.25}


def load_uem(path, uri):
    tl = Timeline(uri=uri)
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                tl.add(Segment(float(parts[2]), float(parts[3])))
    return tl or None


def load_annotation(rttm_path, uri):
    if not Path(rttm_path).exists():
        return Annotation(uri=uri)
    data = load_rttm(str(rttm_path))
    return data.get(uri, Annotation(uri=uri))


def score_file(ref_rttm, hyp_rttm, uri, uem_path=None):
    ref = load_annotation(ref_rttm, uri)
    hyp = load_annotation(hyp_rttm, uri)
    uem = load_uem(uem_path, uri) if uem_path else None

    row = {"uri": uri, "ref_speakers": len(ref.labels()), "hyp_speakers": len(hyp.labels())}
    for name, collar in COLLAR_SETTINGS.items():
        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        der = metric(ref, hyp, uem=uem, detailed=True)
        row[f"der_{name}"] = der["diarization error rate"]
        row[f"miss_{name}"] = der["missed detection"] / der["total"] if der["total"] else 0.0
        row[f"fa_{name}"] = der["false alarm"] / der["total"] if der["total"] else 0.0
        row[f"conf_{name}"] = der["confusion"] / der["total"] if der["total"] else 0.0
    jer_metric = JaccardErrorRate()
    row["jer"] = jer_metric(ref, hyp, uem=uem)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, type=Path, help="dir of {uri}.rttm reference files")
    ap.add_argument("--hyp-dir", required=True, type=Path, help="dir of {uri}.rttm hypothesis files")
    ap.add_argument("--uem-dir", type=Path, default=None, help="dir of {uri}.uem files")
    ap.add_argument("--meetings-file", type=Path, default=None)
    ap.add_argument("--system", required=True, help="system name, recorded in output CSV")
    ap.add_argument("--dataset", required=True, help="dataset name, recorded in output CSV")
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    if args.meetings_file:
        uris = [l.strip() for l in args.meetings_file.read_text().splitlines() if l.strip()]
    else:
        uris = sorted(p.stem for p in args.ref_dir.glob("*.rttm"))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for uri in uris:
        ref_rttm = args.ref_dir / f"{uri}.rttm"
        hyp_rttm = args.hyp_dir / f"{uri}.rttm"
        uem_path = args.uem_dir / f"{uri}.uem" if args.uem_dir else None
        if not ref_rttm.exists():
            print(f"SKIP {uri}: no reference rttm at {ref_rttm}")
            continue
        if not hyp_rttm.exists():
            print(f"SKIP {uri}: no hypothesis rttm at {hyp_rttm}")
            continue
        row = score_file(ref_rttm, hyp_rttm, uri, uem_path)
        row["system"] = args.system
        row["dataset"] = args.dataset
        rows.append(row)
        print(
            f"{uri}: der_no_collar={row['der_no_collar']:.4f} "
            f"der_collar_0.25={row['der_collar_0.25']:.4f} jer={row['jer']:.4f}"
        )

    if not rows:
        print("no files scored")
        return

    fieldnames = ["system", "dataset", "uri", "ref_speakers", "hyp_speakers"]
    for name in COLLAR_SETTINGS:
        fieldnames += [f"der_{name}", f"miss_{name}", f"fa_{name}", f"conf_{name}"]
    fieldnames.append("jer")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
