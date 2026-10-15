#!/usr/bin/env python3
"""Build the IndicVoices Nepali two-party call release: RTTM with anonymous labels A/B, UEMs and a
manifest of source file ids. No speaker ids, no gender, no audio.
Writes data/nepali_real/release/indicvoices_ne_calls/."""
import csv
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data/nepali_real/indicvoices"
OUT = D.parent / "release/indicvoices_ne_calls"
ARCHIVE = {"raw_v1_train": "v1_Nepali_train.tgz", "raw_v2_train": "v2_Nepali_train.tgz",
           "raw_v2_valid": "v2_Nepali_valid.tgz"}


def main():
    uris = [u.strip() for u in (D / "calls.txt").read_text().splitlines() if u.strip()]
    st = {r["uri"]: r for r in csv.DictReader(open(D / "stats.csv"))}
    for sub in ("rttm", "uem"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for u in uris:
        lines = []
        for line in open(D / "rttm" / f"{u}.rttm"):
            f = line.split()
            f[7] = {"side0": "A", "side1": "B"}[f[7]]
            lines.append(" ".join(f))
        (OUT / "rttm" / f"{u}.rttm").write_text("\n".join(lines) + "\n")
        (OUT / "uem" / f"{u}.uem").write_text((D / "uem" / f"{u}.uem").read_text())
        r = st[u]
        rows.append({"call_id": u, "side_A_file": f"{u}_0", "side_B_file": f"{u}_1",
                     "source_archive": ARCHIVE[r["source"]], "duration_s": r["duration_s"],
                     "speech_s": r["speech_s"], "overlap_s": r["overlap_s"]})
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} calls written to {OUT}")


if __name__ == "__main__":
    main()
