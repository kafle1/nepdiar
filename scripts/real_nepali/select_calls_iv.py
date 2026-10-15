#!/usr/bin/env python3
"""Pick the scored IndicVoices calls: speaker-disjoint, so every call is independent.
Walk all paired calls in call-id order (the validation call first) and keep a call only when neither
side's speaker id was used by a call already kept. Drops calls prep flagged with two speaker ids
on one side. Keeps the full list as calls_all.txt and writes the pick to calls.txt."""
from pathlib import Path
import csv

D = Path(__file__).resolve().parents[2] / "data/nepali_real/indicvoices"
VALID = "3e9b98be-4e10-415a-9020-3b62bd21cd56"


def main():
    rows = list(csv.DictReader(open(D / "stats.csv")))
    bad = {ln.split("_")[0] for ln in (D / "check.txt").read_text().splitlines() if "segment speaker ids" in ln}
    rows.sort(key=lambda r: (r["uri"] != VALID, r["uri"]))
    used, pick = set(), []
    for r in rows:
        ids = {r["side0_speaker_id"], r["side1_speaker_id"]}
        if r["uri"] in bad or len(ids) < 2 or ids & used:
            continue
        used |= ids
        pick.append(r["uri"])
    (D / "calls_all.txt").write_text("".join(r["uri"] + "\n" for r in sorted(rows, key=lambda r: r["uri"])))
    (D / "calls.txt").write_text("".join(u + "\n" for u in pick))
    hours = sum(float(r["duration_s"]) for r in rows if r["uri"] in set(pick)) / 3600
    print(f"{len(rows)} paired calls, dropped {sorted(bad)}, kept {len(pick)} speaker-disjoint calls, {hours:.3f} h")


if __name__ == "__main__":
    main()
