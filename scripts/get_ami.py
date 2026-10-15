#!/usr/bin/env python3
"""Download AMI Mix-Headset audio (BUT split, only_words references).

Reads the meeting list from ref/ami/test.meetings.txt by default, or the file
given with --meetings-file (e.g. ref/ami/dev.meetings.txt), both vendored from
BUTSpeechFIT/AMI-diarization-setup, and pulls each meeting's Mix-Headset
wav from the Edinburgh AMI corpus mirror. Idempotent: skips files whose
checksum already matches. Writes sha256 sums to ref/ami/audio.sha256 so
the download is auditable and reproducible.

License: AMI Meeting Corpus is CC BY 4.0
(https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).
"""
import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEETINGS_FILE = ROOT / "ref" / "ami" / "test.meetings.txt"
AUDIO_DIR = ROOT / "data" / "ami" / "audio"
CHECKSUM_FILE = ROOT / "ref" / "ami" / "audio.sha256"
BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_known_checksums() -> dict:
    known = {}
    if CHECKSUM_FILE.exists():
        for line in CHECKSUM_FILE.read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            known[name.strip()] = digest.strip()
    return known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings-file", type=Path, default=MEETINGS_FILE)
    meetings_file = ap.parse_args().meetings_file
    meetings = [m.strip() for m in meetings_file.read_text().splitlines() if m.strip()]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    known = load_known_checksums()
    results = {}
    for i, meeting in enumerate(meetings, 1):
        out = AUDIO_DIR / f"{meeting}.Mix-Headset.wav"
        name = out.name
        if out.exists() and known.get(name) == sha256sum(out):
            print(f"[{i}/{len(meetings)}] {meeting}: already present, checksum ok")
            results[name] = known[name]
            continue
        url = BASE_URL.format(m=meeting)
        print(f"[{i}/{len(meetings)}] {meeting}: downloading {url}")
        t0 = time.time()
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            out.unlink(missing_ok=True)  # urlretrieve leaves a truncated wav behind
            continue
        dt = time.time() - t0
        digest = sha256sum(out)
        results[name] = digest
        size_mb = out.stat().st_size / 1e6
        print(f"  done: {size_mb:.1f} MB in {dt:.1f}s, sha256={digest[:12]}...")

    # keep other splits' sums, this file covers every split downloaded so far
    results = {**known, **results}
    with open(CHECKSUM_FILE, "w") as f:
        for name in sorted(results):
            f.write(f"{results[name]}  {name}\n")
    print(f"\nWrote checksums for {len(results)} files to {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()
