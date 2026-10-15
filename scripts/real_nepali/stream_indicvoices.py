#!/usr/bin/env python3
"""Stream an IndicVoices Nepali tgz straight off the server and keep only the "Conversation"
files (json + wav). The archive itself is never written to disk.

usage: stream_indicvoices.py URL OUT_DIR [--only UUID_PREFIX]
Writes OUT_DIR/{name}.json and .wav for Conversation files, and OUT_DIR/_scan.json with counts
for every scenario seen. Stops if free disk drops under 30 GB.
"""
import argparse
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

MIN_FREE = int(float(__import__("os").environ.get("MIN_FREE_GB", "30")) * 1024**3)


class ResumingReader:
    """file-like over an http body that reconnects with a Range header when the connection drops,
    so a 24 GB gzip stream survives a reset without starting over."""

    def __init__(self, url):
        self.url, self.pos, self.resp, self.tries = url, 0, None, 0
        self._open()

    def _open(self):
        req = urllib.request.Request(self.url)
        if self.pos:
            req.add_header("Range", f"bytes={self.pos}-")
        self.resp = urllib.request.urlopen(req, timeout=120)
        if self.pos and self.resp.status != 206:
            raise RuntimeError(f"server ignored range request, status {self.resp.status}")
        self.total = int(self.resp.headers.get("Content-Length", 0)) + self.pos

    def read(self, n=-1):
        while True:
            try:
                b = self.resp.read(n)
                if not b and n != 0 and self.pos < self.total:
                    # an idle connection (after a disk pause) can end quietly, not with an error
                    raise ConnectionError(f"early end of body at {self.pos} of {self.total}")
                self.pos += len(b)
                self.tries = 0
                return b
            except Exception as e:  # noqa: BLE001  reconnect on any network error
                self.tries += 1
                if self.tries > 20:
                    raise
                print(f"read error at {self.pos}: {e!r}, reconnect {self.tries}", flush=True)
                time.sleep(min(60, 5 * self.tries))
                try:
                    self._open()
                except Exception as e2:  # noqa: BLE001
                    print(f"reconnect failed: {e2!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only", default=None, help="keep only files whose name starts with this")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    src = ResumingReader(a.url)
    keep, scen, scen_dur = set(), Counter(), Counter()
    n_wav = n_json = 0
    t0 = last = time.time()
    with tarfile.open(fileobj=src, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.rsplit("/", 1)[-1]
            stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
            if a.only and not stem.startswith(a.only):
                continue
            if ext == "json":
                n_json += 1
                raw = tf.extractfile(m).read()
                d = json.loads(raw)
                s = d.get("scenario")
                scen[s] += 1
                scen_dur[s] += float(d.get("audio_raw_duration") or 0)
                if s == "Conversation":
                    keep.add(stem)
                    (a.out_dir / name).write_bytes(raw)
            elif ext == "wav":
                n_wav += 1
                if stem in keep:
                    with open(a.out_dir / name, "wb") as f:
                        shutil.copyfileobj(tf.extractfile(m), f)
            if time.time() - last > 60:
                last = time.time()
                free = shutil.disk_usage("/").free
                print(f"{time.strftime('%H:%M:%S')} {src.pos / 1e9:.2f}/{src.total / 1e9:.2f} GB "
                      f"json={n_json} wav={n_wav} conv_kept={len(keep)} free={free / 1024**3:.1f} GB "
                      f"rate={src.pos / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)
                # other jobs share this disk, so wait for room instead of dying; the reader reconnects with Range
                waited = 0
                while shutil.disk_usage("/").free < MIN_FREE:
                    if waited % 300 == 0:
                        print(f"PAUSE: free disk under floor ({shutil.disk_usage('/').free / 1024**3:.1f})", flush=True)
                    if waited > 6 * 3600:
                        print("STOP: free disk under floor for 6 h", flush=True)
                        sys.exit(1)
                    time.sleep(30)
                    waited += 30
    scan = {"url": a.url, "bytes": src.pos, "json": n_json, "wav": n_wav,
            "scenario_files": dict(scen), "scenario_hours": {k: round(v / 3600, 3) for k, v in scen_dur.items()},
            "conversation_files_kept": len(keep)}
    (a.out_dir / "_scan.json").write_text(json.dumps(scan, indent=1, ensure_ascii=False))
    print("DONE", json.dumps(scan, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
