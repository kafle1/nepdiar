"""Shared helpers: RTTM I/O, wav loading, resource limits."""
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def load_mono16k(path, target_sr=16000):
    """Load audio as mono float32 at target_sr. Uses ffmpeg for resample/downmix
    so we don't need librosa as a dependency."""
    info = sf.info(str(path))
    if info.samplerate == target_sr and info.channels == 1:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return audio[:, 0], sr
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")  # results were made with ffmpeg 4
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype="<f4").copy()
    return audio, target_sr


def write_rttm(path, segments, uri):
    """segments: list of (start, end, speaker_label)."""
    with open(path, "w") as f:
        for start, end, spk in sorted(segments, key=lambda s: s[0]):
            dur = end - start
            if dur <= 0:
                continue
            f.write(
                f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )


def peak_rss_mb():
    """Peak RSS of this process and children, in MB.
    macOS reports ru_maxrss in bytes; Linux reports it in KB."""
    import platform
    unit = 1 if platform.system() == "Darwin" else 1024
    self_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
    children_b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit
    return (self_b + children_b) / 1024 / 1024


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


def ref_num_speakers(ref_dir, uri):
    """Distinct speakers in the reference {ref_dir}/{uri}.rttm, for oracle-k runs."""
    lines = (Path(ref_dir) / f"{uri}.rttm").read_text().splitlines()
    n = len({l.split()[7] for l in lines if l.startswith("SPEAKER")})
    if n == 0:
        raise ValueError(f"no speakers in reference rttm for {uri}")
    return n
