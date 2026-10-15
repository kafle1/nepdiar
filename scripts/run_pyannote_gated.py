#!/usr/bin/env python3
"""Gated pyannote pipelines: speaker-diarization-3.1 and
speaker-diarization-community-1, run as released (no retuning).

Both need a Hugging Face login whose account has accepted the model's terms.
The token is read from --hf-token, HF_TOKEN or ~/.cache/huggingface/token and
is never printed or written anywhere. With no token it prints why and exits 0.

Each runs in its own venv so torch never touches the scoring venv:
  3.1:          .venv-pyannote3 (pyannote.audio 3.3.2, torch 2.5.1), the version 3.1 was released for.
                pyannote.audio 4 also pulls community-1's PLDA files even for 3.1, so it needs that gate too.
  community-1:  .venv-pyannote (pyannote.audio 4.0.7)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_mono16k, peak_rss_mb, ref_num_speakers, write_rttm

MODELS = {
    "3.1": "pyannote/speaker-diarization-3.1",
    "community-1": "pyannote/speaker-diarization-community-1",
}


def find_token(cli_token=None):
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        tok = token_file.read_text().strip()
        if tok:
            return tok
    return None


def run_file(pipeline, wav_path, out_rttm, uri, csv_writer, num_speakers, note):
    import torch

    audio, sr = load_mono16k(wav_path)
    duration = len(audio) / sr
    t0 = time.time()
    # in-memory waveform, so the result never depends on which audio decoder is installed
    out = pipeline({"waveform": torch.from_numpy(audio).float()[None], "sample_rate": sr},
                   num_speakers=num_speakers)
    proc_time = time.time() - t0
    # pyannote.audio 4 wraps the annotation; 3.x returns it directly
    diarization = getattr(out, "speaker_diarization", out)

    segments = [
        (segment.start, segment.end, label)
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]
    write_rttm(out_rttm, segments, uri)

    mem_mb = peak_rss_mb()
    n_speakers = len({label for _, _, label in segments})
    print(
        f"{uri}: dur={duration:.1f}s proc={proc_time:.1f}s "
        f"rtf={proc_time / duration:.3f} speakers={n_speakers} peak_rss={mem_mb:.0f}MB",
        flush=True,
    )
    csv_writer.writerow(
        {
            "uri": uri,
            "duration_s": f"{duration:.2f}",
            "proc_time_s": f"{proc_time:.2f}",
            "rtf": f"{proc_time / duration:.4f}",
            "num_speakers_hyp": n_speakers,
            "peak_rss_mb": f"{mem_mb:.0f}",
            "timing_note": note,
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--meetings-file", type=Path, default=None,
                     help="Optional file with one recording id per line; else glob *<suffix>")
    ap.add_argument("--suffix", default=".wav", help="wav filename suffix after the id")
    ap.add_argument("--timing-csv", type=Path, required=True)
    ap.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN / the cached login")
    ap.add_argument("--oracle-ref-dir", type=Path, default=None,
                    help="take the number of speakers from {dir}/{uri}.rttm (oracle-k run)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--timing-note", default="under load, rerun idle")
    args = ap.parse_args()

    token = find_token(args.hf_token)
    if not token:
        print(
            "SKIP: no Hugging Face token found (checked --hf-token, HF_TOKEN, "
            "~/.cache/huggingface/token). Log in with `huggingface-cli login` "
            "using an HF account that has accepted the model's terms, then re-run.",
            file=sys.stderr,
        )
        sys.exit(0)

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        print(
            "SKIP: pyannote.audio is not installed in this venv. Use "
            ".venv-pyannote/bin/python (see the docstring).",
            file=sys.stderr,
        )
        sys.exit(0)

    if args.threads:
        torch.set_num_threads(args.threads)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.timing_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.meetings_file:
        ids = [l.strip() for l in args.meetings_file.read_text().splitlines() if l.strip()]
        wavs = [(mid, args.wav_dir / f"{mid}{args.suffix}") for mid in ids]
    else:
        wavs = [(p.name[: -len(args.suffix)], p) for p in sorted(args.wav_dir.glob(f"*{args.suffix}"))]

    # pyannote.audio 3.x calls it use_auth_token, 4.x calls it token
    import inspect
    kw = "token" if "token" in inspect.signature(Pipeline.from_pretrained).parameters else "use_auth_token"
    pipeline = Pipeline.from_pretrained(MODELS[args.model], **{kw: token})
    if pipeline is None:
        sys.exit(f"could not load {MODELS[args.model]}: accept its terms on huggingface.co with this account")
    pipeline.to(torch.device(args.device))

    write_header = not args.timing_csv.exists()
    with open(args.timing_csv, "a", newline="") as f:
        fieldnames = ["uri", "duration_s", "proc_time_s", "rtf", "num_speakers_hyp", "peak_rss_mb", "timing_note"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for uri, wav_path in wavs:
            if not wav_path.exists():
                print(f"SKIP {uri}: missing {wav_path}", file=sys.stderr)
                continue
            out_rttm = args.out_dir / f"{uri}.rttm"
            if out_rttm.exists():
                print(f"SKIP {uri}: already have {out_rttm}")
                continue
            k = ref_num_speakers(args.oracle_ref_dir, uri) if args.oracle_ref_dir else None
            run_file(pipeline, wav_path, out_rttm, uri, writer, k, args.timing_note)
            f.flush()


if __name__ == "__main__":
    main()
