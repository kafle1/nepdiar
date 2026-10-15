#!/usr/bin/env python3
"""System 1: sherpa-onnx offline speaker diarization.

Segmentation: pyannote/segmentation-3.0, ONNX export shipped by the
sherpa-onnx project (MIT license, no login needed).
Embedding: WeSpeaker en-voxceleb ResNet34 (LM), ONNX export shipped by the
sherpa-onnx project.
Clustering: sherpa-onnx's built-in agglomerative clustering (FastClustering)
with a distance threshold, since the number of speakers per meeting/file is
not known in advance.

Runs one file at a time, under nice, and writes one RTTM per input file
plus a timing/memory row per file to a CSV.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import sherpa_onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_mono16k, peak_rss_mb, ref_num_speakers, write_rttm

SEG_MODEL = ROOT / "models/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL = ROOT / "models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.onnx"
# same weights with feature_normalize_type=global-mean, see scripts/add_cmn_flag.py
EMB_MODEL_CMN = EMB_MODEL.with_suffix(".cmn.onnx")


def build_config(num_speakers=-1, threshold=0.5, num_threads=1, emb_model=EMB_MODEL_CMN):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL)
            ),
            num_threads=num_threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=num_threads, provider="cpu"
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError(f"invalid sherpa-onnx config: {config}")
    return config


def run_file(sd, wav_path, out_rttm, uri, csv_writer, timing_note):
    audio, sr = load_mono16k(wav_path, target_sr=sd.sample_rate)
    duration = len(audio) / sr

    t0 = time.time()
    result = sd.process(audio.tolist())
    rtf_wall = time.time() - t0

    segments = [(s.start, s.end, f"SPK{s.speaker:02d}") for s in result.sort_by_start_time()]
    write_rttm(out_rttm, segments, uri)

    mem_mb = peak_rss_mb()
    print(
        f"{uri}: dur={duration:.1f}s proc={rtf_wall:.1f}s "
        f"rtf={rtf_wall / duration:.3f} speakers={result.num_speakers} peak_rss={mem_mb:.0f}MB"
    )
    csv_writer.writerow(
        {
            "uri": uri,
            "duration_s": f"{duration:.2f}",
            "proc_time_s": f"{rtf_wall:.2f}",
            "rtf": f"{rtf_wall / duration:.4f}",
            "num_speakers_hyp": result.num_speakers,
            "peak_rss_mb": f"{mem_mb:.0f}",
            "timing_note": timing_note,
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--meetings-file", type=Path, default=None,
                     help="Optional file with one recording id per line; else glob *.wav")
    ap.add_argument("--suffix", default=".wav", help="wav filename suffix after the id")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--timing-csv", type=Path, required=True)
    ap.add_argument("--oracle-ref-dir", type=Path, default=None,
                    help="take the number of speakers from {dir}/{uri}.rttm (oracle-k run)")
    ap.add_argument("--timing-note", default="under load, rerun idle")
    ap.add_argument("--num-threads", type=int, default=1, help="onnxruntime threads; timed runs keep 1")
    ap.add_argument("--emb-model", type=Path, default=EMB_MODEL_CMN,
                    help="embedding onnx; pass models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.onnx for the old no-CMN run")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.timing_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.meetings_file:
        ids = [l.strip() for l in args.meetings_file.read_text().splitlines() if l.strip()]
        wavs = [(mid, args.wav_dir / f"{mid}{args.suffix}") for mid in ids]
    else:
        # p.stem only strips the last dot-suffix, which breaks compound
        # suffixes like ".Mix-Headset.wav" (would leave "ES2004a.Mix-Headset"
        # instead of "ES2004a", so it wouldn't match the reference rttm name).
        wavs = [(p.name[: -len(args.suffix)], p) for p in sorted(args.wav_dir.glob(f"*{args.suffix}"))]

    sd = sherpa_onnx.OfflineSpeakerDiarization(build_config(threshold=args.threshold, num_threads=args.num_threads, emb_model=args.emb_model))

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
            if args.oracle_ref_dir:
                # set_config only swaps the clustering part, models stay loaded
                sd.set_config(build_config(ref_num_speakers(args.oracle_ref_dir, uri), args.threshold, args.num_threads, args.emb_model))
            run_file(sd, wav_path, out_rttm, uri, writer, args.timing_note)
            f.flush()


if __name__ == "__main__":
    main()
