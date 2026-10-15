#!/usr/bin/env python3
"""System 2: WeSpeaker embeddings + spectral clustering (VBx-style front end,
without the VB-HMM resegmentation step -- see SYSTEMS.md for why).

VAD: WebRTC VAD (Google, GMM-based, no model download, aggressiveness 3, tuned on dev).
Embedding: WeSpeaker en-voxceleb ResNet34 (LM), same ONNX checkpoint as
system 1, run here on hand-rolled Kaldi-style fbank features
(kaldi-native-fbank, no torch needed).
Clustering: NME-SC auto-tuned spectral clustering (Park et al. 2019). The
cosine affinity is binarized to each row's top-p neighbours and symmetrized,
p and the number of speakers are both picked from the eigengaps of that
graph's Laplacian (capped at MAX_SPEAKERS). This is the part that differs
most from system 1's agglomerative FastClustering.

# ponytail: no VB-HMM resegmentation pass (the "Bx" in VBx); add if the
# spectral-clustering DER trails VBx published numbers by a lot. This is
# closer to a "WeSpeaker + spectral clustering" baseline than full VBx.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import webrtcvad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_mono16k, peak_rss_mb, ref_num_speakers, write_rttm

EMB_MODEL = ROOT / "models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.onnx"
SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
WIN_S = 1.5
HOP_S = 0.75
MAX_SPEAKERS = 10
MIN_SPEECH_S = 0.3
# NME-SC p search: 30 values of p in [1, 0.15 * N], NeMo's NMESC defaults
P_MAX_RATIO = 0.15
P_SEARCH = 30


def run_vad(audio, sr, aggressiveness=3):  # 3 beat 2 on dev speech/non-speech error, 0.124 vs 0.147 (NOTES.md)
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * VAD_FRAME_MS / 1000)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    n_frames = len(pcm16) // frame_len
    flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        chunk = pcm16[i * frame_len:(i + 1) * frame_len].tobytes()
        flags[i] = vad.is_speech(chunk, sr)
    # merge into (start_s, end_s) speech segments, padding 0.2s each side
    segs = []
    in_seg = False
    for i, f in enumerate(flags):
        t = i * VAD_FRAME_MS / 1000
        if f and not in_seg:
            start = t
            in_seg = True
        elif not f and in_seg:
            segs.append((start, t))
            in_seg = False
    if in_seg:
        segs.append((start, n_frames * VAD_FRAME_MS / 1000))
    # pad and merge overlaps
    pad = 0.2
    padded = [(max(0, s - pad), e + pad) for s, e in segs]
    merged = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= MIN_SPEECH_S]


SILERO_MODEL = ROOT / "models/silero/silero_vad.onnx"
# picked on the 18 dev meetings: speech/non-speech error 0.077 vs webrtc's 0.124 (results/raw/ami_dev/vad_sweep_silero.txt)
SILERO_THR, SILERO_PAD, SILERO_BRIDGE = 0.6, 0.2, 0.3


def run_silero_vad(audio, sr):  # same steps as vad_sweep_dev.py, so test gets exactly the dev-picked setting
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(SILERO_MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
    h = c = np.zeros((2, 1, 64), np.float32)
    n, flags = 512, []
    for i in range(len(audio) // n):
        p, h, c = sess.run(None, {"x": audio[None, i * n:(i + 1) * n], "h": h, "c": c})
        flags.append(p[0, 0] > SILERO_THR)
    step = n / sr
    runs, st = [], None
    for i, x in enumerate(flags + [False]):
        if x and st is None:
            st = i
        elif not x and st is not None:
            runs.append((st * step, i * step))
            st = None
    merged = []
    for s, e in [(max(0, s - SILERO_PAD), e + SILERO_PAD) for s, e in runs]:
        if merged and s <= merged[-1][1] + SILERO_BRIDGE:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= MIN_SPEECH_S]


def compute_fbank(wav_chunk, sr=SAMPLE_RATE, num_mel_bins=80):
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sr
    opts.frame_opts.dither = 0.0
    opts.mel_opts.num_bins = num_mel_bins
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sr, wav_chunk.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    if n == 0:
        return None
    feats = np.stack([fbank.get_frame(i) for i in range(n)])
    feats -= feats.mean(axis=0, keepdims=True)  # per-utterance CMN
    return feats.astype(np.float32)


class Embedder:
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, wav_chunk):
        feats = compute_fbank(wav_chunk)
        if feats is None or feats.shape[0] < 5:
            return None
        feats = feats[np.newaxis, :, :]
        (emb,) = self.sess.run(None, {self.in_name: feats})
        v = emb[0]
        return v / (np.linalg.norm(v) + 1e-9)


def sliding_windows(speech_segs, win_s=WIN_S, hop_s=HOP_S):
    """Yield (start, end) windows covering the speech regions."""
    for seg_start, seg_end in speech_segs:
        dur = seg_end - seg_start
        if dur <= win_s:
            yield (seg_start, seg_end)
            continue
        t = seg_start
        while t + win_s <= seg_end:
            yield (t, t + win_s)
            t += hop_s
        if t < seg_end:
            yield (max(seg_end - win_s, seg_start), seg_end)


def binarized_laplacian(sim, order, p):
    a = np.zeros_like(sim)
    np.put_along_axis(a, order[:, :p], 1.0, axis=1)  # keep each row's top-p
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    return np.diag(a.sum(axis=1)) - a


def cluster(embeddings, max_speakers=MAX_SPEAKERS, num_speakers=None):
    # NME-SC: Park et al. 2019, "Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap", IEEE SPL
    from sklearn.cluster import KMeans

    n = embeddings.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    sim = embeddings @ embeddings.T  # cosine, embeddings already L2-normalized
    order = np.argsort(-sim, axis=1)
    kmax = min(max_speakers, n - 1)
    best = None
    for p in np.unique(np.linspace(1, max(2, int(n * P_MAX_RATIO)), min(P_SEARCH, n)).astype(int)):
        lam = np.linalg.eigvalsh(binarized_laplacian(sim, order, p))
        gaps = np.diff(lam[: kmax + 1])
        ratio = p / (gaps.max() / (lam[-1] + 1e-10) + 1e-10)  # p over the normalized max eigengap
        if best is None or ratio < best[0]:
            best = (ratio, p, int(gaps.argmax()) + 1)
    _, p, k = best
    k = min(num_speakers or k, n)
    if k == 1:
        return np.zeros(n, dtype=int)
    _, vecs = np.linalg.eigh(binarized_laplacian(sim, order, p))
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs[:, :k])


def diarize_file(embedder, wav_path, sample_rate=SAMPLE_RATE, num_speakers=None, vad="webrtc"):
    audio, sr = load_mono16k(wav_path, target_sr=sample_rate)
    duration = len(audio) / sr
    speech_segs = run_silero_vad(audio.astype(np.float32), sr) if vad == "silero" else run_vad(audio, sr)
    windows = list(sliding_windows(speech_segs))

    embeddings, kept_windows = [], []
    for (s, e) in windows:
        chunk = audio[int(s * sr):int(e * sr)]
        emb = embedder(chunk)
        if emb is not None:
            embeddings.append(emb)
            kept_windows.append((s, e))

    if not embeddings:
        return [], duration, 0

    embeddings = np.stack(embeddings)
    labels = cluster(embeddings, num_speakers=num_speakers)
    num_speakers = len(set(labels.tolist()))

    # majority-vote per-window labels back onto a frame grid, then read off
    # contiguous same-speaker segments (10ms grid is plenty for RTTM output).
    grid_hop = 0.01
    n_grid = int(duration / grid_hop) + 1
    votes = [[] for _ in range(n_grid)]
    for (s, e), lab in zip(kept_windows, labels):
        gi0, gi1 = int(s / grid_hop), min(int(e / grid_hop), n_grid - 1)
        for gi in range(gi0, gi1):
            votes[gi].append(lab)

    frame_label = [max(set(v), key=v.count) if v else -1 for v in votes]

    segments = []
    cur_label, cur_start = None, None
    for i, lab in enumerate(frame_label):
        t = i * grid_hop
        if lab != cur_label:
            if cur_label is not None and cur_label != -1:
                segments.append((cur_start, t, f"SPK{cur_label:02d}"))
            cur_label, cur_start = lab, t
    if cur_label is not None and cur_label != -1:
        segments.append((cur_start, n_grid * grid_hop, f"SPK{cur_label:02d}"))

    return segments, duration, num_speakers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--meetings-file", type=Path, default=None)
    ap.add_argument("--suffix", default=".wav")
    ap.add_argument("--timing-csv", type=Path, required=True)
    ap.add_argument("--oracle-ref-dir", type=Path, default=None,
                    help="take the number of speakers from {dir}/{uri}.rttm (oracle-k run)")
    ap.add_argument("--timing-note", default="under load, rerun idle")
    ap.add_argument("--vad", default="webrtc", choices=["webrtc", "silero"])
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

    embedder = Embedder(EMB_MODEL)

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
            oracle_k = ref_num_speakers(args.oracle_ref_dir, uri) if args.oracle_ref_dir else None
            t0 = time.time()
            segments, duration, num_speakers = diarize_file(embedder, wav_path, num_speakers=oracle_k, vad=args.vad)
            proc_time = time.time() - t0
            write_rttm(out_rttm, segments, uri)
            mem_mb = peak_rss_mb()
            print(
                f"{uri}: dur={duration:.1f}s proc={proc_time:.1f}s "
                f"rtf={proc_time / duration:.3f} speakers={num_speakers} peak_rss={mem_mb:.0f}MB"
            )
            writer.writerow(
                {
                    "uri": uri,
                    "duration_s": f"{duration:.2f}",
                    "proc_time_s": f"{proc_time:.2f}",
                    "rtf": f"{proc_time / duration:.4f}",
                    "num_speakers_hyp": num_speakers,
                    "peak_rss_mb": f"{mem_mb:.0f}",
                    "timing_note": args.timing_note,
                }
            )
            f.flush()


if __name__ == "__main__":
    main()
