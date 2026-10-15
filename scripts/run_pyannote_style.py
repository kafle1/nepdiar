#!/usr/bin/env python3
"""System 3: the pyannote 3.1 speaker-diarization recipe rebuilt on ONNX, no torch.

- segmentation-3.0 on 10 s windows every STEP_S seconds (powerset output -> 3 local speakers)
- one WeSpeaker ResNet34-LM pass per window, then weighted stats pooling per
  local-speaker mask, overlap frames left out when enough clean frames exist
  (pyannote's exclude_overlap)
- centroid-linkage agglomerative clustering, clusters smaller than
  min_cluster_size folded into the nearest large one, then a Hungarian
  assignment per window so two local speakers never share a cluster
- overlap-aware output: each frame keeps its top-N clusters, N being the
  speaker count the segmentation model predicts for that frame

Same models as system 1 (sherpa-onnx). The difference is the clustering and
reconstruction, which sherpa-onnx does without the minimum cluster size.

Two stages, so clustering can be tuned on dev without redoing the network passes:
  extract: wav -> {cache}/{uri}.npz (binarized segmentations + embeddings)
  cluster: npz -> rttm
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_mono16k, peak_rss_mb, ref_num_speakers, write_rttm

SEG_MODEL = ROOT / "models/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
# the stock WeSpeaker ONNX with its pre-pooling tensor "343" added as an output
EMB_MODEL = ROOT / "models/wespeaker/resnet34_LM.frames.onnx"
EMB_HEAD = ROOT / "models/wespeaker/resnet34_LM.head.npz"
SR = 16000
CHUNK = 10 * SR
STEP_S = 2.0  # pyannote uses 1 s; 2 s halves the ResNet passes (NOTES.md)
FRAME = 270  # segmentation frame shift, samples
FRAME_RF = 991  # segmentation receptive field, samples
FB_HOP, FB_WIN = 160, 400
# pyannote's powerset class order for 3 speakers, at most 2 at once
POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
MIN_POOL_FRAMES = 2  # ResNet output frames (80 ms each) a mask needs before it gets an embedding
BATCH = 8


def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def fbank_all(audio):
    # kaldi frames are local to their 25 ms window, so slicing one long fbank
    # gives exactly the per-window features (snip_edges, no dither)
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SR
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    for i in range(0, len(audio), CHUNK):
        fb.accept_waveform(SR, (audio[i:i + CHUNK] * 32768).tolist())
    fb.input_finished()
    return np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)


def pool(frames, seg, head):
    """frames (B, D, T) ResNet output, seg (B, F, 3) binary masks -> (B, 3, 256) unit embeddings.
    Mirrors pyannote's weighted StatsPool; NaN where a mask is too short."""
    _, _, T = frames.shape
    F = seg.shape[1]
    near = (np.arange(T) * F / T).astype(int)  # nearest-neighbour resize, like F.interpolate
    full = seg[:, near, :].astype(np.float64)
    clean = full * (full.sum(-1, keepdims=True) < 2)
    w = np.where(clean.sum(1, keepdims=True) >= MIN_POOL_FRAMES, clean, full)  # (B, T, 3)
    fr = frames.astype(np.float64)
    s1 = np.einsum("bdt,bts->bsd", fr, w)
    s2 = np.einsum("bdt,bts->bsd", fr * fr, w)
    v1 = w.sum(1)[..., None]
    mean = s1 / np.maximum(v1, 1e-8)
    # binary weights, so pyannote's unbiased denominator v1 - v2/v1 is v1 - 1
    var = np.maximum(s2 - v1 * mean ** 2, 0) / np.maximum(v1 - 1, 1e-8)
    e = np.concatenate([mean, np.sqrt(var)], -1) @ head["w"].T.astype(np.float64) + head["b"] - head["mean_vec"]
    e /= np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12
    e[v1[..., 0] < MIN_POOL_FRAMES] = np.nan
    return e.astype(np.float32)


def extract(wav, seg_sess, emb_sess, head):
    audio, _ = load_mono16k(wav, target_sr=SR)
    n = len(audio)
    step = int(STEP_S * SR)
    n_chunks = 1 if n <= CHUNK else int(np.ceil((n - CHUNK) / step)) + 1
    padded = np.zeros((n_chunks - 1) * step + CHUNK, dtype=np.float32)
    padded[:n] = audio
    fb = fbank_all(padded)
    fb_step = step // FB_HOP
    n_fb = 1 + (CHUNK - FB_WIN) // FB_HOP
    segs, embs = [], []
    for b0 in range(0, n_chunks, BATCH):
        idx = range(b0, min(b0 + BATCH, n_chunks))
        x = np.stack([padded[c * step:c * step + CHUNK] for c in idx])[:, None, :]
        s = POWERSET[seg_sess.run(None, {"x": x})[0].argmax(-1)]  # (B, F, 3)
        feats = np.stack([fb[c * fb_step:c * fb_step + n_fb] for c in idx])
        feats -= feats.mean(axis=1, keepdims=True)  # per-window CMN, as WeSpeaker was trained
        fr = emb_sess.run(["343"], {"feats": feats})[0]
        segs.append(s)
        embs.append(pool(fr.reshape(fr.shape[0], -1, fr.shape[-1]), s, head))
    return np.concatenate(segs), np.concatenate(embs), n, step


def cluster_embeddings(train, threshold, min_cluster_size, num_clusters=None, max_clusters=None, Z=None):
    """pyannote.audio AgglomerativeClustering.cluster (method="centroid") on unit vectors."""
    n = len(train)
    if n == 1:
        return np.zeros(1, dtype=int)
    mcs = min(min_cluster_size, max(1, round(0.1 * n)))
    if Z is None:
        Z = linkage(train, method="centroid", metric="euclidean")
    clusters = fcluster(Z, threshold, criterion="distance") - 1
    uniq, counts = np.unique(clusters, return_counts=True)
    large = uniq[counts >= mcs]
    if num_clusters is None and max_clusters and len(large) > max_clusters:
        num_clusters = max_clusters
    if num_clusters is not None and len(large) != num_clusters:
        # walk merges by index, nearest the threshold first, until the large-cluster count fits
        Zi = Z.copy()
        Zi[:, 2] = np.arange(n - 1)
        best_it, best_n = n - 1, 1
        for it in np.argsort(np.abs(Z[:, 2] - threshold)):
            if Zi[it, 3] < mcs:
                continue
            c = fcluster(Zi, it, criterion="distance") - 1
            nl = int((np.unique(c, return_counts=True)[1] >= mcs).sum())
            if abs(nl - num_clusters) < abs(best_n - num_clusters):
                best_it, best_n = it, nl
            if nl == num_clusters:
                break
        clusters = fcluster(Zi, best_it, criterion="distance") - 1
        uniq, counts = np.unique(clusters, return_counts=True)
        large = uniq[counts >= mcs]
    if len(large) == 0:
        return np.zeros(n, dtype=int)
    small = uniq[counts < mcs]
    if len(small):
        lc = np.stack([train[clusters == k].mean(0) for k in large])
        sc = np.stack([train[clusters == k].mean(0) for k in small])
        for s, li in zip(small, cdist(lc, sc, metric="cosine").argmin(0)):
            clusters[clusters == s] = large[li]
    return np.unique(clusters, return_inverse=True)[1]


def train_set(seg, emb, min_clean_ratio):
    active = seg.sum(1) > 0
    valid = ~np.isnan(emb).any(-1)
    clean = (seg * (seg.sum(-1, keepdims=True) == 1)).sum(1)
    keep = active & valid & (clean >= min_clean_ratio * seg.shape[1])
    if not keep.any():
        keep = active & valid
    return np.nonzero(keep)


def diarize(d, threshold, min_cluster_size, min_clean_ratio=0.0, num_speakers=None,
            max_speakers=None, min_off=0.0, Z=None):
    seg, emb = d["seg"], d["emb"]
    C, F, S = seg.shape
    active = seg.sum(1) > 0
    ci, si = train_set(seg, emb, min_clean_ratio)
    if len(ci) == 0:
        return [], 0
    train = emb[ci, si]
    labels = cluster_embeddings(train, threshold, min_cluster_size, num_speakers, max_speakers, Z)
    K = int(labels.max()) + 1
    centroids = np.stack([train[labels == k].mean(0) for k in range(K)])
    with np.errstate(invalid="ignore"):
        sim = 2 - cdist(emb.reshape(-1, emb.shape[-1]), centroids, metric="cosine").reshape(C, S, K)
    sim = np.nan_to_num(sim, nan=np.nanmin(sim))
    hard = np.full((C, S), -2)
    for c in range(C):
        r, k = linear_sum_assignment(sim[c], maximize=True)
        hard[c, r] = k
    hard[~active] = -2

    off = np.rint(np.arange(C) * int(d["step"]) / FRAME).astype(int)
    G = off[-1] + F
    count = np.zeros(G)
    cover = np.zeros(G)
    act = np.zeros((G, K))
    local = seg.sum(-1)
    for c in range(C):
        sl = slice(off[c], off[c] + F)
        count[sl] += local[c]
        cover[sl] += 1
        for k in set(hard[c].tolist()) - {-2}:
            act[sl, k] += seg[c][:, hard[c] == k].max(1)
    count = np.minimum(np.rint(count / np.maximum(cover, 1)), K).astype(int)
    rank = np.argsort(np.argsort(-act, axis=1, kind="stable"), axis=1)
    binary = rank < count[:, None]

    centers = (np.arange(G) * FRAME + FRAME_RF / 2) / SR
    n_frames = int(np.searchsorted(centers, int(d["n_samples"]) / SR))
    if n_frames == 0:
        return [], 0
    binary, centers = binary[:n_frames], centers[:n_frames]
    segments = []
    for k in range(K):
        edges = np.diff(np.concatenate([[0], binary[:, k].astype(np.int8), [0]]))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        t0 = centers[starts]
        t1 = np.append(centers, centers[-1] + FRAME / SR)[ends]
        runs = []
        for a, b in zip(t0, t1):
            if runs and a - runs[-1][1] < min_off:
                runs[-1][1] = b
            else:
                runs.append([a, b])
        segments += [(a, b, f"SPK{k:02d}") for a, b in runs]
    return segments, len({s[2] for s in segments})


def read_ids(meetings_file):
    return [l.strip() for l in meetings_file.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--wav-dir", required=True, type=Path)
    e.add_argument("--meetings-file", required=True, type=Path)
    e.add_argument("--suffix", default=".Mix-Headset.wav")
    e.add_argument("--cache-dir", required=True, type=Path)
    e.add_argument("--timing-csv", required=True, type=Path)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--timing-note", default="under load, rerun idle")
    c = sub.add_parser("cluster")
    c.add_argument("--cache-dir", required=True, type=Path)
    c.add_argument("--meetings-file", required=True, type=Path)
    c.add_argument("--out-dir", required=True, type=Path)
    c.add_argument("--threshold", type=float, required=True)
    c.add_argument("--min-cluster-size", type=int, required=True)
    c.add_argument("--min-clean-ratio", type=float, default=0.0)
    c.add_argument("--min-off", type=float, default=0.0)
    c.add_argument("--max-speakers", type=int, default=None)
    c.add_argument("--oracle-ref-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.cmd == "extract":
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        args.timing_csv.parent.mkdir(parents=True, exist_ok=True)
        seg_sess = session(SEG_MODEL, 1)
        emb_sess = session(EMB_MODEL, args.threads)
        head = dict(np.load(EMB_HEAD))
        new = not args.timing_csv.exists()
        with open(args.timing_csv, "a", newline="") as f:
            w = csv.DictWriter(f, ["uri", "duration_s", "proc_time_s", "rtf", "threads", "step_s", "peak_rss_mb", "timing_note"])
            if new:
                w.writeheader()
            for uri in read_ids(args.meetings_file):
                out = args.cache_dir / f"{uri}.npz"
                wav = args.wav_dir / f"{uri}{args.suffix}"
                if out.exists():
                    print(f"SKIP {uri}: already have {out}")
                    continue
                if not wav.exists():
                    print(f"SKIP {uri}: missing {wav}", file=sys.stderr)
                    continue
                t0 = time.time()
                seg, emb, n, step = extract(wav, seg_sess, emb_sess, head)
                dt = time.time() - t0
                tmp = out.with_suffix(".tmp.npz")
                np.savez(tmp, seg=seg, emb=emb, n_samples=n, step=step)
                tmp.rename(out)  # no half-written cache if the run is killed
                dur = n / SR
                print(f"{uri}: dur={dur:.1f}s proc={dt:.1f}s rtf={dt / dur:.3f} chunks={len(seg)}", flush=True)
                w.writerow({"uri": uri, "duration_s": f"{dur:.2f}", "proc_time_s": f"{dt:.2f}", "rtf": f"{dt / dur:.4f}",
                            "threads": args.threads, "step_s": STEP_S, "peak_rss_mb": f"{peak_rss_mb():.0f}",
                            "timing_note": args.timing_note})
                f.flush()
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for uri in read_ids(args.meetings_file):
            npz = args.cache_dir / f"{uri}.npz"
            if not npz.exists():
                print(f"SKIP {uri}: no cache at {npz}", file=sys.stderr)
                continue
            k = ref_num_speakers(args.oracle_ref_dir, uri) if args.oracle_ref_dir else None
            segments, n_spk = diarize(np.load(npz), args.threshold, args.min_cluster_size, args.min_clean_ratio,
                                      k, args.max_speakers, args.min_off)
            write_rttm(args.out_dir / f"{uri}.rttm", segments, uri)
            print(f"{uri}: speakers={n_spk}")


if __name__ == "__main__":
    main()
