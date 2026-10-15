# webrtc vad settings on the dev set, scored as speech/non-speech only (miss + false alarm, collar 0.25)
# usage: ./.venv/bin/python scripts/vad_sweep_dev.py [--silero]   (all 18 dev meetings, ~8 min under load)
import sys
from pathlib import Path
import numpy as np, webrtcvad
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_mono16k
from score import load_annotation, load_uem
from pyannote.core import Annotation, Segment
from pyannote.metrics.detection import DetectionErrorRate

def flags(pcm16, sr, aggr, ms=30):
    v = webrtcvad.Vad(aggr); n = int(sr * ms / 1000)
    return np.array([v.is_speech(pcm16[i*n:(i+1)*n].tobytes(), sr) for i in range(len(pcm16) // n)]), ms / 1000

def segs(f, step, pad, min_s=0.3, bridge=0.0):
    out, st = [], None
    for i, x in enumerate(list(f) + [False]):
        if x and st is None: st = i
        elif not x and st is not None: out.append((st*step, i*step)); st = None
    m = []
    for s, e in [(max(0, s - pad), e + pad) for s, e in out]:
        if m and s <= m[-1][1] + bridge: m[-1] = (m[-1][0], max(m[-1][1], e))
        else: m.append((s, e))
    return [(s, e) for s, e in m if e - s >= min_s]

def silero_probs(audio, model):
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    s = ort.InferenceSession(str(model), sess_options=o, providers=["CPUExecutionProvider"])
    h = c = np.zeros((2, 1, 64), np.float32); n = 512; out = []
    for i in range(len(audio) // n):
        p, h, c = s.run(None, {"x": audio[None, i*n:(i+1)*n], "h": h, "c": c})
        out.append(p[0, 0])
    return np.array(out), n / 16000

R = ROOT
uris = open(f"{R}/ref/ami/dev.meetings.txt").read().split()
SILERO = "--silero" in sys.argv
# silero: (threshold, pad, bridge); webrtc: (aggressiveness, pad)
grid = ([(t, p, b) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for p in (0.0, 0.1, 0.2) for b in (0.0, 0.3)] if SILERO
        else [(a, p) for a in (0, 1, 2, 3) for p in (0.0, 0.1, 0.2)])
tot = {g: [0.0, 0.0, 0.0] for g in grid}
for uri in uris:
    audio, sr = load_mono16k(f"{R}/data/ami/audio/{uri}.Mix-Headset.wav", target_sr=16000)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    ref = load_annotation(f"{R}/ref/ami/only_words/rttms/dev/{uri}.rttm", uri)
    uem = load_uem(f"{R}/ref/ami/uems/dev/{uri}.uem", uri)
    if SILERO:
        prob, step = silero_probs(audio, R / "models/silero/silero_vad.onnx")
    else:
        fl = {a: flags(pcm, sr, a) for a in (0, 1, 2, 3)}
    for g in grid:
        if SILERO:
            f, p, b = prob > g[0], g[1], g[2]
        else:
            (f, step), p, b = fl[g[0]], g[1], 0.0
        hyp = Annotation(uri=uri)
        for s, e in segs(f, step, p, bridge=b): hyp[Segment(s, e)] = "speech"
        d = DetectionErrorRate(collar=0.25)(ref, hyp, uem=uem, detailed=True)
        t = tot[g]; t[0] += d["miss"]; t[1] += d["false alarm"]; t[2] += d["total"]
    print(uri, flush=True)
for g, (mi, fa, T) in sorted(tot.items(), key=lambda kv: (kv[1][0] + kv[1][1]) / kv[1][2]):
    print(("thr={} pad={} bridge={}" if SILERO else "aggr={} pad={}").format(*g), f" miss={mi/T:.3f} fa={fa/T:.3f} total={(mi+fa)/T:.3f}")
