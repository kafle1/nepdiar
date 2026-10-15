#!/usr/bin/env python3
"""Pooled DER (as corpus_der.py), per-recording means with a 10,000-resample bootstrap (seed 0),
JER, speaker-count accuracy and RTF for the real Nepali sets, plus DiarBench by speaker count and
AMI test next to them (AMI read only, from its existing score CSVs and RTTMs).
Writes results/summary_nepali_real.csv; the printed text goes to results/corpus_der_nepali_real.txt."""
import csv
import sys
from pathlib import Path

import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402
from score import load_annotation, load_uem  # noqa: E402
from summarise import bootstrap_ci  # noqa: E402

NB = 10000
SYSTEMS = ["pyannote_style", "pyannote_style_oracle", "pyannote31", "pyannote31_oracle", "sherpa_onnx",
           "sherpa_onnx_oracle", "wespeaker_spectral_silero", "wespeaker_spectral_silero_oracle",
           "wespeaker_spectral", "wespeaker_spectral_oracle", "community1", "community1_oracle"]
SETS = {
    "indic_diarbench_ne": (ROOT / "data/nepali_real/diarbench", "clips.txt"),
    "indicvoices_ne_calls": (ROOT / "data/nepali_real/indicvoices", "calls.txt"),
    "ami_test": (ROOT / "ref/ami", "test.meetings.txt"),
}


def paths(ds, s, u):
    if ds == "ami_test":
        return (ROOT / "ref/ami/only_words/rttms/test" / f"{u}.rttm", ROOT / "ref/ami/uems/test" / f"{u}.uem",
                ROOT / "results/raw/ami" / f"{s}_rttm" / f"{u}.rttm")
    d = SETS[ds][0]
    return d / "rttm" / f"{u}.rttm", d / "uem" / f"{u}.uem", ROOT / "results/raw" / ds / f"{s}_rttm" / f"{u}.rttm"


def rtf(ds, s):
    """audio-weighted RTF from the run's timing csv (System 3 adds its extract stage; cluster is tiny)."""
    base = ROOT / "results/raw" / ("ami" if ds == "ami_test" else ds)
    f = base / ("pyannote_style_extract_timing.csv" if s.startswith("pyannote_style") else f"{s}_timing.csv")
    if not f.exists():
        return "", ""
    rows = list(csv.DictReader(open(f)))
    dur, proc = sum(float(r["duration_s"]) for r in rows), sum(float(r["proc_time_s"]) for r in rows)
    return f"{proc / dur:.3f}", rows[0].get("timing_note", "")


def iv_clusters(d, uris):
    """calls linked by any shared speaker id (connected components), from prep_indicvoices stats.csv."""
    parent = {u: u for u in uris}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for r in csv.DictReader(open(d / "stats.csv")):
        if r["uri"] not in parent:
            continue
        for sid in (r["side0_speaker_id"], r["side1_speaker_id"]):
            if sid in first:
                parent[find(r["uri"])] = find(first[sid])
            else:
                first[sid] = r["uri"]
    return [find(u) for u in uris]


def cluster_ci(err, tot, cl, n=NB, seed=0):
    """pooled DER with a bootstrap over clusters of recordings: resample clusters, pool error over speech."""
    keys = sorted(set(cl))
    ce = np.array([sum(e for e, c in zip(err, cl) if c == k) for k in keys])
    ct = np.array([sum(t for t, c in zip(tot, cl) if c == k) for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n, len(keys)))
    b = ce[idx].sum(1) / ct[idx].sum(1)
    return len(keys), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def block(ds, s, uris, per, label, clusters=None):
    m0, m25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
    for u in uris:
        r, e, h = paths(ds, s, u)
        ref, uem = load_annotation(r, u), load_uem(e, u)
        hyp = load_annotation(h, u)
        m0(ref, hyp, uem=uem)
        m25(ref, hyp, uem=uem)
    d0, d25 = m0[:], m25[:]
    out = {"dataset": ds, "group": label, "system": s, "n": len(uris)}
    for tag, d, m in (("no_collar", d0, m0), ("collar_0.25", d25, m25)):
        t = d["total"]
        out[f"pooled_der_{tag}"] = abs(m)
        out[f"pooled_miss_{tag}"] = d["missed detection"] / t
        out[f"pooled_fa_{tag}"] = d["false alarm"] / t
        out[f"pooled_conf_{tag}"] = d["confusion"] / t
        if clusters is not None:
            c = dict(zip(uris, clusters))
            res = [(c[u], cp) for u, cp in m.results_]
            err = [cp["missed detection"] + cp["false alarm"] + cp["confusion"] for _, cp in res]
            out["n_clusters"], out[f"pooled_der_{tag}_cluster_ci95_lo"], out[f"pooled_der_{tag}_cluster_ci95_hi"] = \
                cluster_ci(err, [cp["total"] for _, cp in res], [k for k, _ in res])
    for k in ("der_no_collar", "der_collar_0.25", "jer"):
        vals = [float(per[u][k]) for u in uris]
        mean, lo, hi = bootstrap_ci(vals, n=NB, seed=0)
        out[f"mean_{k}"], out[f"{k}_ci95_lo"], out[f"{k}_ci95_hi"] = mean, lo, hi
        out[f"min_{k}"], out[f"max_{k}"] = min(vals), max(vals)
    ref_k = np.array([int(per[u]["ref_speakers"]) for u in uris])
    hyp_k = np.array([int(per[u]["hyp_speakers"]) for u in uris])
    out["speaker_count_right"] = int((ref_k == hyp_k).sum())
    out["speaker_count_acc"] = float((ref_k == hyp_k).mean())
    out["mean_hyp_minus_ref_speakers"] = float((hyp_k - ref_k).mean())
    out["mean_abs_speaker_count_err"] = float(np.abs(hyp_k - ref_k).mean())
    out["rtf"], out["timing_note"] = rtf(ds, s) if label == "all" else ("", "")
    return out


def main():
    rows = []
    for ds, (d, lst) in SETS.items():
        uris = [u.strip() for u in (d / lst).read_text().splitlines() if u.strip()] if (d / lst).exists() else []
        if not uris:
            continue
        groups = {"all": uris}
        if ds == "indic_diarbench_ne":
            nspk = {u: len({l.split()[7] for l in open(d / "rttm" / f"{u}.rttm")}) for u in uris}
            groups.update({f"{k}spk": [u for u in uris if nspk[u] == k] for k in sorted(set(nspk.values()))})
            st = {r["uri"]: r for r in csv.DictReader(open(d / "stats.csv"))}
            for rec in sorted({st[u]["recording_id"] for u in uris}):
                ru = [u for u in uris if st[u]["recording_id"] == rec]
                groups[f"rec_{rec}"] = ru
                dur = sum(float(st[u]["duration_s"]) for u in ru)
                ov = sum(float(st[u]["overlap_s"]) for u in ru)
                sp = sum(float(st[u]["speech_s"]) for u in ru)
                print(f"[{ds}] source recording {rec}: {len(ru)} clips, {dur / 60:.1f} min, overlap share "
                      f"{ov / sp:.4f} of speech, speakers per clip "
                      f"{', '.join(f'{k}: {sum(nspk[u] == k for u in ru)}' for k in sorted({nspk[u] for u in ru}))}")
        clusters = iv_clusters(d, uris) if ds == "indicvoices_ne_calls" else None
        if clusters:
            print(f"[{ds}] {len(set(clusters))} clusters of calls linked by a shared speaker id, "
                  f"sizes {sorted(clusters.count(c) for c in set(clusters))}")
        for s in SYSTEMS:
            sc = ROOT / "results/raw" / (f"scores_{s}_ami.csv" if ds == "ami_test" else f"scores_{s}_{ds}.csv")
            if not sc.exists() or not all(paths(ds, s, u)[2].exists() for u in uris):
                continue
            per = {r["uri"]: r for r in csv.DictReader(open(sc))}
            if not all(u in per for u in uris):
                continue
            for g, gu in groups.items():
                rows.append(block(ds, s, gu, per, g, clusters if g == "all" else None))
    if not rows:
        print("nothing scored yet")
        return
    hdr = (f"{'system':34} {'n':>3} {'DER0':>6} {'miss':>6} {'fa':>6} {'conf':>6} {'DER25':>6} "
           f"{'mean DER0 [interval]':>24} {'mean DER25 [interval]':>24} {'JER':>6} {'k ok':>6} {'k err':>6} "
           f"{'|k err|':>7} {'RTF':>6}")
    print("\nPooled DER (total error / total speech), miss/fa/conf at no collar. k ok = speaker count right; "
          "k err = mean (found - true); |k err| = mean absolute.\nInterval: AMI = per-recording bootstrap 95% CI "
          f"({NB} resamples, seed 0); IndicVoices = same, plus a pooled-DER CI from a bootstrap over clusters of "
          "calls that share a speaker; DiarBench = per-clip min to max, no interval (25 clips from 2 recordings).")
    last = None
    for r in rows:
        if (r["dataset"], r["group"]) != last:
            last = (r["dataset"], r["group"])
            print(f"\n[{r['dataset']}] {r['group']}, {r['n']} recordings")
            print(hdr)
        rng = r["dataset"] == "indic_diarbench_ne"
        iv = [(f"{r['mean_' + k]:.4f} [{r[('min_' if rng else '') + k + ('' if rng else '_ci95_lo')]:.4f},"
               f"{r[('max_' if rng else '') + k + ('' if rng else '_ci95_hi')]:.4f}]") for k in ("der_no_collar", "der_collar_0.25")]
        print(f"{r['system']:34} {r['n']:>3} {r['pooled_der_no_collar']:6.4f} {r['pooled_miss_no_collar']:6.4f} "
              f"{r['pooled_fa_no_collar']:6.4f} {r['pooled_conf_no_collar']:6.4f} {r['pooled_der_collar_0.25']:6.4f} "
              f"{iv[0]:>24} {iv[1]:>24} {r['mean_jer']:6.4f} {r['speaker_count_right']:>3}/{r['n']:<2} "
              f"{r['mean_hyp_minus_ref_speakers']:+6.2f} {r['mean_abs_speaker_count_err']:7.2f} {r['rtf']:>6}")
        if "n_clusters" in r:
            print(f"{'':38} pooled DER cluster CI ({r['n_clusters']} clusters): no collar "
                  f"[{r['pooled_der_no_collar_cluster_ci95_lo']:.4f},{r['pooled_der_no_collar_cluster_ci95_hi']:.4f}], "
                  f"0.25 s [{r['pooled_der_collar_0.25_cluster_ci95_lo']:.4f},{r['pooled_der_collar_0.25_cluster_ci95_hi']:.4f}]")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    f4 = lambda r, ks: {k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in ks}  # noqa: E731
    with open(ROOT / "results/summary_nepali_real.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(f4(r, keys) for r in rows)
    sk = ["dataset", "system", "n", "speaker_count_right", "speaker_count_acc", "mean_abs_speaker_count_err",
          "mean_hyp_minus_ref_speakers"]
    with open(ROOT / "results/speaker_count_nepali_real.csv", "w", newline="") as f:
        w = csv.DictWriter(f, sk)
        w.writeheader()
        w.writerows(f4(r, sk) for r in rows if r["group"] == "all" and r["dataset"] != "ami_test")


if __name__ == "__main__":
    main()
