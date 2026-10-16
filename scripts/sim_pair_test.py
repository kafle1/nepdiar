#!/usr/bin/env python3
"""Paired Nepali minus English DER test on NepDiar-Sim and its English twin, as fixed in NOTES.md
("Analysis plan fixed before the Nepali tables"). Per configuration and condition: d_i = DER no collar
of nepsim conversation i minus that of engsim conversation i; mean d, 95% percentile bootstrap CI
(10,000 resamples of the 30 pairs, default_rng(0) per test), two-sided p = 2 min(P(mean* <= 0), P(mean* >= 0)),
Holm across the 6 configurations inside each condition. Writes results/sim_pair_test.csv.
Amendment (nepdiar ANALYSIS_PLAN.md): `reading` puts the CI against the 2.0 DER-point bound; sensitivity 1
resamples whole groups of twins linked by a shared LibriSpeech speaker (union-find, default_rng(0));
sensitivity 2 is a paired sign-flip permutation (10,000 sign vectors, default_rng(1), observed counted).
`--selfcheck` runs the new code on a made-up d, never on real scores."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

SYSTEMS = ["sherpa_onnx", "wespeaker_spectral", "wespeaker_spectral_silero", "pyannote_style", "pyannote31", "community1"]
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
CONDITIONS = {"clean": ("nepali_sim", "english_sim", "primary"), "reverb_noise": ("nepali_sim_rn", "english_sim_rn", "secondary")}
N_BOOT = 10000
BOUND = 0.02  # 2.0 DER points; score.py DER is a fraction


def reading(lo, hi):
    if -BOUND < lo and hi < BOUND:
        return "inside"
    if hi <= -BOUND or lo >= BOUND:
        return "outside"
    return "crosses"


def twin_groups(uris):
    """Group index per conversation: twins linked through a shared LibriSpeech speaker, by union-find."""
    conv = {r["nepali_twin"]: r["speakers"].split() for r in csv.DictReader(open(ROOT / "data/english_sim/conversations.tsv"), delimiter="\t")}
    parent = list(range(len(uris)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, u in enumerate(uris):
        for spk in conv[u]:
            if spk in first:
                parent[find(i)] = find(first[spk])
            else:
                first[spk] = i
    roots = sorted({find(i) for i in range(len(uris))})
    return np.array([roots.index(find(i)) for i in range(len(uris))])


def group_ci(d, groups):
    members = [np.where(groups == g)[0] for g in range(groups.max() + 1)]
    rng = np.random.default_rng(0)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(members), size=len(members))
        means[b] = d[np.concatenate([members[g] for g in pick])].mean()
    return np.percentile(means, [2.5, 97.5])


def signflip_p(d):
    signs = np.random.default_rng(1).choice([-1.0, 1.0], size=(N_BOOT, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return (1 + np.sum(perm >= abs(d.mean()) - 1e-12)) / (N_BOOT + 1)


def selfcheck():
    d = np.random.default_rng(7).normal(0.03, 0.02, 30)  # made-up d, clearly above +2 points
    groups = np.repeat(np.arange(10), 3)
    lo, hi = group_ci(d, groups)
    assert lo < d.mean() < hi and lo > 0, (lo, hi)
    assert signflip_p(d) < 0.001 and signflip_p(d - d.mean()) > 0.5
    assert group_ci(d, np.arange(30))[0] > 0
    assert (reading(-0.01, 0.01), reading(-0.01, 0.03), reading(0.021, 0.05), reading(-0.05, -0.02)) == ("inside", "crosses", "outside", "outside")
    print(f"selfcheck ok: fake d mean {d.mean():.4f}, group CI [{lo:.4f},{hi:.4f}], sign-flip p {signflip_p(d):.5f}")


if "--selfcheck" in sys.argv:
    selfcheck()
    sys.exit()


def der(sys_, ds):
    p = ROOT / "results/raw" / f"scores_{sys_}_{ds}.csv"
    if not p.exists():
        return None
    return {r["uri"]: float(r["der_no_collar"]) for r in csv.DictReader(open(p))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


uris = [u.strip() for u in (ROOT / "data/nepali_sim/conversations.txt").read_text().splitlines() if u.strip()]
groups = twin_groups(uris)
print(f"{len(uris)} twin pairs form {groups.max() + 1} groups linked by shared English speakers", file=sys.stderr)
rows = []
for cond, (nds, eds, family) in CONDITIONS.items():
    fam = []
    for s in SYSTEMS:
        n, e = der(s, nds), der(s, eds)
        if n is None or e is None:
            print(f"{s} {cond}: scores missing, family incomplete", file=sys.stderr)
            continue
        twins = [(u, u.replace("nepsim_", "engsim_")) for u in uris]
        missing = [u for u, t in twins if u not in n or t not in e]
        if missing:
            sys.exit(f"{s} {cond}: no score for {missing[:3]}")
        d = np.array([n[u] - e[t] for u, t in twins])
        idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
        glo, ghi = group_ci(d, groups)
        fam.append({"system": s, "config": LABEL[s], "condition": cond, "family": family, "n_pairs": len(d),
                    "mean_der_nepali": round(float(np.mean([n[u] for u, _ in twins])), 4),
                    "mean_der_english": round(float(np.mean([e[t] for _, t in twins])), 4),
                    "mean_diff": round(float(d.mean()), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "p_raw": round(float(p), 4), "reading_2pt": reading(lo, hi),
                    "group_ci_lo": round(float(glo), 4), "group_ci_hi": round(float(ghi), 4),
                    "n_groups": int(groups.max() + 1), "group_reading_2pt": reading(glo, ghi),
                    "signflip_p": round(float(signflip_p(d)), 4)})
    for r, a in zip(fam, holm([r["p_raw"] for r in fam])):
        r["p_holm"] = round(a, 4)
        r["holm_family_size"] = len(fam)
    rows += fam
if not rows:
    sys.exit("no paired scores yet")
with open(ROOT / "results/sim_pair_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['condition']:12} {r['config']:4} nep {r['mean_der_nepali']:.4f} eng {r['mean_der_english']:.4f} "
          f"diff {r['mean_diff']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p {r['p_raw']:.4f} holm {r['p_holm']:.4f} (m={r['holm_family_size']}) "
          f"{r['reading_2pt']}; groups [{r['group_ci_lo']:+.4f},{r['group_ci_hi']:+.4f}] {r['group_reading_2pt']}; sign-flip p {r['signflip_p']:.4f}")
