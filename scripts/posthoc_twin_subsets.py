#!/usr/bin/env python3
"""Post hoc, not in the analysis plan: the paired Nepali minus English gap in DER and in each component
(miss, false alarm, confusion, each over that conversation's reference speech, no collar, from the score CSVs)
on harder subsets of the 30 twin pairs:
- all: the full 30;
- sex_exact: match == exact in results/sim_sex_mix_conversations.csv;
- eng_indep: a largest set of conversations in which no English speaker appears twice (exact, networkx);
- both_indep: the same, with no Nepali speaker appearing twice either.
Percentile bootstrap of the mean d, 10,000 resamples of the pairs, default_rng(0) per test, as sim_pair_test.py,
read at 95%, 99.17% (1 - 0.05/6, Bonferroni over 6 configs) and 90%. Values are DER points.
Writes results/posthoc_twin_subsets.csv."""
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

SYSTEMS = ["sherpa_onnx", "wespeaker_spectral", "wespeaker_spectral_silero", "pyannote_style", "pyannote31", "community1"]
LABEL = {"sherpa_onnx": "S1", "wespeaker_spectral": "S2a", "wespeaker_spectral_silero": "S2b",
         "pyannote_style": "S3", "pyannote31": "S4", "community1": "S5"}
CONDITIONS = {"clean": ("nepali_sim", "english_sim"), "reverb_noise": ("nepali_sim_rn", "english_sim_rn")}
LEVELS = {"95": 95.0, "99.17": 100 * (1 - 0.05 / 6), "90": 90.0}
N_BOOT = 10000


def largest_independent(uris, spk_of):
    """every largest set of conversations with no shared speaker: brute force per component of the conflict graph."""
    g = nx.Graph()
    g.add_nodes_from(uris)
    g.add_edges_from((a, b) for i, a in enumerate(uris) for b in uris[i + 1:] if spk_of[a] & spk_of[b])
    per_comp = []
    for comp in nx.connected_components(g):
        nodes = sorted(comp, key=uris.index)
        assert len(nodes) <= 24, "component too big for brute force"
        masks = np.arange(1 << len(nodes), dtype=np.int64)
        ok = np.ones(len(masks), bool)
        for a, b in g.subgraph(nodes).edges():
            ok &= ~((masks >> nodes.index(a)) & (masks >> nodes.index(b)) & 1).astype(bool)
        size = sum((masks >> i) & 1 for i in range(len(nodes)))
        best = size[ok].max()
        per_comp.append([[nodes[i] for i in range(len(nodes)) if m >> i & 1] for m in masks[ok & (size == best)]])
    combos = [[]]
    for opts in per_comp:
        combos = [c + o for c in combos for o in opts]
    return sorted(sorted(c, key=uris.index) for c in combos)


uris = [u.strip() for u in (ROOT / "data/nepali_sim/conversations.txt").read_text().splitlines() if u.strip()]
conv = {r["nepali_twin"]: r for r in csv.DictReader(open(ROOT / "data/english_sim/conversations.tsv"), delimiter="\t")}
eng = {u: set(conv[u]["speakers"].split()) for u in uris}
nep_rows = {r["uri"]: r for r in csv.DictReader(open(ROOT / "data/nepali_sim/conversations.tsv"), delimiter="\t")}
nep = {u: set(nep_rows[u]["speakers"].split()) for u in uris}
both = {u: {"e" + s for s in eng[u]} | {"n" + s for s in nep[u]} for u in uris}
sex = {r["uri"]: r["match"] for r in csv.DictReader(open(ROOT / "results/sim_sex_mix_conversations.csv"))}

e_sets, b_sets = largest_independent(uris, eng), largest_independent(uris, both)
n_rep = sum(1 for s in {x for u in uris for x in nep[u]} if sum(s in nep[u] for u in uris) > 1)
print(f"Nepali speakers used in two or more conversations: {n_rep}", file=sys.stderr)
print(f"eng_indep: size {len(e_sets[0])}, {len(e_sets)} largest sets; both_indep: size {len(b_sets[0])}, {len(b_sets)} largest sets", file=sys.stderr)
subsets = {"all": uris, "sex_exact": [u for u in uris if sex[u] == "exact"], "eng_indep": e_sets[0], "both_indep": b_sets[0]}
n_alt = {"all": 1, "sex_exact": 1, "eng_indep": len(e_sets), "both_indep": len(b_sets)}
print("eng_indep picked: " + " ".join(subsets["eng_indep"]), file=sys.stderr)


def scores(s, ds):
    return {r["uri"]: r for r in csv.DictReader(open(ROOT / f"results/raw/scores_{s}_{ds}.csv"))}


rows = []
for cond, (nds, eds) in CONDITIONS.items():
    for s in SYSTEMS:
        n, e = scores(s, nds), scores(s, eds)
        for name, sub in subsets.items():
            for comp in ("der", "miss", "fa", "conf"):
                k = f"{comp}_no_collar"
                d = 100 * np.array([float(n[u][k]) - float(e[u.replace("nepsim_", "engsim_")][k]) for u in sub])
                idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
                means = d[idx].mean(axis=1)
                row = {"condition": cond, "config": LABEL[s], "system": s, "subset": name, "n": len(d),
                       "n_largest_sets": n_alt[name], "component": comp, "mean_d_pts": round(float(d.mean()), 2)}
                for tag, lvl in LEVELS.items():
                    lo, hi = np.percentile(means, [(100 - lvl) / 2, 100 - (100 - lvl) / 2])
                    row[f"ci{tag}_lo"], row[f"ci{tag}_hi"] = round(float(lo), 2), round(float(hi), 2)
                rows.append(row)
            # the same d on every other largest set, to show the pick does not drive the answer
            if n_alt[name] > 1:
                for r in rows[-4:]:
                    k = f"{r['component']}_no_collar"
                    alt = []
                    for c in (e_sets if name == "eng_indep" else b_sets):
                        d = 100 * np.array([float(n[u][k]) - float(e[u.replace("nepsim_", "engsim_")][k]) for u in c])
                        means = d[np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
                        alt.append((d.mean(), np.percentile(means, 2.5), np.percentile(means, (100 - LEVELS["99.17"]) / 2)))
                    r["mean_d_min_over_sets"] = round(float(min(a[0] for a in alt)), 2)
                    r["mean_d_max_over_sets"] = round(float(max(a[0] for a in alt)), 2)
                    r["ci95_lo_min_over_sets"] = round(float(min(a[1] for a in alt)), 2)
                    r["ci99.17_lo_min_over_sets"] = round(float(min(a[2] for a in alt)), 2)
with open(ROOT / "results/posthoc_twin_subsets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), restval="")
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["component"] in ("der", "conf"):
        print(f"{r['condition']:12} {r['config']:4} {r['subset']:10} n={r['n']:2} {r['component']:4} {r['mean_d_pts']:+6.2f} "
              f"95 [{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}] 99.17 [{r['ci99.17_lo']:+.2f},{r['ci99.17_hi']:+.2f}] "
              f"90 [{r['ci90_lo']:+.2f},{r['ci90_hi']:+.2f}]"
              + (f" | other sets: d {r['mean_d_min_over_sets']:+.2f} to {r['mean_d_max_over_sets']:+.2f}, lowest 95 lo "
                 f"{r['ci95_lo_min_over_sets']:+.2f}, lowest 99.17 lo {r['ci99.17_lo_min_over_sets']:+.2f}" if "mean_d_min_over_sets" in r else ""))
