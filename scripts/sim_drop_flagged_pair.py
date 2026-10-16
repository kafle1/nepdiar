# post hoc: the twin pair test with and without the one conversation holding the English speaker
# (2277) that the Nepali speaker check would have dropped. writes results/sim_drop_flagged_pair.csv
import csv
import numpy as np

SYSTEMS = (("sherpa_onnx", "S1"), ("wespeaker_spectral", "S2a"), ("wespeaker_spectral_silero", "S2b"),
           ("pyannote_style", "S3"), ("pyannote31", "S4"), ("community1", "S5"))
DROP = "3spk_07"


def load(p):
    return {r["uri"].split("_", 1)[1]: float(r["der_no_collar"]) for r in csv.DictReader(open(p))}


rows = []
for cond, suf in (("clean", ""), ("reverb_noise", "_rn")):
    for s, lab in SYSTEMS:
        ne = load(f"results/raw/scores_{s}_nepali_sim{suf}.csv")
        en = load(f"results/raw/scores_{s}_english_sim{suf}.csv")
        assert sorted(ne) == sorted(en) and DROP in ne
        for subset, keys in (("all_30", sorted(ne)), ("without_" + DROP, [k for k in sorted(ne) if k != DROP])):
            d = np.array([ne[k] - en[k] for k in keys]) * 100
            m = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
            rows.append(dict(condition=cond, config=lab, subset=subset, n=len(d), mean_d=round(d.mean(), 2),
                             ci_lo=round(np.percentile(m, 2.5), 2), ci_hi=round(np.percentile(m, 97.5), 2)))
with open("results/sim_drop_flagged_pair.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows")
