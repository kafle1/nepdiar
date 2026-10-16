# builds tables/*.tex from ../results (read only). values are fractions in the
# source files, printed here as percentages by moving the decimal point, so no
# digit is rounded away. a result file that does not exist yet gives red \todo cells.
import csv
import re
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)
TODO = r"\todo{}"

ORDER = [
    ("pyannote31", r"S4 pyannote 3.1 (released)", "S4"),
    ("pyannote_style", r"S3 pyannote recipe, ONNX", "S3"),
    ("sherpa_onnx", r"S1 sherpa-onnx", "S1"),
    ("wespeaker_spectral", r"S2a spectral, WebRTC VAD", "S2a"),
    ("wespeaker_spectral_silero", r"S2b spectral, Silero VAD", "S2b"),
    ("community1", r"S5 pyannote community-1", "S5"),
]
REAL = [("indic_diarbench_ne", "Indic DiarBench, Nepali", "clips"), ("indicvoices_ne_calls", "IndicVoices, Nepali calls", "calls")]


def pct(s) -> str:
    d = Decimal(str(s)) * 100
    return f"{d:.2f}"  # 4-decimal fraction -> 2-decimal percent, exact


def label(name: str, oracle: bool) -> str:
    return name + (r" + oracle $k$" if oracle else "")


def read_csv(p: Path):
    return list(csv.DictReader(open(p))) if p.exists() else []


def c1_pooled():
    """pooled rows of results/corpus_der_community1.txt: {(dataset, system): [der0, miss, fa, conf, der25]}"""
    p = RES / "corpus_der_community1.txt"
    out, cur = {}, None
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"^(\S+) (community1(?:_oracle)?), (\d+) files$", line)
        if m:
            cur = (m.group(1), m.group(2))
            C1N[cur] = int(m.group(3))
            continue
        m = re.match(r"^\s+pooled: DER0 (\S+) miss (\S+) fa (\S+) conf (\S+) DER25 (\S+)$", line)
        if m and cur:
            out[cur] = list(m.groups())
    return out


C1N = {}  # (dataset, system) -> files scored
C1P = c1_pooled()
C1S = {(r["system"], r["dataset"]): r for r in read_csv(RES / "summary_community1.csv")}

# pooled DER on AMI test, results/corpus_der_ami_test.txt (+ community-1's own report)
pooled = {}
for line in (RES / "corpus_der_ami_test.txt").read_text().splitlines()[1:]:
    parts = line.split()
    if len(parts) == 6:
        pooled[parts[0]] = parts[1:]
for (ds, s), v in C1P.items():
    if ds == "ami_test":
        pooled[s] = v

lines = []
for key, name, _ in ORDER:
    for oracle in (False, True):
        k = key + ("_oracle" if oracle else "")
        if k not in pooled and key != "community1":
            raise SystemExit(f"missing pooled row {k}")
        cells = [pct(v) for v in pooled[k]] if k in pooled else [TODO] * 5
        lines.append(f"{label(name, oracle)} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\botrule"  # \input in a tabular cannot be followed by a rule, so the rule lives here
(OUT / "pooled.tex").write_text("\n".join(lines) + "\n")

# per-meeting means with 95% bootstrap CIs, results/summary.csv (+ summary_community1.csv)
rows = {r["system"]: r for r in read_csv(RES / "summary.csv")}
for (s, ds), r in C1S.items():
    if ds == "ami_test":
        rows[s] = r


def ci(r, m):
    return f"{pct(r[m + '_mean'])} [{pct(r[m + '_ci95_lo'])}, {pct(r[m + '_ci95_hi'])}]"


def per_file_range(ds, s):
    if s.startswith("community1"):
        p = RES / "raw" / ("ami" if ds == "ami_test" else ds) / f"scores_{s}.csv"
    else:
        p = RES / "raw" / f"scores_{s}_{'ami' if ds == 'ami_test' else ds}.csv"
    v = [Decimal(r["der_no_collar"]) for r in read_csv(p)]
    return f"{min(v) * 100:.1f} to {max(v) * 100:.1f}" if v else TODO


lines, comp = [], []
for key, name, short in ORDER:
    for oracle in (False, True):
        k = key + ("_oracle" if oracle else "")
        r = rows.get(k)
        if r is None and key != "community1":
            raise SystemExit(f"missing summary row {k}")
        if r and (r["n_recordings"] != "16" or r["dataset"] != "ami_test"):
            raise SystemExit(f"unexpected row {k}")
        cells = [ci(r, m) for m in ("der_no_collar", "der_collar_0.25", "jer")] if r else [TODO] * 3
        cells.append(per_file_range("ami_test", k) if r else TODO)
        lines.append(f"{label(short, oracle)} & " + " & ".join(cells) + " \\\\")
        cells = [pct(r[f"{m}_{c}_mean"]) for c in ("no_collar", "collar_0.25") for m in ("miss", "fa", "conf")] if r else [TODO] * 6
        comp.append(f"{label(short, oracle)} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
    comp.append(r"\addlinespace")
lines[-1] = comp[-1] = r"\botrule"
(OUT / "permeeting.tex").write_text("\n".join(lines) + "\n")
(OUT / "components.tex").write_text("\n".join(comp) + "\n")

# real Nepali sets: results/summary_nepali_real.csv (scripts/real_nepali/score_real.sh), the one
# source for all six configs. DiarBench gets the per-clip range, no interval: its 25 clips come from
# 2 recordings. IndicVoices gets the per-call mean with its bootstrap interval (calls share no speaker).
real = {(r["dataset"], r["group"], r["system"]): r for r in read_csv(RES / "summary_nepali_real.csv")}


def real_cells(ds, s, group="all"):
    r = real.get((ds, group, s))
    if not r:
        return [TODO] * 7
    cells = [pct(r[f"pooled_{m}_no_collar"]) for m in ("der", "miss", "fa", "conf")] + [pct(r["pooled_der_collar_0.25"])]
    if ds == "indic_diarbench_ne":
        cells.append(f"{pct(r['min_der_no_collar'])} to {pct(r['max_der_no_collar'])}")
    else:
        cells.append(f"{pct(r['mean_der_no_collar'])} [{pct(r['der_no_collar_ci95_lo'])}, {pct(r['der_no_collar_ci95_hi'])}]")
    return cells + [f"{r['speaker_count_right']}/{r['n']}"]


def set_n(ds, group="all"):
    """files scored per row of a set, from the result file; every row of a set must agree"""
    ns = {int(r["n"]) for (d, g, _), r in real.items() if d == ds and g == group}
    if len(ns) > 1:
        raise SystemExit(f"{ds}/{group}: rows scored on different file counts {sorted(ns)}")
    return ns.pop() if ns else None


BLOCKS = REAL + [("indicvoices_ne_calls_tight", "IndicVoices, tighter reference", "calls", "all", False),
                 ("indicvoices_ne_calls", "IndicVoices, certain sync only", "calls", "excl_4_uncertain_sync", False)]
lines = []
for ds, title, unit, *rest in BLOCKS:
    group, with_oracle = rest if rest else ("all", True)
    n = set_n(ds, group)
    title = f"{title} ({n} {unit})" if n else rf"{title} (\todo{{}} {unit})"
    lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
    for key, _, short in ORDER:
        for oracle in ((False, True) if with_oracle else (False,)):
            lines.append(f"{label(short, oracle)} & " + " & ".join(real_cells(ds, key + ('_oracle' if oracle else ''), group)) + " \\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\botrule"
(OUT / "realres.tex").write_text("\n".join(lines) + "\n")

# NepDiar-Sim pooled DER by speaker count: results/summary_nepali_sim_by_count.csv, which
# scripts/nepali_sim_report.py writes next to results/summary_nepali_sim.csv
sim = {(r["dataset"], r["group"], r["system"]): r for r in read_csv(RES / "summary_nepali_sim_by_count.csv")}
lines = []
for key, _, short in ORDER:
    cells = []
    for ds in ("nepali_sim", "nepali_sim_rn"):
        for g in ("2spk", "3spk", "4spk", "all"):
            r = sim.get((ds, g, key))
            cells.append(pct(r["pooled_der_no_collar"]) if r else TODO)
    lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
lines.append(r"\botrule")
(OUT / "simres.tex").write_text("\n".join(lines) + "\n")

# where the error moves: pooled components without a collar, AMI test against the Nepali sets
lines = []
for key, _, short in ORDER:
    for j, (m, name) in enumerate((("miss", "Miss"), ("fa", "FA"), ("conf", "Conf.")), start=1):
        cells = [pct(pooled[key][j]) if key in pooled else TODO]
        cells += [real_cells(ds, key)[j] for ds, *_ in REAL]
        r = sim.get(("nepali_sim", "all", key))
        cells.append(pct(r[f"pooled_{m}"]) if r else TODO)
        head = rf"\multirow{{3}}{{*}}{{{short}}}" if j == 1 else ""
        lines.append(f"{head} & {name} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\botrule"
(OUT / "shift.tex").write_text("\n".join(lines) + "\n")

# english twin, all from final result files. sim_pair_test.csv holds the primary and secondary
# paired tests; the by-count files give pooled DER and components; sim_pause_diagnostic.csv is post hoc.
def neg(x: str) -> str:
    return x.replace("-", "$-$")


def pval(x: str) -> str:
    return "$<$0.001" if float(x) < 0.001 else f"{float(x):.3f}"


READING = {"outside": "difference", "inside": "no difference", "crosses": "inconclusive"}
CONDS = (("clean", "Clean", "nepali_sim", "english_sim"), ("reverb_noise", "Reverb + noise", "nepali_sim_rn", "english_sim_rn"))
pair = {(r["condition"], r["system"]): r for r in read_csv(RES / "sim_pair_test.csv")}
if pair and {c for c, _ in pair} != {c for c, *_ in CONDS}:
    raise SystemExit(f"unexpected conditions in sim_pair_test.csv: {sorted({c for c, _ in pair})}")
lines = []
for c, title, _, _ in CONDS:
    lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
    for key, _, short in ORDER:
        t = pair.get((c, key))
        if not t:
            lines.append(f"{short} & " + " & ".join([TODO] * 5) + " \\\\")
            continue
        cells = [neg(f"{pct(t['mean_diff'])} [{pct(t['ci_lo'])}, {pct(t['ci_hi'])}]"), pval(t["p_holm"]),
                 READING[t["reading_2pt"]], neg(f"[{pct(t['group_ci_lo'])}, {pct(t['group_ci_hi'])}]"), pval(t["signflip_p"])]
        lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\botrule"
(OUT / "twin.tex").write_text("\n".join(lines) + "\n")

# pooled DER and components, Nepali against English, estimated count, no collar
eng = {(r["dataset"], r["group"], r["system"]): r for r in read_csv(RES / "summary_english_sim_by_count.csv")}
lines = []
for key, _, short in ORDER:
    for lang, src, j in (("Nepali", sim, 0), ("English", eng, 1)):
        cells = []
        for _, _, nds, eds in CONDS:
            r = src.get((nds if j == 0 else eds, "all", key))
            cells += [pct(r[f"pooled_{m}"]) for m in ("der_no_collar", "miss", "fa", "conf")] if r else [TODO] * 4
        head = rf"\multirow{{2}}{{*}}{{{short}}}" if j == 0 else ""
        lines.append(f"{head} & {lang} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\botrule"
(OUT / "simcomp.tex").write_text("\n".join(lines) + "\n")

# oracle-k pooled DER without a collar, both simulated sets and conditions, from the same by-count files
lines = []
for key, _, short in ORDER:
    cells = []
    for src, dss in ((sim, ("nepali_sim", "nepali_sim_rn")), (eng, ("english_sim", "english_sim_rn"))):
        for ds in dss:
            r = src.get((ds, "all", key + "_oracle"))
            cells.append(pct(r["pooled_der_no_collar"]) if r else TODO)
    lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
lines.append(r"\botrule")
(OUT / "simoracle.tex").write_text("\n".join(lines) + "\n")

# english twin by speaker count, same layout as simres.tex
lines = []
for key, _, short in ORDER:
    cells = []
    for ds in ("english_sim", "english_sim_rn"):
        for g in ("2spk", "3spk", "4spk", "all"):
            r = eng.get((ds, g, key))
            cells.append(pct(r["pooled_der_no_collar"]) if r else TODO)
    lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
lines.append(r"\botrule")
(OUT / "simres_en.tex").write_text("\n".join(lines) + "\n")

# post hoc pause diagnostic: mean per-conversation difference under each view (already in percent)
diag = {(r["view"], r["condition"], r["system"]): r for r in read_csv(RES / "sim_pause_diagnostic.csv") if r["view"] != "pauses"}
lines = []
for _, _, short in ORDER:
    s2 = "S2" if short == "S2a" else short
    cells = []
    for view, c in (("as_scored", "clean"), ("pause_merged", "clean"), ("quiet_removed", "clean"), ("as_scored", "reverb_noise"), ("pause_merged", "reverb_noise")):
        r = diag.get((view, c, s2))
        ci_txt = f" [{float(r['d_ci_lo']):.2f}, {float(r['d_ci_hi']):.2f}]" if r and view != "pause_merged" else ""
        cells.append(neg(f"{float(r['mean_d']):.2f}{ci_txt}") if r else "--")
    lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
lines.append(r"\botrule")
(OUT / "pausediag.tex").write_text("\n".join(lines) + "\n")

# post hoc reference sensitivity: DER and confusion difference under three references
sens = {(r["view"], r["condition"], r["system"]): r for r in read_csv(RES / "sim_reference_sensitivity.csv") if r["system"]}
lines = []
for c, cname in (("clean", "Clean"), ("reverb_noise", "Reverb + noise")):
    lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{cname}}}}} \\")
    for _, _, short in ORDER:
        rs = [sens[(v, c, short)] for v in ("as_scored", "quiet_removed", "strict_+10")]
        cells = [neg(f"{float(r['mean_d']):.2f}") for r in rs]
        cells += [neg(f"{float(r['d_conf']):.2f}" + ("" if v == "quiet_removed" else f" [{float(r['d_conf_ci_lo']):.2f}, {float(r['d_conf_ci_hi']):.2f}]")) for v, r in zip(("as_scored", "quiet_removed", "strict_+10"), rs)]
        lines.append(f"{short} & " + " & ".join(cells) + " \\\\")
lines.append(r"\botrule")
(OUT / "simref.tex").write_text("\n".join(lines) + "\n")

# post hoc sex mix: paired d on conversation subsets, estimates then intervals
sx = {(r["condition"], r["config"], r["subset"]): r for r in read_csv(RES / "sim_sex_mix_pair.csv")}
SUBS = ("full", "exact_match", "mixed_sex", "exact_match_robust", "mixed_sex_robust")
lines = []
for c, cname in (("clean", "Clean"), ("reverb_noise", "Reverb + noise")):
    lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{cname}}}}} \\")
    for _, _, short in ORDER:
        rs = [sx[(c, "S2" if short == "S2a" else short, k)] for k in SUBS]
        lines.append(f"{short} & " + " & ".join(neg(f"{100 * float(r['mean_diff']):.2f}") for r in rs) + " \\\\")
        lines.append(" & " + " & ".join(neg(f"[{100 * float(r['ci_lo']):.2f}, {100 * float(r['ci_hi']):.2f}]") for r in rs) + " \\\\")
ns = [sx[("clean", "S4", k)]["n"] for k in SUBS]
lines.append(r"\botrule")
(OUT / "sexmix.tex").write_text("\n".join(lines) + "\n")
(OUT / "sexmix_n.tex").write_text(" & ".join(["Conversations"] + ns) + " \\\\\n")
print("ok")

# tab:chars body, from paper/analysis/turn_stats_table.py
ch = read_csv(HERE / "analysis/turn_stats_table.csv")
def spk_range(r):
    lo, hi = r["speakers_min"], r["speakers_max"]
    return lo if lo == hi else f"{lo} to {hi}"
CHAR_ROWS = [
    ("Recordings", lambda r: r["recordings"]),
    ("Audio, hours", lambda r: r["audio_h"]),
    ("Channel", lambda r: r["channel"]),
    ("Speakers per recording", spk_range),
    ("Distinct speakers", lambda r: r["distinct_speakers"]),
    ("Speech, share of audio", lambda r: r["speech_pct"] + r"\%"),
    ("Overlap, share of speech", lambda r: r["overlap_pct"] + r"\%"),
    ("Median segment", lambda r: r["median_segment_s"] + " s"),
    ("Segments under 1 s", lambda r: r["segments_under_1s_pct"] + r"\%"),
    ("Median turn", lambda r: r["median_turn_s"] + " s"),
    ("Speaker changes per minute of speech", lambda r: r["changes_per_speech_min"]),
]
lines = [" & ".join([name] + ([f(r) for r in ch] if ch else [TODO] * 4)) + r" \\" for name, f in CHAR_ROWS]
(OUT / "chars.tex").write_text("\n".join(lines) + "\n")

# post hoc twin subsets (tab:subsets): all 30, the 17 sex-matched and the six largest sets of
# conversations with no repeated English speaker, where we give the range and the worst lower bounds
ts = {(r["condition"], r["config"], r["subset"], r["component"]): r for r in read_csv(RES / "posthoc_twin_subsets.csv")}
lines = []
for c, cname, *_ in CONDS:
    lines.append(r"\multicolumn{4}{@{}l}{\emph{" + cname + r"}} \\")
    for cfg, comp, name in (("S3", "conf", "S3 confusion"), ("S4", "conf", "S4 confusion"), ("S1", "der", "S1 DER")):
        a, x, n = (ts.get((c, cfg, k, comp)) for k in ("all", "sex_exact", "eng_indep"))
        if not (a and x and n):
            lines.append(f"{name} & {TODO} & {TODO} & {TODO} \\\\")
            continue
        est = lambda r: neg(f"{float(r['mean_d_pts']):.2f} [{float(r['ci95_lo']):.2f}, {float(r['ci95_hi']):.2f}]")
        wide = lambda r: neg(f"[{float(r['ci99.17_lo']):.2f}, {float(r['ci99.17_hi']):.2f}]")
        rng = neg(f"{float(n['mean_d_min_over_sets']):.2f} to {float(n['mean_d_max_over_sets']):.2f}")
        lines.append(f"{name} & {est(a)} & {est(x)} & {rng} \\\\")
        lines.append(f" & {wide(a)} & {wide(x)} & " + neg(f"{float(n['ci95_lo_min_over_sets']):.2f} / {float(n['ci99.17_lo_min_over_sets']):.2f}") + " \\\\")
lines.append(r"\botrule")
(OUT / "subsets.tex").write_text("\n".join(lines) + "\n")
