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
REAL = [("indic_diarbench_ne", "Indic DiarBench, Nepali (25 clips)"), ("indicvoices_ne_calls", "IndicVoices, Nepali calls (36 calls)")]


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
        m = re.match(r"^(\S+) (community1(?:_oracle)?), \d+ files$", line)
        if m:
            cur = (m.group(1), m.group(2))
            continue
        m = re.match(r"^\s+pooled: DER0 (\S+) miss (\S+) fa (\S+) conf (\S+) DER25 (\S+)$", line)
        if m and cur:
            out[cur] = list(m.groups())
    return out


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


lines, comp = [], []
for key, name, _ in ORDER:
    for oracle in (False, True):
        k = key + ("_oracle" if oracle else "")
        r = rows.get(k)
        if r is None and key != "community1":
            raise SystemExit(f"missing summary row {k}")
        if r and (r["n_recordings"] != "16" or r["dataset"] != "ami_test"):
            raise SystemExit(f"unexpected row {k}")
        cells = [ci(r, m) for m in ("der_no_collar", "der_collar_0.25", "jer")] if r else [TODO] * 3
        lines.append(f"{label(name, oracle)} & " + " & ".join(cells) + " \\\\")
        cells = [pct(r[f"{m}_{c}_mean"]) for c in ("no_collar", "collar_0.25") for m in ("miss", "fa", "conf")] if r else [TODO] * 6
        comp.append(f"{label(name, oracle)} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
    comp.append(r"\addlinespace")
lines[-1] = comp[-1] = r"\botrule"
(OUT / "permeeting.tex").write_text("\n".join(lines) + "\n")
(OUT / "components.tex").write_text("\n".join(comp) + "\n")

# real Nepali sets: results/summary_nepali_real.csv (scripts/real_nepali/report_real.py), group "all",
# plus the per-recording DER range from the per-file score csvs. no interval: DiarBench's 25 clips
# come from 2 recordings, so they are not independent samples.
real = {(r["dataset"], r["system"]): r for r in read_csv(RES / "summary_nepali_real.csv") if r["group"] == "all"}


def per_file_range(ds, s):
    p = RES / "raw" / (f"{ds}/scores_{s}.csv" if s.startswith("community1") else f"scores_{s}_{ds}.csv")
    v = [Decimal(r["der_no_collar"]) for r in read_csv(p)]
    return f"{min(v) * 100:.1f} to {max(v) * 100:.1f}" if v else TODO


def real_cells(ds, s):
    if s.startswith("community1"):
        v = C1P.get((ds, s))
        return ([pct(x) for x in v] if v else [TODO] * 5) + [per_file_range(ds, s)]
    r = real.get((ds, s))
    if not r:
        return [TODO] * 6
    cells = [pct(r[f"pooled_{m}_no_collar"]) for m in ("der", "miss", "fa", "conf")] + [pct(r["pooled_der_collar_0.25"])]
    return cells + [per_file_range(ds, s)]


lines = []
for ds, title in REAL:
    lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
    for key, _, short in ORDER:
        lines.append(f"{short} & " + " & ".join(real_cells(ds, key)) + " \\\\")
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
        cells += [real_cells(ds, key)[j] for ds, _ in REAL]
        r = sim.get(("nepali_sim", "all", key))
        cells.append(pct(r[f"pooled_{m}"]) if r else TODO)
        head = rf"\multirow{{3}}{{*}}{{{short}}}" if j == 1 else ""
        lines.append(f"{head} & {name} & " + " & ".join(cells) + " \\\\")
    lines.append(r"\addlinespace")
lines[-1] = r"\botrule"
(OUT / "shift.tex").write_text("\n".join(lines) + "\n")
print("ok")
