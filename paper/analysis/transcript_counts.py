# word, tag and english-gloss counts for the two real nepali sets, read only. run: python3 -I transcript_counts.py
# words = whitespace tokens left after deleting every <...> or [...] tag, so a tag stuck to a word still counts as a tag
import json, re
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

R = Path(__file__).resolve().parents[2] / "data/nepali_real"
GLOSS = re.compile(r"\([A-Za-z][^)]*\)")  # diarbench writes a loanword as देवनागरी(english)


def count(texts, tag):
    tags = Counter(m for t in texts for m in tag.findall(t))
    words = [w for t in texts for w in tag.sub(" ", t).split()]
    return words, tags


t = pq.read_table(R / "diarbench/Nepali_test-00000-of-00001.parquet", columns=["annotated_transcript"]).to_pylist()
texts = [s["transcript"] for r in t for s in r["annotated_transcript"]]
words, tags = count(texts, re.compile(r"<[^>]*>"))
g = sum(len(GLOSS.findall(t)) for t in texts)  # a gloss can hold a space, so count it in the text, not per token
print(f"diarbench words {len(words)} glosses {g} per_word {g / len(words):.4f} tags {sum(tags.values())} {tags.most_common(6)}")

calls = {l.strip() for l in (R / "indicvoices/calls.txt").read_text().splitlines() if l.strip()}
texts = []
for d in ("raw_v1_train", "raw_v2_valid"):
    for j in (R / "indicvoices" / d).glob("*.json"):
        if j.stem.rsplit("_", 1)[0] in calls:
            texts += [s["text"] for s in json.loads(j.read_text())["verbatim"]]
words, tags = count(texts, re.compile(r"\[[^\]]*\]"))
lat = sum(1 for w in words if re.search(r"[A-Za-z]", w))
print(f"indicvoices words {len(words)} with_latin {lat} tags {sum(tags.values())} {tags.most_common(8)}")
