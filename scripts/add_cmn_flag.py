#!/usr/bin/env python3
"""Copy the WeSpeaker ONNX model with feature_normalize_type=global-mean added to its metadata.

sherpa-onnx only subtracts the per-utterance fbank mean when the model carries
that key, and the sherpa-onnx export of this model doesn't. WeSpeaker trained
it with that mean subtracted, so without the key the embeddings come out wrong.
Weights are untouched. Needs the onnx package, which isn't in .venv:
  uv run --no-project --with onnx python scripts/add_cmn_flag.py
"""
from pathlib import Path

import onnx

SRC = Path(__file__).resolve().parent.parent / "models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.onnx"
DST = SRC.with_suffix(".cmn.onnx")

m = onnx.load(SRC)
meta = {p.key: p for p in m.metadata_props}
if "feature_normalize_type" in meta:
    meta["feature_normalize_type"].value = "global-mean"
else:
    m.metadata_props.add(key="feature_normalize_type", value="global-mean")
onnx.save(m, DST)
print(f"wrote {DST}")
