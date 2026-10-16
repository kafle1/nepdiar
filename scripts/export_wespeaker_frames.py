#!/usr/bin/env python3
"""Build the two System 3 files from the stock WeSpeaker ResNet34-LM ONNX.

- resnet34_LM.frames.onnx: the same graph with its pre-pooling tensor "343"
  (B, 256, 10, T) added as a first output, so one pass gives the frames
- resnet34_LM.head.npz: the weights after pooling (w, b of the final Gemm and
  mean_vec), so run_pyannote_style.py can pool each local speaker in numpy

Weights are untouched. Needs the onnx package:
  .venv/bin/python scripts/export_wespeaker_frames.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIR = Path(__file__).resolve().parent.parent / "models/wespeaker"
SRC = DIR / "wespeaker_en_voxceleb_resnet34_LM.onnx"
FRAMES = DIR / "resnet34_LM.frames.onnx"
HEAD = DIR / "resnet34_LM.head.npz"

m = onnx.load(SRC)
if all(o.name != "343" for o in m.graph.output):
    m.graph.output.insert(0, helper.make_tensor_value_info("343", TensorProto.FLOAT, ["B", 256, 10, "T"]))
onnx.save(m, FRAMES)
print(f"wrote {FRAMES}")

init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
np.savez(HEAD, w=init["model.seg_1.weight"], b=init["model.seg_1.bias"], mean_vec=init["mean_vec"])
print(f"wrote {HEAD}")
