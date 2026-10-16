#!/usr/bin/env bash
# fetch the ONNX models S1 to S3 load, check their sha256 and build the derived copies. run from the repo root.
set -eu

PY=.venv/bin/python
REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
SEG=models/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
EMB=models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.onnx
VAD=models/silero/silero_vad.onnx

sha() { python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
ok() { [ -f "$1" ] && [ "$(sha "$1")" = "$2" ]; }
check() { ok "$1" "$2" || { echo "sha256 mismatch: $1 (want $2, got $( [ -f "$1" ] && sha "$1" || echo missing))" >&2; exit 1; }; echo "ok $1"; }
fetch() {
  mkdir -p "$(dirname "$2")"
  echo "downloading $1"
  python3 -c 'import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2.part"
  mv "$2.part" "$2"
}

# segmentation-3.0, release tag speaker-segmentation-models
if ! ok $SEG 220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079; then
  TAR=models/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  fetch $REL/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 $TAR
  check $TAR 24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488
  tar -xjf $TAR -C models/sherpa-onnx
  rm $TAR
fi
check $SEG 220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079

# wespeaker resnet34-LM, release tag speaker-recongition-models (their spelling)
ok $EMB e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012 \
  || fetch $REL/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx $EMB
check $EMB e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012

# silero vad, release tag asr-models
ok $VAD 9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6 \
  || fetch $REL/asr-models/silero_vad.onnx $VAD
check $VAD 9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6

# the derived copies need the onnx package, which requirements.txt leaves out
CMN=models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.cmn.onnx
FRAMES=models/wespeaker/resnet34_LM.frames.onnx
HEAD=models/wespeaker/resnet34_LM.head.npz
if ! ok $CMN ce2c5db3fb2c438f1d58de35577c951bc27ea0d41ef74968de4a480a1a59e39b \
  || ! ok $FRAMES 5e4849123d0289af808b744747c734b86f4c18493a2522b634b3e93de849abea \
  || ! ok $HEAD d1ead74571f90474242d665a65eec017ca86b82870ec0014b074d10a01dccc17; then
  "$PY" -c 'import onnx' 2>/dev/null || uv pip install --python .venv onnx==1.23.2
  "$PY" scripts/add_cmn_flag.py
  "$PY" scripts/export_wespeaker_frames.py
fi
check $CMN ce2c5db3fb2c438f1d58de35577c951bc27ea0d41ef74968de4a480a1a59e39b
check $FRAMES 5e4849123d0289af808b744747c734b86f4c18493a2522b634b3e93de849abea
check $HEAD d1ead74571f90474242d665a65eec017ca86b82870ec0014b074d10a01dccc17
echo "models ready"
