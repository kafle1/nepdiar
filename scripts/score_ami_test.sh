#!/bin/bash
# score every AMI test run (each system, estimated and oracle k), then build results/summary.csv
cd "$(dirname "$0")/.." || exit 1
for s in sherpa_onnx sherpa_onnx_oracle wespeaker_spectral wespeaker_spectral_oracle wespeaker_spectral_silero wespeaker_spectral_silero_oracle pyannote_style pyannote_style_oracle pyannote31 pyannote31_oracle; do
  n=$(ls results/raw/ami/${s}_rttm/*.rttm 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = 16 ] || { echo "$s has $n/16 rttms, not scoring it"; continue; }
  ./.venv/bin/python scripts/score.py --ref-dir ref/ami/only_words/rttms/test --hyp-dir results/raw/ami/${s}_rttm \
    --uem-dir ref/ami/uems/test --meetings-file ref/ami/test.meetings.txt \
    --system $s --dataset ami_test --out-csv results/raw/scores_${s}_ami.csv
done
./.venv/bin/python scripts/summarise.py "results/raw/scores_*_ami.csv" --out results/summary.csv
