#!/bin/bash
# score the english twin runs with the same scorer and settings as score_ami_test.sh, then the
# per-recording summary (results/summary_english_sim.csv) and pooled + per-speaker-count tables
# (results/corpus_der_english_sim.txt). AMI files are not touched.
cd "$(dirname "$0")/.." || exit 1
M=data/english_sim/conversations.txt
n_all=$(wc -l < $M | tr -d ' ')
for ds in english_sim english_sim_rn; do
  for s in sherpa_onnx sherpa_onnx_oracle wespeaker_spectral wespeaker_spectral_oracle wespeaker_spectral_silero wespeaker_spectral_silero_oracle pyannote_style pyannote_style_oracle pyannote31 pyannote31_oracle community1 community1_oracle; do
    n=$(ls results/raw/$ds/${s}_rttm/*.rttm 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "$n_all" ] || { echo "$ds $s has $n/$n_all rttms, not scoring it"; continue; }
    ./.venv/bin/python scripts/score.py --ref-dir data/english_sim/rttm --hyp-dir results/raw/$ds/${s}_rttm \
      --uem-dir data/english_sim/uem --meetings-file $M \
      --system $s --dataset $ds --out-csv results/raw/scores_${s}_${ds}.csv > /dev/null
  done
done
./.venv/bin/python scripts/summarise.py "results/raw/scores_*_english_sim.csv" "results/raw/scores_*_english_sim_rn.csv" --out results/summary_english_sim.csv
./.venv/bin/python scripts/nepali_sim_report.py english | tee results/corpus_der_english_sim.txt
