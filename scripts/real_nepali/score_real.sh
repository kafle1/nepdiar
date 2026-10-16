#!/bin/bash
# score the real nepali runs with the same scorer and settings as scripts/score_ami_test.sh, then
# results/summary_nepali_real.csv and results/corpus_der_nepali_real.txt. AMI and sim files untouched.
cd "$(dirname "$0")/../.." || exit 1
for ds in indic_diarbench_ne indicvoices_ne_calls indicvoices_ne_calls_tight; do
  case $ds in
    indic_diarbench_ne) D=data/nepali_real/diarbench; M=$D/clips.txt; R=$D/rttm; H=$ds ;;
    indicvoices_ne_calls) D=data/nepali_real/indicvoices; M=$D/calls.txt; R=$D/rttm; H=$ds ;;
    indicvoices_ne_calls_tight) D=data/nepali_real/indicvoices; M=$D/calls.txt; R=$D/rttm_tight; H=indicvoices_ne_calls ;;
  esac
  [ -f $M ] || continue
  n_all=$(grep -c . $M)
  for s in sherpa_onnx sherpa_onnx_oracle wespeaker_spectral wespeaker_spectral_oracle wespeaker_spectral_silero wespeaker_spectral_silero_oracle pyannote_style pyannote_style_oracle pyannote31 pyannote31_oracle community1 community1_oracle; do
    n=$(while read -r u; do [ -n "$u" ] && [ -f results/raw/$H/${s}_rttm/$u.rttm ] && echo; done < $M | wc -l | tr -d ' ')
    [ "$n" = "$n_all" ] || { echo "$ds $s has $n/$n_all rttms, not scoring it"; continue; }
    ./.venv/bin/python scripts/score.py --ref-dir $R --hyp-dir results/raw/$H/${s}_rttm \
      --uem-dir $D/uem --meetings-file $M \
      --system $s --dataset $ds --out-csv results/raw/scores_${s}_${ds}.csv > /dev/null
  done
done
./.venv/bin/python scripts/real_nepali/report_real.py | tee results/corpus_der_nepali_real.txt
