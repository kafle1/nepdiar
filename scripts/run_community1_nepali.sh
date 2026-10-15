#!/bin/bash
# released pyannote community-1 on the nepali sets, default settings then oracle k, mac gpu.
# same audio, file lists and references as scripts/real_nepali/run_real.sh and scripts/run_nepali_sim.sh.
#   nohup scripts/run_community1_nepali.sh DATASET > logs/community1_DATASET.log 2>&1 &
#   DATASET: indic_diarbench_ne | indicvoices_ne_calls | nepali_sim | nepali_sim_rn
# the runner skips an rttm that already exists, so a rerun picks up where it stopped
cd "$(dirname "$0")/.." || exit 1
ds=$1
case $ds in
  indic_diarbench_ne) D=data/nepali_real/diarbench; WAV=$D/wav; M=$D/clips.txt; REF=$D/rttm ;;
  indicvoices_ne_calls) D=data/nepali_real/indicvoices; WAV=$D/wav; M=$D/calls.txt; REF=$D/rttm ;;
  nepali_sim) WAV=data/nepali_sim/wav; M=data/nepali_sim/conversations.txt; REF=data/nepali_sim/rttm ;;
  nepali_sim_rn) WAV=data/nepali_sim/wav_rn; M=data/nepali_sim/conversations.txt; REF=data/nepali_sim/rttm ;;
  *) echo "unknown dataset $ds" >&2; exit 2 ;;
esac
O=results/raw/$ds
W="--wav-dir $WAV --suffix .wav --meetings-file $M"
PY="nice -n 10 .venv-pyannote/bin/python -u scripts/run_pyannote_gated.py --model community-1 --device mps --threads 2"
mkdir -p $O
echo "=== community-1 $ds started $(date) ==="
$PY $W --out-dir $O/community1_rttm --timing-csv $O/community1_timing.csv --timing-note mps
echo "=== estimated k done $(date) exit=$? ==="
$PY $W --oracle-ref-dir $REF --out-dir $O/community1_oracle_rttm --timing-csv $O/community1_oracle_timing.csv --timing-note oracle-k-mps
echo "=== ALL DONE $ds $(date) exit=$? ==="
