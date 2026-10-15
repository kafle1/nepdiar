#!/bin/bash
# every configuration on both english twin conditions. cpu groups one at a time, only after the nepali
# cpu queues finish; gpu groups one at a time through the shared gpu lock (inside run_english_sim.sh).
#   nohup scripts/run_english_sim_all.sh > logs/english_sim_all.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
until [ -s data/english_sim/wav/SHA256SUMS ] && [ -s data/english_sim/wav_rn/SHA256SUMS ]; do sleep 60; done
echo "twin audio ready $(date)"
(
  for ds in english_sim english_sim_rn; do
    for g in pyannote31 community1; do scripts/run_english_sim.sh $ds $g > logs/english_sim_${ds}_$g.log 2>&1; done
  done
  echo "GPU CHAIN DONE $(date)"
) &
until grep -q "ALL QUEUES DONE" logs/nepali_sim_all.log 2>/dev/null; do sleep 60; done
echo "nepali cpu queues done $(date)"
for ds in english_sim english_sim_rn; do
  for g in sherpa spectral silero pyannote_style; do scripts/run_english_sim.sh $ds $g > logs/english_sim_${ds}_$g.log 2>&1; done
done
echo "CPU CHAIN DONE $(date)"
wait
echo "ENGLISH ALL DONE $(date)"
