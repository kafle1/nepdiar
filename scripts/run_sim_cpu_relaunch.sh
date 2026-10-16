#!/bin/bash
# relaunch the cpu groups the 20 GB disk guard stopped on 2026-10-16. two queues, so at most two cpu jobs at once.
#   nohup scripts/run_sim_cpu_relaunch.sh > logs/sim_cpu_relaunch.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
echo "relaunch started $(date)"
# A: nepali reverb + noise S1, S2, S2b, then english reverb + noise
( for g in sherpa spectral silero; do scripts/run_nepali_sim.sh nepali_sim_rn $g > logs/nepali_sim_rn_${g}_relaunch.log 2>&1; done
  for g in sherpa spectral silero pyannote_style; do scripts/run_english_sim.sh english_sim_rn $g > logs/english_sim_english_sim_rn_$g.log 2>&1; done ) &
# B: nepali reverb + noise S3 (its extract is cached, so that step skips all 30), then english clean
( scripts/run_nepali_sim.sh nepali_sim_rn pyannote_style > logs/nepali_sim_rn_pyannote_style_relaunch.log 2>&1
  for g in sherpa spectral silero pyannote_style; do scripts/run_english_sim.sh english_sim $g > logs/english_sim_english_sim_$g.log 2>&1; done ) &
wait
bad=0
for f in logs/nepali_sim_rn_*_relaunch.log logs/english_sim_english_sim_{,rn_}{sherpa,spectral,silero,pyannote_style}.log; do
  grep -q "=== ALL DONE" "$f" || { echo "NOT DONE: $f"; bad=1; }
done
[ $bad = 0 ] && echo "SIM CPU RELAUNCH DONE $(date)" || echo "SIM CPU RELAUNCH INCOMPLETE $(date)"
