#!/bin/bash
# three queues so cpu-heavy runs don't pile up; pyannote 3.1 rides the gpu. clean first, then reverb + noise
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
wait_rn() { until [ -f data/nepali_sim/wav_rn/SHA256SUMS ]; do sleep 30; done; }
( for g in sherpa spectral silero; do scripts/run_nepali_sim.sh nepali_sim $g; done; wait_rn
  for g in sherpa spectral silero; do scripts/run_nepali_sim.sh nepali_sim_rn $g; done ) > logs/nepali_sim_queue_a.log 2>&1 &
( scripts/run_nepali_sim.sh nepali_sim pyannote_style; wait_rn; scripts/run_nepali_sim.sh nepali_sim_rn pyannote_style ) > logs/nepali_sim_queue_b.log 2>&1 &
( scripts/run_nepali_sim.sh nepali_sim pyannote31; wait_rn; scripts/run_nepali_sim.sh nepali_sim_rn pyannote31 ) > logs/nepali_sim_queue_c.log 2>&1 &
( nice .venv-sim/bin/python -u scripts/make_nepali_sim.py rooms && nice .venv/bin/python -u scripts/make_nepali_sim.py render_rn ) > logs/nepali_sim_rn_build.log 2>&1 &
wait
echo "ALL QUEUES DONE $(date)"
