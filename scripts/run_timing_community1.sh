#!/bin/bash
# idle, 1-thread timing of community-1 on the same 3 test meetings as scripts/run_timing_idle.sh, cpu then mac gpu, no nice.
# start only when nothing else heavy runs:
#   nohup scripts/run_timing_community1.sh > logs/timing_community1.log 2>&1 &
# outputs go to results/raw/timing_idle/community1_*, never over the test RTTMs; the end compares them to the test RTTMs.
cd "$(dirname "$0")/.." || exit 1
O=results/raw/timing_idle
M=$O/community1_meetings.txt
W="--wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file $M"
mkdir -p $O
printf "IS1009a\nES2004a\nTS3003a\n" > $M
note() { echo "idle 1 thread, load $(sysctl -n vm.loadavg | awk '{print $2}') at start"; }

for dev in cpu mps; do
  echo "=== community-1 $dev $(date)"
  .venv-pyannote/bin/python -u scripts/run_pyannote_gated.py --model community-1 $W --device $dev --threads 1 \
    --out-dir $O/community1_${dev}_rttm --timing-csv $O/community1_${dev}_timing.csv --timing-note "$(note)"
done

echo "=== same output as the test run?"
for dev in cpu mps; do
  for u in $(cat $M); do
    cmp -s $O/community1_${dev}_rttm/$u.rttm results/raw/ami/community1_rttm/$u.rttm && echo "community1_$dev $u same" || echo "community1_$dev $u DIFFERS"
  done
done
echo "=== ALL DONE $(date)"
