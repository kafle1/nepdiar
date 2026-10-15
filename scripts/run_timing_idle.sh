#!/bin/bash
# idle, 1-thread timing on 3 test meetings, one system at a time, no nice. Start only when nothing else heavy runs.
#   nohup scripts/run_timing_idle.sh > logs/timing_idle.log 2>&1 &
# outputs go to results/raw/timing_idle/, never over the real test RTTMs; the end compares them to the test RTTMs.
cd "$(dirname "$0")/.." || exit 1
M=results/raw/timing_idle/meetings.txt
O=results/raw/timing_idle
W="--wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file $M"
SHERPA_T=${SHERPA_T:-1.1}
mkdir -p $O
printf "IS1009a\nES2004a\nTS3003a\n" > $M
note() { echo "idle 1 thread, load $(sysctl -n vm.loadavg | awk '{print $2}') at start"; }

echo "=== sherpa $(date)"
./.venv/bin/python -u scripts/run_sherpa_onnx.py $W --num-threads 1 --threshold $SHERPA_T \
  --out-dir $O/sherpa_onnx_rttm --timing-csv $O/sherpa_onnx_timing.csv --timing-note "$(note)"
for v in webrtc silero; do
  echo "=== spectral $v $(date)"
  ./.venv/bin/python -u scripts/run_wespeaker_spectral.py $W --vad $v \
    --out-dir $O/spectral_${v}_rttm --timing-csv $O/spectral_${v}_timing.csv --timing-note "$(note)"
done
echo "=== system 3 extract $(date)"
./.venv/bin/python -u scripts/run_pyannote_style.py extract $W --threads 1 \
  --cache-dir $O/pyannote_style_cache --timing-csv $O/pyannote_style_extract_timing.csv --timing-note "$(note)"
echo "=== system 3 cluster $(date)"
/usr/bin/time -l ./.venv/bin/python scripts/run_pyannote_style.py cluster --cache-dir $O/pyannote_style_cache \
  --meetings-file $M --out-dir $O/pyannote_style_rttm --threshold 0.7 --min-cluster-size 12 --min-clean-ratio 0.2 \
  2> $O/pyannote_style_cluster_time.txt
for dev in cpu mps; do
  echo "=== pyannote 3.1 $dev $(date)"
  .venv-pyannote3/bin/python -u scripts/run_pyannote_gated.py --model 3.1 $W --device $dev --threads 1 \
    --out-dir $O/pyannote31_${dev}_rttm --timing-csv $O/pyannote31_${dev}_timing.csv --timing-note "$(note)"
done

echo "=== same output as the test run?"
for pair in sherpa_onnx:sherpa_onnx spectral_webrtc:wespeaker_spectral spectral_silero:wespeaker_spectral_silero \
            pyannote_style:pyannote_style pyannote31_cpu:pyannote31 pyannote31_mps:pyannote31; do
  a=${pair%%:*}; b=${pair##*:}
  for u in $(cat $M); do
    cmp -s $O/${a}_rttm/$u.rttm results/raw/ami/${b}_rttm/$u.rttm && echo "$a $u same" || echo "$a $u DIFFERS"
  done
done
echo "=== ALL DONE $(date)"
