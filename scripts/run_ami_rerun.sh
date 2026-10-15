#!/bin/bash
# AMI runs, one pipeline per invocation, stops if the disk drops under 20 GB free.
#   scripts/run_ami_rerun.sh dev "0.5 0.6 0.7 0.8"   spectral, then sherpa-onnx at each threshold
#   scripts/run_ami_rerun.sh test spectral           spectral, estimated k then oracle k
#   scripts/run_ami_rerun.sh test 0.7                sherpa-onnx at 0.7, then oracle k (safe to run beside the spectral one)
cd "$(dirname "$0")/.." || exit 1
split=$1
PY="nice -n 10 ./.venv/bin/python -u"
W="--wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file ref/ami/$split.meetings.txt"
REF=ref/ami/only_words/rttms/$split
# 4 threads gave a byte-identical rttm to 1 thread on IS1008a, so untimed runs use it
FAST="--num-threads 4 --timing-note 4-threads-not-a-timing-run"

run() {
  local name=$1 free rc
  shift
  free=$(df -k / | awk 'NR==2 {print $4}')
  if [ "$free" -lt 20971520 ]; then
    echo "=== STOP before $name: only $((free / 1048576)) GB free ==="
    exit 1
  fi
  echo "=== $name started $(date) ==="
  "$@"
  rc=$?
  echo "=== $name finished $(date), exit=$rc ==="
}

if [ "$split" = dev ]; then
  O=results/raw/ami_dev
  run "spectral dev" $PY scripts/run_wespeaker_spectral.py $W --out-dir $O/wespeaker_spectral_rttm --timing-csv $O/wespeaker_spectral_timing.csv --timing-note dev-run-under-load
  for t in $2; do
    run "sherpa-onnx dev t=$t" $PY scripts/run_sherpa_onnx.py $W $FAST --threshold "$t" --out-dir $O/sherpa_onnx_t${t}_rttm --timing-csv $O/sherpa_onnx_t${t}_timing.csv
  done
elif [ "$split" = test ] && [ "$2" = spectral ]; then
  O=results/raw/ami
  run "spectral test" $PY scripts/run_wespeaker_spectral.py $W --out-dir $O/wespeaker_spectral_rttm --timing-csv $O/wespeaker_spectral_timing.csv
  run "spectral test oracle k" $PY scripts/run_wespeaker_spectral.py $W --oracle-ref-dir $REF --out-dir $O/wespeaker_spectral_oracle_rttm --timing-csv $O/wespeaker_spectral_oracle_timing.csv --timing-note oracle-k-under-load
elif [ "$split" = test ] && [ -n "$2" ]; then
  # 1 thread would take ~4.5 h under this load; timings get an idle rerun anyway
  O=results/raw/ami
  run "sherpa-onnx test t=$2" $PY scripts/run_sherpa_onnx.py $W $FAST --threshold "$2" --out-dir $O/sherpa_onnx_rttm --timing-csv $O/sherpa_onnx_timing.csv
  run "sherpa-onnx test oracle k" $PY scripts/run_sherpa_onnx.py $W $FAST --oracle-ref-dir $REF --out-dir $O/sherpa_onnx_oracle_rttm --timing-csv $O/sherpa_onnx_oracle_timing.csv
else
  echo "usage: $0 dev \"THRESHOLDS\" | test spectral | test THRESHOLD" >&2
  exit 2
fi
echo "=== ALL DONE $(date) ==="
