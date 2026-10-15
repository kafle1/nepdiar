#!/bin/bash
# released pyannote community-1 (pyannote.audio 4.0.7) on AMI test, default settings then oracle k, on the mac gpu.
# then one meeting again on cpu, to check the device changes speed only (same check NOTES.md did for 3.1).
#   nohup scripts/run_community1.sh > logs/ami_test_community1.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY="nice -n 10 .venv-pyannote/bin/python -u scripts/run_pyannote_gated.py --model community-1"
W="--wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file ref/ami/test.meetings.txt"
O=results/raw/ami
$PY $W --device mps --threads 2 --out-dir $O/community1_rttm --timing-csv $O/community1_timing.csv --timing-note mps-under-cpu-load
echo "=== estimated k done $(date) exit=$? ==="
$PY $W --device mps --threads 2 --oracle-ref-dir ref/ami/only_words/rttms/test --out-dir $O/community1_oracle_rttm \
  --timing-csv $O/community1_oracle_timing.csv --timing-note oracle-k-mps-under-cpu-load
echo "=== oracle k done $(date) exit=$? ==="
C=$O/community1_cpu_check
mkdir -p $C
echo IS1009a > $C/meetings.txt
$PY --wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file $C/meetings.txt --device cpu --threads 4 \
  --out-dir $C/rttm --timing-csv $C/timing.csv --timing-note cpu-device-check-under-load
cmp -s $C/rttm/IS1009a.rttm $O/community1_rttm/IS1009a.rttm && echo "cpu vs mps IS1009a: byte-identical" || echo "cpu vs mps IS1009a: DIFFERS"
echo "=== ALL DONE $(date) ==="
