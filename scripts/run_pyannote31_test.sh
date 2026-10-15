#!/bin/bash
# released pyannote 3.1 on AMI test, estimated k then oracle k, on the mac gpu (cpu gave the same output on dev IS1008a, see NOTES.md)
cd "$(dirname "$0")/.."
PY="nice -n 10 .venv-pyannote3/bin/python -u scripts/run_pyannote_gated.py --model 3.1 --device mps --threads 2"
W="--wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file ref/ami/test.meetings.txt"
O=results/raw/ami
$PY $W --out-dir $O/pyannote31_rttm --timing-csv $O/pyannote31_timing.csv --timing-note mps-under-cpu-load
echo "=== estimated k done $(date) exit=$? ==="
$PY $W --oracle-ref-dir ref/ami/only_words/rttms/test --out-dir $O/pyannote31_oracle_rttm --timing-csv $O/pyannote31_oracle_timing.csv --timing-note oracle-k-mps-under-cpu-load
echo "=== ALL DONE $(date) exit=$? ==="
