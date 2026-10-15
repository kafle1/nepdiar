#!/bin/bash
# system 2 with the dev-picked silero vad on AMI test, estimated k then oracle k. run once.
cd "$(dirname "$0")/.."
W="--wav-dir data/ami/audio --suffix .Mix-Headset.wav --meetings-file ref/ami/test.meetings.txt --vad silero"
O=results/raw/ami
nice -n 10 ./.venv/bin/python -u scripts/run_wespeaker_spectral.py $W --out-dir $O/wespeaker_spectral_silero_rttm --timing-csv $O/wespeaker_spectral_silero_timing.csv
echo "=== estimated k done $(date) exit=$? ==="
nice -n 10 ./.venv/bin/python -u scripts/run_wespeaker_spectral.py $W --oracle-ref-dir ref/ami/only_words/rttms/test --out-dir $O/wespeaker_spectral_silero_oracle_rttm --timing-csv $O/wespeaker_spectral_silero_oracle_timing.csv --timing-note oracle-k-under-load
echo "=== ALL DONE $(date) exit=$? ==="
