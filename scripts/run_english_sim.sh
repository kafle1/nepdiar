#!/bin/bash
# every system on the english twin of the simulated nepali set with its final AMI settings, unchanged (zero-shot). AMI paths untouched.
#   scripts/run_english_sim.sh DATASET GROUP
#   DATASET: english_sim (clean, data/english_sim/wav) | english_sim_rn (reverb + noise, data/english_sim/wav_rn)
#   GROUP:   sherpa | spectral | silero | pyannote_style | pyannote31 | community1
# each runner skips an rttm that already exists, so a rerun picks up where it stopped
cd "$(dirname "$0")/.." || exit 1
ds=$1
case $ds in
  english_sim) WAV=data/english_sim/wav ;;
  english_sim_rn) WAV=data/english_sim/wav_rn ;;
  *) echo "unknown dataset $ds" >&2; exit 2 ;;
esac
M=data/english_sim/conversations.txt
REF=data/english_sim/rttm
O=results/raw/$ds
W="--wav-dir $WAV --suffix .wav --meetings-file $M"
PY="nice -n 10 ./.venv/bin/python -u"
FAST="--num-threads 4 --timing-note 4-threads-not-a-timing-run"
mkdir -p $O

run() {
  local name=$1 free
  shift
  free=$(df -k / | awk 'NR==2 {print $4}')
  if [ "$free" -lt 12582912 ]; then  # 12 GB: runs write little, the floor guards against swap growth
    echo "=== STOP before $name: only $((free / 1048576)) GB free ==="
    exit 1
  fi
  echo "=== $name started $(date) ==="
  "$@"
  echo "=== $name finished $(date), exit=$? ==="
}

case $2 in
  sherpa)
    run "sherpa-onnx t=1.1" $PY scripts/run_sherpa_onnx.py $W $FAST --threshold 1.1 --out-dir $O/sherpa_onnx_rttm --timing-csv $O/sherpa_onnx_timing.csv
    run "sherpa-onnx oracle k" $PY scripts/run_sherpa_onnx.py $W $FAST --oracle-ref-dir $REF --out-dir $O/sherpa_onnx_oracle_rttm --timing-csv $O/sherpa_onnx_oracle_timing.csv ;;
  spectral)
    run "spectral webrtc" $PY scripts/run_wespeaker_spectral.py $W --out-dir $O/wespeaker_spectral_rttm --timing-csv $O/wespeaker_spectral_timing.csv
    run "spectral webrtc oracle k" $PY scripts/run_wespeaker_spectral.py $W --oracle-ref-dir $REF --out-dir $O/wespeaker_spectral_oracle_rttm --timing-csv $O/wespeaker_spectral_oracle_timing.csv ;;
  silero)
    run "spectral silero" $PY scripts/run_wespeaker_spectral.py $W --vad silero --out-dir $O/wespeaker_spectral_silero_rttm --timing-csv $O/wespeaker_spectral_silero_timing.csv
    run "spectral silero oracle k" $PY scripts/run_wespeaker_spectral.py $W --vad silero --oracle-ref-dir $REF --out-dir $O/wespeaker_spectral_silero_oracle_rttm --timing-csv $O/wespeaker_spectral_silero_oracle_timing.csv ;;
  pyannote_style)
    # same cluster settings as the AMI test rttms
    run "system 3 extract" $PY scripts/run_pyannote_style.py extract $W --threads 4 --cache-dir results/cache/pyannote_style_$ds --timing-csv $O/pyannote_style_extract_timing.csv --timing-note 4-threads-not-a-timing-run
    run "system 3 cluster" $PY scripts/run_pyannote_style.py cluster --cache-dir results/cache/pyannote_style_$ds --meetings-file $M --out-dir $O/pyannote_style_rttm --threshold 0.7 --min-cluster-size 12 --min-clean-ratio 0.2
    run "system 3 cluster oracle k" $PY scripts/run_pyannote_style.py cluster --cache-dir results/cache/pyannote_style_$ds --meetings-file $M --out-dir $O/pyannote_style_oracle_rttm --threshold 0.7 --min-cluster-size 24 --min-clean-ratio 0.2 --oracle-ref-dir $REF ;;
  pyannote31)
    P3="nice -n 10 .venv-pyannote3/bin/python -u scripts/run_pyannote_gated.py --model 3.1 --device mps --threads 2"
    run "pyannote 3.1" $P3 $W --out-dir $O/pyannote31_rttm --timing-csv $O/pyannote31_timing.csv --timing-note mps
    run "pyannote 3.1 oracle k" $P3 $W --oracle-ref-dir $REF --out-dir $O/pyannote31_oracle_rttm --timing-csv $O/pyannote31_oracle_timing.csv --timing-note oracle-k-mps ;;
  community1)
    # same as scripts/run_community1_nepali.sh
    C1="nice -n 10 .venv-pyannote/bin/python -u scripts/run_pyannote_gated.py --model community-1 --device mps --threads 2"
    run "community-1" $C1 $W --out-dir $O/community1_rttm --timing-csv $O/community1_timing.csv --timing-note mps
    run "community-1 oracle k" $C1 $W --oracle-ref-dir $REF --out-dir $O/community1_oracle_rttm --timing-csv $O/community1_oracle_timing.csv --timing-note oracle-k-mps ;;
  *) echo "unknown group $2" >&2; exit 2 ;;
esac
echo "=== ALL DONE $ds $2 $(date) ==="
