#!/bin/bash
# sherpa-onnx on 2 short dev meetings (IS1008a, ES2011a), to find the threshold range before the full dev grid.
#   scripts/sweep_short_dev.sh cmn "oracle 0.7 0.8"   cmn or nocmn picks the embedding model, oracle means oracle k
# full dev: M=ref/ami/dev.meetings.txt O=results/raw/ami_dev_full DS=ami_dev scripts/sweep_short_dev.sh cmn "1.0 1.1"
cd "$(dirname "$0")/.." || exit 1
case $1 in
  cmn) EMB=models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.cmn.onnx ;;
  nocmn) EMB=models/wespeaker/wespeaker_en_voxceleb_resnet34_LM.onnx ;;
  *) echo "usage: $0 cmn|nocmn \"oracle 0.7 0.8 ...\"" >&2; exit 2 ;;
esac
M=${M:-ref/ami/dev_short.meetings.txt}
O=${O:-results/raw/ami_dev_short}
DS=${DS:-ami_dev_short}
for t in $2; do
  if [ "$t" = oracle ]; then name=sherpa_onnx_$1_oracle; extra="--oracle-ref-dir ref/ami/only_words/rttms/dev"
  else name=sherpa_onnx_$1_t$t; extra="--threshold $t"; fi
  echo "=== $name started $(date) ==="
  nice -n 10 ./.venv/bin/python -u scripts/run_sherpa_onnx.py --wav-dir data/ami/audio --suffix .Mix-Headset.wav \
    --meetings-file $M --num-threads 4 --timing-note 4-threads-not-a-timing-run --emb-model $EMB $extra \
    --out-dir $O/${name}_rttm --timing-csv $O/${name}_timing.csv
  nice -n 10 ./.venv/bin/python scripts/score.py --ref-dir ref/ami/only_words/rttms/dev --hyp-dir $O/${name}_rttm \
    --uem-dir ref/ami/uems/dev --meetings-file $M --system $name --dataset $DS --out-csv $O/scores_$name.csv
  echo "=== $name finished $(date) ==="
done
echo "=== ALL DONE $(date) ==="
