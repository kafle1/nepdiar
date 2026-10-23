# NepDiar

Do the open-source speaker diarization systems that work on English meetings hold up on Nepali?
This repo is the benchmark that checks, with the code, references and system outputs to rerun it.

Five systems (six setups) that run on a laptop are tuned on AMI dev, checked on the 16-meeting AMI
test set, then run unchanged on:

- Indic DiarBench, Nepali: 25 clips of online-meeting conversation, 3 to 6 speakers.
- IndicVoices, Nepali phone calls: 92 two-party calls (5.07 h), no speaker in more than one call.
- NepDiar-Sim: 30 simulated Nepali conversations (2, 3 and 4 speakers) built from OpenSLR 54 read
  speech, in a clean and a reverb + noise condition.
- English twin: the same 30 conversation plans (turn order, pauses, overlaps, rooms, noise, SNR) with
  LibriSpeech voices instead of Nepali ones. Comparing each Nepali conversation with its twin holds
  the recording setup fixed.

Scoring is DER and JER from `pyannote.metrics` (no collar and 0.25 s collar, overlap scored, inside
each set's UEM). The main test, fixed in `ANALYSIS_PLAN.md` (commit ea7664f) before any simulated set
was scored, is the paired DER difference between each NepDiar-Sim conversation and its English twin,
per configuration, with a 10,000-resample bootstrap and Holm's correction (`scripts/sim_pair_test.py`).

## Results

DER, no collar, lower is better.

- AMI test: S4 (pyannote 3.1) scores 18.81% pooled DER against its published 18.8%, which checks
  the scoring chain (`results/corpus_der_ami_test.txt`).
- Clean simulated audio: four configurations do worse on Nepali than on the English twin, S1 by
  6.92 points, S5 by 9.75, S3 by 11.91 and S4 by 13.79 (mean paired
  difference over 30 pairs, Holm-corrected, `results/sim_pair_test.csv`). S2a and S2b show no
  clear difference.
- Post hoc, most of that gap is missed speech near the edges of the simulated reference's speech,
  and its size depends on how those edges are drawn (`results/sim_reference_sensitivity.csv`). It
  does not show that Nepali speech is harder to diarize.
- Real Nepali sets (descriptive only): every configuration confuses speakers more on Indic
  DiarBench than on AMI test (`results/corpus_der_nepali_real.txt`).

## Citation

The paper is under review and uses version 1.0.1 of this repo, archived at
https://doi.org/10.5281/zenodo.23412673. The concept DOI https://doi.org/10.5281/zenodo.23412010
always points to the latest version. To cite the code and data, use `CITATION.cff`.

## Systems

| | System | Runner | Env |
|---|---|---|---|
| S1 | sherpa-onnx: pyannote segmentation-3.0 + WeSpeaker ResNet34-LM, agglomerative clustering | `scripts/run_sherpa_onnx.py` | `.venv` |
| S2a | WeSpeaker embeddings + NME-SC spectral clustering, WebRTC VAD | `scripts/run_wespeaker_spectral.py` | `.venv` |
| S2b | same as S2a with Silero VAD (`--vad silero`) | `scripts/run_wespeaker_spectral.py` | `.venv` |
| S3 | the pyannote 3.1 recipe rebuilt on ONNX, no torch | `scripts/run_pyannote_style.py` | `.venv` |
| S4 | pyannote/speaker-diarization-3.1, released settings | `scripts/run_pyannote_gated.py --model 3.1` | `.venv-pyannote3` |
| S5 | pyannote/speaker-diarization-community-1, released settings | `scripts/run_pyannote_gated.py --model community-1` | `.venv-pyannote` |

Each system also has an "oracle k" run that is given the true number of speakers.

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/), and ffmpeg 4 on PATH (or set `FFMPEG`, with `ffprobe`
next to it). Results were made on an Apple M3 Pro.

```
uv venv .venv --python 3.12            && uv pip install --python .venv -r requirements.txt
uv venv .venv-pyannote3 --python 3.12  && uv pip install --python .venv-pyannote3 -r requirements-pyannote3.txt
uv venv .venv-pyannote --python 3.12   && uv pip install --python .venv-pyannote -r requirements-pyannote.txt
uv venv .venv-sim --python 3.12        && uv pip install --python .venv-sim -r requirements-sim.txt
scripts/get_models.sh
```

`.venv` runs S1 to S3 and everything else, `.venv-pyannote3` is pyannote.audio 3.3.2 (S4),
`.venv-pyannote` is pyannote.audio 4.0.7 (S5), and `.venv-sim` is pyroomacoustics for the NepDiar-Sim
`rooms` stage. `scripts/get_models.sh` fetches and sha256-checks the ONNX models from the sherpa-onnx
releases and writes the derived copies S1 and S3 use.

S4 and S5 are gated on Hugging Face. Accept the terms for `pyannote/speaker-diarization-3.1`,
`pyannote/segmentation-3.0` and `pyannote/speaker-diarization-community-1`, then set `HF_TOKEN`, run
`huggingface-cli login`, or pass `--hf-token`.

## Data

No audio is in this repo. It goes under `data/`.

AMI, Mix-Headset, BUT test (16) and dev (18) splits. References, UEMs and meeting lists in
`ref/ami/` are from github.com/BUTSpeechFIT/AMI-diarization-setup at commit
`2509d8933721023fab4def2618aabd5c28eb82e9` (`only_words` references). Audio comes from the
Edinburgh mirror, checked against `ref/ami/audio.sha256`:

```
.venv/bin/python scripts/get_ami.py
.venv/bin/python scripts/get_ami.py --meetings-file ref/ami/dev.meetings.txt
```

Indic DiarBench, Nepali: download `Nepali/test-00000-of-00001.parquet` from
https://huggingface.co/datasets/sarvamai/indic-diarbench as
`data/nepali_real/diarbench/Nepali_test-00000-of-00001.parquet` and convert it (`git diff` checks
the references it writes):

```
uv run --no-project --with pyarrow --with numpy --with soundfile python -I scripts/real_nepali/prep_diarbench.py
```

IndicVoices, Nepali (https://huggingface.co/datasets/ai4bharat/IndicVoices): only the
"Conversation" files are streamed out of the release archives. The v1 train archive is 23.7 GB. The
92 calls are in `data/nepali_real/indicvoices/calls.txt` (91 from v1 train, 1 from v2 valid). The
cleaned references also ship on their own in `data/indicvoices_ne_calls/`, with how the calls were
picked.

```
U=https://iv-release.objectstore.e2enetworks.net/dmu_release
.venv/bin/python scripts/real_nepali/stream_indicvoices.py $U/v1_Nepali_train.tgz data/nepali_real/indicvoices/raw_v1_train
.venv/bin/python scripts/real_nepali/stream_indicvoices.py $U/v2_Nepali_valid.tgz data/nepali_real/indicvoices/raw_v2_valid
.venv/bin/python -I scripts/real_nepali/prep_indicvoices.py data/nepali_real/indicvoices/raw_v1_train data/nepali_real/indicvoices/raw_v2_valid
```

## Rebuild NepDiar-Sim

`data/nepali_sim/` has the references, manifests and checksums. `data/nepali_sim/release/` is the
standalone package.

1. Put `asr_nepali_0.zip` to `asr_nepali_3.zip` from https://www.openslr.org/54/ in
   `data/slr54/zips/`, and `DKITCHEN_16k.zip`, `DLIVING_16k.zip`, `DWASHING_16k.zip`,
   `OOFFICE_16k.zip`, `OHALLWAY_16k.zip` from https://zenodo.org/records/1227121 in
   `data/demand/zips/`. Leave them zipped. Check them from the repo root:
   `shasum -a 256 -c data/nepali_sim/sources.sha256`
2. Copy in the room impulse responses used for the published checksums:
   `cp data/nepali_sim/release/rirs.npz data/nepali_sim/rirs.npz`
   (or remake them from fixed seeds with pyroomacoustics 0.10.1:
   `.venv-sim/bin/python scripts/make_nepali_sim.py rooms`).
3. Render both conditions:
   ```
   .venv/bin/python scripts/make_nepali_sim.py render_clean
   .venv/bin/python scripts/make_nepali_sim.py render_rn
   ```
4. Compare `data/nepali_sim/wav/SHA256SUMS` with `data/nepali_sim/wav.SHA256SUMS`, and
   `wav_rn/SHA256SUMS` with `wav_rn.SHA256SUMS`.

The `stats`, `plan` and `check_nepali_speakers.py` stages remake the manifests, only needed to change
the design.

## Rebuild the English twin

`data/english_sim/` has the references, manifests, trimmed LibriSpeech regions and audio checksums.
`data/english_sim/release/` is the standalone package.

1. Download `dev-clean.tar.gz` and `test-clean.tar.gz` from https://www.openslr.org/12/ and unpack
   both into `data/librispeech/`, so it holds `LibriSpeech/dev-clean`, `LibriSpeech/test-clean` and
   `LibriSpeech/SPEAKERS.TXT`. The DEMAND zips are the same as above.
2. Do NepDiar-Sim step 2, then rekey the room responses (this rewrites identical manifests):
   `.venv/bin/python scripts/make_english_sim.py plan`
3. Render both conditions:
   ```
   .venv/bin/python scripts/make_english_sim.py render_clean
   .venv/bin/python scripts/make_english_sim.py render_rn
   ```
4. Compare `data/english_sim/wav/SHA256SUMS` and `wav_rn/SHA256SUMS` with the lists in
   `data/english_sim/release/wav_sha256/`.

## Rerun and score

Runners skip any recording whose RTTM already exists, so delete the matching
`results/raw/.../*_rttm/` folder for a fresh run. Outputs go to `results/raw/`.

Settings picked on AMI dev:

```
.venv/bin/python scripts/vad_sweep_dev.py            # S2a WebRTC VAD
.venv/bin/python scripts/vad_sweep_dev.py --silero   # S2b Silero VAD
M=ref/ami/dev.meetings.txt O=results/raw/ami_dev_full DS=ami_dev scripts/sweep_short_dev.sh cmn "1.0 1.1 1.2"   # S1 threshold
.venv/bin/python scripts/run_pyannote_style.py extract --wav-dir data/ami/audio --meetings-file ref/ami/dev.meetings.txt \
  --cache-dir results/cache/pyannote_style --timing-csv results/raw/ami_dev/pyannote_style_extract_timing.csv
.venv/bin/python scripts/sweep_pyannote_style_dev.py  # S3 clustering
```

AMI test:

```
scripts/run_ami_rerun.sh test spectral
scripts/run_ami_rerun.sh test 1.1
scripts/run_spectral_silero_test.sh
scripts/run_pyannote31_test.sh
scripts/run_community1.sh
# S3: extract once, then cluster with the dev-picked settings
.venv/bin/python scripts/run_pyannote_style.py extract --wav-dir data/ami/audio --meetings-file ref/ami/test.meetings.txt \
  --cache-dir results/cache/pyannote_style_test --timing-csv results/raw/ami/pyannote_style_extract_timing.csv
.venv/bin/python scripts/run_pyannote_style.py cluster --cache-dir results/cache/pyannote_style_test --meetings-file ref/ami/test.meetings.txt \
  --out-dir results/raw/ami/pyannote_style_rttm --threshold 0.7 --min-cluster-size 12 --min-clean-ratio 0.2
.venv/bin/python scripts/run_pyannote_style.py cluster --cache-dir results/cache/pyannote_style_test --meetings-file ref/ami/test.meetings.txt \
  --out-dir results/raw/ami/pyannote_style_oracle_rttm --threshold 0.7 --min-cluster-size 24 --min-clean-ratio 0.2 \
  --oracle-ref-dir ref/ami/only_words/rttms/test

scripts/score_ami_test.sh                                   # results/summary.csv
.venv/bin/python scripts/corpus_der.py > results/corpus_der_ami_test.txt
.venv/bin/python scripts/run_community1.py                   # results/summary_community1.csv
.venv/bin/python paper/make_tables.py                        # paper/tables/*.tex
```

Speed (one thread, idle machine, 3 AMI test meetings): `scripts/run_timing_idle.sh` and
`scripts/run_timing_community1.sh`. Each checks its RTTMs against the test run.

Real Nepali. DATASET is `indic_diarbench_ne` or `indicvoices_ne_calls`. GROUP is `sherpa`,
`spectral`, `silero`, `pyannote_style` or `pyannote31`.

```
scripts/real_nepali/run_real.sh DATASET GROUP
scripts/run_community1_nepali.sh DATASET
scripts/real_nepali/score_real.sh          # results/summary_nepali_real.csv, results/corpus_der_nepali_real.txt
.venv/bin/python scripts/real_nepali/report_real.py
```

NepDiar-Sim. DATASET is `nepali_sim` (clean) or `nepali_sim_rn` (reverb + noise).

```
scripts/run_nepali_sim.sh DATASET GROUP     # or scripts/run_nepali_sim_all.sh for every group in three queues
scripts/run_community1_nepali.sh DATASET
scripts/score_nepali_sim.sh                 # results/summary_nepali_sim.csv, results/corpus_der_nepali_sim.txt
.venv/bin/python scripts/nepali_sim_report.py
```

English twin. DATASET is `english_sim` or `english_sim_rn`, GROUP as above plus `community1`.

```
scripts/run_english_sim.sh DATASET GROUP    # or scripts/run_english_sim_all.sh for everything
scripts/score_english_sim.sh                # results/summary_english_sim.csv, results/corpus_der_english_sim.txt
.venv/bin/python scripts/sim_pair_test.py   # results/sim_pair_test.csv, the paired Nepali minus English test
.venv/bin/python scripts/sim_twin_stats.py  # results/sim_twin_stats.csv, how close the two sets are
```

## Layout

System RTTMs are in `results/raw/<dataset>/<system>_rttm/`, with per-recording scores and run times
nearby. The tables are `results/summary*.csv` and `results/corpus_der_*.txt`. `paper/analysis/` has
the small scripts behind single numbers in the paper. S3's embedding caches (`results/cache/`) are
left out, the `extract` step rebuilds them.

## Licences

The code is under the GNU AGPL 3.0 (`LICENSE`). Data files carry their source's licence, listed
with citations in `DATA-LICENSES.txt`:

- AMI references and UEMs: CC BY 4.0 (the BUT repo is Apache 2.0,
  `ref/ami/LICENSE-AMI-diarization-setup`).
- Indic DiarBench references: CC BY 4.0.
- IndicVoices call references (`data/indicvoices_ne_calls/`): CC BY 4.0.
- NepDiar-Sim manifests, references and checksums: CC BY-SA 4.0 (derived from OpenSLR 54).
- English twin manifests, references and checksums: CC BY-SA 4.0 (LibriSpeech is CC BY 4.0, the
  rooms, noise plan and timing are NepDiar-Sim's).
- System outputs and scores follow the licence of the set they were run on.

The audio is not redistributed. Get it from the sources above under their terms.
