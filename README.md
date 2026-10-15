# NepDiar

Code, references and system outputs for a speaker diarization benchmark. Five open-source systems
that run on a laptop are scored on the 16-meeting AMI test set. Then the same systems, with the
settings picked on AMI dev and nothing changed, are scored on Nepali:

- **Indic DiarBench, Nepali**: 25 clips of online-meeting conversation, 3 to 6 speakers.
- **IndicVoices, Nepali phone calls**: 36 two-party calls.
- **NepDiar-Sim**: 30 simulated Nepali conversations (2, 3 and 4 speakers) built from OpenSLR 54
  read speech, in a clean and a reverb + noise condition. It is rebuilt from the manifests in
  `data/nepali_sim/`, see below.

Scoring is DER and JER from `pyannote.metrics`, with no collar and with a 0.25 s collar, overlap
scored, inside each set's UEM. We report pooled DER and per-recording means with 95% bootstrap
intervals.

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

Python 3.12, [uv](https://docs.astral.sh/uv/), and ffmpeg 4 on your PATH (or point `FFMPEG` at
the binary; `ffprobe` must sit next to it). Results were made on an Apple M3 Pro.

```
uv venv .venv --python 3.12            && uv pip install --python .venv -r requirements.txt
uv venv .venv-pyannote3 --python 3.12  && uv pip install --python .venv-pyannote3 -r requirements-pyannote3.txt
uv venv .venv-pyannote --python 3.12   && uv pip install --python .venv-pyannote -r requirements-pyannote.txt
uv venv .venv-sim --python 3.12        && uv pip install --python .venv-sim -r requirements-sim.txt
scripts/get_models.sh
```

- `.venv`: S1 to S3, scoring, summaries, data prep and the NepDiar-Sim renderer.
- `.venv-pyannote3`: pyannote.audio 3.3.2 for S4.
- `.venv-pyannote`: pyannote.audio 4.0.7 for S5.
- `.venv-sim`: pyroomacoustics, only for the `rooms` stage of NepDiar-Sim.

`scripts/get_models.sh` downloads the segmentation, embedding and Silero VAD ONNX files from the
sherpa-onnx releases, checks their sha256, and writes the two derived copies (the CMN-flagged
embedding model for S1, and the frame-output model plus numpy head for S3).

S4 and S5 are gated on Hugging Face. Accept the terms on the model pages for
`pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0` and
`pyannote/speaker-diarization-community-1`, then either set `HF_TOKEN` or run
`huggingface-cli login`. The runner reads the token from `--hf-token`, `HF_TOKEN` or the cached
login, and never prints it.

## Data

No audio is in this repo. Everything goes under `data/`, which the scripts expect.

**AMI**, Mix-Headset, the BUT test (16) and dev (18) splits. References, UEMs and meeting lists
are in `ref/ami/`, from github.com/BUTSpeechFIT/AMI-diarization-setup at commit
`2509d8933721023fab4def2618aabd5c28eb82e9` (`only_words` references). Audio comes from the
Edinburgh mirror and is checked against `ref/ami/audio.sha256`:

```
.venv/bin/python scripts/get_ami.py
.venv/bin/python scripts/get_ami.py --meetings-file ref/ami/dev.meetings.txt
```

**Indic DiarBench, Nepali.** Download `Nepali/test-00000-of-00001.parquet` from
https://huggingface.co/datasets/sarvamai/indic-diarbench and save it as
`data/nepali_real/diarbench/Nepali_test-00000-of-00001.parquet`, then convert it. The references this writes are already in the repo, so you can check them with
`git diff`.

```
uv run --no-project --with pyarrow --with numpy --with soundfile python -I scripts/real_nepali/prep_diarbench.py
```

**IndicVoices, Nepali.** The archives are on the IndicVoices release server (see
https://huggingface.co/datasets/ai4bharat/IndicVoices). Only the "Conversation" files are kept,
streamed without saving the archive. The 36 calls used are in `data/nepali_real/indicvoices/calls.txt`
(35 from v1 train, 1 from v2 valid). The references are not shipped; the prep script rebuilds them.

```
U=https://iv-release.objectstore.e2enetworks.net/dmu_release
.venv/bin/python scripts/real_nepali/stream_indicvoices.py $U/v1_Nepali_train.tgz data/nepali_real/indicvoices/raw_v1_train
.venv/bin/python scripts/real_nepali/stream_indicvoices.py $U/v2_Nepali_valid.tgz data/nepali_real/indicvoices/raw_v2_valid
.venv/bin/python -I scripts/real_nepali/prep_indicvoices.py data/nepali_real/indicvoices/raw_v1_train data/nepali_real/indicvoices/raw_v2_valid
```

The v1 train archive is 23.7 GB.

## Rebuild NepDiar-Sim

`data/nepali_sim/` holds the reference RTTMs and UEMs, `placement.tsv` (every speech segment: source
file, start, end, placement, gain), `manifest_rn.tsv` (room, positions, noise, SNR per
conversation), `turn_stats.json`, the speaker check output, `sources.sha256` for the source zips, and
sha256 lists for the rebuilt audio. `data/nepali_sim/release/` is the same set packaged on its own,
with its own README.

1. Put `asr_nepali_0.zip` to `asr_nepali_3.zip` from https://www.openslr.org/54/ in
   `data/slr54/zips/`, and `DKITCHEN_16k.zip`, `DLIVING_16k.zip`, `DWASHING_16k.zip`,
   `OOFFICE_16k.zip`, `OHALLWAY_16k.zip` from https://zenodo.org/records/1227121 in
   `data/demand/zips/`. Do not unzip them. Check all nine from the repo root with
   `shasum -a 256 -c data/nepali_sim/sources.sha256`.
2. Make the room impulse responses. They are not shipped; this stage remakes `rirs.npz` (and an
   identical `manifest_rn.tsv`) from fixed seeds:
   `.venv-sim/bin/python scripts/make_nepali_sim.py rooms`
3. Render both conditions:
   ```
   .venv/bin/python scripts/make_nepali_sim.py render_clean
   .venv/bin/python scripts/make_nepali_sim.py render_rn
   ```
4. Compare `data/nepali_sim/wav/SHA256SUMS` with `data/nepali_sim/wav.SHA256SUMS`, and
   `wav_rn/SHA256SUMS` with `wav_rn.SHA256SUMS`.

The `stats`, `plan` and `check_nepali_speakers.py` stages remake the manifests from scratch and are
only needed to change the design.

## Rerun and score

Every runner skips a recording whose RTTM already exists, so delete the matching
`results/raw/.../*_rttm/` folder first if you want a fresh run. Outputs go to `results/raw/`.

**Settings picked on AMI dev** (dev table):

```
.venv/bin/python scripts/vad_sweep_dev.py            # S2a WebRTC VAD
.venv/bin/python scripts/vad_sweep_dev.py --silero   # S2b Silero VAD
M=ref/ami/dev.meetings.txt O=results/raw/ami_dev_full DS=ami_dev scripts/sweep_short_dev.sh cmn "1.0 1.1 1.2"   # S1 threshold
.venv/bin/python scripts/run_pyannote_style.py extract --wav-dir data/ami/audio --meetings-file ref/ami/dev.meetings.txt \
  --cache-dir results/cache/pyannote_style --timing-csv results/raw/ami_dev/pyannote_style_extract_timing.csv
.venv/bin/python scripts/sweep_pyannote_style_dev.py  # S3 clustering
```

**AMI test** (pooled, per-meeting and components tables):

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

**Speed** (one thread, idle machine, 3 AMI test meetings): `scripts/run_timing_idle.sh` and
`scripts/run_timing_community1.sh`. Each checks its RTTMs against the test run.

**Real Nepali** (DATASET is `indic_diarbench_ne` or `indicvoices_ne_calls`, GROUP is `sherpa`,
`spectral`, `silero`, `pyannote_style` or `pyannote31`):

```
scripts/real_nepali/run_real.sh DATASET GROUP
scripts/run_community1_nepali.sh DATASET
scripts/real_nepali/score_real.sh          # results/summary_nepali_real.csv, results/corpus_der_nepali_real.txt
.venv/bin/python scripts/real_nepali/report_real.py
```

**NepDiar-Sim** (DATASET is `nepali_sim` for clean or `nepali_sim_rn` for reverb + noise):

```
scripts/run_nepali_sim.sh DATASET GROUP     # or scripts/run_nepali_sim_all.sh for every group in three queues
scripts/run_community1_nepali.sh DATASET
scripts/score_nepali_sim.sh                 # results/summary_nepali_sim.csv, results/corpus_der_nepali_sim.txt
.venv/bin/python scripts/nepali_sim_report.py
```

## Layout

- `scripts/`: every script the results depend on. `scripts/real_nepali/` covers the two real sets.
- `ref/ami/`: AMI references, UEMs, split lists and audio checksums.
- `data/nepali_sim/`, `data/nepali_real/diarbench/`: references and manifests we can share.
  `data/nepali_real/indicvoices/` has only the call list.
- `results/raw/<dataset>/<system>_rttm/`: system output RTTMs. `results/raw/**/scores_*.csv`:
  per-recording scores. `results/raw/**/*_timing.csv`: run times. `results/summary*.csv`,
  `results/corpus_der_*.txt`: the tables. `results/raw/ami_dev*`: dev tuning runs and sweeps.
- `paper/make_tables.py`: builds the LaTeX tables from `results/`.

Embedding caches from S3 (`results/cache/`) are left out, they are large and rebuilt by the
`extract` step.

## Data and licences

The code is under the GNU AGPL 3.0, see `LICENSE`. The data files in this repo carry their source's
licence, listed with citations in `DATA-LICENSES.txt`:

- AMI references and UEMs: AMI annotations are CC BY 4.0. The BUT AMI-diarization-setup repo they
  come from is Apache 2.0, its licence is kept in `ref/ami/LICENSE-AMI-diarization-setup`.
- Indic DiarBench references: CC BY 4.0.
- NepDiar-Sim manifests, references and checksums: derived from OpenSLR 54, so CC BY-SA 4.0.
- System output RTTMs and scores on each set follow that set's licence.

The audio itself is not redistributed. Get it from the official sources above under their terms.
