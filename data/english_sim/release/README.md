# English twin of NepDiar-Sim

30 simulated conversations of about 5 minutes (2.52 h in all), 10 each with 2, 3 and 4 speakers, in two conditions: clean, and reverb + noise. Each conversation is the twin of the NepDiar-Sim conversation with the same number (`engsim_3spk_04` is the twin of `nepsim_3spk_04`). The twin keeps everything from the Nepali plan except the voices: speaker count, turn order, every pause and overlap draw, the room, the room impulse responses, the noise clip, its start offset and the SNR. The voices are read English from LibriSpeech dev-clean and test-clean.

It is there to test whether a diarization system does worse on Nepali than on English when the recording setup is held fixed. No audio is shipped. You rebuild it from LibriSpeech and DEMAND with the manifests, bit-exact (compare with `wav_sha256/`).

## Licence

This release (manifests, references, room responses and checksums) is CC BY-SA 4.0 like NepDiar-Sim, since it reuses its room responses, noise plan and timing. The code is AGPL-3.0. The sources keep their own licences:

- LibriSpeech (OpenSLR 12), CC BY 4.0. Panayotov, Chen, Povey and Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015.
- DEMAND, CC BY 4.0. Thiemann, Ito and Vincent, Zenodo record 1227121.

## Files

- `conversations.tsv`: one row per conversation: LibriSpeech speakers, their sex (`sexes`, from LibriSpeech `SPEAKERS.TXT`), length, speech, overlap, silence, segment count, per-speaker talk time, and the Nepali twin with its speakers.
- `conversations*.txt`: conversation lists, all and per speaker count.
- `rttm/`, `uem/`: references. One RTTM line per placed segment; the UEM covers the whole conversation.
- `placement.tsv`: the rebuild manifest. One row per segment in turn order: LibriSpeech utterance ID, start and end in that file (ms), start in the conversation (ms), gain (full precision), the Nepali speaker it stands in for, and the replayed gap (ms).
- `manifest_rn.tsv`: room, speaker and mic positions, RT60, noise clip, offset and SNR. Copied from NepDiar-Sim with the English IDs.
- `rirs.npz`: room impulse responses (float32), keyed `<uri>__<speaker>`. They are the NepDiar-Sim responses, rekeyed.
- `librispeech_utterances.tsv`: the energy-trimmed speech regions and gain of every dev-clean and test-clean utterance. `plan` reads it.
- `turn_stats.json`: the AMI dev turn statistics the Nepali plan drew from.
- `make_english_sim.py`, `make_nepali_sim.py`: the builder; the second holds the shared mixing, reverb and noise code.
- `sources.sha256`: SHA-256 of the source archives. `wav_sha256/`: SHA-256 of every rebuilt WAV.

## Rebuild

1. Download `dev-clean.tar.gz` and `test-clean.tar.gz` from https://www.openslr.org/12/ and unpack both into one folder, so it holds `LibriSpeech/dev-clean`, `LibriSpeech/test-clean` and `LibriSpeech/SPEAKERS.TXT`. Download `DKITCHEN_16k.zip`, `DLIVING_16k.zip`, `DWASHING_16k.zip`, `OOFFICE_16k.zip` and `OHALLWAY_16k.zip` from https://zenodo.org/records/1227121 into another folder. Do not unzip those.
2. Install Python 3.12 with numpy, scipy and soundfile.
3. From this folder, run:

```
python make_english_sim.py render_clean --out . --libri /path/to/librispeech_folder
python make_english_sim.py render_rn --out . --libri /path/to/librispeech_folder --demand-zips /path/to/demand_zips
```

4. This writes `wav/` and `wav_rn/` (16 kHz mono PCM16), each with a `SHA256SUMS` file. Compare them with `wav_sha256/`.

`prep` and `plan` remake the manifests. `prep` needs the speaker-check script from the NepDiar-Sim release; `plan` needs the NepDiar-Sim folder next to it.

## How it was made

1. Every LibriSpeech utterance gets the same energy trim and gain as NepDiar-Sim: 25 ms frames every 10 ms, threshold max(peak minus 35 dB, 10th percentile plus 10 dB), gaps under 0.3 s bridged, regions under 0.2 s dropped, 20 ms padding, and a gain to -26 dBFS RMS over the kept speech. That gives 10,716 regions from 5,323 utterances of 80 speakers. The 3 utterances with 0.1% or more clipped samples are left out. They are longer than the Nepali ones: mean 3.05 s against 1.75 s.
2. Speakers. OpenSLR 54 has no sex labels, so the twin cannot match sex speaker by speaker. Each conversation is sex-balanced instead: 1 female + 1 male with 2 speakers, 2 + 2 with 4, and 2 + 1 alternating between conversations with 3. A seeded shuffle sets which Nepali speaker gets which sex. The busiest Nepali speakers pick first. Each takes the least-used English speaker of that sex who has at least as many regions of 2.5 s or less as the Nepali speaker has segments. That gives 45 female and 45 male slots, filled by 70 distinct speakers (34 female, 36 male). 20 speakers appear in two conversations, and no speaker appears twice in one.
3. Segments. Each Nepali segment slot gets a distinct region of its English speaker. The assignment minimises the total length mismatch per speaker (Hungarian method). A speaker who is in two conversations can reuse a region across them, never inside one. The English length differs from the Nepali one by a median of 0.020 s, with p90 0.160 s, p99 0.655 s and a maximum of 1.835 s. 85.3% are within 0.1 s and 93.5% within 0.25 s. Mean segment length is 1.776 s against 1.752 s for Nepali.
4. Placement. The NepDiar-Sim gap and overlap draws are replayed from the same seed. The replay is checked against every Nepali start time and stops if one differs. The draws are applied to the English lengths with the same rule: start = max(last end + gap, this speaker's last end, 0). So the timing differs from the Nepali twin only through the small length differences. Conversations run from 287.8 to 314.0 s.
5. Mixing, reverb and noise are the NepDiar-Sim code unchanged: 10 ms fades, seeded dither, peak scaled to 0.99, the same room responses per speaker slot, the same DEMAND channel 1 clip and offset, and the same SNR, measured against the reverberant speech.

## How close the twin is

From the reference RTTMs, pooled over the 30 conversations, using the turn definitions in `paper/analysis/turn_stats_table.py`: back-to-back segments of one speaker form a turn, and speaker changes are counted per minute of speech (not of audio). Reverb + noise uses the same references, so its numbers are the same.

| | speech % | overlap % | speaker changes per speech min | median segment (s) | segments under 1 s | median turn (s) |
|---|---|---|---|---|---|---|
| NepDiar-Sim | 66.5 | 20.3 | 28.0 | 1.69 | 10.1% | 2.03 |
| English twin | 66.8 | 20.4 | 27.6 | 1.72 | 10.5% | 2.07 |

`results/sim_twin_stats.csv` in the repository also has per-conversation means, with changes and segments per minute of audio.

## Datasheet

- Version 1.0, 2026-10-16. Report errors as GitHub issues on the NepDiar repository; fixes get a new version number and a changelog line here.
- Source: LibriSpeech dev-clean and test-clean (40 speakers each, 80 in all), CC BY 4.0. No other English speech is used.
- Speaker sex is set by design (see How it was made, step 2). In 3-speaker conversations, odd-numbered ones (`_01`, `_03`, ...) get 2 female + 1 male and even ones 1 female + 2 male.
- 20 speakers appear in two conversations, so twin conversations are not fully independent. The repository's pair test resamples them in linked groups as a sensitivity check.
- The WeSpeaker speaker check used to clean OpenSLR 54 was not applied to the LibriSpeech speakers. LibriSpeech speaker labels are taken as given; only clipping (0.1% or more clipped samples, 3 utterances) and the energy trim filter anything.

## Limits

- Read speech in both languages, cut into short regions and shuffled. It is not conversation, and the turn-taking is not either language's.
- Sex is balanced per conversation in English but unknown in Nepali, so the two sides may differ in sex mix.
- LibriSpeech is audiobook speech read by LibriVox volunteers on their own equipment, while OpenSLR 54 is crowd-sourced read Nepali collected by Google. The recording channel is not held fixed, only the simulated room and noise are.
