# NepDiar-Sim: simulated Nepali conversations for diarization testing

30 simulated conversations of about 5 minutes each (2.50 h in total). There are 10 conversations each with 2, 3 and 4 speakers, and 90 speakers in all. Every speaker appears in exactly one conversation. The set comes in two conditions with the same timing and the same reference:

- **clean**: dry read speech, level balanced, with no noise and no reverb.
- **reverb_noise**: each speaker is convolved with a room impulse response from a random shoebox room. The measured RT60 is 0.23 to 0.67 s, with a median of 0.44 s. DEMAND indoor noise is then added at an SNR of 5.5 to 19.8 dB.

No audio is shipped here. You rebuild the audio from OpenSLR 54 and DEMAND using the manifests, and the rebuild is bit-exact (see `wav_sha256/`).

## Sources and licences

| What | Licence | Used for |
|---|---|---|
| OpenSLR 54, "Large Nepali ASR training data set" (Google, Kjartansson et al., SLTU 2018, doi 10.21437/SLTU.2018-11), parts `asr_nepali_0` to `asr_nepali_3` | CC BY-SA 4.0 | All speech. The rebuilt audio is a derivative, so it is also CC BY-SA 4.0. |
| DEMAND (Thiemann, Ito, Vincent, zenodo 1227121): DKITCHEN, DLIVING, DWASHING, OOFFICE, OHALLWAY, 16 kHz, channel 1 | CC BY 4.0 | Noise in reverb_noise |
| AMI Meeting Corpus, dev set "only_words" reference RTTMs | CC BY 4.0 | Pause and overlap length statistics only (`turn_stats.json`) |
| Landini et al., Interspeech 2022, doi 10.21437/Interspeech.2022-10451; code at github.com/BUTSpeechFIT/EEND_dataprep | Apache 2.0 | Method. It is reimplemented here, not copied. |

The files in this folder (manifests, RTTMs, UEMs, room responses and scripts) follow OpenSLR 54 and are released under CC BY-SA 4.0.

## Files

- `make_nepali_sim.py`: the generator. All seeds are fixed (base seed 2083).
- `check_nepali_speakers.py`: the speaker-ID check that chose the speakers. Its output is in `speaker_check/`.
- `conversations.tsv`: one row per conversation, with its speakers, length, speech, overlap and per-speaker time. The `conversations*.txt` files list the IDs, all of them or split by speaker count.
- `placement.tsv`: the rebuild manifest. Each row is one speech segment, giving the OpenSLR 54 file ID, the start and end in that file (ms), where it goes in the conversation (ms), and the gain applied (full precision).
- `manifest_rn.tsv`: one room per conversation. It gives the room size, target, design and measured RT60, the mic and speaker positions, the noise file and its offset, and the SNR.
- `rirs.npz`: the room impulse responses as float32, keyed `<uri>__<speaker>`. With these you do not need pyroomacoustics to rebuild.
- `rttm/`, `uem/`: the reference, which is exact because it comes from the placement. One UEM covers each whole conversation.
- `turn_stats.json`: the AMI dev pause and overlap lengths the simulator drew from.
- `wav_sha256/`: sha256 of every rebuilt wav for both conditions.
- `sources.sha256`: sha256 of the source zips we used.

## Rebuild

1. Download `asr_nepali_0.zip` to `asr_nepali_3.zip` from https://www.openslr.org/54/ into one folder. Download `DKITCHEN_16k.zip`, `DLIVING_16k.zip`, `DWASHING_16k.zip`, `OOFFICE_16k.zip` and `OHALLWAY_16k.zip` from https://zenodo.org/records/1227121 into another. Do not unzip them.
2. Install Python 3.12 with numpy, scipy and soundfile.
3. From this folder, run:

```
python make_nepali_sim.py render_clean --out . --slr-zips /path/to/slr54_zips
python make_nepali_sim.py render_rn --out . --slr-zips /path/to/slr54_zips --demand-zips /path/to/demand_zips
```

4. This writes `wav/` and `wav_rn/` (16 kHz mono PCM16), each with a `SHA256SUMS` file. Compare them with `wav_sha256/`.

The `stats`, `plan` and `rooms` stages remake the manifests from scratch. They need the AMI references, the speaker-check output and pyroomacoustics 0.10.1.

## How it was made

1. **Speaker check.** 461 OpenSLR 54 speaker IDs had at least 50 files in the four parts. Each utterance got an energy-based trim and a WeSpeaker ResNet34 embedding.
   - 14 IDs were dropped as doubtful. 5 had more than 10% of their utterances closer to another speaker. 12 split into two distinct voice groups (some IDs failed both checks). No ID looked like a duplicate of another.
   - 262 more IDs had under 150 s of usable speech, and 185 IDs remained. 90 of those were drawn at random.
2. **Simulation.** This follows Landini et al. (2022). Each speaker's trimmed speech regions are shuffled into one turn order.
   - A turn after the same speaker gets a pause drawn from AMI same-speaker pauses.
   - A speaker change is a pause with probability 0.389 and an overlap otherwise, each drawn from AMI. Values of 7 s or more are dropped, as in their code.
   - The per-speaker speech budget is tuned so each conversation lasts about 300 s.
   - Every segment is brought to -26 dBFS speech RMS and gets a 10 ms fade.
3. **Reverb and noise.**
   - Rooms are 3 to 8 by 3 to 6 by 2.5 to 3.5 m. A phone-height mic is at least 1 m from the walls, and the speakers sit 0.5 to 2.5 m from it.
   - The image method is calibrated so the measured RT60 matches the drawn target.
   - SNR is measured against the reverberant speech over the reference speech frames.

## Statistics

| | 2 speakers | 3 speakers | 4 speakers | all |
|---|---|---|---|---|
| overlap / speech | 0.137 | 0.216 | 0.249 | 0.203 |
| silence / total | 0.378 | 0.314 | 0.312 | 0.335 |
| segments per conversation | 123 | 143 | 152 | 139 |

There are 4,174 segments from 3,700 utterance files, with a mean length of 1.75 s. Speaker time is close to equal: the smallest speaker share is 0.495 in 2-speaker conversations, 0.327 in 3-speaker and 0.239 in 4-speaker.

## Limits

- This is read speech, not conversation. There are no backchannels, laughter or turn-taking cues.
- The pause and overlap lengths come from English meetings.
- The rooms are simulated, and the noise is a single DEMAND channel.
- OpenSLR 54 has no gender labels, so the gender mix is unknown.
