# IndicVoices Nepali two-party calls: diarization references

Two-speaker reference RTTMs and UEMs for 92 Nepali phone conversations from IndicVoices, built
for the NepDiar evaluation.

## What is here

- `manifest.csv`: one row per call. `call_id` is the IndicVoices file id without the side suffix.
  `side_A_file` and `side_B_file` are the two per-side source files (`<id>_0` and `<id>_1`).
  `source_archive` names the IndicVoices archive the call came from. Duration, speech and overlap
  seconds are also listed.
- `rttm/<call_id>.rttm`: speaker A is side `_0`, speaker B is side `_1`. Each side's segments are
  the verbatim transcription segments, with a side's own overlapping segments merged.
- `uem/<call_id>.uem`: the full call.
- `rttm_tight/<call_id>.rttm`: a tighter second reference, the same segments cut to the frames an energy
  detector on that side's own channel calls speech. Rebuilt by `scripts/real_nepali/sync_tight_iv.py`.

There are no speaker ids, no gender or other speaker metadata, and no audio.

## Numbers

- 92 calls, 5.068 h of audio. Calls run from 58 s to 393 s, with a median of 193 s.
- 184 sides with 184 distinct speakers: no speaker appears in two calls.
- 4.309 h of reference speech. Overlap is 28.6 min, which is 11.05% of speech (median per call 9.3%).
- 91 calls come from `v1_Nepali_train.tgz` and 1 call from `v2_Nepali_valid.tgz`.
- All 184 speakers are from West Bengal, India: 143 sides from Kalimpong and 41 from Darjeeling;
  182 sides are rural. This is Indian Nepali speech, not speech recorded in Nepal.

## How the calls were picked

1. Take every Conversation file in `v1_Nepali_train.tgz`, plus one validation call.
2. Keep a call only when both sides exist. This gives 201 calls.
3. Drop the one call whose side carried two speaker ids.
4. Go through the calls in id order, with the validation call first. Keep a call only when neither
   of its speakers is in a call already kept.

To rebuild the audio, download the named archive from
`https://iv-release.objectstore.e2enetworks.net/dmu_release/<archive>` and take both side wavs, which
are 8 kHz. Start both sides at t=0; the two sides of every call have the same sample length. Sum
them and resample to 16 kHz.

## Known limits of the reference

- Pauses inside segments. The boundaries come from transcription segmentation, not from a
  diarization annotation, so a segment includes pauses inside an utterance. An energy detector on
  each side's own channel calls 17.5% of reference speech time silence (threshold -50 dBFS, 25 ms
  frames). This pads the overlap. If each side's segments are cut down to the frames its own
  channel's energy detector marks as speech, speech falls by 11% and overlap by 38%. Overlap then
  becomes 7.7% of speech. That tighter version is in `rttm_tight/`.
- Sync between the sides. The IndicVoices paper says each role-play call joined two people
  on one telephone bridge and recorded the call. Both sides of every call have exactly the same
  length, which also points to one recording of the call, so offset 0 is the most likely value.
  The sync can't be checked from the audio: there is no crosstalk between the channels, and an
  overlap-based lag search is too noisy (it is just as noisy on recordings known to be in sync).
  Four calls behave differently from the in-sync recordings and are less certain:
  `0567e43d-0fc2-4a77-be64-ed73f905fa99`, `1fca12e8-6d74-4d28-b8da-e1ae4a050bef`,
  `3a87e195-ad44-44d1-813f-a803c244473d` and `4af13f83-fe70-40c2-a6d8-592f76031ca2`.
- The audio is 8 kHz telephone speech.

## Source and licence

The calls come from IndicVoices (AI4Bharat), https://huggingface.co/datasets/ai4bharat/IndicVoices.
IndicVoices is released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). These
references are changed from the original (merged segments, anonymous labels) and are shared under
the same licence.

Javed et al., "IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian
Languages", Findings of ACL 2024, https://arxiv.org/abs/2403.01926

## Errata

Report reference errors as GitHub issues on this repository.
