# IndicVoices Nepali two-party calls: diarization references

Two-speaker reference RTTMs and UEMs for Nepali phone conversations from IndicVoices, built for
the NepDiar evaluation. No audio, no speaker ids, no gender or other speaker metadata.

- `manifest.csv`: one row per call. `call_id` is the IndicVoices file id without the side suffix;
  `side_A_file` and `side_B_file` are the two per-side source files (`<id>_0`, `<id>_1`);
  `source_archive` names the IndicVoices release archive it came from.
- `rttm/<call_id>.rttm`: speaker A is side `_0`, speaker B is side `_1`. Each side's verbatim
  transcription segments, with a side's own overlapping segments merged.
- `uem/<call_id>.uem`: the full call.

To rebuild the audio: download the named archive from
`https://iv-release.objectstore.e2enetworks.net/dmu_release/<archive>`, take both side wavs (8 kHz),
start both at t=0, pad the shorter at the end, sum, and resample to 16 kHz.

Caveats: segment boundaries come from transcription segmentation, not from a diarization
annotation, so they include short pauses inside an utterance. The audio is 8 kHz telephone speech.

## Source and licence

Derived from IndicVoices (AI4Bharat), https://huggingface.co/datasets/ai4bharat/IndicVoices,
released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). These references are a
modified derivative (merged segments, anonymous labels) and are shared under the same licence.

Javed et al., "IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian
Languages", ACL Findings 2024, https://arxiv.org/abs/2403.01926
