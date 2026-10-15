# Analysis plan for NepDiar-Sim and its English twin

Committed here before any NepDiar-Sim or English twin DER was computed. At commit time no
`results/summary_nepali_sim*.csv` or `results/sim_pair_test.csv` existed, no simulated-set RTTM
had been scored, and no English twin system output existed at all. The plan itself was written
on 2026-10-16 at 01:18 (+0545); the amendment below was added later the same day, still before
any of those scores.

## Question

Does the language change DER when the recording setup is held fixed? NepDiar-Sim
(`data/nepali_sim/`) and its English twin (`data/english_sim/`, LibriSpeech dev-clean and
test-clean voices) share the same 30 conversation plans: speaker counts, turn order, gap and
overlap draws, rooms, room responses, noise clips and SNRs. The voices differ, and with them the
speakers, their recording channel, reading style and the sex mix of each conversation, so a
difference is not a pure language effect.

## Configurations

Six: S1 sherpa-onnx, S2a spectral + WebRTC VAD, S2b spectral + Silero VAD, S3 pyannote-style
ONNX, S4 pyannote 3.1, S5 pyannote community-1. Each runs with its final AMI settings and an
estimated speaker count. Oracle-count runs are descriptive only.

## Measure

Per-conversation DER with no collar, scored by `scripts/score.py` (pyannote.metrics 4.1, the UEM
covers the whole conversation).

## Tests

- Primary family (clean condition): for each configuration, d_i = DER(Nepali conversation i)
  minus DER(English twin i) over the 30 pairs. Report the mean of d and a 95% paired bootstrap
  percentile CI (resample the 30 pairs with replacement, 10,000 resamples, numpy
  `default_rng(0)`). Raw p is two-sided from the same bootstrap: 2 x min(share of resampled means
  <= 0, share >= 0), capped at 1. Holm-adjust the raw p across the 6 configurations.
- Secondary family (reverb + noise condition): the same test, with Holm across its own 6.
- Everything else (pooled DER, collar 0.25, JER, per speaker count, oracle runs, AMI comparison)
  is descriptive and gets no test.
- Output: `results/sim_pair_test.csv` with system, condition, mean diff, CI, raw p, Holm p.

## Amendment (2026-10-16, before any simulated-set score)

These do not change the tests above. They decide how the tests are read and add two checks.

- Smallest difference that matters: 2.0 DER points. A configuration whose 95% CI for the mean
  of d lies wholly inside (-2.0, +2.0) is read as "no practical difference". A CI that crosses
  either bound is read as inconclusive, even when the Holm p is above 0.05. A CI wholly outside
  is a practical difference. 2.0 points is larger than the 1.5-point gap between the two best
  systems on AMI test, so a smaller shift would not change which system a user picks.
- Sensitivity 1, shared English speakers: 20 LibriSpeech speakers appear in two twin
  conversations, so the 30 pairs are not fully independent. Group conversations that are
  linked through a shared English speaker (connected components), resample whole groups with
  replacement (10,000 resamples, `default_rng(0)`), and report the CI for the mean of d.
- Sensitivity 2, small-sample check: a paired sign-flip permutation test on the 30 values of d
  (10,000 random sign vectors, `default_rng(1)`, two-sided on the mean, p counted with the
  observed statistic included).
- Both sensitivity results are reported next to the primary ones. If they disagree with the
  primary reading, the paper says so.
