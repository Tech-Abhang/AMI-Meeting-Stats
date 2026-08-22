# AMI corpus statistics — `audio`

**Total utterances scanned:** 266120
**Filenames matching expected pattern:** 14.3%
**Missing .wav:** 0  |  **Missing .txt:** 0
**Filename/folder split mismatches:** 0  |  **mic mismatches:** 0  |  **speaker-id/folder mismatches:** 0

## Duration statistics (derived from filename begin/end)
- Total: 28.13 hours across 38153 utterances
- Mean/median/stdev (s): 2.654 / 1.7 / 2.677
- Min/max (s): 0.02 / 30.08
- Hours by split: {'dev': 2.67, 'eval': 2.21, 'train': 23.25}
- Hours IHM vs SDM: {'sdm': 28.13}
- Duration buckets: {'<1s (likely backchannel)': 13172, '3-10s': 11586, '1-3s': 12599, '>10s': 796}

## Audio header verification (random sample)
- Files opened: 2000
- Sample rates found: {16000: 2000}
- Channel counts found: {1: 2000}
- Mean |filename-duration - actual-duration|: 0.0 s (max 0.0001 s) — near 0 confirms the centisecond assumption

## Speaker / meeting statistics
- Unique speaker IDs: 55 (paper reports 188 for full AMI)
- Unique meetings: 168
- Meeting-ID prefix distribution: {'ES': 10933, 'IB': 1896, 'IS': 6218, 'TS': 11748, 'EN': 4318, 'IN': 3040}
- Gender split (guessed from speaker_id first letter — verify against AMI speaker table): {'f': 9645, 'm': 28508}

## Transcript statistics
- Total words: 1966908
- Mean/median words per utterance: 7.39 / 4.0

## Split integrity (speaker overlap between train/dev/eval)
- **dev_vs_train: 2 speaker(s) overlap** -> ['fie038', 'mio036']

## Enrollment/test feasibility (paper protocol: 6 enroll utts >=~5s, test utts ~2.5s)
- Speaker x split x mic conditions with >= 6 enrollment-length utterances: 56 / 56
- Eligible by split: {'dev': 6, 'eval': 5, 'train': 45}
- Total test-length utterance pool (eligible conditions only): 6865
