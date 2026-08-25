# AMI corpus statistics — `/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio`

**Total utterances scanned:** 266121
**Filenames matching expected pattern:** 100.0%
**Missing .wav:** 1  |  **Missing .txt:** 1
**Filename/folder split mismatches:** 0  |  **speaker-id/folder mismatches:** 0  |  **no-segment (whole-file) entries:** 0

## Zero-duration segments (begin == end in filename)
- None found.

## Duration statistics (derived from filename begin/end)
- Total: 189.08 hours across 266120 utterances
- Mean/median/stdev (s): 2.558 / 1.61 / 2.615
- Min/max (s): 0.02 / 33.18
- Hours by split: {'dev': 17.88, 'eval': 17.36, 'train': 153.83}
- Hours IHM vs SDM: {'ihm': 94.54, 'sdm': 94.54}
- Duration buckets: {'<1s (likely backchannel)': 96766, '3-10s': 77936, '1-3s': 86528, '>10s': 4890}

## Audio header verification (random sample)
- Files opened: 2000
- Sample rates found: {16000: 2000}
- Channel counts found: {1: 2000}
- Mean |filename-duration - actual-duration|: 0.0 s (max 0.0001 s) — near 0 confirms the centisecond assumption

## Speaker statistics
- Unique speaker IDs: 190 (paper reports 188 for full AMI)
- Utterance count per speaker: see files.csv / xlsx 'Per-Speaker' sheet for the full table (min 114, max 4744)

## Meeting (session) statistics
- Unique meetings: 169
- Utterance count per meeting: min 226, max 4838 (full per-meeting table in files.csv / xlsx)
- Meeting-ID prefix distribution: {'ES': 73092, 'IB': 12830, 'IS': 41254, 'TS': 71644, 'EN': 43504, 'IN': 23796}

## Mic-code token in filename, by folder (descriptive — not an error)
Folder location (ihm/ vs sdm/) is the confirmed-reliable indicator of actual mic type — verified by direct listening on paired ihm/sdm files. The token embedded in the filename (sdm, h00, h01, h02, h03, ...) does not always say 'sdm' for files in the sdm/ folder; this is a naming-convention quirk (most likely: which reference channel supplied the segmentation boundaries), not a data-placement error. Counts below are purely descriptive.
- ihm/: {'h00': 38153, 'h01': 33844, 'h03': 30769, 'h02': 29229, 'h04': 1065}
- sdm/: {'sdm': 38153, 'h01': 33844, 'h03': 30769, 'h02': 29229, 'h04': 1065}

## Transcript statistics
- Total words: 1966908
- Mean/median words per utterance: 7.39 / 4.0

## Split integrity (speaker overlap between train/dev/eval)
- **dev_vs_train: 2 speaker(s) overlap** -> ['fie038', 'mio036']

## Enrollment/test feasibility (paper protocol: 6 enroll utts >=~5s, test utts ~2.5s)
- Speaker x split x mic conditions with >= 6 enrollment-length utterances: 384 / 384
- Eligible by split: {'dev': 42, 'eval': 32, 'train': 310}
- Total test-length utterance pool (eligible conditions only): 46934
