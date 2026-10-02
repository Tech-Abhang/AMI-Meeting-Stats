# AMI Speaker-Verification Trial Dataset

Trial-pair dataset built from the AMI Meeting Corpus for a speaker-verification research paper (IIIT Dharwad, P-ECAPA model). This README is the full project recap — corpus prep through final data, in the order it actually happened, including the dead ends.

Everything below runs against data at:

```
/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/
```

**Standing rule for every script in this project:** nothing here ever edits files on your machine directly. Every deliverable is a script you run yourself; you paste the terminal output back so the next step can be built on real results, not assumptions.

---

## 1. Goal

Build genuine (target) and impostor (non-target) trial pairs from AMI for speaker verification evaluation, with:

- A clean, reproducible eval-speaker split (no speaker seen in training leaking into eval).
- An enroll/test architecture that splits by **meeting**, not just by file, so enroll and test audio for a speaker never come from the same recording session.
- A negative (impostor) set that is distribution-matched to the positives on duration and mic-combo, so a verification system can't cheat by learning "short recordings are always genuine" or "ihm-ihm pairs are always genuine."

## 2. Data source

AMI Meeting Corpus, two mic conditions used throughout:

- **ihm** — individual headset mic (one per speaker, cleaner signal)
- **sdm** — single distant mic (one shared mic for the room, noisier)

Enroll audio is always ihm. Test audio can be ihm or sdm (that's what makes "ihm-ihm" vs "ihm-sdm" trial sub-types).

## 3. Pipeline, in build order

### 3.1 Corpus prep (one-time, raw corpus → usable layout)

`code/corpus_prep/`: `organize_ami.py`, `rename_ami.py`, `standardize_ami.py`, `get_sdm_audio.py`, `ami-ihm-kaldi-chunked.py`, `resolve_mismatch.py`, `ami_stats.py`.

One-time setup that laid out the raw corpus into the folder structure everything downstream assumes (`audio/ihm/...`, `audio/sdm/...`), chunked long recordings, and reconciled filename mismatches between metadata and audio files. Not re-run in this phase of the project; mentioned here for completeness since the README should cover start to end.

### 3.2 Speaker split

`ami_split_demographics.py` → `split_demographics.json`

- 190 total AMI speakers.
- 100 held out as the **eval set**: 70 male + 30 female, `seed=42`.
- The remaining speakers are available for training; eval speakers are never used there.

### 3.3 Meeting-split enroll/test architecture

`ami_eval_enroll_test_xlsx.py` → `eval_enroll_test2.xlsx` (sheets: `Enroll`, `Test`, `Meeting_Split_Detail`, `Dropped_Speakers`)

For each of the 100 eval speakers with **≥ 2 meetings**:

1. Sort that speaker's meetings.
2. First `floor(n/2)` meetings → **Enroll** (ihm only).
3. Remaining meetings → **Test** (ihm + sdm).

Speakers with only 1 meeting are dropped (can't split enroll from test without reusing a meeting). This leaves **91 usable eval speakers**.

This replaced an earlier v3 scheme (`eval_split2.py`, now deleted) that put all 100 speakers in both Enroll and Test with deliberate row overlap and no meeting-difference guarantee — the meeting-split redesign is what makes the trial rules in 3.4 meaningful.

### 3.4 Positive (genuine) trials

`create_positive_trials.py` → `trial_outputs/positive_trials.csv`
`encode_positive_trials.py` → `encoded_trials/positive_trials.parquet`, `encoded_trials/path_lookup.csv`

Rules for a positive trial: same speaker, enroll file × test file, where:

- `meeting_id` differs between the enroll file and the test file (enforced by construction — enroll and test meetings never overlap per speaker).

`path_lookup.csv` is a dictionary of all **104,748 unique audio file paths** referenced anywhere in the project (both positive and negative trials), each given an integer id. `encode_positive_trials.py` rewrites the positive table against this dictionary, storing `enroll_id` / `test_id` (int32) instead of path strings — smaller and faster to join than keeping paths.

Result: **27,301,776 positive trial rows.**

### 3.5 Duration distribution + mic-combo ratio of the positives

`positive_trial_full_distribution_stats.py` → `positive_trial_full_distribution_stats.json`
`plot_distribution_bins.py` → `distribution_bins_equal_width.png`, `distribution_bins_quantile.png`
`compute_positive_mic_ratio.py` → `positive_mic_ratio_stats.json`

- Splits the full `duration_diff` range into **50 equal-width bins** and counts positives per bin.
- Separately counts, **inside each bin**, how many positives are ihm-ihm vs ihm-sdm (test file's mic decides this, since enroll is always ihm). Cross-checked against the bin totals above — must match exactly, or the script aborts rather than writing a silently wrong ratio file.

### 3.6 Negative (impostor) trials — v1 → v2

`create_negative_trials.py` (**deleted**, superseded) → `create_negative_trials_v2.py` → `trial_outputs/negative_trials_v2.joblib`, `negative_trials_v2_report.json`

Candidate negative pairs: different speaker, enroll file (A) × test file (B), same gender, `meeting_id` of A's enroll file ≠ `meeting_id` of B's test file. The full candidate pool is ~1.22 billion pairs — too large to materialize, so each candidate is streamed once and routed into a **reservoir sample** (Algorithm R) sized for its `(duration_diff bin, mic-combo)` slot, never fully stored.

v1 matched only the overall per-bin count (10× the positive count in that bin, `ratio=10`). v2 (current) **additionally** matches the ihm-ihm vs ihm-sdm split within each bin, using the ratios measured in 3.5 — so a bin's negative ihm-ihm:ihm-sdm ratio mirrors the real positive ratio for that bin, not just whatever the reservoir happened to catch first.

Result: **273,017,760 negative trial rows** — exactly 10× the positive count, **zero shortfall** in any bin. Took ~22 minutes to generate, 1.58 GB output (joblib, `compress=3`).

Negative trials are stored as `enroll_path` / `test_path` — pandas `category` dtype path strings — **not** the integer ids the positive table uses. Both encodings point at the same `path_lookup.csv` dictionary, just represented two different ways. This mismatch mattered later (see §5).

### 3.7 Verifying the match held

`plot_pos_neg_mic_comparison.py` → `pos_neg_mic_comparison.png`

Two panels, both read straight from `negative_trials_v2_report.json`:

- **Top:** per-bin counts, 4 bars (positive ihm-ihm / positive ihm-sdm / negative ihm-ihm / negative ihm-sdm), log scale.
- **Bottom:** per-bin negative:positive ratio, 3 bars (overall / ihm-ihm-only / ihm-sdm-only) against a dashed line at the target ratio (10×). All three sitting on the line confirms the split was held exactly per bin, not just on average.

### 3.8 Confound check #1 — Duration

`duration_eer_kde.py` → `duration_difference_eer.png`

A verification system should not be able to tell genuine from impostor trials **just by looking at duration_diff** — if it could, that would mean the negative set wasn't actually duration-matched, and any downstream EER on a real model would be partly an artifact of that leak rather than real speaker discrimination.

- Score = `-duration_diff` (smaller gap = more "genuine-like").
- EER computed on the **full population** (hundreds of millions of rows, no subsampling — sorting floats is cheap).
- KDE density curve for the plot uses a random subsample (150,000 per side) — a picture doesn't need 300M points to look right.
- Hand-rolled 1D Gaussian KDE (Scott's rule bandwidth), pure numpy, no scipy/sklearn dependency.

**Result: EER = 0.50.** This is the *good* outcome — chance level, meaning duration_diff alone carries no information to separate genuine from impostor. Confirms the bin-matched negative sampling in §3.6 worked.

### 3.9 Side quest — Energy & SNR confound check

Asked: does the same hold for signal energy and noise level, or could a model cheat on *those* instead?

**`compute_audio_features.py`** → `audio_features.csv`

For each of the 104,748 unique files in `path_lookup.csv`:
- `energy_db` = `10 * log10(mean(x^2))` — RMS power in dB.
- `wada_snr_db` — WADA (Waveform Amplitude Distribution Analysis) blind SNR estimate (Kim & Stern, 2008). The G-statistic `log(mean|x|) − mean(log|x|)` is mapped to dB SNR via a **self-derived Monte Carlo calibration table** (synthetic Gamma(0.4)-speech + Gaussian-noise mixtures across an SNR grid from −20 to +60 dB) — built this way deliberately, rather than copying a literature lookup table from memory, to avoid a transcription error nobody would catch.

Run on real data: parallelized (multiprocessing), resumable, with a `--self-test N` preflight before committing to all 104,748 files. **104,748 files processed in 66.6 minutes (26.2 files/s), 4 files errored.**

Two real bugs hit and fixed along the way — see §5.

**`join_audio_features.py`** → `positive_trials_with_features.parquet`, `negative_trials_v2_with_features.joblib` (new files; originals from §3.4/§3.6 never touched)

Attaches `energy_diff` / `snr_diff` (`abs(test − enroll)`) to every trial row. Had to handle the two different encodings from §3.6 (integer ids on positives, category path strings on negatives) — see §5 for the bug this caused.

- Positive: 27,301,776 rows, 1,064 NaN (from the 4 errored audio files).
- Negative: 273,017,760 rows, 17,430 NaN.

**`feature_eer_kde.py`** (generalized version of `duration_eer_kde.py`) → `energy_difference_eer.png`, `snr_difference_eer.png`

| Feature | EER | Interpretation |
|---|---|---|
| Duration Difference | **0.50** | Chance level — no leak (by design, §3.8) |
| WADA SNR Difference | **0.4987** | Chance level — no leak, good |
| Energy Difference | **0.4067** (≈0.41) | Real, moderate leak — negatives were never energy-matched, only duration/mic-matched |

The energy result is a genuine finding, not a bug: negatives were sampled to match duration and mic-combo, never energy, so there's no reason they'd accidentally match energy too. It's a real limitation of the dataset worth disclosing (or fixing with an energy-matched v3 sampler).

### 3.10 Decision point — professor's call

Plots and numbers above were reviewed by the professor. **Decision: no energy/SNR-matched (v3) negative sampler needed.** The existing path/duration/mic-matched trials (from §3.4 and §3.6, without `energy_diff`/`snr_diff`) are the final dataset for the paper. The energy/SNR side quest (§3.9) stands as a documented confound check, not as a basis for further data engineering.

**Practical effect:** `positive_trials_with_features.parquet` and `negative_trials_v2_with_features.joblib` (the versions *with* energy/snr columns) are no longer needed and are candidates for deletion (§6). `positive_trials.parquet` and `negative_trials_v2.joblib` (*without* those columns) are the real final tables.

### 3.11 Format consistency

`store_both_formats.py`

Professor also asked that both trial tables be available in **both** `.parquet` and `.joblib` (not one format per table). This is a pure load-then-save-in-the-other-format conversion — no columns or rows change, nothing is recomputed:

```
positive_trials.parquet   (already have)  -> + positive_trials.joblib
negative_trials_v2.joblib (already have)  -> + negative_trials_v2.parquet
```

```bash
python store_both_formats.py \
    encoded_trials/positive_trials.parquet \
    trial_outputs/negative_trials_v2.joblib \
    --pos-joblib-out positive_trials.joblib \
    --neg-parquet-out negative_trials_v2.parquet
```

Tested lossless (round-trip verified column-by-column) on synthetic data mirroring the real mixed id/path-category structure before being handed over.

Note: `negative_trials_v2.parquet` came out **larger** than `negative_trials_v2.joblib` (2.0G vs 1.6G) even though it's the same data — parquet's default compression (snappy) is weaker than the `compress=3` passed to joblib. If size matters more than default tooling, re-save with `to_parquet(path, compression="zstd")`.

### 3.12 Codebase reorganization

`reorganize_ami_codebase.sh`

Moves scripts into `code/corpus_prep/`, `code/eval_split_and_trials/`, `code/stats_and_plots/`; generated data into `trial_outputs/`; figures into `figures/`. Uses `git mv` for tracked files, plain `mv` for untracked (large generated outputs are expected to be `.gitignore`'d). Deletes exactly 4 confirmed-superseded files (v1 negative-trial script/output/report, and the old pre-meeting-split `eval_split2.py`) with a reason printed next to each. Anything of unclear purpose (`ami.py`, `generate_transcripts.py`) is left alone. Review with `git status` before committing — nothing is pushed anywhere by the script itself.

---

## 4. Current directory layout

```
AMI_Meeting_Corpus/
├── audio/ihm/..., audio/sdm/...        # raw corpus (corpus_prep output)
├── code/
│   ├── corpus_prep/                    # one-time raw-corpus setup
│   ├── eval_split_and_trials/          # active split + trial-generation pipeline
│   └── stats_and_plots/                # distribution/ratio analysis + plots
├── encoded_trials/
│   ├── path_lookup.csv                 # 104,748-file id dictionary (source of truth)
│   ├── positive_trials.parquet         # 27,301,776 rows (source of truth)
│   └── positive_trials.joblib          # same data, other format
├── trial_outputs/
│   ├── positive_trials.csv             # pre-encoding version (redundant, see §6)
│   ├── negative_trials_v2.joblib       # 273,017,760 rows (source of truth)
│   ├── negative_trials_v2.parquet      # same data, other format
│   ├── negative_trials_v2_report.json
│   └── positive_mic_ratio_stats.json, split_demographics.json, exact_trial_counts.json, ...
├── ami_stats_out/
│   └── files.csv
├── figures/
│   ├── distribution_bins_equal_width.png, distribution_bins_quantile.png
│   ├── pos_neg_mic_comparison.png
│   ├── duration_difference_eer.png
│   ├── energy_difference_eer.png
│   └── snr_difference_eer.png
├── audio_features.csv                  # per-file energy_db / wada_snr_db, 104,748 rows
└── eval_enroll_test2.xlsx              # Enroll/Test/Meeting_Split_Detail/Dropped_Speakers
```

## 5. Bugs hit and fixed along the way

Every one of these was caught by running the real script on real data and pasting the actual terminal output back — none were caught in advance.

1. **Wrong path assumption (`compute_audio_features.py`).** First real run: 104,748/104,748 files errored (`LibsndfileError`). `path_lookup.csv` stores paths relative to the corpus root, but the script opened them as if absolute. Fixed with an `--audio-root` argument + a resolver that strips the leading separator before joining.
2. **Wrong directory depth.** Even with `--audio-root` set, the preflight self-test still failed — the audio files actually live one level deeper, under an `audio/` subdirectory (`.../AMI_Meeting_Corpus/audio/ihm/train/...`), not directly under the corpus root. Fixed by pointing `--audio-root` at the `audio/` subfolder.
3. **Resume logic counted error rows as "done."** Found while fixing #1 — the original resume check would have permanently skipped retrying any file that had previously errored, even after the real bug was fixed. Fixed by rewriting the output file on resume to drop error rows and only count successes as done.
4. **Mismatched join-key assumption (`join_audio_features.py`).** Written assuming both trial tables used the same category-path-string columns. Real run revealed `positive_trials.parquet` actually uses raw integer id columns (`enroll_id`/`test_id`) while `negative_trials_v2.joblib` uses category path strings (`enroll_path`/`test_path`) — two different encodings of the same `path_lookup.csv` dictionary. Fixed by building both a dense-array id lookup and a dict path lookup, and dispatching per table based on which columns it actually has.

## 6. File inventory & cleanup

From the latest `find . -type f -size +25M -exec du -h {} + | sort -hr` on the real machine:

| File | Size | Status |
|---|---|---|
| `trial_outputs/positive_trials.csv` | 3.5G | **Delete candidate** — redundant with `positive_trials.parquet` + `path_lookup.csv`; verify row count matches the parquet first |
| `encoded_trials/negative_trials_v2.parquet` | 2.0G | Keep — new, from §3.11 format conversion |
| `encoded_trials/negative_trials_v2.joblib` | 1.6G | **Keep — source of truth** |
| `ami_stats_out/files.csv` | 47M | Keep (small, cheap) |
| `.git/lfs/objects/.../1bf7177f...` | 33M | Stalled/failed LFS object — stop trying to push large generated files into git (see §7) |
| `encoded_trials/positive_trials.parquet` | 33M | **Keep — source of truth** |
| `encoded_trials/positive_trials.joblib` | 32M | Keep — new, from §3.11 format conversion |

Also already identified for deletion (energy/snr no longer needed per §3.10):

| File | Size | Why delete |
|---|---|---|
| `negative_trials_v2_with_features.joblib` | 7.2G | energy_diff/snr_diff columns no longer needed |
| `positive_trials_with_features.parquet` | 360M | same reason |

**Keep regardless:** `audio_features.csv` (66 minutes to regenerate, cheap to keep), the three EER plots (`duration_difference_eer.png`, `energy_difference_eer.png`, `snr_difference_eer.png` — evidence supporting the professor's decision in §3.10), `path_lookup.csv`, and both source-of-truth trial tables.

## 7. Sharing / upload policy

**Do not put the large binaries on GitHub, with or without LFS.**

- Plain git rejects any file over 100MB outright (warns at 50MB). `positive_trials.csv` (3.5G), `negative_trials_v2.parquet` (2.0G), and `negative_trials_v2.joblib` (1.6G) will all be refused on push — that's core git, nothing to do with LFS.
- Git-LFS's free tier is 1GB storage + 1GB bandwidth/month. Those three files alone total 7.1GB — storage quota is blown on the first push, and bandwidth quota is blown the first time anyone clones.

**Instead:** data files (trial tables, `audio_features.csv`, `path_lookup.csv`) go on Google Drive / institutional storage, shared as a link. GitHub holds code, the README, and a small sample CSV (a few hundred rows) so the schema is visible without downloading gigabytes.

## 8. Scripts reference

| Script | Stage | Output |
|---|---|---|
| `ami_split_demographics.py` | §3.2 | `split_demographics.json` |
| `ami_eval_enroll_test_xlsx.py` | §3.3 | `eval_enroll_test2.xlsx` |
| `create_positive_trials.py` + `encode_positive_trials.py` | §3.4 | `positive_trials.csv`, `positive_trials.parquet`, `path_lookup.csv` |
| `positive_trial_full_distribution_stats.py` + `plot_distribution_bins.py` + `compute_positive_mic_ratio.py` | §3.5 | `positive_trial_full_distribution_stats.json`, `positive_mic_ratio_stats.json`, 2 PNGs |
| `create_negative_trials_v2.py` | §3.6 | `negative_trials_v2.joblib`, `negative_trials_v2_report.json` |
| `plot_pos_neg_mic_comparison.py` | §3.7 | `pos_neg_mic_comparison.png` |
| `duration_eer_kde.py` | §3.8 | `duration_difference_eer.png` (EER=0.50) |
| `compute_audio_features.py` | §3.9 | `audio_features.csv` |
| `join_audio_features.py` | §3.9 | `*_with_features.{parquet,joblib}` |
| `feature_eer_kde.py` | §3.9 | `energy_difference_eer.png` (EER=0.41), `snr_difference_eer.png` (EER=0.4987) |
| `store_both_formats.py` | §3.11 | `positive_trials.joblib`, `negative_trials_v2.parquet` |
| `reorganize_ami_codebase.sh` | §3.12 | reorganized directory tree |

## 9. Status / open items

- Core dataset (positive + negative trials, meeting-split, duration+mic matched): **done**, both formats available.
- Energy/SNR confound check: **done**, reviewed by professor, v3 explicitly not pursued.
- Cleanup: delete candidates identified (§6), not yet deleted — pending your go-ahead and the CSV-vs-parquet row-count verification.
- Upload: not yet done — needs to go to Drive/institutional storage per §7, not git.