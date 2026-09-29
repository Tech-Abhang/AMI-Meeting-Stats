#!/usr/bin/env bash
#
# reorganize_ami_codebase.sh
#
# Reorganizes the AMI_Meeting_Corpus working directory: moves files into
# a proper folder structure, and deletes ONLY the files this conversation
# has direct, confident knowledge are fully superseded.
#
# SAFE BY DESIGN
# ---------------
#   - Uses `git mv` when the file is tracked by git (preserves history),
#     falls back to plain `mv` for untracked files (your large generated
#     outputs are likely .gitignore'd).
#   - Deletes only 3 files (see "deleting confirmed-superseded files"
#     below), each with a one-line reason next to it.
#   - Everything of uncertain purpose is MOVED, never deleted -- a wrong
#     guess about what a file does costs you nothing, you can move it
#     back.
#   - 3 files are left completely untouched at the root because this
#     conversation has no knowledge of what they do: ami.py,
#     eval_split2.py, generate_transcripts.py. Check them yourself before
#     deciding anything.
#   - 1 file (eval_enroll_test.xlsx at the root) is flagged but NOT
#     deleted -- it looks superseded by eval_xlsx_out/eval_enroll_test2.xlsx
#     but this script can't read either file to prove it.
#
# Run from inside AMI_Meeting_Corpus/:
#     bash reorganize_ami_codebase.sh
#
# Review with `git status` / `git diff --stat` before committing.
# Nothing is pushed anywhere.

set -euo pipefail

safe_mv() {
    local src="$1" dst="$2"
    if [ ! -e "$src" ]; then
        echo "  skip (not found): $src"
        return
    fi
    if git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
        git mv "$src" "$dst"
    else
        mv "$src" "$dst"
    fi
    echo "  moved: $src -> $dst"
}

safe_rm() {
    local f="$1" reason="$2"
    if [ ! -e "$f" ]; then
        echo "  skip (not found): $f"
        return
    fi
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        git rm "$f"
    else
        rm "$f"
    fi
    echo "  deleted: $f   ($reason)"
}

echo "Creating folder structure..."
mkdir -p code/corpus_prep code/eval_split_and_trials code/stats_and_plots
mkdir -p trial_outputs figures

echo
echo "== code/corpus_prep  (one-time raw-corpus setup, guessed from filename -- moved, not deleted) =="
safe_mv organize_ami.py            code/corpus_prep/
safe_mv rename_ami.py              code/corpus_prep/
safe_mv standardize_ami.py         code/corpus_prep/
safe_mv get_sdm_audio.py           code/corpus_prep/
safe_mv ami-ihm-kaldi-chunked.py   code/corpus_prep/
safe_mv resolve_mismatch.py        code/corpus_prep/
safe_mv ami_stats.py               code/corpus_prep/

echo
echo "== code/eval_split_and_trials  (the active speaker-split + trial-generation pipeline) =="
safe_mv ami_split_demographics.py         code/eval_split_and_trials/
safe_mv ami_eval_enroll_test_xlsx.py      code/eval_split_and_trials/
safe_mv ami_exact_trial_count.py          code/eval_split_and_trials/
safe_mv create_positive_trials.py         code/eval_split_and_trials/
safe_mv encode_positive_trials.py         code/eval_split_and_trials/
safe_mv compute_positive_mic_ratio.py     code/eval_split_and_trials/
safe_mv create_negative_trials_v2.py      code/eval_split_and_trials/

echo
echo "== code/stats_and_plots  (analysis / reporting scripts) =="
safe_mv positive_trial_full_distribution_stats.py     code/stats_and_plots/
safe_mv plot_distribution_bins.py                     code/stats_and_plots/
safe_mv plot_pos_neg_mic_comparison.py                code/stats_and_plots/

echo
echo "== trial_outputs  (generated data files, not code) =="
safe_mv split_demographics.json                       trial_outputs/
safe_mv exact_trial_counts.json                        trial_outputs/
safe_mv positive_trials.csv                             trial_outputs/
safe_mv positive_trial_full_distribution_stats.json      trial_outputs/
safe_mv positive_mic_ratio_stats.json                      trial_outputs/
safe_mv negative_trials_v2.joblib                            trial_outputs/
safe_mv negative_trials_v2_report.json                        trial_outputs/

echo
echo "== figures =="
safe_mv distribution_bins_equal_width.png    figures/
safe_mv distribution_bins_quantile.png       figures/
safe_mv pos_neg_mic_comparison.png           figures/

echo
echo "== deleting confirmed-superseded files =="
safe_rm create_negative_trials.py   "superseded by create_negative_trials_v2.py -- v1 never tracked/guaranteed the ihm-ihm vs ihm-sdm ratio, which is the whole reason v2 exists"
safe_rm negative_trials.joblib      "v1's output; negative_trials_v2.joblib covers the same rows plus the ratio guarantee v1 lacked"
safe_rm negative_trials_report.json "v1's report; all analysis moved to negative_trials_v2_report.json"

echo
echo "== NOT touched -- purpose unclear from this conversation, please check yourself =="
echo "  ami.py                  -- name too generic to guess safely; grep other scripts for 'import ami' first"
echo "  eval_split2.py           -- possibly an earlier draft of ami_eval_enroll_test_xlsx.py, unconfirmed"
echo "  generate_transcripts.py  -- doesn't appear related to the trial-pair pipeline as far as known here"

echo
echo "== flagged, not auto-deleted -- please confirm before removing by hand =="
echo "  eval_enroll_test.xlsx (root) -- looks like the PRE meeting-split workbook. Every current script"
echo "  reads eval_xlsx_out/eval_enroll_test2.xlsx instead. Confirm that's the one you actually use, then:"
echo "      git rm eval_enroll_test.xlsx"

echo
echo "Done. Run 'git status' to review everything before committing."
