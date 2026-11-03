#!/usr/bin/env bash
# run_all_fixes.sh -- re-embed WITHOUT padding, score 6 variants, EER for each.
# Trials are never touched. Original embeddings.npy / score files are never
# overwritten (everything new goes to *_v2 names).
#
# Run from the folder that holds encoded_trials/ and trial_outputs/:
#     bash run_all_fixes.sh 2>&1 | tee run_all_fixes.log
#
# Resume-safe: if it dies mid-way, run it again; embedding step resumes.
set -euo pipefail

AUDIO="/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"
POS="encoded_trials/positive_trials.parquet"
NEG="encoded_trials/negative_trials_v2.joblib"
LOOK="encoded_trials/path_lookup.csv"
DEV="cuda:0"

# ---- 1. embeddings, length-sorted (no padding) -- skipped if already done ----
if [ -f embeddings_v2_done.npy ] && [ "${FORCE_EMBED:-0}" != "1" ]; then
    echo "embeddings_v2.npy already exists -> skipping step 1 (FORCE_EMBED=1 to redo)"
else
    python compute_embeddings.py "$LOOK" \
        --audio-root "$AUDIO" \
        --out embeddings_v2.npy --done-out embeddings_v2_done.npy \
        --errors-out embeddings_v2_errors.csv \
        --device "$DEV" --batch-size 64 --workers 8 \
        --sort-by-length --self-test 3
fi

# ---- 2. scores: raw + centering + whitening + AS-norm, one pass ----
python compute_cosine_scores_v2.py "$POS" "$NEG" "$LOOK" \
    embeddings_v2.npy embeddings_v2_done.npy \
    --out-dir scores_v2 --device "$DEV" \
    --variants raw,center,centermic,whiten,centermic+asnorm,whiten+asnorm

# ---- 3. EER + plots per variant (ihm-ihm / ihm-sdm / overall) ----
for V in raw center centermic whiten centermic_asnorm whiten_asnorm; do
    echo "=================  $V  ================="
    python cosine_eer_mic_breakdown.py "$POS" "$NEG" "$LOOK" \
        "scores_v2/positive_cosine_scores_${V}.npy" \
        "scores_v2/negative_cosine_scores_${V}.npy" \
        --out-prefix "scores_v2/eer_${V}" 2>&1 | grep -E "Summary|EER" || true
done

echo
echo "Baseline (v1 embeddings, plain cosine): overall 28.46%  ihm-ihm 24.21%  ihm-sdm 31.87%"