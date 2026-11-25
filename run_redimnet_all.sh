#!/usr/bin/env bash
# run_redimnet_all.sh -- full ReDimNet (b6 / vox2 / ft_lm) pipeline on the SAME final trials.
# Everything new goes into redimnet_b6/ ; ECAPA files are never touched, trials never read-modified.
# Run from the folder that holds encoded_trials/ :   bash run_redimnet_all.sh 2>&1 | tee run_redimnet_all.log
# Resume-safe: the embedding step resumes; finished steps are skipped.
set -euo pipefail

AUDIO="/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"
POS="encoded_trials/positive_trials.parquet"
NEG="encoded_trials/negative_trials_v2.joblib"
LOOK="encoded_trials/path_lookup.csv"
DEV="cuda:0"
OUT="redimnet_b6"
EMB="$OUT/embeddings_redimnet_b6_vox2_ft_lm.npy"
DONE="$OUT/embeddings_redimnet_b6_vox2_ft_lm_done.npy"
PS="$OUT/positive_cosine_scores.npy"
NS="$OUT/negative_cosine_scores.npy"
mkdir -p "$OUT"

# 1. embeddings (resumes if interrupted)
python compute_embeddings_redimnet.py "$LOOK" --audio-root "$AUDIO" \
    --model-name b6 --train-type ft_lm --dataset vox2 \
    --out "$EMB" --done-out "$DONE" --errors-out "$OUT/embeddings_errors.csv" \
    --device "$DEV" --workers 8

# 2. cosine score per trial row (explicit output names: do NOT overwrite the ECAPA score files)
if [ -f "$PS" ] && [ -f "$NS" ]; then
    echo "scores already exist in $OUT -> skipping step 2"
else
    python compute_cosine_scores.py "$POS" "$NEG" "$LOOK" "$EMB" "$DONE" \
        --pos-out "$PS" --neg-out "$NS"
fi

# 3. EER: overall / ihm-ihm / ihm-sdm
python cosine_eer_mic_breakdown.py "$POS" "$NEG" "$LOOK" "$PS" "$NS" --out-prefix "$OUT/eer"

# 4. EER by clip length + the graphs
python eer_by_duration.py          "$POS" "$NEG" "$LOOK" "$PS" "$NS" --out-prefix "$OUT/eer_by_duration"
python eer_duration_analysis.py    "$POS" "$NEG" "$LOOK" "$PS" "$NS" --out-prefix "$OUT/eer_dur"
python eer_both_ge4.py             "$POS" "$NEG" "$LOOK" "$PS" "$NS" --out-prefix "$OUT/eer_dur_4_both_ge4"
python eer_both_ge4_diff.py        "$POS" "$NEG" "$LOOK" "$PS" "$NS" --out-prefix "$OUT/eer_dur_5_both_ge4_diff"

echo
echo "ECAPA baseline: overall 28.46%  ihm-ihm 24.21%  ihm-sdm 31.87%"