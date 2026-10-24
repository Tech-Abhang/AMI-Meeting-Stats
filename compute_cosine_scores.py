#!/usr/bin/env python3
"""
compute_cosine_scores.py

Computes one cosine-similarity score per TRIAL ROW (not per file) --
the real verification score, option (b) from earlier: the specific
enroll file named in that row vs. the specific test file named in that
row, using the P-ECAPA embeddings compute_embeddings.py already built.

WHY THIS DOESN'T WRITE "_with_cosine" COPIES OF YOUR TRIAL TABLES
----------------------------------------------------------------------
join_audio_features.py (the energy/snr step) wrote whole new copies of
positive_trials.parquet / negative_trials_v2.joblib with extra columns
added -- which produced the 360MB and 7.2GB files that later had to be
deleted once the professor said energy/snr wasn't needed. Don't repeat
that: this script writes ONLY the score column, as a plain float32
array aligned by ROW POSITION to the original table:

    positive_cosine_scores.npy  -- float32, one score per row of
                                   positive_trials.parquet, same order
    negative_cosine_scores.npy  -- float32, one score per row of
                                   negative_trials_v2.joblib, same order

~109MB and ~1.09GB respectively -- a fraction of a full table copy.
Your original trial tables are never touched, never reordered, never
filtered. The only rule this creates: don't re-sort/re-filter
positive_trials.parquet or negative_trials_v2.joblib after this, or
these score arrays go out of alignment with the rows they describe.

WHY CHUNKED, NOT ONE GIANT VECTORIZED STEP
----------------------------------------------
Unlike energy/snr (one number per file), an embedding is 192 numbers
per file. Naively doing embeddings[all_273_million_test_ids] in one
shot would materialize a 273,000,000 x 192 float32 array -- about
210GB, more RAM than this machine has. So each table is processed in
chunks (--chunk-size rows at a time): small enough to stay well under
your 67GB RAM, large enough that numpy's vectorized math still does
all the real work (this is plain linear algebra, no GPU needed here --
the GPU's job was the embedding extraction, already done).

HANDLES BOTH KEY ENCODINGS, SAME AS join_audio_features.py
-----------------------------------------------------------
positive_trials.parquet uses enroll_id/test_id (int32).
negative_trials_v2.joblib uses enroll_path/test_path (category dtype).
Both are resolved down to a file id compatible with embeddings.npy's
row index before scoring -- same "operate on the small category list,
not every row" trick used throughout this project.

MISSING EMBEDDINGS
--------------------
A trial row gets NaN if its enroll or test file's embedding was never
computed (the embeddings_done.npy mask is False for that file id --
your 4 errored files from compute_embeddings.py, for example). These
NaNs are not dropped here -- the EER/KDE script drops them, same
pattern as feature_eer_kde.py.

Usage:
    python compute_cosine_scores.py \\
        encoded_trials/positive_trials.parquet \\
        trial_outputs/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        embeddings.npy \\
        embeddings_done.npy \\
        --pos-out positive_cosine_scores.npy \\
        --neg-out negative_cosine_scores.npy \\
        --chunk-size 5000000
"""

import argparse
import csv
import sys
import time
from pathlib import Path


def load_path_to_id(path_lookup_csv):
    path_to_id = {}
    with open(path_lookup_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                path_to_id[row[1]] = int(row[0])
    return path_to_id


def find_pair_columns(columns, label):
    """Same detection as join_audio_features.py -- returns (enroll_col,
    test_col, key_type), key_type 'id' (raw int columns) or 'path'
    (category-dtype path strings). Aborts loudly rather than guessing."""
    cols_lower = {c.lower(): c for c in columns}
    for suffix, key_type in (("path", "path"), ("id", "id")):
        enroll_candidates = [c for lc, c in cols_lower.items()
                              if "enroll" in lc and lc.endswith(suffix)]
        test_candidates = [c for lc, c in cols_lower.items()
                            if "test" in lc and lc.endswith(suffix)]
        if len(enroll_candidates) == 1 and len(test_candidates) == 1:
            return enroll_candidates[0], test_candidates[0], key_type
    print(f"\nERROR: could not identify enroll/test id or path columns in {label}.\n"
          f"  Columns found: {list(columns)}", file=sys.stderr)
    sys.exit(1)


def get_file_ids(df, col_name, key_type, path_to_id, n_embeddings, np):
    """Resolves a trial-table column down to a (file_id array, valid mask)
    pair, where file_id indexes directly into embeddings.npy."""
    if key_type == "id":
        ids = df[col_name].to_numpy().astype(np.int64)
        valid = (ids >= 0) & (ids < n_embeddings)
    else:
        col = df[col_name]
        codes = col.cat.codes.to_numpy()
        categories = col.cat.categories
        cat_to_fileid = np.array([path_to_id.get(p, -1) for p in categories], dtype=np.int64)
        safe_codes = np.clip(codes, 0, len(cat_to_fileid) - 1)
        ids = np.where(codes >= 0, cat_to_fileid[safe_codes], -1)
        valid = (ids >= 0) & (ids < n_embeddings)
    ids_safe = np.where(valid, ids, 0)  # clip invalid to a harmless index; valid mask excludes them from scoring
    return ids_safe, valid


def chunked_cosine(enroll_ids, test_ids, valid_mask, embeddings, done_mask,
                    chunk_size, label, np):
    n = len(enroll_ids)
    scores = np.full(n, np.nan, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        e_ids = enroll_ids[start:end]
        t_ids = test_ids[start:end]
        v = valid_mask[start:end] & done_mask[e_ids] & done_mask[t_ids]

        e = embeddings[e_ids]       # (chunk, emb_dim)
        t = embeddings[t_ids]
        dot = np.einsum("ij,ij->i", e, t)
        ne = np.linalg.norm(e, axis=1)
        nt = np.linalg.norm(t, axis=1)
        denom = ne * nt
        ok = v & (denom > 0)

        chunk_scores = np.full(end - start, np.nan, dtype=np.float32)
        chunk_scores[ok] = (dot[ok] / denom[ok]).astype(np.float32)
        scores[start:end] = chunk_scores

        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        print(f"  [{label}] {end:,}/{n:,} rows scored ({rate:,.0f} rows/s)",
              file=sys.stderr)

    n_nan = int(np.isnan(scores).sum())
    print(f"  [{label}] done. {n_nan:,}/{n:,} rows are NaN (enroll or test "
          f"file had no embedding).", file=sys.stderr)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("embeddings_npy")
    ap.add_argument("embeddings_done_npy")
    ap.add_argument("--pos-out", default="positive_cosine_scores.npy")
    ap.add_argument("--neg-out", default="negative_cosine_scores.npy")
    ap.add_argument("--chunk-size", type=int, default=5_000_000,
                     help="Rows processed per chunk. Lower this if you see "
                          "a MemoryError; 5,000,000 rows x 192 floats x 2 "
                          "(enroll+test) x 4 bytes = ~7.7GB per chunk.")
    args = ap.parse_args()

    for p in (args.positive_trials_parquet, args.negative_trials_joblib,
              args.path_lookup_csv, args.embeddings_npy, args.embeddings_done_npy):
        if not Path(p).is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    try:
        import numpy as np
        import pandas as pd
        import joblib
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install pandas numpy joblib pyarrow --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.embeddings_npy} / {args.embeddings_done_npy} ...", file=sys.stderr)
    embeddings = np.load(args.embeddings_npy)
    done_mask = np.load(args.embeddings_done_npy)
    n_embeddings = embeddings.shape[0]
    print(f"  {n_embeddings:,} file ids, embedding dim {embeddings.shape[1]}, "
          f"{int(done_mask.sum()):,} with a real embedding "
          f"({n_embeddings - int(done_mask.sum()):,} missing/errored).",
          file=sys.stderr)

    print(f"\nLoading {args.path_lookup_csv} ...", file=sys.stderr)
    path_to_id = load_path_to_id(args.path_lookup_csv)
    print(f"  {len(path_to_id):,} paths.", file=sys.stderr)

    print(f"\nLoading {args.positive_trials_parquet} ...", file=sys.stderr)
    pos_df = pd.read_parquet(args.positive_trials_parquet)
    print(f"  {len(pos_df):,} rows. Columns: {list(pos_df.columns)}", file=sys.stderr)
    pos_enroll_col, pos_test_col, pos_key_type = find_pair_columns(pos_df.columns, "positive_trials")
    print(f"  enroll='{pos_enroll_col}', test='{pos_test_col}' (key_type={pos_key_type})",
          file=sys.stderr)

    pos_enroll_ids, pos_enroll_valid = get_file_ids(
        pos_df, pos_enroll_col, pos_key_type, path_to_id, n_embeddings, np)
    pos_test_ids, pos_test_valid = get_file_ids(
        pos_df, pos_test_col, pos_key_type, path_to_id, n_embeddings, np)
    pos_valid = pos_enroll_valid & pos_test_valid
    del pos_df

    print(f"\nScoring positive trials ...", file=sys.stderr)
    pos_scores = chunked_cosine(pos_enroll_ids, pos_test_ids, pos_valid,
                                 embeddings, done_mask, args.chunk_size, "positive", np)
    np.save(args.pos_out, pos_scores)
    print(f"  saved {Path(args.pos_out).resolve()}", file=sys.stderr)
    del pos_enroll_ids, pos_test_ids, pos_valid, pos_scores

    print(f"\nLoading {args.negative_trials_joblib} ...", file=sys.stderr)
    neg_df = joblib.load(args.negative_trials_joblib)
    print(f"  {len(neg_df):,} rows. Columns: {list(neg_df.columns)}", file=sys.stderr)
    neg_enroll_col, neg_test_col, neg_key_type = find_pair_columns(neg_df.columns, "negative_trials_v2")
    print(f"  enroll='{neg_enroll_col}', test='{neg_test_col}' (key_type={neg_key_type})",
          file=sys.stderr)

    neg_enroll_ids, neg_enroll_valid = get_file_ids(
        neg_df, neg_enroll_col, neg_key_type, path_to_id, n_embeddings, np)
    neg_test_ids, neg_test_valid = get_file_ids(
        neg_df, neg_test_col, neg_key_type, path_to_id, n_embeddings, np)
    neg_valid = neg_enroll_valid & neg_test_valid
    del neg_df

    print(f"\nScoring negative trials ...", file=sys.stderr)
    neg_scores = chunked_cosine(neg_enroll_ids, neg_test_ids, neg_valid,
                                 embeddings, done_mask, args.chunk_size, "negative", np)
    np.save(args.neg_out, neg_scores)
    print(f"  saved {Path(args.neg_out).resolve()}", file=sys.stderr)

    print("\nAll done. These two .npy files are score arrays ALIGNED BY ROW "
          "POSITION to positive_trials.parquet and negative_trials_v2.joblib "
          "-- don't reorder or filter those tables without regenerating these.",
          file=sys.stderr)


if __name__ == "__main__":
    main()