#!/usr/bin/env python3
"""
join_audio_features.py

Attaches energy_diff and snr_diff to positive_trials and negative_trials_v2,
using audio_features.csv (one energy_db/wada_snr_db per unique file, keyed
by the same path_lookup.csv id every table in this project shares).

WRITES NEW FILES -- NEVER OVERWRITES YOUR ORIGINALS
-----------------------------------------------------
positive_trials_with_features.parquet and
negative_trials_v2_with_features.joblib are written fresh, alongside your
existing positive_trials.parquet / negative_trials_v2.joblib, which are
left untouched. negative_trials_v2.joblib took 22 minutes and 1.58GB to
build with zero shortfall -- no join script gets to risk that file on its
first run.

SCHEMA IS PRINTED, NEVER ASSUMED
-----------------------------------
This project has already hit silent wrong-assumption bugs this session
(the audio root, the audio/ subfolder). To not make another one blind,
this script prints every column name and dtype in both tables BEFORE
doing anything else, and aborts with a clear message if it can't find
id-like or path-like columns to join on -- it does not guess and
silently produce wrong numbers.

HOW THE JOIN WORKS (two different encodings, same path_lookup.csv)
-----------------------------------------------------------------------
This project turned out to encode the same path_lookup.csv dictionary
two different ways in two different files:
  - positive_trials.parquet stores raw integer ids (enroll_id/test_id).
  - negative_trials_v2.joblib stores pandas 'category' path strings
    (enroll_path/test_path).
Both are handled here. For id columns: energy/snr are loaded into a
dense numpy array indexed directly by integer id, then fancy-indexed --
no per-row dict lookups. For category columns: energy/snr are mapped
onto the column's small category list ONCE, then fancy-indexed by code,
same "operate on the small dictionary, not the big table" trick
create_negative_trials_v2.py already uses. A plain (non-category) path
column falls back to .map(), slower but correct.

MISSING ENERGY/SNR (the 4 files that errored in compute_audio_features.py)
-----------------------------------------------------------------------------
Any trial whose enroll or test file has no energy/snr value (because that
file errored) gets NaN in energy_diff/snr_diff for that row. These rows
are NOT dropped from the output file -- you may want them for other
columns -- but they must be dropped before computing EER/KDE on
energy_diff or snr_diff (NaN breaks both). The plot script does this
automatically and reports how many rows it dropped.

Usage:
    python join_audio_features.py \\
        positive_trials.parquet \\
        negative_trials_v2.joblib \\
        audio_features.csv \\
        path_lookup.csv \\
        --pos-out positive_trials_with_features.parquet \\
        --neg-out negative_trials_v2_with_features.joblib
"""

import argparse
import csv
import sys
from pathlib import Path


def load_feature_lookups(path_lookup_csv, audio_features_csv, np):
    """Returns (id_to_energy_arr, id_to_snr_arr, path_to_energy, path_to_snr).

    Two trial tables in this project encode the same path_lookup.csv
    dictionary two different ways: positive_trials.parquet stores raw
    integer ids (enroll_id/test_id), negative_trials_v2.joblib stores
    pandas 'category' path strings (enroll_path/test_path). Both are
    built here so attach_diffs() can join whichever kind a table has.

    id_to_energy_arr / id_to_snr_arr are dense numpy arrays indexed
    directly by integer id (fast fancy-indexing, no dict lookups per row).
    path_to_energy / path_to_snr are dicts keyed by path string, used for
    the category-encoded table via its small category list.
    """
    id_to_path = {}
    max_id = -1
    with open(path_lookup_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                fid = int(row[0])
                id_to_path[fid] = row[1]
                max_id = max(max_id, fid)

    id_to_energy_arr = np.full(max_id + 1, np.nan, dtype=np.float64)
    id_to_snr_arr = np.full(max_id + 1, np.nan, dtype=np.float64)
    path_to_energy = {}
    path_to_snr = {}
    n_rows = 0
    n_error = 0
    with open(audio_features_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            n_rows += 1
            fid_s, energy_s, snr_s, err = row[0], row[1], row[2], row[3] if len(row) > 3 else ""
            if err or energy_s == "" or snr_s == "":
                n_error += 1
                continue
            fid = int(fid_s)
            e, s = float(energy_s), float(snr_s)
            id_to_energy_arr[fid] = e
            id_to_snr_arr[fid] = s
            p = id_to_path.get(fid)
            if p is not None:
                path_to_energy[p] = e
                path_to_snr[p] = s

    print(f"  audio_features.csv: {n_rows:,} rows, {n_error:,} errored/blank "
          f"(those files will produce NaN diffs wherever referenced).",
          file=sys.stderr)
    print(f"  Built id-indexed and path-indexed lookups for "
          f"{len(path_to_energy):,} files.", file=sys.stderr)
    return id_to_energy_arr, id_to_snr_arr, path_to_energy, path_to_snr


def attach_diffs(df, lookups, enroll_col, test_col, key_type, label, np, pd):
    """Adds energy_diff / snr_diff columns to df in place.
    key_type is 'id' (raw integer columns, dense array lookup) or
    'path' (category dtype, small-category-list + fancy indexing)."""
    id_to_energy_arr, id_to_snr_arr, path_to_energy, path_to_snr = lookups
    feat_pairs = (("energy", id_to_energy_arr, path_to_energy),
                  ("snr", id_to_snr_arr, path_to_snr))
    for feat_name, id_arr, path_dict in feat_pairs:
        out_col = f"{feat_name}_diff"
        if key_type == "id":
            enroll_vals = _feature_by_id(df[enroll_col], id_arr, np)
            test_vals = _feature_by_id(df[test_col], id_arr, np)
        else:
            enroll_vals = _feature_by_path(df[enroll_col], path_dict, np)
            test_vals = _feature_by_path(df[test_col], path_dict, np)
        df[out_col] = np.abs(test_vals - enroll_vals)
        n_nan = int(np.isnan(df[out_col].to_numpy()).sum())
        print(f"  [{label}] {out_col}: {n_nan:,} / {len(df):,} rows are NaN "
              f"(enroll or test file missing energy/snr).", file=sys.stderr)
    return df


def _feature_by_id(col, id_arr, np):
    ids = col.to_numpy()
    out = np.full(len(ids), np.nan, dtype=np.float64)
    valid = (ids >= 0) & (ids < len(id_arr))
    out[valid] = id_arr[ids[valid]]
    return out


def _feature_by_path(col, feat_dict, np):
    if hasattr(col, "cat"):
        categories = col.cat.categories
        cat_lookup = np.array([feat_dict.get(c, np.nan) for c in categories], dtype=np.float64)
        codes = col.cat.codes.to_numpy()
        out = np.where(codes >= 0, cat_lookup[np.clip(codes, 0, len(cat_lookup) - 1)], np.nan)
        return out
    else:
        return col.map(feat_dict).astype(np.float64).to_numpy()


def find_path_columns(columns, label):
    """Returns (enroll_col, test_col, key_type) -- key_type is 'id' if the
    table stores raw integer ids (enroll_id/test_id), or 'path' if it
    stores category-encoded path strings (enroll_path/test_path)."""
    cols_lower = {c.lower(): c for c in columns}
    for suffix, key_type in (("path", "path"), ("id", "id")):
        enroll_candidates = [c for lc, c in cols_lower.items()
                              if "enroll" in lc and lc.endswith(suffix)]
        test_candidates = [c for lc, c in cols_lower.items()
                            if "test" in lc and lc.endswith(suffix)]
        if len(enroll_candidates) == 1 and len(test_candidates) == 1:
            return enroll_candidates[0], test_candidates[0], key_type
    print(f"\nERROR: could not identify enroll/test id or path columns in {label}.\n"
          f"  Columns found: {list(columns)}\n"
          f"Not guessing -- tell me the real column names and I'll fix the script.",
          file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("audio_features_csv")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("--pos-out", default="positive_trials_with_features.parquet")
    ap.add_argument("--neg-out", default="negative_trials_v2_with_features.joblib")
    ap.add_argument("--inspect-only", action="store_true",
                     help="Print schemas and exit without writing anything -- use this first "
                          "if you're not sure the column names match what this script expects.")
    args = ap.parse_args()

    for p in (args.positive_trials_parquet, args.negative_trials_joblib,
              args.audio_features_csv, args.path_lookup_csv):
        if not Path(p).is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    try:
        import numpy as np
        import pandas as pd
        import joblib
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install pandas numpy joblib pyarrow --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.positive_trials_parquet} ...", file=sys.stderr)
    pos_df = pd.read_parquet(args.positive_trials_parquet)
    print(f"  {len(pos_df):,} rows. Columns: {[(c, str(pos_df[c].dtype)) for c in pos_df.columns]}",
          file=sys.stderr)

    print(f"\nLoading {args.negative_trials_joblib} ...", file=sys.stderr)
    neg_df = joblib.load(args.negative_trials_joblib)
    print(f"  {len(neg_df):,} rows. Columns: {[(c, str(neg_df[c].dtype)) for c in neg_df.columns]}",
          file=sys.stderr)

    pos_enroll_col, pos_test_col, pos_key_type = find_path_columns(pos_df.columns, "positive_trials")
    neg_enroll_col, neg_test_col, neg_key_type = find_path_columns(neg_df.columns, "negative_trials_v2")
    print(f"\nUsing positive columns: enroll='{pos_enroll_col}', test='{pos_test_col}' (key_type={pos_key_type})",
          file=sys.stderr)
    print(f"Using negative columns: enroll='{neg_enroll_col}', test='{neg_test_col}' (key_type={neg_key_type})",
          file=sys.stderr)

    if args.inspect_only:
        print("\n--inspect-only set, stopping here without writing anything.", file=sys.stderr)
        return

    print(f"\nLoading {args.path_lookup_csv} and {args.audio_features_csv} ...", file=sys.stderr)
    lookups = load_feature_lookups(args.path_lookup_csv, args.audio_features_csv, np)

    print(f"\nAttaching energy_diff / snr_diff to positive trials ({len(pos_df):,} rows)...",
          file=sys.stderr)
    attach_diffs(pos_df, lookups, pos_enroll_col, pos_test_col, pos_key_type, "positive", np, pd)
    print(f"Writing {args.pos_out} ...", file=sys.stderr)
    pos_df.to_parquet(args.pos_out)
    print(f"  done. {Path(args.pos_out).resolve()}", file=sys.stderr)

    print(f"\nAttaching energy_diff / snr_diff to negative trials ({len(neg_df):,} rows)...",
          file=sys.stderr)
    attach_diffs(neg_df, lookups, neg_enroll_col, neg_test_col, neg_key_type, "negative", np, pd)
    print(f"Writing {args.neg_out} ...", file=sys.stderr)
    joblib.dump(neg_df, args.neg_out)
    print(f"  done. {Path(args.neg_out).resolve()}", file=sys.stderr)

    print("\nAll done. Originals untouched:", file=sys.stderr)
    print(f"  {args.positive_trials_parquet}  (unchanged)", file=sys.stderr)
    print(f"  {args.negative_trials_joblib}  (unchanged)", file=sys.stderr)


if __name__ == "__main__":
    main()