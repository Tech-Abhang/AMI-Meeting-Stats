#!/usr/bin/env python3
"""
store_both_formats.py

Your professor said: no energy/SNR, path-only trials are enough. That
means positive_trials.parquet and negative_trials_v2.joblib (the ones
WITHOUT energy_diff/snr_diff) are already the final data -- nothing to
regenerate. This script just saves each one in the OTHER format too, so
both trial tables exist as both .parquet and .joblib:

    positive_trials.parquet   (already have)  -> + positive_trials.joblib
    negative_trials_v2.joblib (already have)  -> + negative_trials_v2.parquet

This is a straight load-then-write-other-format. No columns change, no
rows change, nothing is computed. Cheap: dominated by how fast your
drive reads/writes ~1.6GB, not by any real work.

Usage:
    python store_both_formats.py \
        encoded_trials/positive_trials.parquet \
        trial_outputs/negative_trials_v2.joblib \
        --pos-joblib-out positive_trials.joblib \
        --neg-parquet-out negative_trials_v2.parquet
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("--pos-joblib-out", default="positive_trials.joblib")
    ap.add_argument("--neg-parquet-out", default="negative_trials_v2.parquet")
    args = ap.parse_args()

    for p in (args.positive_trials_parquet, args.negative_trials_joblib):
        if not Path(p).is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    try:
        import pandas as pd
        import joblib
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install pandas joblib pyarrow --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.positive_trials_parquet} ...", file=sys.stderr)
    pos_df = pd.read_parquet(args.positive_trials_parquet)
    print(f"  {len(pos_df):,} rows, columns: {list(pos_df.columns)}", file=sys.stderr)
    print(f"Writing {args.pos_joblib_out} ...", file=sys.stderr)
    joblib.dump(pos_df, args.pos_joblib_out, compress=3)
    print(f"  done. {Path(args.pos_joblib_out).resolve()}", file=sys.stderr)

    print(f"\nLoading {args.negative_trials_joblib} ...", file=sys.stderr)
    neg_df = joblib.load(args.negative_trials_joblib)
    print(f"  {len(neg_df):,} rows, columns: {list(neg_df.columns)}", file=sys.stderr)
    print(f"Writing {args.neg_parquet_out} ...", file=sys.stderr)
    neg_df.to_parquet(args.neg_parquet_out)
    print(f"  done. {Path(args.neg_parquet_out).resolve()}", file=sys.stderr)

    print("\nAll done. You now have, for each trial table, one .parquet and one .joblib "
          "copy -- same rows, same columns, just two container formats.", file=sys.stderr)


if __name__ == "__main__":
    main()