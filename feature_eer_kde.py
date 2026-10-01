#!/usr/bin/env python3
"""
feature_eer_kde.py

Generalized version of duration_eer_kde.py -- same EER + KDE machinery,
but takes the column name and title as arguments instead of hard-coding
"duration_diff_cs". Runs the Energy Difference and WADA SNR Difference
plots (step 3/4 of the 3-plot plan) against the files join_audio_features.py
produces, without duplicating the EER/KDE code a second and third time.

duration_eer_kde.py is left as-is (already run, already verified on your
real data, EER=0.50) -- this is a new script, not a replacement.

WHAT'S DIFFERENT FROM duration_eer_kde.py
--------------------------------------------
1. --pos-column / --neg-column instead of a hard-coded "duration_diff_cs"
   (duration was stored in centiseconds as an int; energy_diff/snr_diff
   are already plain float dB values in both tables, so no /100.0
   conversion here -- pass --scale if some future column needs one).
2. Drops NaN rows before computing anything, and reports how many were
   dropped. NaNs happen because 4 of your 104,748 files errored in
   compute_audio_features.py -- any trial that references one of those
   4 files has no energy_diff/snr_diff and must be excluded, same as
   you'd exclude a missing value from any statistic.
3. Everything else -- full-population EER, subsampled KDE picture, plot
   styling -- is identical to duration_eer_kde.py.

Usage (run once per feature):
    python feature_eer_kde.py \\
        positive_trials_with_features.parquet \\
        negative_trials_v2_with_features.joblib \\
        --pos-column energy_diff --neg-column energy_diff \\
        --title "Energy Difference" \\
        --out energy_difference_eer.png

    python feature_eer_kde.py \\
        positive_trials_with_features.parquet \\
        negative_trials_v2_with_features.joblib \\
        --pos-column snr_diff --neg-column snr_diff \\
        --title "WADA SNR Difference" \\
        --out snr_difference_eer.png
"""

import argparse
import gc
import sys
from pathlib import Path


def gaussian_kde_1d(samples, x_grid, np):
    """Hand-rolled 1D Gaussian KDE -- no scipy dependency. Identical to
    duration_eer_kde.py's version. Scott's rule bandwidth."""
    n = len(samples)
    std = np.std(samples)
    h = std * (n ** (-1.0 / 5.0))
    if h <= 0:
        h = 1e-6

    density = np.empty_like(x_grid, dtype=np.float64)
    norm_const = 1.0 / (n * h * np.sqrt(2 * np.pi))
    for i, xg in enumerate(x_grid):
        z = (xg - samples) / h
        density[i] = norm_const * np.sum(np.exp(-0.5 * z * z))
    return density


def compute_eer(target_scores, nontarget_scores):
    """Pure numpy EER. Identical to duration_eer_kde.py's version."""
    import numpy as np

    n_tar = len(target_scores)
    n_non = len(nontarget_scores)

    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([
        np.ones(n_tar, dtype=np.int8),
        np.zeros(n_non, dtype=np.int8),
    ])

    order = np.argsort(-scores, kind="stable")
    labels_sorted = labels[order]

    cum_tar_accepted = np.cumsum(labels_sorted == 1)
    cum_non_accepted = np.cumsum(labels_sorted == 0)

    far = cum_non_accepted / n_non
    frr = (n_tar - cum_tar_accepted) / n_tar

    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2.0
    threshold = scores[order][idx]
    return float(eer), float(threshold)


def load_column_dropna(path, column, label, pd, np, scale=1.0, is_parquet=True):
    import joblib
    if is_parquet:
        table = pd.read_parquet(path, columns=[column])
        vals = table[column].to_numpy(dtype=np.float64) * scale
        del table
    else:
        df = joblib.load(path)
        vals = df[column].to_numpy(dtype=np.float64) * scale
        del df
        gc.collect()

    n_total = len(vals)
    mask = ~np.isnan(vals)
    n_nan = n_total - int(mask.sum())
    vals = vals[mask]
    print(f"  {label}: {n_total:,} rows loaded, {n_nan:,} dropped (NaN -- missing "
          f"energy/snr for enroll or test file), {len(vals):,} usable.", file=sys.stderr)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_parquet", help="positive_trials_with_features.parquet")
    ap.add_argument("negative_joblib", help="negative_trials_v2_with_features.joblib")
    ap.add_argument("--pos-column", required=True, help="e.g. energy_diff or snr_diff")
    ap.add_argument("--neg-column", required=True, help="e.g. energy_diff or snr_diff")
    ap.add_argument("--title", required=True, help='e.g. "Energy Difference"')
    ap.add_argument("--scale", type=float, default=1.0,
                     help="Multiply loaded values by this (duration_diff_cs needed /100 -- "
                          "pass 0.01 for that case; energy_diff/snr_diff need no scaling, default 1.0)")
    ap.add_argument("--out", default="feature_difference_eer.png")
    ap.add_argument("--sample-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    pos_path = Path(args.positive_parquet)
    neg_path = Path(args.negative_joblib)
    for p in (pos_path, neg_path):
        if not p.is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install pandas numpy joblib matplotlib pyarrow --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"Loading {pos_path} ...", file=sys.stderr)
    pos_diff = load_column_dropna(pos_path, args.pos_column, "Target (positive)", pd, np,
                                   scale=args.scale, is_parquet=True)
    print(f"  range {pos_diff.min():.3f} to {pos_diff.max():.3f}", file=sys.stderr)

    print(f"\nLoading {neg_path} ...", file=sys.stderr)
    neg_diff = load_column_dropna(neg_path, args.neg_column, "Non-Target (negative)", pd, np,
                                   scale=args.scale, is_parquet=False)
    print(f"  range {neg_diff.min():.3f} to {neg_diff.max():.3f}", file=sys.stderr)

    print("\nComputing EER on the full (NaN-dropped) population...", file=sys.stderr)
    eer, threshold = compute_eer(-pos_diff, -neg_diff)
    print(f"  EER = {eer:.4f}   (score = -{args.pos_column}, threshold = {-threshold:.3f})",
          file=sys.stderr)
    print(f"  Reminder: EER near 0.50 means this feature alone carries almost no information "
          f"to separate Target from Non-Target -- that's the GOOD, expected confound-check "
          f"result, same interpretation as the duration_diff plot. EER pulled toward 0 or 1 "
          f"would mean negatives still carry a {args.pos_column} shortcut worth investigating.",
          file=sys.stderr)

    n_pos_sample = min(args.sample_size, len(pos_diff))
    n_neg_sample = min(args.sample_size, len(neg_diff))
    pos_sample = rng.choice(pos_diff, size=n_pos_sample, replace=False)
    neg_sample = rng.choice(neg_diff, size=n_neg_sample, replace=False)
    print(f"\nDrawing KDE from a subsample: {n_pos_sample:,} Target, "
          f"{n_neg_sample:,} Non-Target points.", file=sys.stderr)

    combined = np.concatenate([pos_diff, neg_diff])
    x_min = float(np.percentile(combined, 0.5))
    x_max = float(np.percentile(combined, 99.5))
    x_grid = np.linspace(x_min, x_max, 500)

    pos_kde = gaussian_kde_1d(pos_sample, x_grid, np)
    neg_kde = gaussian_kde_1d(neg_sample, x_grid, np)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(x_grid, pos_kde, linestyle=":", linewidth=2.5, color="#3B5FA0", label="Target")
    ax.plot(x_grid, neg_kde, linestyle="-.", linewidth=2.5, color="#B5451B", label="Non-Target")
    ax.fill_between(x_grid, np.minimum(pos_kde, neg_kde), color="#C8A951", alpha=0.6)
    ax.plot([], [], color="black", linewidth=1.5, label=f"EER = {eer:.2f}")

    ax.set_title(args.title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Difference", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()