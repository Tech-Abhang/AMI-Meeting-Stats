#!/usr/bin/env python3
"""
duration_eer_kde.py

Step 1 of 3 toward the Target/Non-Target density plots (Duration Difference,
Energy Difference, WADA SNR Difference) you showed as reference images.
This script does the Duration Difference one FIRST, because duration_diff
already exists for every positive and negative trial -- no new audio
processing needed. It also builds and tests the EER + KDE machinery that
Energy Difference and WADA SNR Difference will reuse later.

WHAT IT DOES
------------
  1. Loads duration_diff for every positive trial (Target) from the
     encoded parquet, and every negative trial (Non-Target) from
     negative_trials_v2.joblib.
  2. Computes EER on the FULL population of both (no subsampling -- time
     isn't the constraint, and EER should reflect all the data, not a
     sample of it). Uses "-duration_diff" as the score (small difference
     = more Target-like), pure numpy, no sklearn dependency.
  3. For the PICTURE only, draws a random subsample of each side
     (default 150,000 each -- plenty for a smooth density curve, and
     fast to render) and fits a KDE density to each.
  4. Plots both densities overlaid, Target vs Non-Target, styled like
     your reference images, with the EER printed in the legend.

WHY EER IS COMPUTED ON EVERYTHING BUT THE PICTURE USES A SAMPLE
-----------------------------------------------------------------
EER is a single number -- getting it right matters, and sorting a few
hundred million floats is cheap (well under a minute). A density curve
is a picture -- a few hundred thousand points draw an identical-looking
curve to 300 million, so there's no reason to make matplotlib work that
hard.

Usage:
    python duration_eer_kde.py \\
        encoded_trials/positive_trials.parquet \\
        negative_trials_v2.joblib \\
        --out duration_difference_eer.png
"""

import argparse
import gc
import sys
from pathlib import Path


def gaussian_kde_1d(samples, x_grid, np):
    """Hand-rolled 1D Gaussian KDE -- no scipy dependency.

    Bandwidth via Scott's rule (same default scipy.stats.gaussian_kde
    uses for 1D data): h = std(samples) * n**(-1/5).
    Loops over grid points (typically ~500) rather than building one
    giant (grid x samples) matrix, so memory stays small regardless of
    how many samples are passed in.
    """
    n = len(samples)
    std = np.std(samples)
    h = std * (n ** (-1.0 / 5.0))
    if h <= 0:
        h = 1e-6  # degenerate case guard, shouldn't happen on real data

    density = np.empty_like(x_grid, dtype=np.float64)
    norm_const = 1.0 / (n * h * np.sqrt(2 * np.pi))
    for i, xg in enumerate(x_grid):
        z = (xg - samples) / h
        density[i] = norm_const * np.sum(np.exp(-0.5 * z * z))
    return density


def compute_eer(target_scores, nontarget_scores):
    """Pure numpy EER -- no sklearn dependency.

    target_scores    : scores for Target (positive/genuine) trials
    nontarget_scores : scores for Non-Target (negative/impostor) trials
    Higher score = more Target-like, by convention of whatever the
    caller passed in (here, -duration_diff).

    Returns (eer, threshold_used).
    """
    import numpy as np

    n_tar = len(target_scores)
    n_non = len(nontarget_scores)

    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([
        np.ones(n_tar, dtype=np.int8),
        np.zeros(n_non, dtype=np.int8),
    ])

    # sort by score DESCENDING -- walking down this list is the same as
    # sweeping the accept/reject threshold from high to low
    order = np.argsort(-scores, kind="stable")
    labels_sorted = labels[order]

    # cumulative count of targets and non-targets accepted so far
    # (as we lower the threshold, more points get "accepted")
    cum_tar_accepted = np.cumsum(labels_sorted == 1)
    cum_non_accepted = np.cumsum(labels_sorted == 0)

    far = cum_non_accepted / n_non                    # false accepts / all non-targets
    frr = (n_tar - cum_tar_accepted) / n_tar           # missed targets / all targets

    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2.0
    threshold = scores[order][idx]
    return float(eer), float(threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_parquet", help="encoded_trials/positive_trials.parquet")
    ap.add_argument("negative_joblib", help="negative_trials_v2.joblib")
    ap.add_argument("--out", default="duration_difference_eer.png")
    ap.add_argument("--sample-size", type=int, default=150_000,
                     help="Points per class used ONLY for drawing the KDE curve "
                          "(EER always uses the full population). Default 150,000.")
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
        import joblib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    conda install -y pandas numpy joblib matplotlib\n"
              f"    pip install pyarrow --break-system-packages", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # ---- Target (positive) ----
    print(f"Loading {pos_path} ...", file=sys.stderr)
    pos_table = pd.read_parquet(pos_path, columns=["duration_diff_cs"])
    pos_diff = pos_table["duration_diff_cs"].to_numpy(dtype=np.float32) / 100.0
    del pos_table
    print(f"  {len(pos_diff):,} Target (positive) trials, "
          f"duration_diff range {pos_diff.min():.2f}s to {pos_diff.max():.2f}s",
          file=sys.stderr)

    # ---- Non-Target (negative) ----
    print(f"Loading {neg_path} ...", file=sys.stderr)
    neg_df = joblib.load(neg_path)
    neg_diff = neg_df["duration_diff"].to_numpy(dtype=np.float32)
    del neg_df
    gc.collect()
    print(f"  {len(neg_diff):,} Non-Target (negative) trials, "
          f"duration_diff range {neg_diff.min():.2f}s to {neg_diff.max():.2f}s",
          file=sys.stderr)

    # ---- EER on the FULL population ----
    print("\nComputing EER on the full population (this is the accurate number, "
          "not a sample)...", file=sys.stderr)
    eer, threshold = compute_eer(-pos_diff, -neg_diff)
    print(f"  EER = {eer:.4f}   (score = -duration_diff, threshold = {-threshold:.3f}s)",
          file=sys.stderr)
    print(f"  For reference: EER near 0.50 means duration_diff alone carries almost "
          f"no information about Target vs Non-Target -- which is the expected, "
          f"GOOD result here, since negatives were deliberately built to match the "
          f"positive duration_diff distribution per bin.", file=sys.stderr)

    # ---- subsample for the picture only ----
    n_pos_sample = min(args.sample_size, len(pos_diff))
    n_neg_sample = min(args.sample_size, len(neg_diff))
    pos_sample = rng.choice(pos_diff, size=n_pos_sample, replace=False)
    neg_sample = rng.choice(neg_diff, size=n_neg_sample, replace=False)
    print(f"\nDrawing KDE from a subsample: {n_pos_sample:,} Target, "
          f"{n_neg_sample:,} Non-Target points.", file=sys.stderr)

    x_max = float(np.percentile(np.concatenate([pos_diff, neg_diff]), 99.5))
    x_grid = np.linspace(0, x_max, 500)

    pos_kde = gaussian_kde_1d(pos_sample, x_grid, np)
    neg_kde = gaussian_kde_1d(neg_sample, x_grid, np)

    # ---- plot, styled like the reference images ----
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(x_grid, pos_kde, linestyle=":", linewidth=2.5, color="#3B5FA0",
            label="Target")
    ax.plot(x_grid, neg_kde, linestyle="-.", linewidth=2.5, color="#B5451B",
            label="Non-Target")
    ax.fill_between(x_grid, np.minimum(pos_kde, neg_kde), color="#C8A951", alpha=0.6)
    ax.plot([], [], color="black", linewidth=1.5, label=f"EER = {eer:.2f}")

    ax.set_title("Duration Difference", fontsize=16, fontweight="bold")
    ax.set_xlabel("Difference", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()