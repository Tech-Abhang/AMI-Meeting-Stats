#!/usr/bin/env python3
"""
cosine_eer_kde.py

Same EER + KDE machinery as duration_eer_kde.py / feature_eer_kde.py,
now pointed at the real thing: P-ECAPA cosine similarity scores from
compute_cosine_scores.py. This is the number that actually reports
system performance -- unlike duration/energy/snr EER (where 0.50 was
the GOOD result, proving those features carried no shortcut), here
LOW EER is good and EER near 0.50 would mean the embeddings aren't
separating speakers at all.

ONE REAL DIFFERENCE FROM THE EARLIER EER SCRIPTS
------------------------------------------------------
duration/energy/snr all used a NEGATED difference as the score
(-duration_diff, etc.) because for those, SMALLER difference meant
MORE genuine-like. Cosine similarity doesn't need that flip: it's
already "higher score = more genuine-like" by construction, so this
script passes the scores straight through, unnegated, into
compute_eer(). Flagging this explicitly so it isn't a silent
copy-paste mismatch from the earlier scripts.

INPUT
-----
Reads positive_cosine_scores.npy / negative_cosine_scores.npy directly
-- NOT the trial tables (compute_cosine_scores.py already pulled the
one column that matters out of those multi-gigabyte files, so there's
nothing left in the tables this step needs).

Usage:
    python cosine_eer_kde.py \\
        positive_cosine_scores.npy \\
        negative_cosine_scores.npy \\
        --out cosine_similarity_eer.png
"""

import argparse
import sys
from pathlib import Path


def gaussian_kde_1d(samples, x_grid, np):
    """Hand-rolled 1D Gaussian KDE -- identical to the earlier scripts.
    Scott's rule bandwidth, no scipy dependency."""
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
    """Pure numpy EER -- identical to the earlier scripts. Higher score
    = more target-like (true here without negation, unlike -duration_diff)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_scores_npy")
    ap.add_argument("negative_scores_npy")
    ap.add_argument("--out", default="cosine_similarity_eer.png")
    ap.add_argument("--sample-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    pos_path = Path(args.positive_scores_npy)
    neg_path = Path(args.negative_scores_npy)
    for p in (pos_path, neg_path):
        if not p.is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install numpy matplotlib --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"Loading {pos_path} ...", file=sys.stderr)
    pos_raw = np.load(pos_path)
    pos_mask = ~np.isnan(pos_raw)
    pos = pos_raw[pos_mask].astype(np.float64)
    print(f"  {len(pos_raw):,} rows, {len(pos_raw) - len(pos):,} dropped (NaN), "
          f"{len(pos):,} usable. range {pos.min():.4f} to {pos.max():.4f}",
          file=sys.stderr)

    print(f"\nLoading {neg_path} ...", file=sys.stderr)
    neg_raw = np.load(neg_path)
    neg_mask = ~np.isnan(neg_raw)
    neg = neg_raw[neg_mask].astype(np.float64)
    print(f"  {len(neg_raw):,} rows, {len(neg_raw) - len(neg):,} dropped (NaN), "
          f"{len(neg):,} usable. range {neg.min():.4f} to {neg.max():.4f}",
          file=sys.stderr)

    print("\nComputing EER on the full (NaN-dropped) population...", file=sys.stderr)
    eer, threshold = compute_eer(pos, neg)  # NOT negated -- see module docstring
    print(f"  EER = {eer:.4f}   (score = cosine similarity, threshold = {threshold:.4f})",
          file=sys.stderr)
    print(f"  Reminder: unlike the duration/energy/snr confound checks, LOW EER is "
          f"the goal here. Near 0 = embeddings strongly separate genuine from "
          f"impostor trials. Near 0.50 = embeddings carry no more speaker "
          f"information than a coin flip -- that would be a real problem to "
          f"dig into, not an expected baseline.", file=sys.stderr)

    n_pos_sample = min(args.sample_size, len(pos))
    n_neg_sample = min(args.sample_size, len(neg))
    pos_sample = rng.choice(pos, size=n_pos_sample, replace=False)
    neg_sample = rng.choice(neg, size=n_neg_sample, replace=False)
    print(f"\nDrawing KDE from a subsample: {n_pos_sample:,} Target, "
          f"{n_neg_sample:,} Non-Target points.", file=sys.stderr)

    combined = np.concatenate([pos, neg])
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

    ax.set_title("Cosine Similarity (P-ECAPA)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Cosine Similarity", fontsize=13)
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