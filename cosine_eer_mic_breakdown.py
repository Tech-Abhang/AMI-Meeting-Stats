#!/usr/bin/env python3
"""
cosine_eer_mic_breakdown.py

Two things in one script, both requested together:

1. A histogram+KDE plot in the style of your reference image (filled
   bars + a smooth KDE line on top, "True Scores" / "False Scores"
   legend) instead of the KDE-only line plot cosine_eer_kde.py made.
   Same data, different picture.

2. EER computed three ways instead of one lumped number:
     - overall   (what cosine_eer_kde.py already gave you: 0.2846)
     - ihm-ihm   (test file recorded on the close headset mic)
     - ihm-sdm   (test file recorded on the far single distant mic)
   This answers the open question from the 0.2846 result: is the far
   mic dragging the blended number up?

HOW ihm vs sdm IS DETERMINED
--------------------------------
Enroll is always ihm (by this project's meeting-split rule -- see
README.md section 3.3), so a trial's mic-combo is fully decided by its
TEST file alone: "/ihm/" in the path means ihm-ihm, "/sdm/" means
ihm-sdm. Same rule compute_positive_mic_ratio.py already used for the
duration-bin mic-ratio matching. Works whether the trial table stores
the test file as a raw id (positive_trials.parquet) or a category path
string (negative_trials_v2.joblib) -- both get resolved to a file id
via path_lookup.csv first, then id_is_sdm[] (built once from
path_lookup.csv) tells you the mic for that id.

INPUT
-----
Reads positive_cosine_scores.npy / negative_cosine_scores.npy (already
computed) PLUS the test-file column from positive_trials.parquet /
negative_trials_v2.joblib (only that one column each, not the whole
table) to know each row's mic-combo. Row order must still match --
same alignment rule as compute_cosine_scores.py.

Usage:
    python cosine_eer_mic_breakdown.py \\
        encoded_trials/positive_trials.parquet \\
        trial_outputs/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        positive_cosine_scores.npy \\
        negative_cosine_scores.npy \\
        --out-prefix cosine_eer
"""

import argparse
import csv
import sys
from pathlib import Path


def load_path_lookup(path_lookup_csv):
    """Returns id_to_path list AND id_is_sdm boolean array (built once,
    reused for both tables -- same 'small dictionary, not the big
    table' pattern as every join script in this project)."""
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
    id_to_path_list = [id_to_path.get(i, "") for i in range(max_id + 1)]
    return id_to_path_list


def find_pair_columns(columns, label):
    cols_lower = {c.lower(): c for c in columns}
    for suffix, key_type in (("path", "path"), ("id", "id")):
        enroll_candidates = [c for lc, c in cols_lower.items()
                              if "enroll" in lc and lc.endswith(suffix)]
        test_candidates = [c for lc, c in cols_lower.items()
                            if "test" in lc and lc.endswith(suffix)]
        if len(enroll_candidates) == 1 and len(test_candidates) == 1:
            return enroll_candidates[0], test_candidates[0], key_type
    print(f"\nERROR: could not identify test id/path column in {label}.\n"
          f"  Columns found: {list(columns)}", file=sys.stderr)
    sys.exit(1)


def get_test_is_sdm(df, test_col, key_type, path_to_id, id_is_sdm, np):
    """Returns a boolean array, one per row: True = ihm-sdm trial,
    False = ihm-ihm trial."""
    if key_type == "id":
        ids = df[test_col].to_numpy().astype(np.int64)
        valid = (ids >= 0) & (ids < len(id_is_sdm))
        ids_safe = np.where(valid, ids, 0)
        return id_is_sdm[ids_safe], valid
    else:
        col = df[test_col]
        codes = col.cat.codes.to_numpy()
        categories = col.cat.categories
        cat_is_sdm = np.array(["/sdm/" in c for c in categories], dtype=bool)
        cat_valid = np.array([c in path_to_id for c in categories], dtype=bool)
        safe_codes = np.clip(codes, 0, len(cat_is_sdm) - 1)
        valid = (codes >= 0) & cat_valid[safe_codes]
        return cat_is_sdm[safe_codes], valid


def compute_eer(target_scores, nontarget_scores, np):
    n_tar, n_non = len(target_scores), len(nontarget_scores)
    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(n_tar, dtype=np.int8), np.zeros(n_non, dtype=np.int8)])
    order = np.argsort(-scores, kind="stable")
    labels_sorted = labels[order]
    cum_tar = np.cumsum(labels_sorted == 1)
    cum_non = np.cumsum(labels_sorted == 0)
    far = cum_non / n_non
    frr = (n_tar - cum_tar) / n_tar
    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2.0
    return float(eer)


def gaussian_kde_1d(samples, x_grid, np):
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


def plot_hist_kde(pos, neg, eer, title, out_path, np, plt, rng, sample_size, dpi):
    """Matches the style of the reference image: filled histogram bars
    for both classes plus a smooth KDE line on top, 'True Scores' /
    'False Scores' legend."""
    combined = np.concatenate([pos, neg])
    x_min = float(np.percentile(combined, 0.5))
    x_max = float(np.percentile(combined, 99.5))
    bins = np.linspace(x_min, x_max, 51)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(pos, bins=bins, density=True, color="#3B5FA0", alpha=0.55,
            edgecolor="black", linewidth=0.4, label="True Scores")
    ax.hist(neg, bins=bins, density=True, color="#C0392B", alpha=0.55,
            edgecolor="black", linewidth=0.4, label="False Scores")

    x_grid = np.linspace(x_min, x_max, 400)
    pos_sample = rng.choice(pos, size=min(sample_size, len(pos)), replace=False)
    neg_sample = rng.choice(neg, size=min(sample_size, len(neg)), replace=False)
    ax.plot(x_grid, gaussian_kde_1d(pos_sample, x_grid, np), color="#2B4A86", linewidth=2.2)
    ax.plot(x_grid, gaussian_kde_1d(neg_sample, x_grid, np), color="#8E2419", linewidth=2.2)

    ax.set_title(f"{title}  (EER = {eer*100:.2f}%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("positive_scores_npy")
    ap.add_argument("negative_scores_npy")
    ap.add_argument("--out-prefix", default="cosine_eer")
    ap.add_argument("--sample-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    for p in (args.positive_trials_parquet, args.negative_trials_joblib,
              args.path_lookup_csv, args.positive_scores_npy, args.negative_scores_npy):
        if not Path(p).is_file():
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
              f"    pip install pandas numpy joblib matplotlib pyarrow --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.path_lookup_csv} ...", file=sys.stderr)
    id_to_path = load_path_lookup(args.path_lookup_csv)
    id_is_sdm = np.array(["/sdm/" in p for p in id_to_path], dtype=bool)
    path_to_id = {p: i for i, p in enumerate(id_to_path)}
    print(f"  {len(id_to_path):,} ids.", file=sys.stderr)

    print(f"\nLoading {args.positive_scores_npy} / {args.negative_scores_npy} ...", file=sys.stderr)
    pos_scores = np.load(args.positive_scores_npy)
    neg_scores = np.load(args.negative_scores_npy)

    print(f"\nLoading test-file column from {args.positive_trials_parquet} ...", file=sys.stderr)
    pos_df = pd.read_parquet(args.positive_trials_parquet)
    _, pos_test_col, pos_key_type = find_pair_columns(pos_df.columns, "positive_trials")
    pos_is_sdm, pos_valid = get_test_is_sdm(pos_df, pos_test_col, pos_key_type, path_to_id, id_is_sdm, np)
    del pos_df

    print(f"Loading test-file column from {args.negative_trials_joblib} ...", file=sys.stderr)
    neg_df = joblib.load(args.negative_trials_joblib)
    _, neg_test_col, neg_key_type = find_pair_columns(neg_df.columns, "negative_trials_v2")
    neg_is_sdm, neg_valid = get_test_is_sdm(neg_df, neg_test_col, neg_key_type, path_to_id, id_is_sdm, np)
    del neg_df

    pos_ok = ~np.isnan(pos_scores) & pos_valid
    neg_ok = ~np.isnan(neg_scores) & neg_valid

    groups = {
        "overall": (pos_scores[pos_ok], neg_scores[neg_ok]),
        "ihm_ihm": (pos_scores[pos_ok & ~pos_is_sdm], neg_scores[neg_ok & ~neg_is_sdm]),
        "ihm_sdm": (pos_scores[pos_ok & pos_is_sdm], neg_scores[neg_ok & neg_is_sdm]),
    }

    print()
    results = {}
    for name, (pos, neg) in groups.items():
        eer = compute_eer(pos.astype(np.float64), neg.astype(np.float64), np)
        results[name] = eer
        print(f"  {name:10s}  Target={len(pos):,}  Non-Target={len(neg):,}  EER={eer*100:.2f}%",
              file=sys.stderr)

    titles = {
        "overall": "Cosine Similarity -- All Trials",
        "ihm_ihm": "Cosine Similarity -- ihm-ihm Trials Only",
        "ihm_sdm": "Cosine Similarity -- ihm-sdm Trials Only",
    }
    for name, (pos, neg) in groups.items():
        out_path = Path(f"{args.out_prefix}_{name}.png")
        plot_hist_kde(pos.astype(np.float64), neg.astype(np.float64), results[name],
                      titles[name], out_path, np, plt, rng, args.sample_size, args.dpi)
        print(f"  saved {out_path.resolve()}", file=sys.stderr)

    print(f"\nSummary: overall EER={results['overall']*100:.2f}%, "
          f"ihm-ihm EER={results['ihm_ihm']*100:.2f}%, "
          f"ihm-sdm EER={results['ihm_sdm']*100:.2f}%", file=sys.stderr)


if __name__ == "__main__":
    main()