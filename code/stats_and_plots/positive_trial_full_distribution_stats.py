#!/usr/bin/env python3
"""
positive_trial_full_distribution_stats.py

Computes BOTH ways of binning the duration_diff column from
positive_trials.csv, so you have everything needed to decide how to
duration-match negatives, without guessing:

  1. quantile_bins  -- equal-COUNT bins (edges = percentiles). Each bin
     holds roughly the same number of positive trials. Width in seconds
     varies: narrow where the data is dense (small diffs), wide where the
     data is sparse (the long tail of large diffs). This is the "ideal"
     view for distribution matching -- no bin is starved of data.

  2. equal_width_bins -- fixed-size slices in seconds (e.g. 50 of them),
     same idea as a plain histogram. Easy to read in seconds, but the
     tail bins end up nearly empty because most trials cluster near 0.

Nothing is dropped or assumed -- summary stats, a fine percentile grid,
and both full bin tables (edges + counts) all go into ONE json file, so
you can slice the data differently later without re-running this script.
Also draws ONE png with both views stacked, so you can see the difference
directly:
  - top:    the equal-width histogram, with the quantile-bin edges drawn
            as thin orange dashed lines on top of it (shows where the
            equal-count bins land relative to the plain histogram)
  - bottom: a bar chart of each quantile bin's width in seconds (shows
            how much the equal-count bins stretch out in the tail)

Uses only the standard library + numpy + matplotlib (no pandas), and
streams the CSV once into a compact array.array('f') buffer, same as
positive_trial_duration_stats.py -- safe on 27M+ rows without blowing up
memory.

Usage:
    pip install numpy matplotlib --break-system-packages
    python positive_trial_full_distribution_stats.py positive_trials.csv
    python positive_trial_full_distribution_stats.py positive_trials.csv \
        --out-prefix mystats --quantile-bins 20 --equal-width-bins 50
"""

import argparse
import array
import csv
import json
import sys
from pathlib import Path

EXPECTED_HEADER = ["enroll_path", "test_path", "duration_diff"]


def load_diffs(csv_path: Path):
    diffs = array.array("f")
    n_total = 0
    n_blank = 0

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != EXPECTED_HEADER:
            print(f"WARNING: header is {header}, expected {EXPECTED_HEADER}. "
                  f"Proceeding anyway, assuming column 3 (index 2) is duration_diff.",
                  file=sys.stderr)
        diff_col = header.index("duration_diff") if "duration_diff" in header else 2

        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            n_total += 1
            val = row[diff_col]
            if val == "" or val is None:
                n_blank += 1
                continue
            try:
                diffs.append(float(val))
            except ValueError:
                n_blank += 1
            if i % 5_000_000 == 0:
                print(f"  ... {i:,} rows read so far", file=sys.stderr)

    return diffs, n_total, n_blank


def build_quantile_bins(arr, n_bins):
    """Equal-COUNT bins: edges are percentiles spaced n_bins apart."""
    import numpy as np
    raw_edges = np.percentile(arr, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(raw_edges)  # guard: if many trials share an exact value,
                                   # some percentile edges can collide -- drop dupes
    counts, edges = np.histogram(arr, bins=edges)
    bins = []
    for i in range(len(counts)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        bins.append({
            "index": i,
            "lower_seconds": round(lo, 4),
            "upper_seconds": round(hi, 4),
            "width_seconds": round(hi - lo, 4),
            "count": int(counts[i]),
        })
    if len(edges) - 1 < n_bins:
        print(f"NOTE: requested {n_bins} quantile bins but only {len(edges) - 1} "
              f"distinct edges exist (some percentile values are identical) -- "
              f"this is normal near duration_diff = 0.0 where many trials tie.",
              file=sys.stderr)
    return bins, edges


def build_equal_width_bins(arr, n_bins):
    """Equal-WIDTH bins: fixed-size slices in seconds from min to max."""
    import numpy as np
    counts, edges = np.histogram(arr, bins=n_bins)
    bins = []
    for i in range(len(counts)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        bins.append({
            "index": i,
            "lower_seconds": round(lo, 4),
            "upper_seconds": round(hi, 4),
            "width_seconds": round(hi - lo, 4),
            "count": int(counts[i]),
        })
    return bins, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to positive_trials.csv")
    ap.add_argument("--out-prefix", default="positive_trial_full_distribution",
                     help="Prefix for the output files (json + png)")
    ap.add_argument("--quantile-bins", type=int, default=20,
                     help="Number of equal-COUNT bins (default 20 = ventiles)")
    ap.add_argument("--equal-width-bins", type=int, default=50,
                     help="Number of equal-WIDTH bins (default 50)")
    ap.add_argument("--percentile-step", type=int, default=5,
                     help="Step for the fine percentile grid recorded in the json "
                          "(default 5 -> p5,p10,...,p95, plus p1 and p99 always included)")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
    except ImportError:
        print("Missing dependency. Run: pip install numpy matplotlib --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print("Reading duration_diff column...", file=sys.stderr)
    diffs, n_total, n_blank = load_diffs(csv_path)
    n_valid = len(diffs)

    if n_valid == 0:
        print("ERROR: no valid duration_diff values found -- nothing to report.", file=sys.stderr)
        sys.exit(1)

    arr = np.frombuffer(diffs, dtype=np.float32)

    print("Computing fine percentile grid...", file=sys.stderr)
    grid_points = sorted(set([1] + list(range(args.percentile_step, 100, args.percentile_step)) + [99]))
    percentiles = {f"p{p}": float(np.percentile(arr, p)) for p in grid_points}

    print(f"Computing {args.quantile_bins} quantile (equal-count) bins...", file=sys.stderr)
    quantile_bins, q_edges = build_quantile_bins(arr, args.quantile_bins)

    print(f"Computing {args.equal_width_bins} equal-width bins...", file=sys.stderr)
    equal_width_bins, ew_edges = build_equal_width_bins(arr, args.equal_width_bins)

    stats = {
        "input_csv": str(csv_path.resolve()),
        "total_rows": n_total,
        "valid_duration_diff_rows": n_valid,
        "blank_or_unparseable_rows": n_blank,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "percentiles": percentiles,
        "quantile_bins": {
            "description": ("Equal-COUNT bins -- each bin holds roughly the same number "
                             "of positive trials. Width in seconds varies: narrow where "
                             "data is dense (small diffs), wide where data is sparse "
                             "(large diffs). Use these edges/counts as the target when "
                             "duration-matching negatives -- no bin is starved of data "
                             "by construction."),
            "n_bins_requested": args.quantile_bins,
            "n_bins_actual": len(quantile_bins),
            "bins": quantile_bins,
        },
        "equal_width_bins": {
            "description": ("Equal-WIDTH bins -- fixed seconds-wide slices from min to "
                             "max. Easy to read in plain seconds, but counts vary a lot: "
                             "packed near small diffs, nearly empty far out in the tail. "
                             "Kept here for comparison against the quantile bins above."),
            "n_bins": args.equal_width_bins,
            "bins": equal_width_bins,
        },
    }

    stats_path = Path(f"{args.out_prefix}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to {stats_path}", file=sys.stderr)

    # ---- diagram: two stacked views (dataviz skill: single blue series for
    # the histogram, orange as the secondary/accent series for the quantile
    # view, light surface, hairline gridlines, no legend needed) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE = "#fcfcfb"
    SERIES = "#2a78d6"
    ACCENT = "#eb6834"
    GRID = "#e1e0d9"
    INK = "#0b0b0b"
    MUTED = "#898781"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # top: equal-width histogram with quantile-bin edges overlaid
    ax1.set_facecolor(SURFACE)
    ax1.hist(arr, bins=args.equal_width_bins, color=SERIES, edgecolor=SURFACE,
              linewidth=0.3, zorder=2)
    for edge in q_edges[1:-1]:
        ax1.axvline(float(edge), color=ACCENT, linewidth=0.6, linestyle="--",
                     alpha=0.6, zorder=3)
    ax1.set_title(f"Equal-width histogram ({args.equal_width_bins} bins) with "
                   f"{len(quantile_bins)} quantile-bin edges overlaid (orange)",
                   color=INK, fontsize=12, pad=10)
    ax1.set_xlabel("duration_diff (seconds)", color=MUTED, fontsize=10)
    ax1.set_ylabel("number of trials", color=MUTED, fontsize=10)
    ax1.tick_params(colors=MUTED, labelsize=9)
    ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.spines["bottom"].set_color(GRID)

    # bottom: quantile bin widths -- shows how width shrinks/grows across bins
    ax2.set_facecolor(SURFACE)
    widths = [b["width_seconds"] for b in quantile_bins]
    idx = list(range(len(widths)))
    ax2.bar(idx, widths, color=ACCENT, zorder=2)
    ax2.set_title("Width (seconds) of each equal-count quantile bin -- "
                   "short bar = dense region, tall bar = sparse tail",
                   color=INK, fontsize=12, pad=10)
    ax2.set_xlabel("quantile bin index (each bin holds ~the same number of trials)",
                    color=MUTED, fontsize=10)
    ax2.set_ylabel("bin width (seconds)", color=MUTED, fontsize=10)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    png_path = Path(f"{args.out_prefix}_diagram.png")
    fig.savefig(png_path, facecolor=SURFACE)
    print(f"Diagram written to {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()