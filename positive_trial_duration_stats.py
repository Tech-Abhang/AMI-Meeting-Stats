#!/usr/bin/env python3
"""
positive_trial_duration_stats.py — reads positive_trials.csv (produced by
create_positive_trials.py, 3 columns: enroll_path, test_path, duration_diff)
and reports the distribution and summary statistics of the duration_diff
column, plus a histogram image.

Streams the CSV once (a csv.reader, not pandas) collecting duration_diff
into a compact array.array('f') buffer -- for 27M rows that's ~110MB, not
the ~1-2GB a plain Python list of floats would cost, so this stays well
within memory on an ordinary machine even at real corpus scale.

OUTPUTS:
  - printed summary to stdout
  - <out-prefix>_stats.json   -- count, mean, std, min, max, median,
                                  percentiles (10/25/50/75/90/95/99), plus
                                  the <1 / 1-3 / 3-10 / >10 breakdown for
                                  continuity with the earlier bucket scheme
                                  (computed live from the raw numbers now,
                                  not stored in the CSV)
  - <out-prefix>_histogram.png -- the distribution graph

Usage:
    pip install matplotlib numpy --break-system-packages
    python positive_trial_duration_stats.py positive_trials.csv
    python positive_trial_duration_stats.py positive_trials.csv --out-prefix mystats --bins 60
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


def bucket_counts_from(np_arr):
    return {
        "<1": int((np_arr < 1).sum()),
        "1-3": int(((np_arr >= 1) & (np_arr < 3)).sum()),
        "3-10": int(((np_arr >= 3) & (np_arr < 10)).sum()),
        ">10": int((np_arr >= 10).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to positive_trials.csv")
    ap.add_argument("--out-prefix", default="positive_trial_duration",
                     help="Prefix for the output files (stats JSON + histogram PNG)")
    ap.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default 50)")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
    except ImportError:
        print("Missing dependency. Run: pip install numpy --break-system-packages", file=sys.stderr)
        sys.exit(1)

    print("Reading duration_diff column...", file=sys.stderr)
    diffs, n_total, n_blank = load_diffs(csv_path)
    n_valid = len(diffs)

    if n_valid == 0:
        print("ERROR: no valid duration_diff values found -- nothing to report.", file=sys.stderr)
        sys.exit(1)

    arr = np.frombuffer(diffs, dtype=np.float32)

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
        "percentiles": {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        },
        "bucket_breakdown": bucket_counts_from(arr),
    }

    stats_path = Path(f"{args.out_prefix}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nStats written to {stats_path}", file=sys.stderr)

    # ---- histogram (dataviz skill: single-series -> one sequential hue,
    # light chart surface, hairline gridlines, no legend needed) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE = "#fcfcfb"
    SERIES = "#2a78d6"
    GRID = "#e1e0d9"
    INK = "#0b0b0b"
    MUTED = "#898781"

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    counts, bin_edges, _ = ax.hist(arr, bins=args.bins, color=SERIES, edgecolor=SURFACE, linewidth=0.3)

    mean_v = stats["mean"]
    median_v = stats["median"]
    ax.axvline(mean_v, color=INK, linewidth=1.2, linestyle="--")
    ax.text(mean_v, ax.get_ylim()[1] * 0.97, f" mean {mean_v:.2f}s", color=INK, fontsize=9, va="top")
    if abs(median_v - mean_v) > (arr.max() - arr.min()) * 0.01:
        ax.axvline(median_v, color=MUTED, linewidth=1.0, linestyle=":")
        ax.text(median_v, ax.get_ylim()[1] * 0.90, f" median {median_v:.2f}s", color=MUTED, fontsize=9, va="top")

    ax.set_title("Distribution of duration_diff across positive trials", color=INK, fontsize=13, pad=12)
    ax.set_xlabel("|enroll duration - test duration| (seconds)", color=MUTED, fontsize=10)
    ax.set_ylabel("number of trials", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    png_path = Path(f"{args.out_prefix}_histogram.png")
    fig.savefig(png_path, facecolor=SURFACE)
    print(f"Histogram written to {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()