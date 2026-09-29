#!/usr/bin/env python3
"""
compute_positive_mic_ratio.py

Answers one question that nothing else in this project has measured yet:
inside each of the 50 duration_diff bins, how many REAL positive trials
are ihm-ihm versus ihm-sdm?

This has to be measured before the negative-trial script can match that
ratio -- right now it isn't stored anywhere, because create_positive_trials.py
only ever split trials by duration, never by mic-combo.

HOW IT WORKS
------------
Every row in positive_trials.csv has an enroll_path and a test_path.
Enroll is always ihm (by construction of the Enroll sheet -- see the
project's meeting-split rule), so the mic-combo of a trial is fully
decided by the test_path alone:

    "/ihm/" appears in test_path   ->  this trial is ihm-ihm
    "/sdm/" appears in test_path   ->  this trial is ihm-sdm

This script streams positive_trials.csv one row at a time (it never
loads the whole multi-GB file into memory), puts each row into the
SAME 50 equal-width bins already used everywhere else in this project
(read directly from positive_trial_full_distribution_stats.json, not
re-derived here), and counts ihm-ihm vs ihm-sdm separately inside each
bin.

OUTPUT
------
positive_mic_ratio_stats.json -- one entry per bin:
    bin_index, lower_seconds, upper_seconds,
    total_count, ihm_ihm_count, ihm_sdm_count

Also cross-checks that ihm_ihm_count + ihm_sdm_count exactly equals the
count already recorded for that bin in
positive_trial_full_distribution_stats.json -- if it doesn't, something
is wrong and this script says so loudly instead of writing a silently
bad file.

Usage:
    python compute_positive_mic_ratio.py \\
        positive_trials.csv \\
        positive_trial_full_distribution_stats.json \\
        --out positive_mic_ratio_stats.json
"""

import argparse
import bisect
import csv
import json
import sys
import time
from pathlib import Path


def load_bin_edges(stats_json_path: Path):
    with open(stats_json_path) as f:
        stats = json.load(f)
    bins = stats["equal_width_bins"]["bins"]
    lowers = [b["lower_seconds"] for b in bins]
    uppers = [b["upper_seconds"] for b in bins]
    pos_counts = [b["count"] for b in bins]
    return lowers, uppers, pos_counts


def find_bin(diff_s, lowers, max_upper):
    """Same rule as create_negative_trials.py: half-open [lower, upper)
    except the last bin, which also includes its own upper edge."""
    if diff_s > max_upper:
        return None
    idx = bisect.bisect_right(lowers, diff_s) - 1
    if idx < 0:
        idx = 0
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_csv")
    ap.add_argument("positive_stats_json",
                     help="positive_trial_full_distribution_stats.json")
    ap.add_argument("--out", default="positive_mic_ratio_stats.json")
    ap.add_argument("--progress-every", type=int, default=2_000_000)
    args = ap.parse_args()

    csv_path = Path(args.positive_trials_csv)
    stats_path = Path(args.positive_stats_json)
    for p in (csv_path, stats_path):
        if not p.is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading bin edges from {stats_path}...", file=sys.stderr)
    lowers, uppers, pos_counts = load_bin_edges(stats_path)
    n_bins = len(lowers)
    max_upper = uppers[-1]
    print(f"  {n_bins} bins, 0.0s to {max_upper:.2f}s", file=sys.stderr)

    ihm_ihm = [0] * n_bins
    ihm_sdm = [0] * n_bins
    n_other = 0            # test_path matched neither marker (should be 0)
    n_out_of_range = 0
    n_rows = 0

    print(f"\nStreaming {csv_path} ...", file=sys.stderr)
    t0 = time.time()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_test = header.index("test_path")
        idx_diff = header.index("duration_diff")

        for row in reader:
            n_rows += 1
            test_path = row[idx_test]
            diff_s = float(row[idx_diff])

            b_idx = find_bin(diff_s, lowers, max_upper)
            if b_idx is None:
                n_out_of_range += 1
            elif "/ihm/" in test_path:
                ihm_ihm[b_idx] += 1
            elif "/sdm/" in test_path:
                ihm_sdm[b_idx] += 1
            else:
                n_other += 1

            if n_rows % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = n_rows / elapsed if elapsed > 0 else 0
                print(f"  ... {n_rows:,} rows read, {rate:,.0f}/s",
                      file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone. {n_rows:,} rows read in {elapsed/60:.1f} minutes.",
          file=sys.stderr)
    if n_other:
        print(f"WARNING: {n_other:,} rows had a test_path matching "
              f"neither '/ihm/' nor '/sdm/' -- check the path format "
              f"before trusting this output.", file=sys.stderr)
    if n_out_of_range:
        print(f"NOTE: {n_out_of_range:,} rows fell outside the known bin "
              f"range (expected to be 0 for positives; flagging just in "
              f"case).", file=sys.stderr)

    bins_report = []
    mismatches = 0
    for i in range(n_bins):
        total = ihm_ihm[i] + ihm_sdm[i]
        if total != pos_counts[i]:
            mismatches += 1
        bins_report.append({
            "bin_index": i,
            "lower_seconds": lowers[i],
            "upper_seconds": uppers[i],
            "total_count": total,
            "ihm_ihm_count": ihm_ihm[i],
            "ihm_sdm_count": ihm_sdm[i],
            "stats_json_count": pos_counts[i],
        })

    if mismatches:
        print(f"WARNING: {mismatches} bin(s) do not exactly match the "
              f"counts already in {stats_path}. A small mismatch can come "
              f"from floating point edge rounding; a large one means "
              f"something is wrong -- check before trusting the ratio.",
              file=sys.stderr)
    else:
        print("Cross-check passed: every bin's total_count matches "
              "positive_trial_full_distribution_stats.json exactly.",
              file=sys.stderr)

    report = {
        "source_csv": str(csv_path.resolve()),
        "n_rows": n_rows,
        "n_bins": n_bins,
        "n_out_of_range": n_out_of_range,
        "n_unrecognized_mic": n_other,
        "bins": bins_report,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()