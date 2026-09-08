#!/usr/bin/env python3
"""
plot_distribution_bins.py

Draws TWO separate, large, self-explanatory diagrams from the json that
positive_trial_full_distribution_stats.py already produced -- one for the
50 equal-width bins, one for the 20 quantile (equal-count) bins. Does NOT
re-read the 27M-row CSV -- all the numbers it needs are already sitting in
the stats json, so this runs in under a second.

OUTPUTS (two files, not one combined figure -- each stands on its own):
  <out-prefix>_equal_width.png
      - log-scale y-axis, because counts span 5 orders of magnitude
        (7 million down to 82) -- on a normal (linear) axis, every bin
        past the first few would look like a flat line at zero
      - every one of the 50 bars is labeled with its real count so small
        bars are still readable even when they're too short to judge by
        eye
      - mean/median marked as reference lines

  <out-prefix>_quantile.png
      - bars are drawn at their TRUE width in seconds (not evenly spaced)
        so you can see with your own eyes how bin 0 is a sliver and the
        last bin swallows a huge stretch of seconds -- that visual IS the
        explanation for why quantile bins trade fine resolution in the
        tail for a reliable count
      - every bar labeled with its count and its width in seconds
      - the last bin (the wide one) is outlined in red and annotated,
        since that is the bin most likely to cause trouble when you use
        it as a negative-sampling target
      - mean/median marked as reference lines

Usage:
    pip install matplotlib --break-system-packages
    python plot_distribution_bins.py positive_trial_full_distribution_stats.json
    python plot_distribution_bins.py positive_trial_full_distribution_stats.json --out-prefix mydiagrams
"""

import argparse
import json
import sys
from pathlib import Path


SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
CRITICAL = "#d03b3b"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"


def fmt_count(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def plot_equal_width(stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = stats["equal_width_bins"]["bins"]
    n = len(bins)
    centers = [(b["lower_seconds"] + b["upper_seconds"]) / 2 for b in bins]
    widths = [b["width_seconds"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(22, 11), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(centers, counts, width=[w * 0.92 for w in widths],
                   color=SERIES_BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    ax.set_yscale("log")

    # label every single bar with its real count -- log scale compresses
    # tall bars, so without this the tiny tail bars are unreadable
    for bar, c in zip(bars, counts):
        if c <= 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, c * 1.15, fmt_count(c),
                 rotation=90, ha="center", va="bottom", fontsize=7.5, color=INK)

    mean_v, median_v = stats["mean"], stats["median"]
    ax.axvline(mean_v, color=INK, linewidth=1.4, linestyle="--", zorder=3)
    ax.axvline(median_v, color=SERIES_ORANGE, linewidth=1.4, linestyle=":", zorder=3)
    ymax = ax.get_ylim()[1]
    ax.text(mean_v, ymax * 0.55, f" mean {mean_v:.2f}s", color=INK, fontsize=11,
             va="top", fontweight="bold")
    ax.text(median_v, ymax * 0.30, f" median {median_v:.2f}s", color=SERIES_ORANGE,
             fontsize=11, va="top", fontweight="bold")

    ax.set_title(
        f"Positive trials by EQUAL-WIDTH duration_diff bin  "
        f"({n} bins, each {widths[0]:.4f}s wide)  --  log scale y-axis\n"
        f"Every bin is the same width in seconds, but the data is NOT spread evenly: "
        f"bin 0 alone holds {fmt_count(counts[0])} trials, the smallest tail bins hold "
        f"only dozens.",
        color=INK, fontsize=15, pad=18)
    ax.set_xlabel("duration_diff (seconds)", color=MUTED, fontsize=12)
    ax.set_ylabel("number of positive trials (log scale)", color=MUTED, fontsize=12)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", which="both", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_quantile(stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = stats["quantile_bins"]["bins"]
    n = len(bins)
    lowers = [b["lower_seconds"] for b in bins]
    widths = [b["width_seconds"] for b in bins]
    counts = [b["count"] for b in bins]
    last = n - 1
    max_c = max(counts)

    # The last bin is far wider (seconds) than all the others combined --
    # drawing every bin on one continuous true-width x-axis would let that
    # one bar eat ~90% of the canvas and crush bins 0..n-2 into an
    # unreadable sliver. So this is a BROKEN x-axis: left panel = bins
    # 0..n-2 at true width (still varies, but now readable), right panel =
    # just the last bin, drawn in its own space. A diagonal break mark
    # between them signals the scale is not continuous.
    fig = plt.figure(figsize=(22, 11), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.04)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)

    # --- left panel: bins 0..n-2, true width, real seconds axis ---
    bodyL = axL.bar(lowers[:last], counts[:last], width=widths[:last], align="edge",
                     color=SERIES_ORANGE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    for bar, c in zip(bodyL, counts[:last]):
        cx = bar.get_x() + bar.get_width() / 2
        axL.text(cx, c + max_c * 0.02, fmt_count(c), ha="center", va="bottom",
                  rotation=90, fontsize=8, color=INK, fontweight="bold")
    axL.set_xlim(lowers[0], lowers[last])

    # --- right panel: just the last (wide) bin ---
    axR.bar([0], [counts[last]], width=[1], align="edge",
            color=SERIES_ORANGE, edgecolor=CRITICAL, linewidth=3.0, zorder=2)
    axR.text(0.5, counts[last] + max_c * 0.02, fmt_count(counts[last]),
              ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    axR.set_xlim(0, 1)
    axR.set_xticks([0.5])
    axR.set_xticklabels([f"{lowers[last]:.2f}s\nto\n{lowers[last] + widths[last]:.2f}s"])
    axR.annotate(
        f"bin {last}: {widths[last]:.2f} seconds wide\njust to collect "
        f"{fmt_count(counts[last])} trials\n\n(positive data runs thin past "
        f"{lowers[last]:.2f}s --\nthis bin trades resolution\nfor a reliable count)",
        xy=(0.5, counts[last] * 0.5), xycoords="data",
        ha="center", va="center", fontsize=10.5, color=CRITICAL, fontweight="bold")

    mean_v, median_v = stats["mean"], stats["median"]
    axL.axvline(mean_v, color=INK, linewidth=1.4, linestyle="--", zorder=3)
    axL.axvline(median_v, color=SERIES_BLUE, linewidth=1.4, linestyle=":", zorder=3)
    ymax = max_c * 1.25
    axL.text(mean_v, ymax * 0.97, f" mean {mean_v:.2f}s", color=INK, fontsize=11,
              va="top", fontweight="bold")
    axL.text(median_v, ymax * 0.90, f" median {median_v:.2f}s", color=SERIES_BLUE,
              fontsize=11, va="top", fontweight="bold")
    axL.set_ylim(0, ymax)

    fig.suptitle(
        f"Positive trials by QUANTILE (equal-COUNT) duration_diff bin  ({n} bins)\n"
        f"Every bar holds roughly the same number of trials (~{fmt_count(int(sum(counts)/n))} each) "
        f"-- bar WIDTH stretches out for the sparse, high-diff bins instead of the count shrinking.\n"
        f"Right panel is a separate, broken-axis zoom on just the last (widest) bin -- "
        f"drawn at true width it would swallow the whole chart.",
        color=INK, fontsize=14.5, y=1.02)

    axL.set_xlabel("duration_diff (seconds) -- bar width IS the real seconds range of that bin",
                    color=MUTED, fontsize=12)
    axL.set_ylabel("number of positive trials", color=MUTED, fontsize=12)
    axR.set_xlabel("bin 19 only\n(broken axis)", color=MUTED, fontsize=10)

    for ax in (axL, axR):
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axR.tick_params(labelleft=False)

    # diagonal break marks between the two panels
    d = 0.012
    kwargs = dict(color=MUTED, clip_on=False, linewidth=1.2)
    axL.plot((1 - d, 1 + d), (-d, +d), transform=axL.transAxes, **kwargs)
    axL.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axL.transAxes, **kwargs)
    axR.plot((-d * 3, +d * 3), (-d, +d), transform=axR.transAxes, **kwargs)
    axR.plot((-d * 3, +d * 3), (1 - d, 1 + d), transform=axR.transAxes, **kwargs)

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats_json", help="Path to positive_trial_full_distribution_stats.json")
    ap.add_argument("--out-prefix", default="distribution_bins")
    args = ap.parse_args()

    stats_path = Path(args.stats_json)
    if not stats_path.is_file():
        print(f"ERROR: {stats_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    for key in ("equal_width_bins", "quantile_bins"):
        if key not in stats:
            print(f"ERROR: '{key}' missing from {stats_path} -- "
                  f"this must be the json produced by positive_trial_full_distribution_stats.py",
                  file=sys.stderr)
            sys.exit(1)

    ew_path = Path(f"{args.out_prefix}_equal_width.png")
    q_path = Path(f"{args.out_prefix}_quantile.png")

    print("Drawing equal-width diagram...", file=sys.stderr)
    plot_equal_width(stats, ew_path)
    print(f"  -> {ew_path}", file=sys.stderr)

    print("Drawing quantile diagram...", file=sys.stderr)
    plot_quantile(stats, q_path)
    print(f"  -> {q_path}", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()