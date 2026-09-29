#!/usr/bin/env python3
"""
plot_pos_neg_mic_comparison.py

Two SEPARATE large graphs, each answering one of the two things you
asked about:

  Graph 1 (positive_vs_negative_counts_ratio.png)
      Top    : positive trial count vs negative trial count, per bin,
               log scale (counts span ~7,000,000 down to a few hundred,
               so log scale keeps the tail bins visible)
      Bottom : negative/positive ratio per bin, with a dashed line at
               the target ratio (e.g. 10x) -- bars sitting exactly on
               the dashed line means every bin hit its target with zero
               shortfall

  Graph 2 (negative_mic_ratio_match.png)
      Top    : ihm-ihm count vs ihm-sdm count, positives vs negatives,
               per bin, log scale -- the raw numbers behind the split
      Bottom : ihm-ihm SHARE of the total (ihm-ihm / (ihm-ihm+ihm-sdm)),
               positives vs negatives, per bin, as two lines -- lines
               sitting exactly on top of each other means the mic-combo
               ratio was preserved

INPUT
-----
Only negative_trials_v2_report.json is needed -- it already carries the
positive counts (positive_total, positive_ihm_ihm, positive_ihm_sdm)
alongside the negative actuals for every one of the 50 bins.

Usage:
    python plot_pos_neg_mic_comparison.py negative_trials_v2_report.json \\
        --out-prefix pos_neg
    (writes pos_neg_counts_ratio.png and pos_neg_mic_ratio.png)
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json", help="negative_trials_v2_report.json")
    ap.add_argument("--out-prefix", default="pos_neg",
                     help="Output files are <prefix>_counts_ratio.png and "
                          "<prefix>_mic_ratio.png (default 'pos_neg')")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    report_path = Path(args.report_json)
    if not report_path.is_file():
        print(f"ERROR: {report_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing dependency. Run: pip install matplotlib --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    bins = report["bins"]
    ratio = report["ratio"]
    n_bins = len(bins)

    bin_idx = [b["bin_index"] for b in bins]
    lower = [b["lower_seconds"] for b in bins]
    upper = [b["upper_seconds"] for b in bins]

    pos_total = [b["positive_total"] for b in bins]
    pos_ihm_ihm = [b["positive_ihm_ihm"] for b in bins]
    pos_ihm_sdm = [b["positive_ihm_sdm"] for b in bins]

    neg_total = [b["actual_total"] for b in bins]
    neg_ihm_ihm = [b["actual_ihm_ihm"] for b in bins]
    neg_ihm_sdm = [b["actual_ihm_sdm"] for b in bins]

    neg_pos_ratio = [(n / p if p else 0.0) for n, p in zip(neg_total, pos_total)]
    pos_ihm_share = [(i / t if t else 0.0) for i, t in zip(pos_ihm_ihm, pos_total)]
    neg_ihm_share = [(i / t if t else 0.0) for i, t in zip(neg_ihm_ihm, neg_total)]

    xtick_pos = sorted(set(list(range(0, n_bins, 5)) + [n_bins - 1]))
    xtick_labels = [f"{lower[i]:.1f}-{upper[i]:.1f}s" for i in xtick_pos]

    def apply_xticks(ax):
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=60, ha="right", fontsize=10)
        ax.set_xlabel("duration_diff bin (0 = shortest gap, 49 = longest gap)",
                       fontsize=13)

    # =====================================================================
    # GRAPH 1: raw counts + ratio
    # =====================================================================
    fig1, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(22, 16), gridspec_kw={"height_ratios": [2, 1]})

    width = 0.4
    ax_a.bar([i - width / 2 for i in bin_idx], pos_total, width=width,
              color="#2E86AB", label="positive trials", zorder=3)
    ax_a.bar([i + width / 2 for i in bin_idx], neg_total, width=width,
              color="#E76F51", label="negative trials (capped)", zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("trial count (log scale)", fontsize=14)
    ax_a.set_title(
        f"Positive vs Negative trials per duration bin  "
        f"(target ratio = {ratio}x, {n_bins} bins)",
        fontsize=18, fontweight="bold")
    ax_a.legend(fontsize=13, loc="upper right")
    ax_a.grid(True, which="both", axis="y", alpha=0.3, zorder=0)
    ax_a.tick_params(axis="x", labelbottom=False)

    ax_b.bar(bin_idx, neg_pos_ratio, color="#8A5A44", width=0.8, zorder=3)
    ax_b.axhline(ratio, color="black", linestyle="--", linewidth=2,
                 label=f"target ratio = {ratio}x", zorder=4)
    ax_b.set_ylabel("negatives / positives", fontsize=14)
    ax_b.set_title("How many times bigger the negative pool is, per bin",
                    fontsize=15)
    ax_b.legend(fontsize=12, loc="lower right")
    ax_b.grid(True, axis="y", alpha=0.3, zorder=0)
    ax_b.set_ylim(0, max(ratio * 1.3, (max(neg_pos_ratio) * 1.1 if neg_pos_ratio else ratio)))
    apply_xticks(ax_b)

    fig1.tight_layout()
    out1 = Path(f"{args.out_prefix}_counts_ratio.png")
    fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out1.resolve()}", file=sys.stderr)

    # =====================================================================
    # GRAPH 2: mic-combo split, positives vs negatives
    # =====================================================================
    fig2, (ax_c, ax_d) = plt.subplots(
        2, 1, figsize=(22, 16), gridspec_kw={"height_ratios": [2, 1]})

    w = 0.2
    ax_c.bar([i - 1.5 * w for i in bin_idx], pos_ihm_ihm, width=w,
              color="#2E86AB", label="positive ihm-ihm", zorder=3)
    ax_c.bar([i - 0.5 * w for i in bin_idx], pos_ihm_sdm, width=w,
              color="#7FB3D5", label="positive ihm-sdm", zorder=3)
    ax_c.bar([i + 0.5 * w for i in bin_idx], neg_ihm_ihm, width=w,
              color="#E76F51", label="negative ihm-ihm", zorder=3)
    ax_c.bar([i + 1.5 * w for i in bin_idx], neg_ihm_sdm, width=w,
              color="#F4A261", label="negative ihm-sdm", zorder=3)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("trial count (log scale)", fontsize=14)
    ax_c.set_title(
        "ihm-ihm vs ihm-sdm counts: positives vs negatives, per bin",
        fontsize=18, fontweight="bold")
    ax_c.legend(fontsize=12, loc="upper right", ncol=2)
    ax_c.grid(True, which="both", axis="y", alpha=0.3, zorder=0)
    ax_c.tick_params(axis="x", labelbottom=False)

    ax_d.plot(bin_idx, pos_ihm_share, marker="o", markersize=5,
              color="#2E86AB", linewidth=2.5, label="positives: ihm-ihm share")
    ax_d.plot(bin_idx, neg_ihm_share, marker="x", markersize=6,
              color="#E76F51", linewidth=2.5, linestyle="--",
              label="negatives: ihm-ihm share")
    ax_d.set_ylabel("ihm-ihm share of total", fontsize=14)
    ax_d.set_title(
        "ihm-ihm share of total: positives vs negatives, per bin "
        "(lines on top of each other = ratio matched exactly)",
        fontsize=15)
    ax_d.set_ylim(0, 1)
    ax_d.legend(fontsize=12, loc="center right")
    ax_d.grid(True, axis="y", alpha=0.3)
    apply_xticks(ax_d)

    fig2.tight_layout()
    out2 = Path(f"{args.out_prefix}_mic_ratio.png")
    fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out2.resolve()}", file=sys.stderr)

    max_dev = max(abs(n - ratio) for n in neg_pos_ratio)
    max_mic_gap = max(abs(p - n) for p, n in zip(pos_ihm_share, neg_ihm_share))
    print(f"\nSanity check:", file=sys.stderr)
    print(f"  largest deviation from target ratio ({ratio}x) across all bins : "
          f"{max_dev:.4f}", file=sys.stderr)
    print(f"  largest gap between positive and negative ihm-ihm share       : "
          f"{max_mic_gap:.6f}", file=sys.stderr)
    print(f"  (both should be ~0 if the run had zero shortfall)", file=sys.stderr)


if __name__ == "__main__":
    main()