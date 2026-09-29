#!/usr/bin/env python3
"""
plot_pos_neg_mic_comparison.py

ONE graph, two panels, from negative_trials_v2_report.json only.

Top panel    : counts per bin, but split into 4 bars instead of 2 --
               positive ihm-ihm, positive ihm-sdm, negative ihm-ihm,
               negative ihm-sdm. Log scale (counts span ~7,000,000 down
               to a few hundred). Comparing the two positive bars (or
               the two negative bars) within a bin shows the ihm-ihm vs
               ihm-sdm ratio directly.

Bottom panel : negative/positive ratio per bin, but as 3 bars instead
               of 1 -- overall ratio, ihm-ihm-only ratio, ihm-sdm-only
               ratio -- plus a dashed line at the target ratio (e.g.
               10x). If all three bars sit on the dashed line for every
               bin, the split was preserved exactly, not just on
               average.

INPUT
-----
Only negative_trials_v2_report.json -- it already carries positive
counts (positive_total, positive_ihm_ihm, positive_ihm_sdm) alongside
negative actuals for every one of the 50 bins.

Usage:
    python plot_pos_neg_mic_comparison.py negative_trials_v2_report.json \\
        --out pos_neg_mic_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json", help="negative_trials_v2_report.json")
    ap.add_argument("--out", default="pos_neg_mic_comparison.png")
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

    pos_ihm_ihm = [b["positive_ihm_ihm"] for b in bins]
    pos_ihm_sdm = [b["positive_ihm_sdm"] for b in bins]
    neg_ihm_ihm = [b["actual_ihm_ihm"] for b in bins]
    neg_ihm_sdm = [b["actual_ihm_sdm"] for b in bins]

    pos_total = [p + s for p, s in zip(pos_ihm_ihm, pos_ihm_sdm)]
    neg_total = [p + s for p, s in zip(neg_ihm_ihm, neg_ihm_sdm)]

    ratio_total = [(n / p if p else 0.0) for n, p in zip(neg_total, pos_total)]
    ratio_ihm_ihm = [(n / p if p else 0.0) for n, p in zip(neg_ihm_ihm, pos_ihm_ihm)]
    ratio_ihm_sdm = [(n / p if p else 0.0) for n, p in zip(neg_ihm_sdm, pos_ihm_sdm)]

    xtick_pos = sorted(set(list(range(0, n_bins, 5)) + [n_bins - 1]))
    xtick_labels = [f"{lower[i]:.1f}-{upper[i]:.1f}s" for i in xtick_pos]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(22, 16), gridspec_kw={"height_ratios": [2, 1]})

    # ---------------- Panel A: 4 bars per bin ----------------
    w = 0.2
    ax_a.bar([i - 1.5 * w for i in bin_idx], pos_ihm_ihm, width=w,
              color="#2E86AB", label="positive ihm-ihm", zorder=3)
    ax_a.bar([i - 0.5 * w for i in bin_idx], pos_ihm_sdm, width=w,
              color="#7FB3D5", label="positive ihm-sdm", zorder=3)
    ax_a.bar([i + 0.5 * w for i in bin_idx], neg_ihm_ihm, width=w,
              color="#E76F51", label="negative ihm-ihm", zorder=3)
    ax_a.bar([i + 1.5 * w for i in bin_idx], neg_ihm_sdm, width=w,
              color="#F4A261", label="negative ihm-sdm", zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("trial count (log scale)", fontsize=14)
    ax_a.set_title(
        f"Positive vs Negative trials per bin, split by mic-combo  "
        f"(target ratio = {ratio}x, {n_bins} bins)",
        fontsize=18, fontweight="bold")
    ax_a.legend(fontsize=12, loc="upper right", ncol=2)
    ax_a.grid(True, which="both", axis="y", alpha=0.3, zorder=0)
    ax_a.tick_params(axis="x", labelbottom=False)

    # ---------------- Panel B: 3 ratio bars per bin ----------------
    w2 = 0.25
    ax_b.bar([i - w2 for i in bin_idx], ratio_total, width=w2,
              color="#8A5A44", label="overall ratio", zorder=3)
    ax_b.bar(bin_idx, ratio_ihm_ihm, width=w2,
              color="#2E86AB", label="ihm-ihm ratio", zorder=3)
    ax_b.bar([i + w2 for i in bin_idx], ratio_ihm_sdm, width=w2,
              color="#E76F51", label="ihm-sdm ratio", zorder=3)
    ax_b.axhline(ratio, color="black", linestyle="--", linewidth=2,
                 label=f"target ratio = {ratio}x", zorder=4)
    ax_b.set_ylabel("negatives / positives", fontsize=14)
    ax_b.set_xlabel("duration_diff bin (0 = shortest gap, 49 = longest gap)",
                     fontsize=13)
    ax_b.set_title(
        "Negative/positive ratio per bin -- overall vs each mic-combo "
        "(all on the dashed line = ratio held exactly, not just on average)",
        fontsize=15)
    ax_b.legend(fontsize=11, loc="lower right", ncol=2)
    ax_b.grid(True, axis="y", alpha=0.3, zorder=0)
    all_ratios = ratio_total + ratio_ihm_ihm + ratio_ihm_sdm
    ax_b.set_ylim(0, max(ratio * 1.3, (max(all_ratios) * 1.1 if all_ratios else ratio)))
    ax_b.set_xticks(xtick_pos)
    ax_b.set_xticklabels(xtick_labels, rotation=60, ha="right", fontsize=10)

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}", file=sys.stderr)

    max_dev = max(abs(r - ratio) for r in all_ratios)
    print(f"\nSanity check:", file=sys.stderr)
    print(f"  largest deviation from target ratio ({ratio}x), across overall/"
          f"ihm-ihm/ihm-sdm and all bins: {max_dev:.4f}", file=sys.stderr)
    print(f"  (should be ~0 if the run had zero shortfall)", file=sys.stderr)


if __name__ == "__main__":
    main()