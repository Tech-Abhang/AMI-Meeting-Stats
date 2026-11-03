#!/usr/bin/env python3
"""
eer_by_duration.py   (READ-ONLY: no trial is changed, dropped or added)

Question it answers: is the high EER driven by SHORT CLIPS?

Every trial gets the length of its SHORTER clip, min(enroll_dur, test_dur)
(a trial is only as good as its weakest clip), and is put in a length
bucket. EER is then computed inside each bucket, separately for ihm-ihm and
ihm-sdm, from the score files you already have.

  * EER falls steeply as clips get longer  -> short clips are the driver.
  * EER stays flat across buckets          -> they are NOT; suspect
                                              cross-talk / domain mismatch /
                                              trial construction instead.

Clip length comes from the filename (..._0000408-0000459.wav = 0.51 s), so
no audio is read. Scores are histogrammed (20,000 bins) so 273M rows stream
through in chunks; EER error from binning is ~1e-4.

Output: printed table, <out-prefix>.csv, <out-prefix>.png

Usage:
    python eer_by_duration.py \\
        encoded_trials/positive_trials.parquet \\
        encoded_trials/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        scores_v2/positive_cosine_scores_centermic_asnorm.npy \\
        scores_v2/negative_cosine_scores_centermic_asnorm.npy \\
        --out-prefix scores_v2/eer_by_duration_centermic_asnorm
"""

import argparse
import csv
import sys
from pathlib import Path

NBINS = 20000


def load_lookup(path_lookup_csv):
    id_to_path = {}
    with open(path_lookup_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                id_to_path[int(row[0])] = row[1]
    return id_to_path


def duration_from_name(p):
    """Seconds from the trailing 'begin-end' (centiseconds); nan if no parse."""
    try:
        b, e = Path(p).stem.rsplit("_", 1)[-1].split("-")
        d = (int(e) - int(b)) / 100.0
        return d if d > 0 else float("nan")
    except Exception:
        return float("nan")


def find_pair_columns(columns, label):
    low = {c.lower(): c for c in columns}
    for suffix, kt in (("path", "path"), ("id", "id")):
        e = [c for lc, c in low.items() if "enroll" in lc and lc.endswith(suffix)]
        t = [c for lc, c in low.items() if "test" in lc and lc.endswith(suffix)]
        if len(e) == 1 and len(t) == 1:
            return e[0], t[0], kt
    print(f"ERROR: cannot find enroll/test columns in {label}: {list(columns)}",
          file=sys.stderr)
    sys.exit(1)


def file_ids(df, col, kt, path_to_id, n, np):
    """-> int32 file ids, -1 where unknown."""
    if kt == "id":
        ids = df[col].to_numpy().astype(np.int64)
    else:
        c = df[col]
        codes = c.cat.codes.to_numpy()
        cat2id = np.array([path_to_id.get(p, -1) for p in c.cat.categories], dtype=np.int64)
        ids = np.where(codes >= 0, cat2id[np.clip(codes, 0, len(cat2id) - 1)], -1)
    ids = np.where((ids >= 0) & (ids < n), ids, -1)
    return ids.astype(np.int32)


def trial_groups(enroll, test, dur, is_sdm, edges, np):
    """group = bucket*2 + (1 if test is sdm); -1 for unusable rows."""
    ok = (enroll >= 0) & (test >= 0)
    e = np.where(ok, enroll, 0)
    t = np.where(ok, test, 0)
    d = np.minimum(dur[e], dur[t])
    ok &= ~np.isnan(d)
    b = np.searchsorted(edges, d, side="right")      # 0..len(edges)
    g = b * 2 + is_sdm[t].astype(np.int64)
    return np.where(ok, g, -1)


def accumulate(scores, groups, lo, hi, n_groups, np, chunk=20_000_000):
    """-> (n_groups, NBINS) int64 histogram of scores per group."""
    H = np.zeros(n_groups * NBINS, dtype=np.int64)
    for s in range(0, len(scores), chunk):
        sc = scores[s:s + chunk]
        g = groups[s:s + chunk]
        m = ~np.isnan(sc) & (g >= 0)
        b = np.clip(((sc[m] - lo) / (hi - lo) * NBINS).astype(np.int64), 0, NBINS - 1)
        H += np.bincount(g[m].astype(np.int64) * NBINS + b, minlength=n_groups * NBINS)
        print(f"    {min(s + chunk, len(scores)):,}/{len(scores):,}", file=sys.stderr)
    return H.reshape(n_groups, NBINS)


def eer_from_hist(hp, hn, lo, hi, np):
    """EER where FAR == FRR (accept if score >= threshold)."""
    P, N = hp.sum(), hn.sum()
    if P < 50 or N < 50:
        return float("nan"), float("nan"), float("nan")
    frr = np.concatenate([[0.0], np.cumsum(hp) / P])        # genuine rejected
    far = 1.0 - np.concatenate([[0.0], np.cumsum(hn) / N])  # impostor accepted
    i = int(np.argmin(np.abs(far - frr)))
    eer = (far[i] + frr[i]) / 2
    med = lambda h: lo + (np.searchsorted(np.cumsum(h), h.sum() / 2) + 0.5) * (hi - lo) / NBINS
    return float(eer), float(med(hp)), float(med(hn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("positive_scores_npy")
    ap.add_argument("negative_scores_npy")
    ap.add_argument("--edges", default="1,2,4",
                    help="Bucket edges in seconds. '1,2,4' -> <1, 1-2, 2-4, >=4.")
    ap.add_argument("--out-prefix", default="eer_by_duration")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import joblib

    edges = np.array([float(x) for x in args.edges.split(",")])
    labels = ([f"<{edges[0]:g}s"] +
              [f"{edges[i]:g}-{edges[i+1]:g}s" for i in range(len(edges) - 1)] +
              [f">={edges[-1]:g}s"])
    nb = len(labels)
    n_groups = nb * 2

    id_to_path = load_lookup(args.path_lookup_csv)
    n = max(id_to_path) + 1
    path_to_id = {p: i for i, p in id_to_path.items()}
    dur = np.array([duration_from_name(id_to_path.get(i, "")) for i in range(n)])
    is_sdm = np.array(["/sdm/" in id_to_path.get(i, "") for i in range(n)])
    print(f"{n:,} files. Unparseable durations: {int(np.isnan(dur).sum())}. "
          f"Clip length median {np.nanmedian(dur):.2f}s; "
          + ", ".join(f"{l}: {100*np.mean((np.searchsorted(edges, dur[~np.isnan(dur)], side='right')==k)):.0f}%"
                      for k, l in enumerate(labels)) + " of files", file=sys.stderr)

    # ---- group labels for every trial row (no table is modified) ----
    print("\nPositive trials ...", file=sys.stderr)
    df = pd.read_parquet(args.positive_trials_parquet)
    ec, tc, kt = find_pair_columns(df.columns, "positive")
    pg = trial_groups(file_ids(df, ec, kt, path_to_id, n, np),
                      file_ids(df, tc, kt, path_to_id, n, np), dur, is_sdm, edges, np)
    del df
    ps = np.load(args.positive_scores_npy, mmap_mode="r")
    if len(ps) != len(pg):
        print(f"ERROR: positive scores ({len(ps):,}) != trials ({len(pg):,}).", file=sys.stderr)
        sys.exit(1)

    print("Negative trials ...", file=sys.stderr)
    df = joblib.load(args.negative_trials_joblib)
    ec, tc, kt = find_pair_columns(df.columns, "negative")
    ng = trial_groups(file_ids(df, ec, kt, path_to_id, n, np),
                      file_ids(df, tc, kt, path_to_id, n, np), dur, is_sdm, edges, np)
    del df
    ns = np.load(args.negative_scores_npy, mmap_mode="r")
    if len(ns) != len(ng):
        print(f"ERROR: negative scores ({len(ns):,}) != trials ({len(ng):,}).", file=sys.stderr)
        sys.exit(1)

    # ---- histogram range from a sample (variant-agnostic: works for AS-norm too) ----
    rng = np.random.default_rng(0)
    samp = np.concatenate([np.asarray(ps[rng.integers(0, len(ps), 2_000_000)]),
                           np.asarray(ns[rng.integers(0, len(ns), 2_000_000)])])
    samp = samp[~np.isnan(samp)]
    lo, hi = np.percentile(samp, [0.01, 99.99])
    print(f"Score range used for histograms: [{lo:.3f}, {hi:.3f}]", file=sys.stderr)

    print("\nBinning positive scores ...", file=sys.stderr)
    HP = accumulate(ps, pg, lo, hi, n_groups, np)
    print("Binning negative scores ...", file=sys.stderr)
    HN = accumulate(ns, ng, lo, hi, n_groups, np)

    # ---- results ----
    rows = []
    def add(name, bucket_label, gset):
        hp = HP[gset].sum(0); hn = HN[gset].sum(0)
        eer, mp, mn = eer_from_hist(hp, hn, lo, hi, np)
        rows.append((name, bucket_label, int(hp.sum()), int(hn.sum()), eer, mp, mn))
    for k, l in enumerate(labels):
        add("ihm_ihm", l, [k * 2])
        add("ihm_sdm", l, [k * 2 + 1])
        add("overall", l, [k * 2, k * 2 + 1])
    add("overall", "ALL", list(range(n_groups)))

    print(f"\n{'mic':8s} {'shorter clip':>13s} {'#genuine':>12s} {'#impostor':>13s} "
          f"{'EER':>8s} {'med.genuine':>12s} {'med.impostor':>13s}")
    for name, l, p, q, e, mp, mn in rows:
        es = "   n/a" if e != e else f"{e*100:6.2f}%"
        print(f"{name:8s} {l:>13s} {p:12,d} {q:13,d} {es:>8s} {mp:12.3f} {mn:13.3f}")

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mic", "shorter_clip", "n_genuine", "n_impostor", "eer",
                    "median_genuine", "median_impostor"])
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, c in (("ihm_ihm", "tab:blue"), ("ihm_sdm", "tab:red"), ("overall", "k")):
            ys = [next(r[4] for r in rows if r[0] == name and r[1] == l) * 100 for l in labels]
            ax.plot(labels, ys, marker="o", color=c, label=name.replace("_", "-"),
                    ls="--" if name == "overall" else "-")
            for x, y in zip(labels, ys):
                if y == y:
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=8, color=c)
        ax.set_xlabel("length of the SHORTER clip in the trial")
        ax.set_ylabel("EER (%)")
        ax.set_title("EER vs clip length")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out}.png", dpi=150)
        print(f"\nSaved {out}.csv and {out}.png", file=sys.stderr)
    except ImportError:
        print(f"\nSaved {out}.csv (matplotlib missing, no plot).", file=sys.stderr)


if __name__ == "__main__":
    main()