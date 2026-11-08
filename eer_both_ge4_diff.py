#!/usr/bin/env python3
"""
eer_both_ge4_diff.py   (READ-ONLY: no trial table or score file is changed)

Graph 5: ONLY trials where enroll >= 4 s AND test >= 4 s, then EER vs the
ABSOLUTE length difference |enroll - test| (seconds). Lines = ihm-ihm,
ihm-sdm, overall (mic of the TEST file; enroll is always ihm).
Question: with both clips long, does a length gap still hurt?

Clip length comes from the filename (..._0000408-0000459.wav = 0.51 s).
Scores are binned (20,000 bins), binning error ~1e-4.

Output: <out-prefix>.png, <out-prefix>.csv

Usage:
    python eer_both_ge4_diff.py \\
        encoded_trials/positive_trials.parquet \\
        encoded_trials/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        positive_cosine_scores.npy negative_cosine_scores.npy \\
        --out-prefix eer_dur_5_both_ge4_diff
"""

import argparse
import csv
import sys
from pathlib import Path

NBINS = 20000
MIN_LEN = 4.0          # both clips must be >= this many seconds


def load_lookup(path_lookup_csv):
    d = {}
    with open(path_lookup_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                d[int(row[0])] = row[1]
    return d


def duration_from_name(p):
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
    print(f"ERROR: cannot find enroll/test columns in {label}: {list(columns)}", file=sys.stderr)
    sys.exit(1)


def file_ids(df, col, kt, path_to_id, n, np):
    if kt == "id":
        ids = df[col].to_numpy().astype(np.int64)
    else:
        c = df[col]
        codes = c.cat.codes.to_numpy()
        cat2id = np.array([path_to_id.get(p, -1) for p in c.cat.categories], dtype=np.int64)
        ids = np.where(codes >= 0, cat2id[np.clip(codes, 0, len(cat2id) - 1)], -1)
    return np.where((ids >= 0) & (ids < n), ids, -1).astype(np.int32)



def accumulate(scores, eid, tid, dur, is_sdm, edges, n_groups, lo, hi, np, chunk=20_000_000):
    """group = diff_bucket*2 + (1 if test is sdm); only rows with both clips >= MIN_LEN."""
    H = np.zeros(n_groups * NBINS, dtype=np.int64)
    edges = np.array(edges)
    for s in range(0, len(scores), chunk):
        sc = np.asarray(scores[s:s + chunk])
        e, t = eid[s:s + chunk], tid[s:s + chunk]
        ok = ~np.isnan(sc) & (e >= 0) & (t >= 0)
        e, t, sc = e[ok], t[ok], sc[ok]
        de, dt = dur[e], dur[t]
        keep = (de >= MIN_LEN) & (dt >= MIN_LEN)       # NaN durations fail this test
        t, sc, de, dt = t[keep], sc[keep], de[keep], dt[keep]
        b = np.clip(((sc - lo) / (hi - lo) * NBINS).astype(np.int64), 0, NBINS - 1)
        g = np.searchsorted(edges, np.abs(dt - de), "right") * 2 + is_sdm[t].astype(np.int64)
        H += np.bincount(g * NBINS + b, minlength=n_groups * NBINS)
        print(f"    {min(s + chunk, len(scores)):,}/{len(scores):,}", file=sys.stderr)
    return H.reshape(n_groups, NBINS)


def eer_from_hist(hp, hn, np):
    P, N = hp.sum(), hn.sum()
    if P < 50 or N < 50:
        return float("nan")
    frr = np.concatenate([[0.0], np.cumsum(hp) / P])
    far = 1.0 - np.concatenate([[0.0], np.cumsum(hn) / N])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("positive_scores_npy")
    ap.add_argument("negative_scores_npy")
    ap.add_argument("--edges", default="0.5,1,2,4",
                    help="Edges (s) for |enroll-test|. '0.5,1,2,4' -> <0.5, 0.5-1, 1-2, 2-4, >=4.")
    ap.add_argument("--out-prefix", default="eer_dur_5_both_ge4_diff")
    ap.add_argument("--min-trials", type=int, default=1000,
                    help="Fewer genuine OR impostor trials than this -> shown as n/a.")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import joblib

    edges = [float(x) for x in args.edges.split(",")]
    labels = ([f"<{edges[0]:g}s"] + [f"{edges[i]:g}-{edges[i+1]:g}s" for i in range(len(edges) - 1)]
              + [f">={edges[-1]:g}s"])
    n_groups = len(labels) * 2

    id_to_path = load_lookup(args.path_lookup_csv)
    n = max(id_to_path) + 1
    path_to_id = {p: i for i, p in id_to_path.items()}
    dur = np.array([duration_from_name(id_to_path.get(i, "")) for i in range(n)], dtype=np.float32)
    is_sdm = np.array(["/sdm/" in id_to_path.get(i, "") for i in range(n)])
    print(f"{n:,} files, {int(np.isnan(dur).sum())} with unparseable duration.", file=sys.stderr)

    def load_ids(df, label):
        ec, tc, kt = find_pair_columns(df.columns, label)
        return (file_ids(df, ec, kt, path_to_id, n, np),
                file_ids(df, tc, kt, path_to_id, n, np))

    print("\nLoading positive trials ...", file=sys.stderr)
    pe, pt = load_ids(pd.read_parquet(args.positive_trials_parquet), "positive")
    ps = np.load(args.positive_scores_npy, mmap_mode="r")
    print("Loading negative trials ...", file=sys.stderr)
    ne, nt = load_ids(joblib.load(args.negative_trials_joblib), "negative")
    ns = np.load(args.negative_scores_npy, mmap_mode="r")
    for a, b, l in ((ps, pe, "positive"), (ns, ne, "negative")):
        if len(a) != len(b):
            print(f"ERROR: {l} scores ({len(a):,}) != trials ({len(b):,}).", file=sys.stderr)
            sys.exit(1)

    rng = np.random.default_rng(0)
    samp = np.concatenate([np.asarray(ps[rng.integers(0, len(ps), 2_000_000)]),
                           np.asarray(ns[rng.integers(0, len(ns), 2_000_000)])])
    samp = samp[~np.isnan(samp)]
    lo, hi = np.percentile(samp, [0.01, 99.99])
    print(f"Score range for histograms: [{lo:.3f}, {hi:.3f}]", file=sys.stderr)

    print("\nBinning positive scores ...", file=sys.stderr)
    HP = accumulate(ps, pe, pt, dur, is_sdm, edges, n_groups, lo, hi, np)
    print("Binning negative scores ...", file=sys.stderr)
    HN = accumulate(ns, ne, nt, dur, is_sdm, edges, n_groups, lo, hi, np)

    MICS = (("ihm_ihm", 0), ("ihm_sdm", 1), ("overall", None))

    def cell(k, mic):
        gs = [k * 2 + m for m in ((0, 1) if mic is None else (mic,))]
        hp, hn = HP[gs].sum(0), HN[gs].sum(0)
        e = eer_from_hist(hp, hn, np)
        ng, nn = int(hp.sum()), int(hn.sum())
        ok = ng >= args.min_trials and nn >= args.min_trials and e == e
        return (e if ok else float("nan")), ng, nn

    res = {m: [cell(k, mic) for k in range(len(labels))] for m, mic in MICS}

    print(f"\n== Enroll >= {MIN_LEN:g} s AND test >= {MIN_LEN:g} s: EER by |enroll - test| ==")
    print(f"{'|diff|':10s}" + "".join(f"{m:>28s}" for m, _ in MICS))
    rows = []
    for i, l in enumerate(labels):
        line = f"{l:10s}"
        for m, _ in MICS:
            e, ng, nn = res[m][i]
            line += f"{('n/a' if e != e else f'{e*100:.2f}%'):>9s} ({ng:>8,d}g/{nn:>10,d}i)"
            rows.append((l, m, ng, nn, e))
        print(line)

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["abs_diff_bucket", "mic", "n_genuine", "n_impostor", "eer"])
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: CSV only.", file=sys.stderr)
        return
    colors = {"ihm_ihm": "tab:blue", "ihm_sdm": "tab:red", "overall": "k"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = list(range(len(labels)))
    for m, _ in MICS:
        ys = [e * 100 if e == e else float("nan") for e, _, _ in res[m]]
        ax.plot(xs, ys, marker="o", color=colors[m], label=m.replace("_", "-"),
                ls="--" if m == "overall" else "-")
        for x, y in zip(xs, ys):
            if y == y:
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color=colors[m])
    ov = res["overall"]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n(n={ov[i][1]/1e6:.2f}M genuine)" for i, l in enumerate(labels)],
                       fontsize=8)
    ax.set_xlabel("absolute difference between enroll and test clip length")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"EER vs |enroll - test| length, both clips >= {MIN_LEN:g} s")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=150)
    print(f"Saved {out}.png and {out}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()