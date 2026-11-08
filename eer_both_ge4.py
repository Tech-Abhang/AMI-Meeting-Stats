#!/usr/bin/env python3
"""
eer_both_ge4.py   (READ-ONLY: no trial table or score file is changed)

Graph 4: ONLY trials where the enroll clip >= 4 s AND the test clip >= 4 s.
Bars = ihm-ihm, ihm-sdm, overall (mic of the TEST file; enroll is always ihm).
This is the "best case" EER: both clips are long, so what is left is the
model + microphone mismatch, not clip length.

Also prints the full 4x4 table (enroll length x test length) for overall EER,
so you can put the >=4/>=4 corner next to all the other cells.

Clip length comes from the filename (..._0000408-0000459.wav = 0.51 s).
Scores are binned (20,000 bins), binning error ~1e-4.

Output: <out-prefix>.png, <out-prefix>.csv

Usage:
    python eer_both_ge4.py \\
        encoded_trials/positive_trials.parquet \\
        encoded_trials/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        positive_cosine_scores.npy negative_cosine_scores.npy \\
        --out-prefix eer_dur_4_both_ge4
"""

import argparse
import csv
import sys
from pathlib import Path

NBINS = 20000
EDGES = [1.0, 2.0, 4.0]
LABELS = ["<1s", "1-2s", "2-4s", ">=4s"]
NB = len(LABELS)


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


N_GROUPS = NB * NB * 2          # (enroll bucket, test bucket, test-is-sdm)


def accumulate(scores, eid, tid, dur, is_sdm, lo, hi, np, chunk=20_000_000):
    H = np.zeros(N_GROUPS * NBINS, dtype=np.int64)
    edges = np.array(EDGES)
    for s in range(0, len(scores), chunk):
        sc = np.asarray(scores[s:s + chunk])
        e, t = eid[s:s + chunk], tid[s:s + chunk]
        ok = ~np.isnan(sc) & (e >= 0) & (t >= 0)
        e, t, sc = e[ok], t[ok], sc[ok]
        de, dt = dur[e], dur[t]
        good = ~np.isnan(de) & ~np.isnan(dt)
        e, t, sc, de, dt = e[good], t[good], sc[good], de[good], dt[good]
        b = np.clip(((sc - lo) / (hi - lo) * NBINS).astype(np.int64), 0, NBINS - 1)
        g = ((np.searchsorted(edges, de, "right") * NB + np.searchsorted(edges, dt, "right")) * 2
             + is_sdm[t].astype(np.int64))
        H += np.bincount(g * NBINS + b, minlength=N_GROUPS * NBINS)
        print(f"    {min(s + chunk, len(scores)):,}/{len(scores):,}", file=sys.stderr)
    return H.reshape(N_GROUPS, NBINS)


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
    ap.add_argument("--out-prefix", default="eer_dur_4_both_ge4")
    ap.add_argument("--min-trials", type=int, default=1000,
                    help="Fewer genuine OR impostor trials than this -> shown as n/a.")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import joblib

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
    HP = accumulate(ps, pe, pt, dur, is_sdm, lo, hi, np)
    print("Binning negative scores ...", file=sys.stderr)
    HN = accumulate(ns, ne, nt, dur, is_sdm, lo, hi, np)

    def cell(ei, ti, mic):
        gs = [(ei * NB + ti) * 2 + m for m in ((0, 1) if mic is None else (mic,))]
        hp, hn = HP[gs].sum(0), HN[gs].sum(0)
        e = eer_from_hist(hp, hn, np)
        ng, nn = int(hp.sum()), int(hn.sum())
        ok = ng >= args.min_trials and nn >= args.min_trials and e == e
        return (e if ok else float("nan")), ng, nn

    MICS = (("ihm_ihm", 0), ("ihm_sdm", 1), ("overall", None))
    last = NB - 1
    res = {m: cell(last, last, mic) for m, mic in MICS}

    print("\n== Enroll >= 4 s AND test >= 4 s ==")
    print(f"{'mic':10s} {'EER':>8s} {'#genuine':>12s} {'#impostor':>13s}")
    for m, _ in MICS:
        e, ng, nn = res[m]
        es = "n/a" if e != e else f"{e*100:.2f}%"
        print(f"{m:10s} {es:>8s} {ng:12,d} {nn:13,d}")

    print("\n== Overall EER, enroll length (rows) x test length (columns) ==")
    print(f"{'enroll \\ test':14s}" + "".join(f"{l:>10s}" for l in LABELS))
    rows = []
    for ei, el in enumerate(LABELS):
        line = f"{el:14s}"
        for ti, tl in enumerate(LABELS):
            for m, mic in MICS:
                e, ng, nn = cell(ei, ti, mic)
                rows.append((el, tl, m, ng, nn, e))
            e = cell(ei, ti, None)[0]
            line += f"{('n/a' if e != e else f'{e*100:.1f}%'):>10s}"
        print(line)

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["enroll_bucket", "test_bucket", "mic", "n_genuine", "n_impostor", "eer"])
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: CSV only.", file=sys.stderr)
        return
    colors = {"ihm_ihm": "tab:blue", "ihm_sdm": "tab:red", "overall": "0.25"}
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for x, (m, _) in enumerate(MICS):
        e, ng, nn = res[m]
        if e != e:
            ax.text(x, 0.5, "n/a", ha="center")
            continue
        ax.bar(x, e * 100, color=colors[m], width=0.6)
        ax.annotate(f"{e*100:.1f}%", (x, e * 100), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10)
    ax.set_xticks(range(len(MICS)))
    ax.set_xticklabels([f"{m.replace('_', '-')}\n(n={res[m][1]/1e6:.2f}M genuine)" for m, _ in MICS],
                       fontsize=9)
    ax.set_ylabel("EER (%)")
    ax.set_title("EER when enroll >= 4 s AND test >= 4 s")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=150)
    print(f"Saved {out}.png and {out}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()