#!/usr/bin/env python3
"""
eer_duration_analysis.py   (READ-ONLY: no trial table is changed, no score file is changed)

Makes 4 figures + 1 CSV from the existing trial tables and score files.
Each figure looks at a SUBSET of trials; the trial tables themselves are
never touched, filtered on disk, or rewritten.

  Graph 1   EER vs duration DIFFERENCE between enroll and test clip
            1  = absolute difference  |enroll - test|  (seconds)
            1b = signed difference     test - enroll    (negative = test shorter)
  Graph 2   enroll clip fixed at >= 4 s;  test clip in <1 / 1-2 / 2-4 / >=4 s
  Graph 3   test clip   fixed at >= 4 s;  enroll clip in <1 / 1-2 / 2-4 / >=4 s

Every graph is split into ihm-ihm / ihm-sdm / overall (mic of the TEST file;
enroll is always ihm). Clip length comes from the filename
(..._0000408-0000459.wav = 0.51 s), no audio is read. Scores are binned
(20,000 bins) so 273M rows stream through in chunks; binning error ~1e-4.

Output: <out-prefix>_1_diff_abs.png, _1b_diff_signed.png,
        _2_enroll_ge4.png, _3_test_ge4.png, <out-prefix>.csv

Usage:
    python eer_duration_analysis.py \\
        encoded_trials/positive_trials.parquet \\
        encoded_trials/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        positive_cosine_scores.npy negative_cosine_scores.npy \\
        --out-prefix eer_dur
"""

import argparse
import csv
import sys
from pathlib import Path

NBINS = 20000
CLIP_EDGES = [1.0, 2.0, 4.0]
CLIP_LABELS = ["<1s", "1-2s", "2-4s", ">=4s"]
ABS_EDGES = [0.5, 1.0, 2.0, 4.0]
ABS_LABELS = ["<0.5s", "0.5-1s", "1-2s", "2-4s", ">=4s"]
SGN_EDGES = [-2.0, -0.5, 0.5, 2.0]
SGN_LABELS = ["test >2s shorter", "test 0.5-2s shorter", "similar (+-0.5s)",
              "test 0.5-2s longer", "test >2s longer"]


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


# 3 groupings, each group = (bucket..., mic):  mic = 1 if TEST file is sdm
N_PAIR = len(CLIP_LABELS) * len(CLIP_LABELS) * 2   # (enroll bucket, test bucket, mic)
N_ABS = len(ABS_LABELS) * 2
N_SGN = len(SGN_LABELS) * 2
GROUPINGS = {"pair": N_PAIR, "abs": N_ABS, "sgn": N_SGN}


def accumulate(scores, eid, tid, dur, is_sdm, lo, hi, np, chunk=20_000_000):
    H = {k: np.zeros(v * NBINS, dtype=np.int64) for k, v in GROUPINGS.items()}
    ce, ca, cs = (np.array(x) for x in (CLIP_EDGES, ABS_EDGES, SGN_EDGES))
    for s in range(0, len(scores), chunk):
        sc = np.asarray(scores[s:s + chunk])
        e, t = eid[s:s + chunk], tid[s:s + chunk]
        ok = ~np.isnan(sc) & (e >= 0) & (t >= 0)
        e, t, sc = e[ok], t[ok], sc[ok]
        de, dt = dur[e], dur[t]
        good = ~np.isnan(de) & ~np.isnan(dt)
        e, t, sc, de, dt = e[good], t[good], sc[good], de[good], dt[good]
        b = np.clip(((sc - lo) / (hi - lo) * NBINS).astype(np.int64), 0, NBINS - 1)
        mic = is_sdm[t].astype(np.int64)
        gp = (np.searchsorted(ce, de, "right") * len(CLIP_LABELS) +
              np.searchsorted(ce, dt, "right")) * 2 + mic
        ga = np.searchsorted(ca, np.abs(dt - de), "right") * 2 + mic
        gs = np.searchsorted(cs, dt - de, "right") * 2 + mic
        for k, g in (("pair", gp), ("abs", ga), ("sgn", gs)):
            H[k] += np.bincount(g * NBINS + b, minlength=GROUPINGS[k] * NBINS)
        print(f"    {min(s + chunk, len(scores)):,}/{len(scores):,}", file=sys.stderr)
    return {k: v.reshape(GROUPINGS[k], NBINS) for k, v in H.items()}


def eer_from_hist(hp, hn, np):
    P, N = hp.sum(), hn.sum()
    if P < 50 or N < 50:
        return float("nan")
    frr = np.concatenate([[0.0], np.cumsum(hp) / P])
    far = 1.0 - np.concatenate([[0.0], np.cumsum(hn) / N])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def cell(HP, HN, groups, np):
    """EER and counts for the union of `groups` (indices into the grouping)."""
    hp, hn = HP[groups].sum(0), HN[groups].sum(0)
    return eer_from_hist(hp, hn, np), int(hp.sum()), int(hn.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("positive_scores_npy")
    ap.add_argument("negative_scores_npy")
    ap.add_argument("--out-prefix", default="eer_dur")
    ap.add_argument("--min-trials", type=int, default=1000,
                    help="Cells with fewer genuine OR impostor trials than this are "
                         "drawn as 'n/a' (EER from a handful of trials is noise).")
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

    # ---------- build the tables ----------
    nc = len(CLIP_LABELS)
    MICS = (("ihm_ihm", 0), ("ihm_sdm", 1), ("overall", None))

    def groups_for(mic, base):               # base: list of mic-less group numbers
        if mic is None:
            return [g * 2 + m for g in base for m in (0, 1)]
        return [g * 2 + mic for g in base]

    analyses = {}   # name -> (title, xlabel, labels, {mic: [(eer, ng, nn)]})
    def build(name, title, xlabel, labels, grouping, bases):
        res = {}
        for mname, m in MICS:
            res[mname] = [cell(HP[grouping], HN[grouping], groups_for(m, b), np) for b in bases]
        analyses[name] = (title, xlabel, labels, res)

    build("1_diff_abs", "EER vs |enroll length - test length|",
          "absolute difference between enroll and test clip length", ABS_LABELS, "abs",
          [[i] for i in range(len(ABS_LABELS))])
    build("1b_diff_signed", "EER vs (test length - enroll length)",
          "test minus enroll clip length", SGN_LABELS, "sgn",
          [[i] for i in range(len(SGN_LABELS))])
    build("2_enroll_ge4", "Enroll clip >= 4 s, test clip varies",
          "length of the TEST clip", CLIP_LABELS, "pair",
          [[(nc - 1) * nc + j] for j in range(nc)])        # enroll bucket = last
    build("3_test_ge4", "Test clip >= 4 s, enroll clip varies",
          "length of the ENROLL clip", CLIP_LABELS, "pair",
          [[i * nc + (nc - 1)] for i in range(nc)])        # test bucket = last

    # ---------- print + CSV ----------
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (title, xl, labels, res) in analyses.items():
        print(f"\n== {title} ==")
        print(f"{'':22s}" + "".join(f"{m:>26s}" for m, _ in MICS))
        for i, l in enumerate(labels):
            line = f"{l:22s}"
            for mname, _ in MICS:
                e, ng, nn = res[mname][i]
                ok = ng >= args.min_trials and nn >= args.min_trials and e == e
                line += f"{(f'{e*100:.2f}%' if ok else 'n/a'):>9s} ({ng:>9,d}g){'':4s}"
                rows.append((name, mname, l, ng, nn, e if ok else float('nan')))
            print(line)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "mic", "bucket", "n_genuine", "n_impostor", "eer"])
        w.writerows(rows)

    # ---------- figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: CSV only.", file=sys.stderr)
        return
    colors = {"ihm_ihm": "tab:blue", "ihm_sdm": "tab:red", "overall": "k"}
    for name, (title, xl, labels, res) in analyses.items():
        fig, ax = plt.subplots(figsize=(8, 4.8))
        xs = list(range(len(labels)))
        for mname, _ in MICS:
            ys = []
            for e, ng, nn in res[mname]:
                ok = ng >= args.min_trials and nn >= args.min_trials and e == e
                ys.append(e * 100 if ok else float("nan"))
            ax.plot(xs, ys, marker="o", color=colors[mname], label=mname.replace("_", "-"),
                    ls="--" if mname == "overall" else "-")
            for x, y in zip(xs, ys):
                if y == y:
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=8, color=colors[mname])
        ov = res["overall"]
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{l}\n(n={ov[i][1]/1e6:.2f}M genuine)" for i, l in enumerate(labels)],
                           fontsize=8)
        ax.set_xlabel(xl)
        ax.set_ylabel("EER (%)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out}_{name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out}_{name}.png", file=sys.stderr)
    print(f"Saved {out}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()