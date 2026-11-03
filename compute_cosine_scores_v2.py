#!/usr/bin/env python3
"""
compute_cosine_scores_v2.py

Same job as compute_cosine_scores.py (one score per trial ROW, file-vs-file,
float32 arrays aligned by row position, trial tables never touched), but:

  * scores SEVERAL post-processing "variants" in ONE pass over the trial
    tables (loading the 273M-row table is the slow part, so do it once);
  * runs the gather + dot product on the GPU (torch) -> minutes, not an hour.

NOTHING HERE CHANGES, DROPS OR ADDS A TRIAL. All variants are
transformations of the 104,748 embeddings, computed without trial labels.

VARIANT SYNTAX   <transform>[+asnorm]
---------------------------------------
transforms:
  raw        plain cosine (the baseline)
  center     subtract the global mean embedding, then cosine
  centermic  subtract a SEPARATE mean for ihm files and for sdm files
             (removes the "this is a far-field recording" offset)
  whiten     global centering + ZCA whitening (decorrelate the 192 dims)
+asnorm      adaptive S-norm (AS-norm) on top: each file's score is
             re-expressed relative to how it scores against a cohort of
             impostor files -> removes per-file bias (some files just score
             high or low against everyone).

    e.g.  --variants raw,center,centermic,whiten,centermic+asnorm,whiten+asnorm

LABEL USE (state this in the paper): the only place speaker identity is
touched is AS-norm's cohort -- cohort files from the SAME speaker as the
file being normalised are masked out (speaker = filename prefix before the
first '_', e.g. fee019_es2005a_...). That uses the speaker tag of the files
but not the trial labels. Use --no-exclude-same-speaker to disable it.
Mean/whitening statistics use all embedded files, no labels.

OUTPUT   <out-dir>/positive_cosine_scores_<variant>.npy
         <out-dir>/negative_cosine_scores_<variant>.npy
(variant name has '+' written as '_'). Original score files are not touched.

Usage:
    python compute_cosine_scores_v2.py \\
        encoded_trials/positive_trials.parquet \\
        trial_outputs/negative_trials_v2.joblib \\
        encoded_trials/path_lookup.csv \\
        embeddings_v2.npy embeddings_v2_done.npy \\
        --out-dir scores_v2 --device cuda:0 \\
        --variants raw,center,centermic,whiten,centermic+asnorm,whiten+asnorm
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path


# ----------------------------------------------------------------------------
# table helpers (same logic as compute_cosine_scores.py)
# ----------------------------------------------------------------------------
def load_paths(path_lookup_csv):
    id_to_path = {}
    with open(path_lookup_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                id_to_path[int(row[0])] = row[1]
    return id_to_path


def find_pair_columns(columns, label):
    cols_lower = {c.lower(): c for c in columns}
    for suffix, key_type in (("path", "path"), ("id", "id")):
        e = [c for lc, c in cols_lower.items() if "enroll" in lc and lc.endswith(suffix)]
        t = [c for lc, c in cols_lower.items() if "test" in lc and lc.endswith(suffix)]
        if len(e) == 1 and len(t) == 1:
            return e[0], t[0], key_type
    print(f"\nERROR: could not identify enroll/test columns in {label}: {list(columns)}",
          file=sys.stderr)
    sys.exit(1)


def get_file_ids(df, col_name, key_type, path_to_id, n_embeddings, np):
    if key_type == "id":
        ids = df[col_name].to_numpy().astype(np.int64)
        valid = (ids >= 0) & (ids < n_embeddings)
    else:
        col = df[col_name]
        codes = col.cat.codes.to_numpy()
        cats = col.cat.categories
        cat2id = np.array([path_to_id.get(p, -1) for p in cats], dtype=np.int64)
        ids = np.where(codes >= 0, cat2id[np.clip(codes, 0, len(cat2id) - 1)], -1)
        valid = (ids >= 0) & (ids < n_embeddings)
    return np.where(valid, ids, 0), valid


# ----------------------------------------------------------------------------
# embedding transforms (all label-free, all on the small 104k x 192 matrix)
# ----------------------------------------------------------------------------
def l2(x, np):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def build_transform(name, E, done, is_sdm, np):
    """Returns an (n, d) float32 matrix, rows L2-normalised, so that
    cosine(a, b) == dot(row_a, row_b). Rows that are not `done` stay zero."""
    X = l2(E.astype(np.float64), np)
    if name == "raw":
        pass
    elif name == "center":
        X = X - X[done].mean(axis=0, keepdims=True)
    elif name == "centermic":
        for m in (False, True):
            sel = done & (is_sdm == m)
            if sel.any():
                X[is_sdm == m] -= X[sel].mean(axis=0, keepdims=True)
    elif name == "whiten":
        mu = X[done].mean(axis=0, keepdims=True)
        X = X - mu
        C = np.cov(X[done], rowvar=False)
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 0.0) + 1e-6 * w.max()          # tame near-zero eigenvalues
        W = (V / np.sqrt(w)) @ V.T                        # ZCA
        X = X @ W
    else:
        print(f"ERROR: unknown transform '{name}'", file=sys.stderr)
        sys.exit(1)
    X = l2(X, np)
    X[~done] = 0.0
    return X.astype(np.float32)


# ----------------------------------------------------------------------------
# AS-norm statistics, one (mean, std) pair per FILE (not per trial)
# ----------------------------------------------------------------------------
def speaker_keys(id_to_path, n):
    keys = []
    for i in range(n):
        p = id_to_path.get(i)
        keys.append(Path(p).name.split("_")[0] if p else "")
    return keys


def asnorm_stats(X, done, spk_codes, cohort_ids, top_n, exclude_same_speaker,
                 device, torch, block=4096):
    """For every file: scores against the cohort -> keep the top_n highest
    (the impostors that look MOST like it) -> mean and std."""
    n, d = X.shape
    Xt = torch.from_numpy(X).to(device)
    C = Xt[torch.from_numpy(cohort_ids).to(device)]               # (K, d)
    c_spk = torch.from_numpy(spk_codes[cohort_ids]).to(device)    # (K,)
    s_all = torch.from_numpy(spk_codes).to(device)
    mu = torch.zeros(n, device=device)
    sd = torch.ones(n, device=device)
    for s in range(0, n, block):
        e = min(s + block, n)
        sc = Xt[s:e] @ C.T                                        # (B, K)
        if exclude_same_speaker:
            sc = sc.masked_fill(s_all[s:e, None] == c_spk[None, :], float("-inf"))
        top = torch.topk(sc, k=min(top_n, sc.shape[1]), dim=1).values
        top = torch.where(torch.isfinite(top), top, torch.nan)
        mu[s:e] = torch.nanmean(top, dim=1)
        dev = top - mu[s:e, None]
        sd[s:e] = torch.sqrt(torch.nanmean(dev * dev, dim=1)).clamp_min(1e-6)
    return mu, sd


# ----------------------------------------------------------------------------
def score_table(label, enroll_ids, test_ids, valid, variants, mats, stats, done_t,
                chunk_size, device, np, torch):
    n = len(enroll_ids)
    out = {v["name"]: np.full(n, np.nan, dtype=np.float32) for v in variants}
    t0 = time.time()
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        ei = torch.from_numpy(enroll_ids[s:e]).to(device)
        ti = torch.from_numpy(test_ids[s:e]).to(device)
        ok = torch.from_numpy(valid[s:e]).to(device) & done_t[ei] & done_t[ti]
        for v in variants:
            X = mats[v["transform"]]
            sc = torch.einsum("ij,ij->i", X[ei], X[ti])
            if v["asnorm"]:
                mu, sd = stats[v["transform"]]
                sc = 0.5 * ((sc - mu[ei]) / sd[ei] + (sc - mu[ti]) / sd[ti])
            sc = torch.where(ok, sc, torch.nan)
            out[v["name"]][s:e] = sc.cpu().numpy()
        el = time.time() - t0
        print(f"  [{label}] {e:,}/{n:,} rows x {len(variants)} variants "
              f"({e/el:,.0f} rows/s)", file=sys.stderr)
    return out


def parse_variants(spec):
    res = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        asn = tok.endswith("+asnorm")
        tr = tok[:-len("+asnorm")] if asn else tok
        res.append({"name": tok, "transform": tr, "asnorm": asn,
                    "fname": tok.replace("+", "_")})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positive_trials_parquet")
    ap.add_argument("negative_trials_joblib")
    ap.add_argument("path_lookup_csv")
    ap.add_argument("embeddings_npy")
    ap.add_argument("embeddings_done_npy")
    ap.add_argument("--out-dir", default="scores_v2")
    ap.add_argument("--variants",
                    default="raw,center,centermic,whiten,centermic+asnorm,whiten+asnorm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-size", type=int, default=4_000_000)
    ap.add_argument("--cohort-size", type=int, default=3000,
                    help="Random files used as the AS-norm impostor cohort.")
    ap.add_argument("--cohort-top", type=int, default=300,
                    help="AS-norm: use the top-N highest cohort scores per file.")
    ap.add_argument("--no-exclude-same-speaker", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for p in (args.positive_trials_parquet, args.negative_trials_joblib,
              args.path_lookup_csv, args.embeddings_npy, args.embeddings_done_npy):
        if not Path(p).is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    import numpy as np
    import pandas as pd
    import joblib
    import torch

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"WARNING: {args.device} unavailable, using CPU.", file=sys.stderr)
    variants = parse_variants(args.variants)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    E = np.load(args.embeddings_npy)
    done = np.load(args.embeddings_done_npy).astype(bool)
    n = E.shape[0]
    id_to_path = load_paths(args.path_lookup_csv)
    path_to_id = {p: i for i, p in id_to_path.items()}
    is_sdm = np.array([("/sdm/" in id_to_path.get(i, "")) for i in range(n)], dtype=bool)
    print(f"{n:,} ids, dim {E.shape[1]}, {int(done.sum()):,} embedded "
          f"({int((done & is_sdm).sum()):,} sdm / {int((done & ~is_sdm).sum()):,} ihm).",
          file=sys.stderr)

    # ---- build the matrices the requested variants need ----
    needed = sorted({v["transform"] for v in variants})
    mats = {}
    for t in needed:
        print(f"Building transform '{t}' ...", file=sys.stderr)
        mats[t] = torch.from_numpy(build_transform(t, E, done, is_sdm, np)).to(device)
    done_t = torch.from_numpy(done).to(device)

    # ---- AS-norm stats ----
    stats = {}
    asn_transforms = sorted({v["transform"] for v in variants if v["asnorm"]})
    if asn_transforms:
        keys = speaker_keys(id_to_path, n)
        cnt = Counter(k for k, d_ in zip(keys, done) if d_)
        print(f"Speaker keys parsed from filenames: {len(cnt)} distinct. "
              f"Examples: {[k for k, _ in cnt.most_common(5)]}", file=sys.stderr)
        if len(cnt) < 20 or len(cnt) > 20000:
            print("  WARNING: that looks wrong for a speaker tag -- check before "
                  "trusting AS-norm (or pass --no-exclude-same-speaker).", file=sys.stderr)
        uniq = {k: i for i, k in enumerate(sorted(cnt))}
        spk_codes = np.array([uniq.get(k, -1) for k in keys], dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        cohort_ids = np.sort(rng.choice(np.flatnonzero(done),
                                        size=min(args.cohort_size, int(done.sum())),
                                        replace=False)).astype(np.int64)
        for t in asn_transforms:
            t0 = time.time()
            stats[t] = asnorm_stats(mats[t].cpu().numpy(), done, spk_codes, cohort_ids,
                                    args.cohort_top, not args.no_exclude_same_speaker,
                                    device, torch)
            print(f"  AS-norm stats for '{t}' ({len(cohort_ids)} cohort files, top "
                  f"{args.cohort_top}) in {time.time()-t0:.0f}s", file=sys.stderr)

    # ---- positive trials ----
    print(f"\nLoading {args.positive_trials_parquet} ...", file=sys.stderr)
    df = pd.read_parquet(args.positive_trials_parquet)
    ec, tc, kt = find_pair_columns(df.columns, "positive_trials")
    ei, ev = get_file_ids(df, ec, kt, path_to_id, n, np)
    ti, tv = get_file_ids(df, tc, kt, path_to_id, n, np)
    nrows = len(df)
    del df
    print(f"  {nrows:,} rows (enroll='{ec}', test='{tc}', {kt}).", file=sys.stderr)
    res = score_table("positive", ei, ti, ev & tv, variants, mats, stats, done_t,
                      args.chunk_size, device, np, torch)
    for v in variants:
        f = out_dir / f"positive_cosine_scores_{v['fname']}.npy"
        np.save(f, res[v["name"]])
        print(f"  saved {f}  mean={np.nanmean(res[v['name']]):.4f}", file=sys.stderr)
    del res, ei, ti, ev, tv

    # ---- negative trials ----
    print(f"\nLoading {args.negative_trials_joblib} ...", file=sys.stderr)
    df = joblib.load(args.negative_trials_joblib)
    ec, tc, kt = find_pair_columns(df.columns, "negative_trials_v2")
    ei, ev = get_file_ids(df, ec, kt, path_to_id, n, np)
    ti, tv = get_file_ids(df, tc, kt, path_to_id, n, np)
    nrows = len(df)
    del df
    print(f"  {nrows:,} rows (enroll='{ec}', test='{tc}', {kt}).", file=sys.stderr)
    res = score_table("negative", ei, ti, ev & tv, variants, mats, stats, done_t,
                      args.chunk_size, device, np, torch)
    for v in variants:
        f = out_dir / f"negative_cosine_scores_{v['fname']}.npy"
        np.save(f, res[v["name"]])
        print(f"  saved {f}  mean={np.nanmean(res[v['name']]):.4f}", file=sys.stderr)

    print("\nAll done. Score arrays are aligned BY ROW POSITION to the original "
          "trial tables (untouched).", file=sys.stderr)


if __name__ == "__main__":
    main()