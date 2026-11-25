#!/usr/bin/env python3
"""
compute_embeddings_redimnet.py

Same job as compute_embeddings.py (ECAPA), but with ReDimNet
(https://github.com/IDRnD/redimnet): one embedding per UNIQUE file in
path_lookup.csv (104,748 files), stored in an id-indexed dense array.

  embeddings_redimnet_<tag>.npy        float32 (n_ids, emb_dim); row i = file id i
  embeddings_redimnet_<tag>_done.npy   bool mask (True = real embedding)
  embeddings_redimnet_<tag>_errors.csv file id, path, error (only if something failed)

<tag> = <model_name>_<dataset>_<train_type>, e.g. b6_vox2_ft_lm.
NOTHING existing is touched (ECAPA files keep their own names); trials are not read at all.

HOW THIS DIFFERS FROM THE ECAPA SCRIPT (on purpose)
---------------------------------------------------
* ReDimNet takes a raw waveform [N, T] and has NO documented length/mask
  argument, so padding is not allowed to happen at all. Batches contain
  clips of EXACTLY the same number of samples (clip length is read from the
  filename in centiseconds = exact multiples of 160 samples at 16 kHz; any
  file whose real length differs by a sample is split off into its own
  sub-batch after loading). Padding used: 0.0 % by construction.
* Audio goes in as the README does it: raw samples, mono, 16 kHz. NO DC
  removal and NO peak normalisation (the ECAPA script did those; they were
  part of that notebook, not of ReDimNet's recipe). Use --peak-norm only if
  you want to test that deliberately.
* fp32 by default (README uses fp16 autocast on GPU; --fp16 turns it on).

PILOT FIRST (recommended)
-------------------------
  --pilot 2000   runs ~2,000 files (the 100 shortest, the 100 longest, rest random),
                 writes to *_pilot.npy files, and prints: embedding size, speed,
                 projected time for all files, any failures (watch the shortest
                 clips), batch-vs-single agreement, and mean cosine of
                 same-speaker vs different-speaker pairs (must be clearly higher).

Usage (pilot, then full):
    python compute_embeddings_redimnet.py encoded_trials/path_lookup.csv \\
        --audio-root "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \\
        --model-name b6 --train-type ft_lm --dataset vox2 \\
        --device cuda:0 --pilot 2000

    python compute_embeddings_redimnet.py encoded_trials/path_lookup.csv \\
        --audio-root "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \\
        --model-name b6 --train-type ft_lm --dataset vox2 \\
        --device cuda:0 --workers 8

Needs: pip install torch torchaudio soundfile numpy   (hubconf.py of ReDimNet imports torchaudio)
First model load downloads the ReDimNet code + weights from GitHub (needs internet);
torch.hub caches them under ~/.cache/torch/hub. --trust-repo is passed to torch.hub, so
code from the IDRnD repo is executed; use --repo <local clone> --source local to avoid that.
"""

import argparse
import csv
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SR = 16000


def _resolve_path(audio_root: Path, stored_path: str) -> Path:
    return audio_root / stored_path.lstrip("/")


def load_path_lookup(path_lookup_csv):
    id_to_path, max_id = {}, -1
    with open(path_lookup_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                fid = int(row[0])
                id_to_path[fid] = row[1]
                max_id = max(max_id, fid)
    return [id_to_path.get(i) for i in range(max_id + 1)]


def duration_cs(stored_path, audio_root):
    """Clip length in centiseconds from the filename's trailing 'begin-end'
    (..._0000408-0000459.wav = 51 cs). Falls back to file size
    (16-bit mono 16 kHz = 320 bytes per centisecond). Free: no audio read."""
    stem = Path(stored_path).stem
    try:
        b, e = stem.rsplit("_", 1)[-1].split("-")
        d = int(e) - int(b)
        if d > 0:
            return d
    except Exception:
        pass
    try:
        return max((_resolve_path(audio_root, stored_path).stat().st_size - 44) // 320, 0)
    except Exception:
        return 0


def speaker_of(stored_path):
    """speaker_meeting_split_mic_begin-end.wav -> speaker"""
    return Path(stored_path).stem.split("_")[0]


def plan_batches(ids, dur_cs, max_items, max_batch_seconds):
    """Group by EXACT filename duration, then cut into batches of at most
    max_items files and at most max_batch_seconds of total audio."""
    by_len = {}
    for i in ids:
        by_len.setdefault(dur_cs[i], []).append(i)
    batches = []
    for d in sorted(by_len):
        secs = max(d, 5) / 100.0
        per = max(1, min(max_items, int(max_batch_seconds // secs)))
        g = by_len[d]
        for s in range(0, len(g), per):
            batches.append(g[s:s + per])
    return batches


def load_audio(path, np, sf, peak_norm=False):
    """Raw mono 16 kHz float32. soundfile first (no resampling, fast);
    librosa fallback only if the file is not 16 kHz mono."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR or data.shape[1] != 1:
        import librosa
        data, _ = librosa.load(str(path), sr=SR, mono=True)
    else:
        data = data[:, 0]
    if peak_norm and len(data):
        data = data - data.mean()
        peak = np.abs(data).max()
        if peak > 0:
            data = data / (1.01 * peak)
    return np.ascontiguousarray(data, dtype=np.float32)


def load_model(args, torch):
    model = torch.hub.load(args.repo, "ReDimNet", model_name=args.model_name,
                           train_type=args.train_type, dataset=args.dataset,
                           source=args.source, trust_repo=True)
    return model


def embed(model, wavs, device, fp16, torch, np):
    """wavs: list of 1-D float32 arrays of EXACTLY the same length -> (n, dim) float32."""
    x = torch.from_numpy(np.stack(wavs)).to(device)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=torch.float16,
                                         enabled=(fp16 and dev_type == "cuda")):
        out = model(x)
    return out.float().cpu().numpy()


def pick_pilot(ids, dur_cs, n, seed=0):
    import random
    order = sorted(ids, key=lambda i: dur_cs[i])
    k = min(100, n // 4)
    chosen = set(order[:k]) | set(order[-k:])
    rest = [i for i in ids if i not in chosen]
    random.Random(seed).shuffle(rest)
    chosen |= set(rest[:max(0, n - len(chosen))])
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path_lookup_csv")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--model-name", default="b6", help="b0..b6, M, S")
    ap.add_argument("--train-type", default="ft_lm", help="ptn | ft_lm | ft_mix")
    ap.add_argument("--dataset", default="vox2", help="vox2 | vb2 | vb2+vox2+cnc")
    ap.add_argument("--repo", default="IDRnD/ReDimNet")
    ap.add_argument("--source", default="github", choices=["github", "local"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--done-out", default=None)
    ap.add_argument("--errors-out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp16", action="store_true", help="autocast fp16 on GPU (default fp32)")
    ap.add_argument("--peak-norm", action="store_true",
                    help="DC removal + peak normalisation (ECAPA-script style). Off = raw, as in ReDimNet's README.")
    ap.add_argument("--batch-size", type=int, default=64, help="max files per batch")
    ap.add_argument("--max-batch-seconds", type=float, default=240.0,
                    help="cap on total audio seconds per batch (GPU memory guard; lower it if you hit OOM)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--pilot", type=int, default=0,
                    help="N>0: only N files (100 shortest, 100 longest, rest random), "
                         "outputs get a _pilot suffix, prints sanity checks")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    tag = f"{args.model_name}_{args.dataset}_{args.train_type}".replace("+", "-")
    base = f"embeddings_redimnet_{tag}" + ("_pilot" if args.pilot else "")
    out_path = Path(args.out or f"{base}.npy")
    done_path = Path(args.done_out or f"{base}_done.npy")
    err_path = Path(args.errors_out or f"{base}_errors.csv")

    lookup, audio_root = Path(args.path_lookup_csv), Path(args.audio_root)
    if not lookup.is_file():
        sys.exit(f"ERROR: {lookup} not found.")
    if not audio_root.is_dir():
        sys.exit(f"ERROR: --audio-root {audio_root} is not a directory.")
    try:
        import numpy as np
        import torch
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"Missing dependency ({e}). Install: pip install torch torchaudio soundfile numpy")

    id_to_path = load_path_lookup(lookup)
    n_ids = len(id_to_path)
    print(f"{n_ids:,} file ids.", file=sys.stderr)
    dur = {i: duration_cs(id_to_path[i], audio_root) for i in range(n_ids)}
    total_audio_s = sum(dur.values()) / 100.0
    print(f"Total audio in lookup: {total_audio_s/3600:.1f} h.", file=sys.stderr)

    print(f"\nLoading ReDimNet {args.model_name} / {args.dataset} / {args.train_type} "
          f"on {args.device} ...", file=sys.stderr)
    model = load_model(args, torch).to(args.device).eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters.", file=sys.stderr)

    # embedding size from a 2 s noise probe (no disk needed)
    probe = embed(model, [np.random.default_rng(0).standard_normal(2 * SR).astype(np.float32) * 0.05],
                  args.device, args.fp16, torch, np)
    emb_dim = probe.shape[-1]
    print(f"Embedding dimension: {emb_dim}", file=sys.stderr)

    # ---- output arrays (resume-safe) ----
    if out_path.is_file() and done_path.is_file():
        embeddings, done = np.load(out_path), np.load(done_path)
        if embeddings.shape != (n_ids, emb_dim) or len(done) != n_ids:
            sys.exit(f"ERROR: existing {out_path} does not match (n_ids, dim)=({n_ids},{emb_dim}). "
                     f"Delete it (and {done_path}) to start fresh.")
        print(f"Resuming from {out_path}.", file=sys.stderr)
    else:
        embeddings = np.zeros((n_ids, emb_dim), dtype=np.float32)
        done = np.zeros(n_ids, dtype=bool)

    universe = list(range(n_ids))
    if args.pilot:
        universe = pick_pilot(universe, dur, args.pilot)
        print(f"PILOT: {len(universe):,} files.", file=sys.stderr)
    todo = [i for i in universe if not done[i]]
    print(f"{len(universe) - len(todo):,} already done, {len(todo):,} to do.", file=sys.stderr)

    batches = plan_batches(todo, dur, args.batch_size, args.max_batch_seconds)
    todo_audio_s = sum(dur[i] for i in todo) / 100.0
    print(f"{len(batches):,} same-length batches.", file=sys.stderr)

    errors, n_split = [], 0
    n_done_now, last_report = 0, 0
    t0 = time.time()

    def save():
        np.save(out_path, embeddings)
        np.save(done_path, done)

    def _load(fid):
        try:
            return fid, load_audio(_resolve_path(audio_root, id_to_path[fid]), np, sf, args.peak_norm), None
        except Exception as e:
            return fid, None, f"load: {e}"

    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending, it = deque(), iter(batches)

    def enqueue():
        b = next(it, None)
        if b is not None:
            pending.append((b, [pool.submit(_load, f) for f in b]))

    for _ in range(max(1, args.prefetch)):
        enqueue()

    def store(fid, vec):
        if not np.all(np.isfinite(vec)):
            errors.append((fid, id_to_path[fid], "non-finite embedding"))
            return
        embeddings[fid] = vec
        done[fid] = True

    while pending:
        batch_ids, futs = pending.popleft()
        enqueue()
        by_len = {}
        for fu in futs:
            fid, wav, err = fu.result()
            if err:
                errors.append((fid, id_to_path[fid], err))
            elif len(wav) == 0:
                errors.append((fid, id_to_path[fid], "empty audio"))
            else:
                by_len.setdefault(len(wav), []).append((fid, wav))
        if len(by_len) > 1:
            n_split += len(by_len) - 1      # filename length != real length for some files
        for L, items in by_len.items():
            fids = [f for f, _ in items]
            try:
                vecs = embed(model, [w for _, w in items], args.device, args.fp16, torch, np)
                for f, v in zip(fids, vecs):
                    store(f, v)
            except Exception:
                for f, w in items:                       # fall back: one file at a time
                    try:
                        store(f, embed(model, [w], args.device, args.fp16, torch, np)[0])
                    except Exception as e2:
                        errors.append((f, id_to_path[f], f"{L} samples: {str(e2)[:200]}"))
        n_done_now += len(batch_ids)
        if n_done_now - last_report >= args.progress_every:
            last_report = n_done_now
            el = time.time() - t0
            rate = n_done_now / el
            print(f"  ... {n_done_now:,}/{len(todo):,}  {rate:.1f} files/s  "
                  f"~{(len(todo) - n_done_now) / rate / 60:.1f} min left", file=sys.stderr)
            save()

    pool.shutdown(wait=True)
    save()
    el = time.time() - t0
    if errors:
        with open(err_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_id", "path", "error"])
            w.writerows(errors)

    print(f"\nDone: {len(todo):,} files in {el/60:.1f} min, {len(errors)} failed. "
          f"Padding used: 0.0% (exact-length batches; {n_split} files differed by a sample "
          f"from their filename length and were batched separately).", file=sys.stderr)
    print(f"Saved {out_path.resolve()} {embeddings.shape} and {done_path.resolve()}", file=sys.stderr)
    if errors:
        print(f"Errors -> {err_path}", file=sys.stderr)

    # ---------------- pilot sanity checks ----------------
    if args.pilot:
        ok = [i for i in universe if done[i]]
        print("\n=== PILOT CHECKS ===", file=sys.stderr)
        proj = total_audio_s / (todo_audio_s / el) / 60 if el > 0 and todo_audio_s > 0 else float("nan")
        print(f"1. Speed: {len(todo)/el:.1f} files/s, {todo_audio_s/el:.0f} audio-seconds per second. "
              f"Projected time for ALL {n_ids:,} files: ~{proj:.0f} min "
              f"(pilot has extra long/short clips, so treat as rough).", file=sys.stderr)
        print(f"2. Failures: {len(errors)} of {len(universe)}. "
              + (f"First errors: {errors[:3]}" if errors else "None."), file=sys.stderr)
        if errors:
            ed = sorted(dur[e[0]] for e in errors)
            print(f"   failed clip lengths (cs): min {ed[0]}, max {ed[-1]}", file=sys.stderr)

        # batch vs single agreement on one real group of equal-length files
        chk = None
        for grp in plan_batches(ok, dur, 8, 1e9):
            if len(grp) < 2:
                continue
            wavs = [load_audio(_resolve_path(audio_root, id_to_path[i]), np, sf, args.peak_norm) for i in grp]
            if len({len(w) for w in wavs}) == 1:
                chk = wavs
                break
        if chk:
            eb = embed(model, chk, args.device, args.fp16, torch, np)
            es = np.stack([embed(model, [w], args.device, args.fp16, torch, np)[0] for w in chk])
            cs = np.sum(eb * es, 1) / (np.linalg.norm(eb, axis=1) * np.linalg.norm(es, axis=1))
            print(f"3. Batch vs one-by-one ({len(chk)} files, {len(chk[0])} samples): "
                  f"min cosine {cs.min():.6f} (should be ~1.0)", file=sys.stderr)
        else:
            print("3. Batch-vs-single check skipped (no equal-length group in the pilot).", file=sys.stderr)
        # same vs different speaker
        spk = np.array([speaker_of(id_to_path[i]) for i in ok])
        E = embeddings[ok]
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, len(ok), 300_000), rng.integers(0, len(ok), 300_000)
        keep = a != b
        a, b = a[keep], b[keep]
        c = np.sum(E[a] * E[b], 1)
        same = spk[a] == spk[b]
        if same.any() and (~same).any():
            print(f"4. Mean cosine, same speaker {c[same].mean():.3f} ({int(same.sum()):,} pairs) vs "
                  f"different speaker {c[~same].mean():.3f}. Same must be clearly higher.", file=sys.stderr)
        else:
            print("4. Not enough same-speaker pairs in this pilot sample to compare.", file=sys.stderr)
        print(f"5. Embedding std across files (per dim, mean): {E.std(0).mean():.4f} "
              f"(near 0 would mean every file gets the same vector).", file=sys.stderr)


if __name__ == "__main__":
    main()