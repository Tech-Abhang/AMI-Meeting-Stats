#!/usr/bin/env python3
"""
compute_embeddings.py

Extracts one P-ECAPA speaker embedding per UNIQUE audio file in
path_lookup.csv (104,748 files) -- not per trial row. Same "operate on
the small dictionary, not the big table" pattern as compute_audio_features.py:
27.3M positive rows + 273M negative rows reuse the same ~104,748 files
over and over, so embeddings are computed once per file and looked up
by id afterwards (the next script, compute_cosine_scores.py, will do
that lookup+join -- not built yet, this is embeddings only, as asked).

ONE DELIBERATE DIFFERENCE FROM THE REFERENCE NOTEBOOK
---------------------------------------------------------
The notebook's dynamic_padding() passes raw sample counts as wav_lens
to encode_batch(). SpeechBrain's documented convention is that wav_lens
is a RELATIVE length (0.0-1.0, fraction of the padded batch length),
used internally to mask out the padding before pooling. This script
normalizes lengths to that 0-1 fraction before calling encode_batch --
if that's not done, encode_batch doesn't know how much of a padded
waveform is real audio vs. zeros, so short+long files mixed in the same
batch could cross-contaminate. Flagging this now so it isn't a silent
mismatch with the notebook you're using as reference.

OUTPUT
------
embeddings.npy    -- dense float32 array, shape (max_id+1, emb_dim).
                     Row i = embedding for path_lookup.csv's file id i.
                     Same "id-indexed dense array" trick as
                     id_to_energy_arr in join_audio_features.py -- fast
                     numpy fancy-indexing later, no dict-per-row lookups
                     across 300M trial rows.
embeddings_done.npy -- boolean mask, same length, True = row is a real
                     embedding, False = never computed or errored.
                     This is what makes --resume work.
embeddings_errors.csv -- file id, path, error message, for any file
                     that failed (same spirit as compute_audio_features.py's
                     4-file error list).

Usage (CPU, what you already ran):
    python compute_embeddings.py \\
        encoded_trials/path_lookup.csv \\
        --audio-root "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \\
        --out embeddings.npy \\
        --model-source speechbrain/spkrec-ecapa-voxceleb \\
        --batch-size 16 \\
        --self-test 5

Usage (GPU -- this machine has 3x Tesla V100 32GB, use one):
    python compute_embeddings.py \\
        encoded_trials/path_lookup.csv \\
        --audio-root "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \\
        --out embeddings.npy \\
        --model-source speechbrain/spkrec-ecapa-voxceleb \\
        --device cuda:0 \\
        --batch-size 64 \\
        --workers 8 \\
        --self-test 5

If you have your own pruned P-ECAPA checkpoint (not the stock
speechbrain/spkrec-ecapa-voxceleb used in the reference notebook), point
--model-source at that local directory/hyperparams file instead -- the
rest of the script doesn't care which checkpoint it is, as long as
SpeechBrain's SpeakerRecognition.from_hparams() can load it and
encode_batch() returns a fixed-length vector per file.

WHY IT WAS SLOW ON CPU (3 files/s, ~9.5 hours projected)
------------------------------------------------------------
The disk-read probe (check_specs.py) showed this drive can read ~32
files/s -- 10x faster than the 3 files/s the run was actually getting.
So the disk wasn't the bottleneck; a 22M-parameter neural net doing
forward passes on CPU, one small batch at a time, was. With 3 Tesla
V100 GPUs available, --device cuda:0 should remove that bottleneck
almost entirely -- GPUs do exactly this kind of batched matrix math
orders of magnitude faster than CPU. --workers adds a thread pool that
loads/decodes the NEXT batch's audio files off disk while the GPU is
still busy with the CURRENT batch, so the (now much faster) GPU isn't
left waiting on librosa.load() between batches.
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _resolve_path(audio_root: Path, stored_path: str) -> Path:
    """Same fix as compute_audio_features.py: path_lookup.csv stores paths
    like '/ihm/train/...' -- joining that naively with os.path.join would
    discard audio_root because the second argument looks absolute."""
    return audio_root / stored_path.lstrip("/")


def load_path_lookup(path_lookup_csv):
    """id -> path string, in id order (id 0, 1, 2, ... with no gaps
    expected, same as every other script in this project)."""
    id_to_path = {}
    max_id = -1
    with open(path_lookup_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                fid = int(row[0])
                id_to_path[fid] = row[1]
                max_id = max(max_id, fid)
    id_to_path_list = [id_to_path.get(i) for i in range(max_id + 1)]
    return id_to_path_list


def load_wav(path, np, librosa):
    """Same preprocessing as the reference notebook's load_timit_wav:
    mono, 16kHz, DC-offset removed, peak-normalized to just under 1.0."""
    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (1.01 * peak)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path_lookup_csv")
    ap.add_argument("--audio-root", required=True,
                     help="Prefix to join with each stored path, e.g. "
                          ".../AMI_Meeting_Corpus/audio")
    ap.add_argument("--out", default="embeddings.npy")
    ap.add_argument("--done-out", default="embeddings_done.npy")
    ap.add_argument("--errors-out", default="embeddings_errors.csv")
    ap.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb",
                     help="HuggingFace id or local path SpeechBrain can load "
                          "with SpeakerRecognition.from_hparams(). Point this "
                          "at your own P-ECAPA checkpoint if you have one.")
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda:0' / 'cuda:1' / 'cuda:2'")
    ap.add_argument("--batch-size", type=int, default=16,
                     help="16 was sized for CPU. On a V100, 64 is a reasonable start -- "
                          "32GB VRAM for a 22M-param model has a lot of headroom.")
    ap.add_argument("--workers", type=int, default=4,
                     help="Thread pool size for loading/decoding wav files off disk in "
                          "parallel with GPU inference. Keep modest (4-8) -- more threads "
                          "doesn't help once the disk itself is saturated.")
    ap.add_argument("--self-test", type=int, default=5,
                     help="Run the model on this many real files before "
                          "committing to the full set. 0 to skip.")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    lookup_path = Path(args.path_lookup_csv)
    audio_root = Path(args.audio_root)
    if not lookup_path.is_file():
        print(f"ERROR: {lookup_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not audio_root.is_dir():
        print(f"ERROR: --audio-root {audio_root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
        import torch
        import librosa
        import torch.nn.functional as F
        from speechbrain.inference import SpeakerRecognition
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install speechbrain librosa torch numpy --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {lookup_path} ...", file=sys.stderr)
    id_to_path = load_path_lookup(lookup_path)
    n_ids = len(id_to_path)
    print(f"  {n_ids:,} file ids (0..{n_ids - 1}).", file=sys.stderr)

    print(f"\nLoading model '{args.model_source}' on {args.device} ...", file=sys.stderr)
    # Pass the device at LOAD time, not just via .to() afterward.
    # SpeakerRecognition.from_hparams() auto-detects a device internally and
    # encode_batch() later moves the INPUT audio to that internally-tracked
    # device -- which is a separate flag from where the model's weights
    # actually live. Calling .to(args.device) only moves the weights, so if
    # auto-detection picked "cuda" (even wrongly, as it did here), inputs and
    # weights end up on different devices and encode_batch() throws exactly
    # the "input type cuda, weight type cpu" error seen on the real run.
    # run_opts={"device": ...} sets that internal flag correctly up front.
    spk_model = SpeakerRecognition.from_hparams(
        source=args.model_source, run_opts={"device": args.device})
    spk_model = spk_model.to(args.device)
    n_params = sum(p.numel() for p in spk_model.parameters())
    print(f"  loaded. {n_params:,} parameters.", file=sys.stderr)

    # ---- self-test on a few real files before committing to all of them ----
    if args.self_test > 0:
        print(f"\nSelf-test: trying {args.self_test} real files ...", file=sys.stderr)
        n_ok = 0
        for fid in range(min(args.self_test, n_ids)):
            p = _resolve_path(audio_root, id_to_path[fid])
            try:
                wav = load_wav(p, np, librosa)
                wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(args.device)
                emb = spk_model.encode_batch(wav_t)
                n_ok += 1
                print(f"  OK  id={fid}  {p.name}  -> embedding shape {tuple(emb.shape)}",
                      file=sys.stderr)
            except Exception as e:
                print(f"  FAIL id={fid}  {p}\n       {e}", file=sys.stderr)
        if n_ok == 0:
            print("\nERROR: 0/%d self-test files worked. Fix --audio-root before "
                  "running on all %d files." % (args.self_test, n_ids), file=sys.stderr)
            sys.exit(1)
        print(f"  {n_ok}/{args.self_test} self-test files OK. Proceeding.\n", file=sys.stderr)

    # ---- figure out embedding dimension from one real file ----
    probe_id = 0
    probe_wav = load_wav(_resolve_path(audio_root, id_to_path[probe_id]), np, librosa)
    probe_t = torch.tensor(probe_wav, dtype=torch.float32).unsqueeze(0).to(args.device)
    emb_dim = spk_model.encode_batch(probe_t).shape[-1]
    print(f"Embedding dimension: {emb_dim}", file=sys.stderr)

    out_path = Path(args.out)
    done_path = Path(args.done_out)
    if out_path.is_file() and done_path.is_file():
        print(f"\nResuming from existing {out_path} / {done_path} ...", file=sys.stderr)
        embeddings = np.load(out_path)
        done = np.load(done_path)
        if embeddings.shape != (n_ids, emb_dim) or len(done) != n_ids:
            print(f"ERROR: existing output shape doesn't match current "
                  f"path_lookup.csv / model. Delete {out_path} and {done_path} "
                  f"to start fresh.", file=sys.stderr)
            sys.exit(1)
    else:
        embeddings = np.zeros((n_ids, emb_dim), dtype=np.float32)
        done = np.zeros(n_ids, dtype=bool)

    todo = [fid for fid in range(n_ids) if not done[fid]]
    print(f"{n_ids - len(todo):,} already done, {len(todo):,} remaining.", file=sys.stderr)

    errors = []
    t0 = time.time()
    n_processed = 0

    def save_checkpoint():
        np.save(out_path, embeddings)
        np.save(done_path, done)

    def _try_load(fid):
        try:
            wav = load_wav(_resolve_path(audio_root, id_to_path[fid]), np, librosa)
            return fid, torch.tensor(wav, dtype=torch.float32), None
        except Exception as e:
            return fid, None, str(e)

    pool = ThreadPoolExecutor(max_workers=args.workers)

    for batch_start in range(0, len(todo), args.batch_size):
        batch_ids = todo[batch_start:batch_start + args.batch_size]
        # Load+decode this batch's files concurrently (I/O + librosa decode
        # release the GIL for most of their work, so threads genuinely help
        # here) instead of one file at a time.
        waveforms, ok_ids = [], []
        for fid, wav_t, err in pool.map(_try_load, batch_ids):
            if err is None:
                waveforms.append(wav_t)
                ok_ids.append(fid)
            else:
                errors.append((fid, id_to_path[fid], err))

        if not waveforms:
            continue

        try:
            max_len = max(len(w) for w in waveforms)
            padded = torch.stack([F.pad(w, (0, max_len - len(w))) for w in waveforms])
            # relative lengths (0-1 fraction of max_len) -- SpeechBrain's
            # documented convention, see module docstring
            rel_lens = torch.tensor([len(w) / max_len for w in waveforms], dtype=torch.float32)
            with torch.no_grad():
                emb_batch = spk_model.encode_batch(padded.to(args.device), wav_lens=rel_lens.to(args.device))
            emb_batch = emb_batch.squeeze(1).cpu().numpy()
            for fid, emb in zip(ok_ids, emb_batch):
                embeddings[fid] = emb
                done[fid] = True
        except Exception as e:
            # whole batch failed -- fall back to one file at a time so a
            # single bad file doesn't lose the rest of the batch
            for fid in ok_ids:
                try:
                    wav = load_wav(_resolve_path(audio_root, id_to_path[fid]), np, librosa)
                    wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        emb = spk_model.encode_batch(wav_t).squeeze().cpu().numpy()
                    embeddings[fid] = emb
                    done[fid] = True
                except Exception as e2:
                    errors.append((fid, id_to_path[fid], str(e2)))

        n_processed += len(batch_ids)
        if n_processed % args.progress_every < args.batch_size:
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - n_processed) / rate if rate > 0 else float("inf")
            print(f"  ... {n_processed:,}/{len(todo):,} done, {rate:.1f} files/s, "
                  f"~{remaining/60:.1f} min remaining", file=sys.stderr)
            save_checkpoint()

    pool.shutdown(wait=True)
    save_checkpoint()
    if errors:
        with open(args.errors_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_id", "path", "error"])
            w.writerows(errors)

    elapsed_total = time.time() - t0
    print(f"\nDone. {n_processed:,} files attempted in {elapsed_total/60:.1f} minutes. "
          f"{len(errors)} errored.", file=sys.stderr)
    print(f"Saved: {out_path.resolve()}  (shape {embeddings.shape})", file=sys.stderr)
    print(f"Saved: {done_path.resolve()}", file=sys.stderr)
    if errors:
        print(f"Errors written to {args.errors_out}", file=sys.stderr)


if __name__ == "__main__":
    main()