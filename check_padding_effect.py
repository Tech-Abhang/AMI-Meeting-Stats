#!/usr/bin/env python3
"""
check_padding_effect.py

Answers one question, cheaply: did batching + zero-padding distort the
embeddings in embeddings.npy?

For N files, this script re-embeds each file ALONE (batch of 1, so there
is no padding at all) and compares that to what is already stored in
embeddings.npy (made in padded batches of 64). For each file it prints
the cosine between the two versions of the SAME file's embedding:

    ~1.00  -> padding changed nothing, no point re-running the embeddings
    well below 1.00 -> padding is distorting embeddings, worth re-running

Results are also broken out by clip length, since short clips padded up
to a long batch-mate are where distortion would show first.

Read-only: touches nothing on disk, writes nothing.

Usage:
    python check_padding_effect.py \\
        encoded_trials/path_lookup.csv \\
        embeddings.npy embeddings_done.npy \\
        --audio-root "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \\
        --device cuda:0 --start-id 50000 --n-files 300
"""

import argparse
import csv
import sys
from pathlib import Path


def load_path_lookup(path_lookup_csv):
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
    return [id_to_path.get(i) for i in range(max_id + 1)]


def load_wav(path, np, librosa):
    """Identical preprocessing to compute_embeddings.py."""
    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (1.01 * peak)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path_lookup_csv")
    ap.add_argument("embeddings_npy")
    ap.add_argument("embeddings_done_npy")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start-id", type=int, default=50000,
                     help="First file id to test. Keep this well past ~3000: the "
                          "first ~2000 ids were embedded earlier on CPU in a "
                          "different batch size.")
    ap.add_argument("--n-files", type=int, default=300)
    args = ap.parse_args()

    for p in (args.path_lookup_csv, args.embeddings_npy, args.embeddings_done_npy):
        if not Path(p).is_file():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    try:
        import numpy as np
        import torch
        import librosa
        from speechbrain.inference import SpeakerRecognition
    except ImportError as e:
        print(f"Missing dependency ({e}).", file=sys.stderr)
        sys.exit(1)

    audio_root = Path(args.audio_root)
    id_to_path = load_path_lookup(args.path_lookup_csv)
    stored = np.load(args.embeddings_npy, mmap_mode="r")
    done = np.load(args.embeddings_done_npy)

    print(f"Loading model on {args.device} ...", file=sys.stderr)
    model = SpeakerRecognition.from_hparams(
        source=args.model_source, run_opts={"device": args.device})

    cosines, seconds = [], []
    end_id = min(args.start_id + args.n_files, len(id_to_path))
    for fid in range(args.start_id, end_id):
        if not done[fid] or id_to_path[fid] is None:
            continue
        try:
            wav = load_wav(audio_root / id_to_path[fid].lstrip("/"), np, librosa)
            wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(args.device)
            with torch.no_grad():
                solo = model.encode_batch(wav_t).squeeze().cpu().numpy()
        except Exception as e:
            print(f"  skip id={fid}: {e}", file=sys.stderr)
            continue
        batched = np.asarray(stored[fid], dtype=np.float64)
        solo = solo.astype(np.float64)
        denom = np.linalg.norm(batched) * np.linalg.norm(solo)
        if denom == 0:
            continue
        cosines.append(float(np.dot(batched, solo) / denom))
        seconds.append(len(wav) / 16000.0)

    if not cosines:
        print("No files could be compared.", file=sys.stderr)
        sys.exit(1)

    c = np.array(cosines)
    s = np.array(seconds)
    print(f"\nCompared {len(c)} files: padded-batch embedding vs. alone (no padding)")
    print(f"  cosine between the two versions: mean={c.mean():.4f}  "
          f"min={c.min():.4f}  5th percentile={np.percentile(c, 5):.4f}")

    print("\n  by clip length:")
    for lo, hi, name in ((0, 1, "< 1 s"), (1, 2, "1-2 s"), (2, 4, "2-4 s"), (4, 1e9, "> 4 s")):
        m = (s >= lo) & (s < hi)
        if m.any():
            print(f"    {name:6s}  n={int(m.sum()):4d}  mean cosine={c[m].mean():.4f}")

    print()
    if c.mean() >= 0.99 and np.percentile(c, 5) >= 0.97:
        print("Verdict: padding is NOT the problem (embeddings barely change without "
              "it). Re-running the embeddings would not help.")
    elif c.mean() >= 0.95:
        print("Verdict: padding changes embeddings a little. Probably a minor "
              "contributor, not the main cause of the overlap.")
    else:
        print("Verdict: padding is visibly distorting embeddings. Worth re-running "
              "the embeddings without it.")


if __name__ == "__main__":
    main()