#!/usr/bin/env python3
"""
compute_audio_features.py

Step 2 of the 3-plot plan: computes ENERGY and WADA-SNR for every unique
audio file, so Energy Difference and WADA SNR Difference plots can be
built the same way Duration Difference was (duration_eer_kde.py).

WHY THIS RUNS ONCE PER FILE, NOT ONCE PER TRIAL
------------------------------------------------
Positive trials: ~27.3M rows. Negative trials: ~273M rows. But both are
built from only 104,748 UNIQUE audio files (path_lookup.csv -- the same
dictionary already reused across create_positive_trials.py,
encode_positive_trials.py, and create_negative_trials_v2.py). Computing
energy/SNR per unique file and looking it up is ~2,600x less work than
computing it per trial, for identical output.

This script does ONLY that: audio_features.csv, keyed by the lookup id.
Attaching energy_diff/snr_diff columns to positive_trials and
negative_trials_v2 is the NEXT script (not this one) -- one step at a
time, same as duration.

TWO FEATURES COMPUTED PER FILE
-------------------------------
1. energy_db -- RMS energy in dB: 10*log10(mean(x^2) + eps).
   Straightforward, no assumptions.

2. wada_snr_db -- WADA (Waveform Amplitude Distribution Analysis) blind
   SNR estimate, Kim & Stern (2008).

   BE HONEST ABOUT WHAT THIS IS: the original paper ships a precomputed
   lookup table (SNR-in-dB -> a "G statistic") built once offline from a
   Gamma-vs-Gaussian simulation. That exact table is NOT something safe
   to reproduce from memory here -- copying one digit wrong would
   silently corrupt every downstream number with no error thrown. So
   instead of trusting a remembered table, this script REBUILDS the same
   calibration from the same theory, via Monte Carlo, every run, with a
   fixed seed (reproducible, and you can inspect exactly what it does --
   see build_wada_calibration_table()). It uses the same model the paper
   uses (speech magnitude ~ Gamma(shape=0.4), noise ~ Gaussian), so it is
   the same estimator, just self-derived instead of copy-pasted.

   PRACTICAL CONSEQUENCE: absolute dB values from this may not match a
   reference WADA-SNR implementation exactly. That does not matter for
   what you need it for -- snr_diff (test - enroll) computed the SAME
   way for every one of the 104,748 files is internally consistent, and
   internal consistency is all the EER/KDE confound-check plot needs.
   If you later want literature-exact calibration, swap in a vetted
   package's table -- the rest of this pipeline (energy_db, the join,
   the plot) does not change.

RESUMABLE
---------
Writes rows to --out as they complete (not buffered until the end). If
interrupted, re-running with the same --out drops any rows that
previously ERRORED (they're retried) and keeps rows that SUCCEEDED, so
re-running never permanently "remembers" a systematic failure as done
and never accumulates duplicate rows for the same id.

Usage:
    python compute_audio_features.py \\
        path_lookup.csv \\
        --audio-root "/path/to/AMI_Meeting_Corpus" \\
        --out audio_features.csv \\
        --workers 8
"""

import argparse
import csv
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------
# WADA calibration: built once in the main process, then handed to every
# worker via the Pool initializer (cheap: ~160 floats, not megabytes).
# ---------------------------------------------------------------------
def build_wada_calibration_table(np, alpha_speech=0.4, n_samples=400_000, seed=42):
    """Monte Carlo calibration curve G(SNR_db), SNR from -20 to 60 dB.

    Model: 'speech' magnitude ~ signed Gamma(shape=alpha_speech), unit
    power; additive Gaussian 'noise'. For each candidate SNR, mix the two
    at that ratio and measure G = log(mean|y|) - mean(log|y|) (the log of
    the arithmetic-mean-to-geometric-mean ratio of |y| -- always >= 0 by
    AM-GM, and larger for more sharply-peaked / heavy-tailed
    distributions). G increases monotonically with SNR under this model
    (more Gamma-like speech, less Gaussian-like noise = peakier signal).
    Averaging over n_samples per grid point keeps the curve smooth.
    """
    rng = np.random.default_rng(seed)
    snr_grid_db = np.arange(-20, 61, 1.0)
    g_grid = np.empty_like(snr_grid_db)
    eps = 1e-10

    for i, snr_db in enumerate(snr_grid_db):
        speech_mag = rng.gamma(shape=alpha_speech, scale=1.0 / alpha_speech, size=n_samples)
        speech = speech_mag * rng.choice([-1.0, 1.0], size=n_samples)
        speech_power = np.mean(speech ** 2)

        noise = rng.normal(0.0, 1.0, size=n_samples)
        noise_power = np.mean(noise ** 2)

        # scale noise so speech_power / noise_power_scaled == 10**(snr_db/10)
        desired_ratio = 10 ** (snr_db / 10.0)
        scale = np.sqrt(speech_power / (noise_power * desired_ratio))
        y = speech + noise * scale

        y_abs = np.abs(y) + eps
        g_grid[i] = np.log(np.mean(y_abs)) - np.mean(np.log(y_abs))

    # enforce monotonic non-decreasing (guards against MC noise causing a
    # tiny local dip, which would make np.interp's inversion ambiguous)
    g_grid = np.maximum.accumulate(g_grid)
    return snr_grid_db, g_grid


def estimate_wada_snr(x, snr_grid_db, g_grid, np):
    eps = 1e-10
    x_abs = np.abs(x).astype(np.float64) + eps
    g = float(np.log(np.mean(x_abs)) - np.mean(np.log(x_abs)))
    g_clipped = min(max(g, g_grid[0]), g_grid[-1])
    snr_db = float(np.interp(g_clipped, g_grid, snr_grid_db))
    return snr_db


# ---------------------------------------------------------------------
# Per-worker state: built ONCE per worker process (not per file).
# ---------------------------------------------------------------------
_worker_state = {}


def _worker_init(snr_grid_db_list, g_grid_list):
    import numpy as np
    _worker_state["np"] = np
    _worker_state["snr_grid_db"] = np.array(snr_grid_db_list)
    _worker_state["g_grid"] = np.array(g_grid_list)


def _resolve_path(audio_root, stored_path):
    """path_lookup.csv stores paths shortened relative to the corpus root
    (the '/{mic_type}/' convention used since encode_positive_trials.py --
    e.g. '/ihm/train/fee019/...'). That leading '/' means naive
    os.path.join(audio_root, stored_path) would DISCARD audio_root
    entirely (that's how os.path.join works when the second argument
    looks absolute) and silently try to open a file at filesystem root.
    Strip the leading separator first so the join actually anchors under
    audio_root."""
    if audio_root is None:
        return stored_path
    return str(Path(audio_root) / stored_path.lstrip("/\\"))


def _process_one(args):
    path_id, path_str, audio_root = args
    np = _worker_state["np"]
    try:
        import soundfile as sf
        full_path = _resolve_path(audio_root, path_str)
        x, _sr = sf.read(full_path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size == 0:
            return (path_id, None, None, "empty_file")
        eps = 1e-12
        energy_db = float(10.0 * np.log10(np.mean(x.astype(np.float64) ** 2) + eps))
        snr_db = estimate_wada_snr(x, _worker_state["snr_grid_db"], _worker_state["g_grid"], np)
        return (path_id, energy_db, snr_db, "")
    except Exception as e:
        return (path_id, None, None, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path_lookup_csv", help="path_lookup.csv (columns: id, path -- or similar; "
                                              "first two columns are used positionally if names differ)")
    ap.add_argument("--audio-root", default=None,
                     help="Corpus root directory to prepend to each stored path -- REQUIRED if "
                          "path_lookup.csv stores paths shortened relative to the corpus root "
                          "(e.g. '/ihm/train/...' rather than a full filesystem path). "
                          "e.g. --audio-root \"/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus\"")
    ap.add_argument("--out", default="audio_features.csv")
    ap.add_argument("--workers", type=int, default=0, help="0 = use all CPU cores")
    ap.add_argument("--progress-every", type=int, default=2000)
    ap.add_argument("--self-test", type=int, default=5,
                     help="Before the full run, open this many files from path_lookup.csv and report "
                          "success/failure -- catches a bad --audio-root immediately instead of "
                          "100k files in. 0 disables.")
    args = ap.parse_args()

    lookup_path = Path(args.path_lookup_csv)
    if not lookup_path.is_file():
        print(f"ERROR: {lookup_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
        import soundfile  # noqa: F401 -- import check only, used inside workers
        from multiprocessing import Pool, cpu_count
    except ImportError as e:
        print(f"Missing dependency ({e}). Install with:\n"
              f"    pip install soundfile numpy --break-system-packages", file=sys.stderr)
        sys.exit(1)

    workers = args.workers if args.workers > 0 else cpu_count()

    print(f"Loading {lookup_path} ...", file=sys.stderr)
    with open(lookup_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [(row[0], row[1]) for row in reader if row]
    print(f"  {len(rows):,} files in lookup table.", file=sys.stderr)

    out_path = Path(args.out)
    done_ids = set()
    write_header = True
    if out_path.is_file():
        # Resume rule: an id only counts as "done" if it SUCCEEDED (error
        # column blank). Error rows are dropped and retried -- otherwise a
        # systematic failure (wrong --audio-root, drive not mounted, ...)
        # would get permanently "remembered" as done and never retried.
        # The file is rewritten to keep only successes, so re-running
        # never accumulates duplicate rows for the same id.
        kept_rows = []
        n_prev_errors = 0
        with open(out_path, newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                if len(row) >= 4 and row[3] == "" and row[1] != "":
                    kept_rows.append(row)
                    done_ids.add(row[0])
                else:
                    n_prev_errors += 1
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "energy_db", "wada_snr_db", "error"])
            writer.writerows(kept_rows)
        write_header = False
        print(f"  Resuming: {len(done_ids):,} files already succeeded in {out_path} (kept), "
              f"{n_prev_errors:,} previous error rows dropped and will be retried.",
              file=sys.stderr)

    todo = [(pid, p) for pid, p in rows if pid not in done_ids]
    print(f"  {len(todo):,} files left to process, {workers} workers.", file=sys.stderr)
    if not todo:
        print("Nothing to do -- already complete.", file=sys.stderr)
        return

    if args.self_test > 0:
        print(f"\nSelf-test: opening {min(args.self_test, len(todo))} file(s) before the full "
              f"run (checks --audio-root is correct)...", file=sys.stderr)
        import soundfile as sf
        n_ok = 0
        for pid, p in todo[:args.self_test]:
            full_path = _resolve_path(args.audio_root, p)
            try:
                sf.read(full_path, dtype="float32", frames=1)
                n_ok += 1
                print(f"  OK   {full_path}", file=sys.stderr)
            except Exception as e:
                print(f"  FAIL {full_path}  ({type(e).__name__}: {e})", file=sys.stderr)
        if n_ok == 0:
            print(f"\nERROR: 0/{min(args.self_test, len(todo))} self-test files opened. "
                  f"Do not proceed -- this would fail on all {len(todo):,} files again. "
                  f"Check --audio-root points at the corpus root (the directory that directly "
                  f"contains 'ihm/' and 'sdm/'), and that the drive is mounted.", file=sys.stderr)
            sys.exit(1)
        print(f"Self-test passed: {n_ok}/{min(args.self_test, len(todo))} opened. Proceeding.\n",
              file=sys.stderr)

    print("\nBuilding WADA-SNR calibration table (Monte Carlo, seed=42, "
          "~a few seconds)...", file=sys.stderr)
    t0 = time.time()
    snr_grid_db, g_grid = build_wada_calibration_table(np)
    print(f"  done in {time.time() - t0:.1f}s. "
          f"G ranges {g_grid[0]:.4f} (at -20dB) to {g_grid[-1]:.4f} (at +60dB), "
          f"monotonic={bool(np.all(np.diff(g_grid) >= 0))}", file=sys.stderr)

    snr_grid_list = snr_grid_db.tolist()
    g_grid_list = g_grid.tolist()

    n_errors = 0
    n_done = 0
    t0 = time.time()

    with open(out_path, "a", newline="") as fout:
        writer = csv.writer(fout)
        if write_header:
            writer.writerow(["id", "energy_db", "wada_snr_db", "error"])
            fout.flush()

        todo_with_root = [(pid, p, args.audio_root) for pid, p in todo]
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(snr_grid_list, g_grid_list)) as pool:
            for path_id, energy_db, snr_db, err in pool.imap_unordered(_process_one, todo_with_root, chunksize=64):
                n_done += 1
                if err:
                    n_errors += 1
                writer.writerow([path_id,
                                  "" if energy_db is None else f"{energy_db:.4f}",
                                  "" if snr_db is None else f"{snr_db:.4f}",
                                  err])
                if n_done % args.progress_every == 0:
                    fout.flush()
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = (len(todo) - n_done) / rate if rate > 0 else float("inf")
                    print(f"  ... {n_done:,}/{len(todo):,} done, {rate:,.1f} files/s, "
                          f"{n_errors} errors so far, ~{remaining/60:.1f} min left",
                          file=sys.stderr)
        fout.flush()

    elapsed = time.time() - t0
    print(f"\nDone. {n_done:,} files processed in {elapsed/60:.1f} minutes "
          f"({n_done/elapsed:,.1f} files/s).", file=sys.stderr)
    if n_errors:
        print(f"WARNING: {n_errors:,} files errored (unreadable/corrupt/missing) -- "
              f"their energy_db/wada_snr_db are blank in {out_path}, 'error' column says why. "
              f"Check those before joining onto trial tables.", file=sys.stderr)
    else:
        print("No errors.", file=sys.stderr)
    print(f"Output: {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()