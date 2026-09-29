#!/usr/bin/env python3
#!/usr/bin/env python3
"""
encode_positive_trials.py

Makes a NEW, much smaller copy of positive_trials.csv.
Your original CSV is opened READ-ONLY and is never modified or deleted.

WHY THE CSV IS SO BIG
---------------------
Your CSV stores the same long file paths over and over as plain text.
There are only about 105,000 unique .wav files in the whole Enroll+Test
set, but 27.3 million trial rows -- so on average every path string is
written out about 260 times. That repetition is most of your 3.4 GB.

WHAT THIS SCRIPT DOES (two separate savings, stacked)
-----------------------------------------------------
1. DICTIONARY ENCODING (the big win)
   Every unique path gets a number. Instead of writing
       /ihm/train/fee019/fee019_es2005a_train_h03_0000408-0000459.wav
   (about 65 characters) in every row, the row just stores
       412
   The number-to-path mapping is saved once, in a small separate file.

2. COMPACT NUMBER TYPES
   duration_diff is always an exact multiple of 0.01 seconds (it comes
   from centisecond timestamps in the filename), so 2.47 seconds is
   stored as the whole number 247. Nothing is rounded or lost -- this is
   exactly reversible, not lossy.

3. PARQUET + COMPRESSION
   The rows are then written as Parquet instead of CSV. Parquet stores
   each column separately and compresses it, which works extremely well
   here because the ID columns contain long runs of repeated values.

OUTPUT FILES (all inside --out-dir, nothing else is written)
-----------------------------------------------------------
  positive_trials.parquet   the encoded trials: 3 columns,
                            enroll_id (int) | test_id (int) | duration_diff_cs (int)
  path_lookup.csv           id,path  -- small and human-readable, open it
                            in any editor or Excel to see what an ID means
  encoding_manifest.json    row counts, file sizes, the size ratio, and
                            the exact rule for decoding back to seconds

To look inside the .parquet afterwards, use read_encoded_trials.py --
Parquet is a binary file, so you cannot just `cat` it like a CSV.

Uses pyarrow directly, NOT pandas and NOT numpy -- pandas is broken on
this machine, and this script's own first run showed pyarrow and this
machine's numpy don't hand data to each other cleanly either (an ABI
mismatch). So this never passes a numpy array into pyarrow: it builds
arrow arrays straight from Python's built-in array.array, and uses
pyarrow's own pyarrow.compute module for min/max/mean instead of numpy.

Usage:
    pip install pyarrow --break-system-packages
    python encode_positive_trials.py positive_trials.csv
    python encode_positive_trials.py positive_trials.csv --out-dir encoded_trials
"""

import argparse
import array
import csv
import json
import sys
import time
from pathlib import Path

EXPECTED_HEADER = ["enroll_path", "test_path", "duration_diff"]
MISSING = -1  # sentinel for a blank / unparseable duration_diff


def human_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024


def encode(csv_path: Path):
    """Single streaming pass over the CSV. Returns the encoded columns plus
    the id->path table. Never loads the CSV text into memory all at once."""
    path_to_id = {}
    id_to_path = []

    enroll_ids = array.array("i")
    test_ids = array.array("i")
    diffs_cs = array.array("i")   # centiseconds; downcast to int16 at write time if it fits

    n_rows = 0
    n_missing = 0
    t0 = time.time()

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != EXPECTED_HEADER:
            print(f"WARNING: header is {header}, expected {EXPECTED_HEADER}. "
                  f"Assuming columns are (enroll_path, test_path, duration_diff).",
                  file=sys.stderr)

        for row in reader:
            if not row:
                continue
            e_path, t_path, diff = row[0], row[1], row[2]

            e_id = path_to_id.get(e_path)
            if e_id is None:
                e_id = len(id_to_path)
                path_to_id[e_path] = e_id
                id_to_path.append(e_path)

            t_id = path_to_id.get(t_path)
            if t_id is None:
                t_id = len(id_to_path)
                path_to_id[t_path] = t_id
                id_to_path.append(t_path)

            if diff == "" or diff is None:
                cs = MISSING
                n_missing += 1
            else:
                try:
                    # exact: duration_diff is always a whole number of centiseconds
                    cs = int(round(float(diff) * 100))
                except ValueError:
                    cs = MISSING
                    n_missing += 1

            enroll_ids.append(e_id)
            test_ids.append(t_id)
            diffs_cs.append(cs)

            n_rows += 1
            if n_rows % 5_000_000 == 0:
                print(f"  ... {n_rows:,} rows encoded "
                      f"({time.time() - t0:.0f}s, {len(id_to_path):,} unique paths so far)",
                      file=sys.stderr)

    return enroll_ids, test_ids, diffs_cs, id_to_path, n_rows, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to positive_trials.csv (read-only, never modified)")
    ap.add_argument("--out-dir", default="encoded_trials",
                     help="Directory for the new encoded files (created if missing)")
    ap.add_argument("--compression", default="zstd",
                     choices=["zstd", "snappy", "gzip", "brotli", "none"],
                     help="Parquet compression codec (default zstd -- best ratio here)")
    ap.add_argument("--row-group-size", type=int, default=1_000_000,
                     help="Rows per Parquet row group (default 1,000,000)")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"Missing dependency ({e}). Run: pip install pyarrow --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_size = csv_path.stat().st_size
    print(f"Reading {csv_path} ({human_size(csv_size)})...", file=sys.stderr)

    t0 = time.time()
    enroll_ids, test_ids, diffs_cs, id_to_path, n_rows, n_missing = encode(csv_path)
    encode_secs = time.time() - t0

    if n_rows == 0:
        print("ERROR: no data rows found.", file=sys.stderr)
        sys.exit(1)

    n_paths = len(id_to_path)
    print(f"\nEncoded {n_rows:,} rows using {n_paths:,} unique paths "
          f"(each path text appears ~{(n_rows * 2) / n_paths:.0f}x in the CSV)",
          file=sys.stderr)

    # Build straight from the Python array.array buffers -- NOT via numpy.
    # (On this machine, numpy<->pyarrow handoff has already shown one ABI
    # mismatch (the pandas issue), and it just showed a second one here:
    # passing a numpy array into pa.array() raised "Input object was not a
    # NumPy array" even though it plainly was one -- a classic sign of
    # pyarrow being compiled against a different numpy ABI than what's
    # installed. pyarrow can build arrays directly from array.array/plain
    # Python sequences and do its own min/max/mean via pyarrow.compute, so
    # numpy is never in the loop for this step.)
    raw_enroll = pa.array(enroll_ids)   # infers int64 from the 'i' typecode
    raw_test = pa.array(test_ids)
    raw_diff = pa.array(diffs_cs)

    d_max = pc.max(raw_diff).as_py()
    if d_max <= 32767:
        diff_type = pa.int16()
    else:
        diff_type = pa.int32()
        print(f"NOTE: max duration_diff is {d_max/100:.2f}s -- too big for int16, "
              f"using int32 for that column.", file=sys.stderr)

    id_type = pa.int16() if n_paths <= 32767 else pa.int32()

    # pc.cast(..., safe=True) checks every value fits before narrowing --
    # if it didn't fit, this raises loudly instead of silently corrupting data
    table = pa.table({
        "enroll_id": pc.cast(raw_enroll, id_type, safe=True),
        "test_id": pc.cast(raw_test, id_type, safe=True),
        "duration_diff_cs": pc.cast(raw_diff, diff_type, safe=True),
    })

    parquet_path = out_dir / "positive_trials.parquet"
    codec = None if args.compression == "none" else args.compression
    print(f"Writing {parquet_path} (compression={args.compression})...", file=sys.stderr)
    try:
        pq.write_table(table, parquet_path, compression=codec,
                        row_group_size=args.row_group_size)
    except Exception as e:
        print(f"WARNING: {args.compression} failed ({e}); falling back to snappy.",
              file=sys.stderr)
        pq.write_table(table, parquet_path, compression="snappy",
                        row_group_size=args.row_group_size)
        args.compression = "snappy"

    lookup_path = out_dir / "path_lookup.csv"
    print(f"Writing {lookup_path}...", file=sys.stderr)
    with open(lookup_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_id", "path"])
        for i, p in enumerate(id_to_path):
            w.writerow([i, p])

    parquet_size = parquet_path.stat().st_size
    lookup_size = lookup_path.stat().st_size
    total_new = parquet_size + lookup_size

    manifest = {
        "source_csv": str(csv_path.resolve()),
        "source_csv_bytes": csv_size,
        "source_csv_human": human_size(csv_size),
        "rows": n_rows,
        "unique_paths": n_paths,
        "blank_or_unparseable_duration_rows": n_missing,
        "encode_seconds": round(encode_secs, 1),
        "output": {
            "parquet_file": str(parquet_path.resolve()),
            "parquet_bytes": parquet_size,
            "parquet_human": human_size(parquet_size),
            "lookup_file": str(lookup_path.resolve()),
            "lookup_bytes": lookup_size,
            "lookup_human": human_size(lookup_size),
            "total_bytes": total_new,
            "total_human": human_size(total_new),
            "compression": args.compression,
        },
        "size_ratio": f"{csv_size / total_new:.1f}x smaller",
        "space_saved_human": human_size(csv_size - total_new),
        "how_to_decode": {
            "enroll_path": "look up enroll_id in path_lookup.csv",
            "test_path": "look up test_id in path_lookup.csv",
            "duration_diff_seconds": "duration_diff_cs / 100.0",
            "missing_marker": f"duration_diff_cs == {MISSING} means the original "
                               f"value was blank/unparseable",
            "lossless": "every original value is exactly recoverable -- "
                         "nothing was rounded away",
        },
    }

    manifest_path = out_dir / "encoding_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 66, file=sys.stderr)
    print(f"  original CSV      {human_size(csv_size):>12}", file=sys.stderr)
    print(f"  new parquet       {human_size(parquet_size):>12}", file=sys.stderr)
    print(f"  new lookup csv    {human_size(lookup_size):>12}", file=sys.stderr)
    print(f"  ------------------------------", file=sys.stderr)
    print(f"  new total         {human_size(total_new):>12}   "
          f"({csv_size / total_new:.1f}x smaller)", file=sys.stderr)
    print("=" * 66, file=sys.stderr)
    print(f"\nManifest: {manifest_path}", file=sys.stderr)
    print(f"Your original CSV was not modified.", file=sys.stderr)
    print(f"To look inside the parquet: python read_encoded_trials.py {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()