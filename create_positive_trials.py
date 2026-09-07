#!/usr/bin/env python3
"""
create_positive_trials.py — builds the full, uncapped POSITIVE trial list
(same speaker, enroll utterance x test utterance) from eval_enroll_test2.xlsx
(the meeting-split Enroll/Test workbook). Negatives are NOT touched here —
per your instruction, positives first; capped negatives come later.

Meeting-diff is automatic and needs no filtering: Enroll and Test meetings
are disjoint per speaker by construction (verified earlier, 0 exclusion for
every speaker), so every Enroll x Test combination for a given speaker is
already a valid trial.

OUTPUT: exactly one CSV, 3 columns, nothing else written:
  enroll_path    - shortened wav path, e.g. /ihm/train/fee019/fee019_es2005a_train_h03_0000408-0000459.wav
                   (full path minus everything before "/ihm/" or "/sdm/" --
                   that's unique on its own, no need for the full disk path)
  test_path      - same shortening, for the test-side file
  duration_diff  - the ABSOLUTE DIFFERENCE, in seconds, between the two
                   files' own durations (each derived from the filename's
                   begin-end timestamp: duration_s = (end - begin) / 100.0,
                   centiseconds -- this is the same formula validated
                   against real audio in the original ami_stats.py project).
                   Raw number, not bucketed -- use
                   positive_trial_duration_stats.py to get the distribution
                   and stats out of this column afterward.
                   Blank if either file's duration couldn't be parsed.

Usage:
    python create_positive_trials.py eval_xlsx_out/eval_enroll_test2.xlsx --out positive_trials.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DUR_RE = re.compile(r"_(?P<begin>\d+)-(?P<end>\d+)\.wav$", re.IGNORECASE)
REQUIRED_COLS = {"wav_path", "speaker_id", "meeting_id", "mic_type"}


def shorten_path(wav_path: str, mic_type: str) -> str:
    """Cut everything before /ihm/ or /sdm/ -- that suffix alone is unique."""
    marker = f"/{mic_type}/"
    idx = wav_path.find(marker)
    if idx == -1:
        # fallback: shouldn't happen on real data, but don't silently corrupt -- keep full path
        return wav_path
    return wav_path[idx:]


def duration_from_path(wav_path: str):
    """duration_s = (end - begin) / 100.0 -- centiseconds, per the filename
    convention validated in ami_stats.py against real audio (near-zero diff
    against the actual wav duration). Returns None if the filename doesn't
    carry a begin-end timestamp (whole-file / no-segment variant)."""
    m = DUR_RE.search(wav_path)
    if not m:
        return None
    begin = int(m.group("begin"))
    end = int(m.group("end"))
    return (end - begin) / 100.0


def load_sheet(wb, sheet_name):
    """Returns list of dicts: {short_path, duration, speaker_id, meeting_id} for that sheet."""
    if sheet_name not in wb.sheetnames:
        print(f"ERROR: no '{sheet_name}' sheet found. Sheets present: {wb.sheetnames}", file=sys.stderr)
        sys.exit(1)
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows_iter)]
    missing = REQUIRED_COLS - set(header)
    if missing:
        print(f"ERROR: sheet '{sheet_name}' missing columns {missing}.", file=sys.stderr)
        sys.exit(1)
    idx = {name: header.index(name) for name in header}

    out = []
    n_no_duration = 0
    for row in rows_iter:
        if row is None:
            continue
        wav_path = row[idx["wav_path"]]
        sid = row[idx["speaker_id"]]
        meeting = row[idx["meeting_id"]]
        mic = row[idx["mic_type"]]
        if wav_path is None or sid is None:
            continue
        dur = duration_from_path(str(wav_path))
        if dur is None:
            n_no_duration += 1
        out.append({
            "short_path": shorten_path(str(wav_path), str(mic).strip().lower()),
            "duration": dur,
            "speaker_id": str(sid),
        })
    if n_no_duration:
        print(f"WARNING: {n_no_duration} rows in '{sheet_name}' had no parseable "
              f"begin-end timestamp in the filename -- their trials will show "
              f"duration_diff_bucket = 'unknown'.", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="Path to eval_enroll_test2.xlsx (the meeting-split workbook)")
    ap.add_argument("--out", default="positive_trials.csv")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_file():
        print(f"ERROR: {xlsx_path} not found.", file=sys.stderr)
        sys.exit(1)

    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)

    print("Reading Enroll sheet...", file=sys.stderr)
    enroll_rows = load_sheet(wb, "Enroll")
    print("Reading Test sheet...", file=sys.stderr)
    test_rows = load_sheet(wb, "Test")
    wb.close()

    # group by speaker
    enroll_by_speaker = {}
    for r in enroll_rows:
        enroll_by_speaker.setdefault(r["speaker_id"], []).append(r)
    test_by_speaker = {}
    for r in test_rows:
        test_by_speaker.setdefault(r["speaker_id"], []).append(r)

    speakers = sorted(set(enroll_by_speaker) & set(test_by_speaker))
    print(f"{len(speakers)} speakers have both enroll and test rows. Writing trials...", file=sys.stderr)

    out_path = Path(args.out)
    n_written = 0
    n_unknown = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["enroll_path", "test_path", "duration_diff"])
        for sid in speakers:
            e_list = enroll_by_speaker[sid]
            t_list = test_by_speaker[sid]
            for e in e_list:
                e_path = e["short_path"]
                e_dur = e["duration"]
                for t in t_list:
                    if e_dur is None or t["duration"] is None:
                        diff_str = ""
                        n_unknown += 1
                    else:
                        diff_str = round(abs(e_dur - t["duration"]), 2)
                    writer.writerow([e_path, t["short_path"], diff_str])
                    n_written += 1
            if n_written % 2_000_000 < len(t_list) * len(e_list):
                print(f"  ... {n_written:,} trials written so far", file=sys.stderr)

    print(f"\nDone. {n_written:,} positive trials written to {out_path}", file=sys.stderr)
    if n_unknown:
        print(f"  ({n_unknown:,} of those have a blank duration_diff -- "
              f"one or both files had no parseable begin-end timestamp)", file=sys.stderr)
    print("Run positive_trial_duration_stats.py on this file to get the distribution/stats/graph.",
          file=sys.stderr)


if __name__ == "__main__":
    main()