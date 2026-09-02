#!/usr/bin/env python3
"""
ami_eval_enroll_test_xlsx.py — builds the eval-group ENROLL/TEST Excel
sheet per your latest spec. Does NOT touch, move, or rename anything on
disk — reads the existing ihm/ and sdm/ folders only, writes one new
.xlsx with wav file paths + metadata for future mapping.

LOCKED-IN SPEC (v3 — updated per your "some change in architecture" message):

  1. Eval group = 100 of 190 speakers: 70 male + 30 female, drawn
     randomly (seed 42, reproducible) from the full pool of 130 male /
     60 female speakers. The other ~90 speakers go into a separate
     "train group" speaker list — NOT processed further, per your
     standing instruction to defer train. Unchanged from v2.

  2. ENROLL: near-field (ihm) ONLY. ALL 100 eval speakers — both male
     and female now (this is the change: v2 had enroll = male-only).
     ALL their meetings, ALL their audio files, no filtering.

  3. TEST: near-field AND far-field (ihm + sdm). ALL 100 eval speakers —
     both male and female (v2 had test = female-only). Same rule —
     every utterance, all meetings, no filtering.

  CONSEQUENCE, confirmed by you as intentional ("yes there will be same
  datapoint thats fine"): every eval speaker's ihm utterances now appear
  in BOTH the Enroll sheet and the Test sheet (Test = ihm+sdm is a
  superset of Enroll = ihm, over the same 100 speakers). That overlap is
  the "duplicates are fine" you confirmed — not a bug, kept as-is.

Output (.xlsx, written next to this script unless --out-dir is given):
  Sheet "Enroll"   — one row per ihm utterance, all 100 eval speakers
  Sheet "Test"     — one row per ihm+sdm utterance, all 100 eval speakers
                     (includes the same ihm rows that are also in Enroll)
  Sheet "Train_Group_Speakers" — speaker-level list of the deferred ~90
  Sheet "Summary"  — counts, some via COUNTIFS formulas so they stay
                     correct if you manually prune rows later in Excel

Columns (Enroll/Test sheets):
  wav_path, speaker_id, gender, location, language, participant_number,
  role_suffix, native_split, meeting_id, mic_type

Usage:
    pip install openpyxl --break-system-packages
    python ami_eval_enroll_test_xlsx.py "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \
        --out-dir ./eval_xlsx_out
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    print("Missing dependency. Run: pip install openpyxl --break-system-packages", file=sys.stderr)
    raise

STEM_RE = re.compile(
    r"^(?P<sid>[a-zA-Z0-9]+)_(?P<meeting>[a-zA-Z0-9]+)_"
    r"(?P<split_in_name>train|dev|eval)_(?P<miccode>[a-zA-Z0-9]+)_"
    r"(?P<begin>\d+)-(?P<end>\d+)$"
)
NOSEG_RE = re.compile(
    r"^(?P<sid>[a-zA-Z0-9]+)_(?P<meeting>[a-zA-Z0-9]+)_"
    r"(?P<split_in_name>train|dev|eval)_(?P<miccode>[a-zA-Z0-9]+)$"
)

LOC = {"i": "Idiap", "e": "Edinburgh", "t": "TNO"}
LANG = {"e": "English", "d": "Dutch", "o": "Other"}
SID_TAIL_RE = re.compile(r"^(\d+)([a-zA-Z]*)$")

SEED = 42
EVAL_MALE_TARGET = 70
EVAL_FEMALE_TARGET = 30
FONT_NAME = "Arial"


def decode_speaker_id(sid: str):
    if len(sid) < 4:
        return None
    g, loc, lang = sid[0].lower(), sid[1].lower(), sid[2].lower()
    rest = sid[3:]
    if g not in ("f", "m") or loc not in LOC or lang not in LANG:
        return None
    m = SID_TAIL_RE.match(rest)
    if not m:
        return None
    return {
        "gender": "Female" if g == "f" else "Male",
        "location": LOC[loc],
        "language": LANG[lang],
        "number": m.group(1),
        "role_suffix": m.group(2) or None,
    }


def find_child_ci(parent: Path, name: str):
    if not parent.is_dir():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def locate_audio_root(user_path: Path):
    candidates = [user_path, user_path / "audio", user_path.parent]
    tried = []
    for cand in candidates:
        if not cand.is_dir():
            tried.append((cand, "does not exist / not a directory"))
            continue
        ihm = find_child_ci(cand, "ihm")
        sdm = find_child_ci(cand, "sdm")
        top_level = sorted(p.name for p in cand.iterdir())[:25]
        tried.append((cand, f"ihm found: {bool(ihm)}, sdm found: {bool(sdm)}; top-level: {top_level}"))
        if ihm and sdm:
            return cand, ihm, sdm
    print("ERROR: could not locate both an 'ihm' and an 'sdm' folder from the path you gave.", file=sys.stderr)
    for cand, msg in tried:
        print(f"  - {cand}  ->  {msg}", file=sys.stderr)
    return None, None, None


def walk_mic_dir(mic_dir: Path, mic_label: str):
    rows = []
    for split_name in ("train", "dev", "eval"):
        split_dir = find_child_ci(mic_dir, split_name)
        if split_dir is None:
            continue
        for speaker_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            speaker_id = speaker_dir.name
            for wav_path in speaker_dir.iterdir():
                if not (wav_path.is_file() and wav_path.suffix.lower() == ".wav"):
                    continue
                stem = wav_path.stem
                m = STEM_RE.match(stem) or NOSEG_RE.match(stem)
                meeting_id = m.group("meeting") if m else None
                rows.append({
                    "wav_path": str(wav_path.resolve()),
                    "speaker_id": speaker_id,
                    "native_split": split_dir.name,
                    "mic_type": mic_label,
                    "meeting_id": meeting_id,
                    "parse_ok": bool(m),
                })
    return rows


def style_header(ws, ncols):
    header_font = Font(name=FONT_NAME, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = header_font
        cell.fill = header_fill
    ws.freeze_panes = "A2"


def autosize(ws, ncols, sample_rows=200):
    for c in range(1, ncols + 1):
        col_letter = get_column_letter(c)
        max_len = 0
        for r, cell in enumerate(ws[col_letter]):
            if r > sample_rows:
                break
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 60)


def write_data_sheet(wb, name, rows, columns):
    ws = wb.create_sheet(name)
    ws.append(columns)
    for r in rows:
        ws.append([r.get(c, "") if r.get(c) is not None else "" for c in columns])
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.font = Font(name=FONT_NAME)
    style_header(ws, len(columns))
    autosize(ws, len(columns))
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to (or near) the 'audio' folder containing ihm/ and sdm/")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    resolved_root, ihm_dir, sdm_dir = locate_audio_root(Path(args.root))
    if resolved_root is None:
        sys.exit(1)
    print(f"Using root: {resolved_root}\n  ihm -> {ihm_dir}\n  sdm -> {sdm_dir}", file=sys.stderr)

    ihm_rows = walk_mic_dir(ihm_dir, "ihm")
    sdm_rows = walk_mic_dir(sdm_dir, "sdm")
    all_rows = ihm_rows + sdm_rows
    if not all_rows:
        print("ERROR: found ihm/sdm folders but zero .wav files under them. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    all_speaker_ids = sorted(set(r["speaker_id"] for r in all_rows))
    decode_map = {}
    decode_failures = []
    for sid in all_speaker_ids:
        dec = decode_speaker_id(sid)
        if dec is None:
            decode_failures.append(sid)
        else:
            decode_map[sid] = dec

    male_ids = sorted(sid for sid, d in decode_map.items() if d["gender"] == "Male")
    female_ids = sorted(sid for sid, d in decode_map.items() if d["gender"] == "Female")

    if len(male_ids) < EVAL_MALE_TARGET or len(female_ids) < EVAL_FEMALE_TARGET:
        print(f"ERROR: not enough speakers to hit the target split. "
              f"Found {len(male_ids)} male (need {EVAL_MALE_TARGET}), "
              f"{len(female_ids)} female (need {EVAL_FEMALE_TARGET}).", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)
    eval_male = sorted(rng.sample(male_ids, EVAL_MALE_TARGET))
    eval_female = sorted(rng.sample(female_ids, EVAL_FEMALE_TARGET))
    eval_male_set, eval_female_set = set(eval_male), set(eval_female)

    train_male = sorted(set(male_ids) - eval_male_set)
    train_female = sorted(set(female_ids) - eval_female_set)

    columns = ["wav_path", "speaker_id", "gender", "location", "language",
               "participant_number", "role_suffix", "native_split",
               "meeting_id", "mic_type"]

    def to_row(r):
        dec = decode_map[r["speaker_id"]]
        return {
            "wav_path": r["wav_path"], "speaker_id": r["speaker_id"],
            "gender": dec["gender"], "location": dec["location"], "language": dec["language"],
            "participant_number": dec["number"], "role_suffix": dec["role_suffix"] or "",
            "native_split": r["native_split"], "meeting_id": r["meeting_id"] or "",
            "mic_type": r["mic_type"],
        }

    eval_all_set = eval_male_set | eval_female_set
    enroll_rows = [to_row(r) for r in ihm_rows if r["speaker_id"] in eval_all_set]
    test_rows = [to_row(r) for r in all_rows if r["speaker_id"] in eval_all_set]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "eval_enroll_test.xlsx"

    wb = Workbook()
    wb.remove(wb.active)

    write_data_sheet(wb, "Enroll", enroll_rows, columns)
    write_data_sheet(wb, "Test", test_rows, columns)

    train_cols = ["speaker_id", "gender", "location", "language", "participant_number", "role_suffix"]
    train_rows = []
    for sid in sorted(train_male + train_female):
        dec = decode_map[sid]
        train_rows.append({
            "speaker_id": sid, "gender": dec["gender"], "location": dec["location"],
            "language": dec["language"], "participant_number": dec["number"],
            "role_suffix": dec["role_suffix"] or "",
        })
    write_data_sheet(wb, "Train_Group_Speakers", train_rows, train_cols)

    # ---- Summary sheet ----
    ws = wb.create_sheet("Summary", 0)
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 16
    bold = Font(name=FONT_NAME, bold=True)
    normal = Font(name=FONT_NAME)
    n_enroll = len(enroll_rows) + 1  # +1 for header row offset
    n_test = len(test_rows) + 1

    rows_to_write = [
        ("AMI eval group — enroll/test summary", None, True),
        ("", None, False),
        ("Eval group size (target)", 100, False),
        ("  eval male speakers", EVAL_MALE_TARGET, False),
        ("  eval female speakers", EVAL_FEMALE_TARGET, False),
        ("Train group size (deferred)", len(train_male) + len(train_female), False),
        ("  train male speakers", len(train_male), False),
        ("  train female speakers", len(train_female), False),
        ("Random seed used", SEED, False),
        ("", None, False),
        ("Enroll rows (Enroll sheet)", f"=COUNTA(Enroll!A2:A{max(n_enroll,2)})", False),
        ("  enroll rows, mic_type=ihm (sanity check, should equal total)",
         f"=COUNTIFS(Enroll!J2:J{max(n_enroll,2)},\"ihm\")", False),
        ("Test rows (Test sheet)", f"=COUNTA(Test!A2:A{max(n_test,2)})", False),
        ("  test rows, mic_type=ihm", f"=COUNTIFS(Test!J2:J{max(n_test,2)},\"ihm\")", False),
        ("  test rows, mic_type=sdm", f"=COUNTIFS(Test!J2:J{max(n_test,2)},\"sdm\")", False),
        ("", None, False),
        ("Speaker decode failures (not classified, excluded everywhere)",
         "; ".join(decode_failures) if decode_failures else "none", False),
        ("", None, False),
        ("Notes", None, True),
        ("Enroll = near-field (ihm) only, ALL 100 eval speakers (male+female), all meetings/all files.", None, False),
        ("Test = near-field + far-field (ihm+sdm), ALL 100 eval speakers (male+female), all meetings/all files.", None, False),
        ("Enroll and Test draw from the SAME 100 speakers, so every enroll (ihm) row also appears in", None, False),
        ("Test (since Test = ihm+sdm for the same speakers). Confirmed intentional — duplicates are fine.", None, False),
        ("No .wav/.txt files were moved, renamed, or modified — this workbook only records paths.", None, False),
    ]
    for i, (label, value, is_bold) in enumerate(rows_to_write, start=1):
        c1 = ws.cell(row=i, column=1, value=label)
        c1.font = bold if is_bold else normal
        if value is not None:
            c2 = ws.cell(row=i, column=2, value=value)
            c2.font = normal

    wb.save(xlsx_path)

    summary = {
        "resolved_root": str(resolved_root),
        "decode_failures": decode_failures,
        "eval_male_speakers": len(eval_male),
        "eval_female_speakers": len(eval_female),
        "train_male_speakers": len(train_male),
        "train_female_speakers": len(train_female),
        "enroll_rows": len(enroll_rows),
        "test_rows": len(test_rows),
        "test_rows_by_mic": {
            "ihm": sum(1 for r in test_rows if r["mic_type"] == "ihm"),
            "sdm": sum(1 for r in test_rows if r["mic_type"] == "sdm"),
        },
        "output_file": str(xlsx_path.resolve()),
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()