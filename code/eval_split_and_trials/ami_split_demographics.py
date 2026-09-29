#!/usr/bin/env python3
"""
ami_split_demographics.py (v2) — prerequisite stats for the eval/train
work. THIS IS THE ONLY THING BEING BUILT RIGHT NOW — no CSV, no
enroll/test construction. Just numbers.

v2 fixes a "0 files found" failure from v1: v1 assumed the root you pass
in is exactly the folder that directly contains ihm/ and sdm/, and it
matched folder names case-sensitively. If your path pointed one level off
(e.g. you passed the AMI_Meeting_Corpus folder instead of its audio/
subfolder, or vice versa) or a folder was capitalized differently, v1
silently found nothing useful in the JSON and only put the real reason in
a stderr warning that's easy to miss.

v2 self-diagnoses:
  - tries the path you give AND path/"audio" AND path's parent, picks
    whichever one actually contains ihm/ and sdm/ (case-insensitive)
  - matches ihm/sdm and train/dev/eval directory names case-insensitively
  - matches *.wav case-insensitively (.wav / .WAV)
  - if it still can't find ihm/sdm anywhere, it PRINTS what it did find at
    each candidate location so you can see exactly what's mismatched,
    instead of quietly returning zeroes

Answers, separately for the ihm/ and sdm/ folders (cross-check — both
should describe the same 190 speakers):
  - unique speakers + utterance counts by GENDER x native dev/eval/train split
  - same for RECORDING LOCATION (Idiap / Edinburgh / TNO)
  - same for NATIVE LANGUAGE (English / Dutch / Other)
  - corpus-wide (all-splits-combined) totals for each, so you have the
    "grand total" alongside the per-split breakdown

No charts. Usage:
    python ami_split_demographics.py "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \
        --out ./split_demographics.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

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
    """Case-insensitive lookup of a directory named `name` directly under
    parent. Returns the Path or None."""
    if not parent.is_dir():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def locate_audio_root(user_path: Path):
    """Try a few reasonable interpretations of what the user passed in and
    return (resolved_root, ihm_dir, sdm_dir) for whichever one actually has
    both. Returns (None, None, None) if none work, after printing
    diagnostics for every candidate tried."""
    candidates = [user_path, user_path / "audio", user_path.parent]
    tried = []
    for cand in candidates:
        cand = cand
        if not cand.is_dir():
            tried.append((cand, "does not exist / not a directory", None))
            continue
        ihm = find_child_ci(cand, "ihm")
        sdm = find_child_ci(cand, "sdm")
        top_level = sorted(p.name for p in cand.iterdir())[:25] if cand.is_dir() else []
        tried.append((cand, f"ihm found: {bool(ihm)}, sdm found: {bool(sdm)}; top-level entries (up to 25): {top_level}", (ihm, sdm)))
        if ihm and sdm:
            return cand, ihm, sdm

    print("ERROR: could not locate both an 'ihm' and an 'sdm' folder from the path you gave.", file=sys.stderr)
    print("Here's what I checked:", file=sys.stderr)
    for cand, msg, _ in tried:
        print(f"  - {cand}  ->  {msg}", file=sys.stderr)
    print("\nPass the exact folder that directly contains ihm/ and sdm/ as subfolders.", file=sys.stderr)
    return None, None, None


def walk_mic_dir(mic_dir: Path, mic_label: str):
    """Walk <mic_dir>/{train,dev,eval}/<speaker_dir>/*.wav (case-insensitive
    at every level). Returns list of dicts: split, speaker_id, stem, parse_ok."""
    rows = []
    found_any_split_dir = False
    for split_name in ("train", "dev", "eval"):
        split_dir = find_child_ci(mic_dir, split_name)
        if split_dir is None:
            continue
        found_any_split_dir = True
        for speaker_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            speaker_id = speaker_dir.name
            wavs = [p for p in speaker_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
            for wav_path in wavs:
                stem = wav_path.stem
                m = STEM_RE.match(stem) or NOSEG_RE.match(stem)
                rows.append({
                    "split": split_name,
                    "speaker_id": speaker_id,
                    "stem": stem,
                    "parse_ok": bool(m),
                })
    if not found_any_split_dir:
        sub = sorted(p.name for p in mic_dir.iterdir())[:25] if mic_dir.is_dir() else []
        print(f"WARNING: no train/dev/eval subfolder found under {mic_dir} "
              f"(top-level entries there: {sub})", file=sys.stderr)
    elif not rows:
        print(f"WARNING: found train/dev/eval under {mic_dir} but zero .wav files inside "
              f"any speaker subfolder — check that speaker folders sit directly under "
              f"{mic_dir}/<split>/ and contain .wav files.", file=sys.stderr)
    return rows


def build_stats_for_mic(rows, mic_label):
    decode_failures = sorted({r["speaker_id"] for r in rows
                               if decode_speaker_id(r["speaker_id"]) is None})

    speakers_by_split = defaultdict(set)
    for r in rows:
        speakers_by_split[r["split"]].add(r["speaker_id"])

    def crosstab(attr_key):
        utt_tab = defaultdict(lambda: defaultdict(int))
        spk_tab = defaultdict(lambda: defaultdict(set))
        utt_total = defaultdict(int)
        spk_total = defaultdict(set)
        for r in rows:
            dec = decode_speaker_id(r["speaker_id"])
            if dec is None:
                continue
            val = dec[attr_key]
            utt_tab[r["split"]][val] += 1
            spk_tab[r["split"]][val].add(r["speaker_id"])
            utt_total[val] += 1
            spk_total[val].add(r["speaker_id"])
        utt_out = {sp: dict(vals) for sp, vals in utt_tab.items()}
        spk_out = {sp: {v: len(s) for v, s in vals.items()} for sp, vals in spk_tab.items()}
        return {
            "utterance_counts_by_split": utt_out,
            "unique_speaker_counts_by_split": spk_out,
            "utterance_counts_all_splits_combined": dict(utt_total),
            "unique_speaker_counts_all_splits_combined": {v: len(s) for v, s in spk_total.items()},
        }

    return {
        "mic_folder": mic_label,
        "total_wav_files": len(rows),
        "parse_ok_pct": round(100.0 * sum(r["parse_ok"] for r in rows) / len(rows), 2) if rows else None,
        "unique_speakers_total": len(set(r["speaker_id"] for r in rows)),
        "unique_speakers_by_split": {sp: len(s) for sp, s in speakers_by_split.items()},
        "decode_failures": decode_failures,
        "gender": crosstab("gender"),
        "location": crosstab("location"),
        "language": crosstab("language"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to (or near) the 'audio' folder containing ihm/ and sdm/")
    ap.add_argument("--out", default="split_demographics.json")
    args = ap.parse_args()

    resolved_root, ihm_dir, sdm_dir = locate_audio_root(Path(args.root))
    if resolved_root is None:
        sys.exit(1)

    print(f"Using root: {resolved_root}", file=sys.stderr)
    print(f"  ihm -> {ihm_dir}", file=sys.stderr)
    print(f"  sdm -> {sdm_dir}", file=sys.stderr)

    ihm_rows = walk_mic_dir(ihm_dir, "ihm")
    sdm_rows = walk_mic_dir(sdm_dir, "sdm")

    result = {
        "resolved_root": str(resolved_root),
        "ihm": build_stats_for_mic(ihm_rows, "ihm"),
        "sdm": build_stats_for_mic(sdm_rows, "sdm"),
    }
    ihm_spk = result["ihm"]["unique_speakers_by_split"]
    sdm_spk = result["sdm"]["unique_speakers_by_split"]
    result["cross_check_ihm_vs_sdm_speaker_counts_match"] = (ihm_spk == sdm_spk)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("AMI split demographics (native dev/eval/train split, ihm vs sdm)")
    print("=" * 70)
    for mic in ("ihm", "sdm"):
        r = result[mic]
        print(f"\n--- {mic.upper()} ---")
        print(f"total wav files: {r['total_wav_files']}  (parsed OK: {r['parse_ok_pct']}%)")
        print(f"unique speakers total: {r['unique_speakers_total']}")
        print(f"unique speakers by split: {r['unique_speakers_by_split']}")
        print(f"decode failures: {r['decode_failures'] or 'none'}")
        for attr in ("gender", "location", "language"):
            print(f"\n  {attr.upper()} - ALL SPLITS COMBINED (unique speakers): "
                  f"{r[attr]['unique_speaker_counts_all_splits_combined']}")
            print(f"  {attr.upper()} - ALL SPLITS COMBINED (utterances): "
                  f"{r[attr]['utterance_counts_all_splits_combined']}")
            print(f"  {attr.upper()} - by split (unique speakers):")
            for sp, vals in r[attr]["unique_speaker_counts_by_split"].items():
                print(f"    {sp}: {vals}")
            print(f"  {attr.upper()} - by split (utterances):")
            for sp, vals in r[attr]["utterance_counts_by_split"].items():
                print(f"    {sp}: {vals}")

    print("\n" + "=" * 70)
    print(f"ihm vs sdm unique-speaker-by-split counts match: "
          f"{result['cross_check_ihm_vs_sdm_speaker_counts_match']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()