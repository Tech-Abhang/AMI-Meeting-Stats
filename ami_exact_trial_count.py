#!/usr/bin/env python3
"""
ami_exact_trial_count.py — computes the EXACT total positive and negative
trial-pair counts (no cap, no sampling, no row generation) under your two
constraints:
  1. gender must match between the enroll row and the test row
  2. meeting_id must differ between the enroll row and the test row

This does NOT generate the 5-billion-row trial list. It only counts it,
using per-(speaker, meeting, mic_type) aggregates — the same trick used
throughout this project to get real numbers out of ~100-190 rows of
aggregate data instead of scanning every one of millions of files.

WHY THIS IS EXACT, NOT AN ESTIMATE:
Enroll = ihm-only rows for the 100 eval speakers.
Test   = ihm+sdm rows for the SAME 100 eval speakers (Enroll's ihm rows
are a subset of Test's rows — duplicates are expected and fine, per your
spec). So everything the calculation needs — how many ihm files and how
many sdm files each speaker has, broken down by meeting_id — is fully
determined by the Test sheet alone. No sampling, no approximation:
   Enroll(speaker, meeting) = Test(speaker, meeting, mic=ihm)
   Test(speaker, meeting)   = Test(speaker, meeting, mic=ihm) + Test(speaker, meeting, mic=sdm)
This script also cross-checks that assumption against the Enroll sheet
and warns you if it doesn't hold on your real file.

WHAT IT COMPUTES (per speaker s, over their meetings):
  Positive pairs for s  = Enroll_total(s) * Test_total(s)
                           - sum over meetings m of Enroll(s,m) * Test(s,m)
  broken into the ihm-enroll->ihm-test and ihm-enroll->sdm-test components.

  Negative pairs for an unordered same-gender pair (s1, s2), both directions:
    Enroll(s1) x Test(s2)  minus sum over meetings SHARED by s1 and s2 of
                                  Enroll(s1,m) * Test(s2,m)
    Enroll(s2) x Test(s1)  minus the mirror image
  summed over every same-gender pair of the 100 eval speakers.

No pandas anywhere (this environment has a known pandas/numpy ABI bug —
see ami_way_b_feasibility_check.py). openpyxl + plain dict/Counter only.

Usage:
    python ami_exact_trial_count.py eval_xlsx_out/eval_enroll_test.xlsx
    python ami_exact_trial_count.py eval_xlsx_out/eval_enroll_test.xlsx --out exact_trial_counts.json
    python ami_exact_trial_count.py --selftest      # runs a built-in check against
                                                      # hand-verified toy numbers, no xlsx needed
"""

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_TEST_COLS = {"speaker_id", "meeting_id", "mic_type", "gender"}
REQUIRED_ENROLL_COLS = {"speaker_id", "meeting_id", "mic_type"}


# ----------------------------------------------------------------------
# Reading (openpyxl only — do not touch pandas, see module docstring)
# ----------------------------------------------------------------------

def read_sheet_rows(wb, sheet_name):
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows_iter)]
    idx = {name: header.index(name) for name in header}
    for row in rows_iter:
        if row is None:
            continue
        yield row, idx


def build_speaker_meeting_mic(wb, sheet_name, required_cols):
    ws_rows = read_sheet_rows(wb, sheet_name)
    first = None
    idx = None
    # peek header via a fresh pass (read_sheet_rows already consumed header internally)
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows_iter)]
    missing = required_cols - set(header)
    if missing:
        print(f"ERROR: sheet '{sheet_name}' missing columns {missing}. "
              f"Found columns: {header}", file=sys.stderr)
        sys.exit(1)
    sid_i = header.index("speaker_id")
    mtg_i = header.index("meeting_id")
    mic_i = header.index("mic_type")
    gender_i = header.index("gender") if "gender" in header else None

    # speaker -> meeting_id -> {"ihm": n, "sdm": n}
    by_speaker = defaultdict(lambda: defaultdict(lambda: {"ihm": 0, "sdm": 0}))
    gender_by_speaker = {}

    n_rows = 0
    n_bad_mic = 0
    for row in rows_iter:
        if row is None:
            continue
        sid = row[sid_i]
        mtg = row[mtg_i]
        mic = row[mic_i]
        if sid is None or mtg is None or mic is None:
            continue
        mic_norm = str(mic).strip().lower()
        if mic_norm not in ("ihm", "sdm"):
            n_bad_mic += 1
            continue
        sid = str(sid)
        mtg = str(mtg)
        by_speaker[sid][mtg][mic_norm] += 1
        n_rows += 1
        if gender_i is not None and sid not in gender_by_speaker:
            g = row[gender_i]
            if g is not None:
                gender_by_speaker[sid] = str(g)

    if n_bad_mic:
        print(f"WARNING: {n_bad_mic} rows in '{sheet_name}' had a mic_type "
              f"that wasn't 'ihm' or 'sdm' and were skipped.", file=sys.stderr)

    return by_speaker, gender_by_speaker, n_rows


def decode_gender_from_speaker_id(sid: str):
    if not sid:
        return None
    g = sid[0].lower()
    if g == "f":
        return "Female"
    if g == "m":
        return "Male"
    return None


# ----------------------------------------------------------------------
# Exact positive-pair count
# ----------------------------------------------------------------------

def compute_positive_pairs(test_by_speaker):
    """test_by_speaker: speaker_id -> meeting_id -> {"ihm": n, "sdm": n}.
    Enroll(s, m) := test_by_speaker[s][m]["ihm"]  (Enroll is ihm-only, and
    is the same ihm rows that appear inside Test)."""
    per_speaker = {}
    total = 0
    total_ihm_to_ihm = 0
    total_ihm_to_sdm = 0

    for sid, meetings in test_by_speaker.items():
        enroll_total = sum(m["ihm"] for m in meetings.values())
        test_total = sum(m["ihm"] + m["sdm"] for m in meetings.values())
        raw = enroll_total * test_total
        same_meeting_excl = sum(m["ihm"] * (m["ihm"] + m["sdm"]) for m in meetings.values())
        positives = raw - same_meeting_excl

        ihm_to_ihm = sum(m["ihm"] * m["ihm"] for m in meetings.values())
        raw_ihm_to_ihm = enroll_total * sum(m["ihm"] for m in meetings.values())
        comp_ihm_ihm = raw_ihm_to_ihm - ihm_to_ihm

        ihm_to_sdm_raw = enroll_total * sum(m["sdm"] for m in meetings.values())
        ihm_to_sdm_excl = sum(m["ihm"] * m["sdm"] for m in meetings.values())
        comp_ihm_sdm = ihm_to_sdm_raw - ihm_to_sdm_excl

        per_speaker[sid] = {
            "n_meetings": len(meetings),
            "enroll_total_ihm": enroll_total,
            "test_total_ihm_plus_sdm": test_total,
            "positive_pairs": positives,
            "component_ihm_enroll_to_ihm_test": comp_ihm_ihm,
            "component_ihm_enroll_to_sdm_test": comp_ihm_sdm,
        }
        total += positives
        total_ihm_to_ihm += comp_ihm_ihm
        total_ihm_to_sdm += comp_ihm_sdm

    return {
        "total_positive_pairs": total,
        "total_component_ihm_to_ihm": total_ihm_to_ihm,
        "total_component_ihm_to_sdm": total_ihm_to_sdm,
        "per_speaker": per_speaker,
    }


# ----------------------------------------------------------------------
# Exact negative-pair count
# ----------------------------------------------------------------------

def directional_negative(enroll_meetings, test_meetings):
    """enroll_meetings: meeting_id -> ihm_count (this speaker's enroll side).
    test_meetings: meeting_id -> ihm+sdm_count (the other speaker's test side).
    Returns (raw, excluded_by_shared_meeting, surviving)."""
    e_total = sum(enroll_meetings.values())
    t_total = sum(test_meetings.values())
    raw = e_total * t_total
    shared = set(enroll_meetings) & set(test_meetings)
    excl = sum(enroll_meetings[m] * test_meetings[m] for m in shared)
    return raw, excl, raw - excl


def compute_negative_pairs(test_by_speaker, gender_by_speaker):
    # precompute per-speaker enroll (ihm-per-meeting) and test (ihm+sdm-per-meeting)
    enroll_pm = {}
    test_pm = {}
    for sid, meetings in test_by_speaker.items():
        enroll_pm[sid] = {m: v["ihm"] for m, v in meetings.items()}
        test_pm[sid] = {m: v["ihm"] + v["sdm"] for m, v in meetings.items()}

    speakers_with_gender = [s for s in test_by_speaker if gender_by_speaker.get(s)]
    missing_gender = [s for s in test_by_speaker if not gender_by_speaker.get(s)]
    if missing_gender:
        print(f"WARNING: {len(missing_gender)} speakers had no usable gender value "
              f"and are EXCLUDED from the negative-pair count: {missing_gender}",
              file=sys.stderr)

    by_gender = defaultdict(list)
    for s in speakers_with_gender:
        by_gender[gender_by_speaker[s]].append(s)

    total_negative = 0
    total_raw_no_filter = 0
    total_excluded = 0
    per_gender_totals = {}
    teammate_pairs = 0
    stranger_pairs = 0
    pair_details = []  # only kept for pairs with nonzero exclusion, to keep JSON small

    for gender, speakers in by_gender.items():
        g_total = 0
        g_raw = 0
        for s1, s2 in itertools.combinations(sorted(speakers), 2):
            r12, x12, s12 = directional_negative(enroll_pm[s1], test_pm[s2])
            r21, x21, s21 = directional_negative(enroll_pm[s2], test_pm[s1])
            pair_total = s12 + s21
            pair_raw = r12 + r21
            pair_excl = x12 + x21
            g_total += pair_total
            g_raw += pair_raw
            if pair_excl > 0:
                teammate_pairs += 1
                pair_details.append({
                    "speaker_1": s1, "speaker_2": s2,
                    "raw": pair_raw, "excluded_shared_meeting": pair_excl,
                    "surviving": pair_total,
                })
            else:
                stranger_pairs += 1
        per_gender_totals[gender] = {
            "n_speakers": len(speakers),
            "n_pairs": len(speakers) * (len(speakers) - 1) // 2,
            "raw_no_meeting_filter": g_raw,
            "surviving_negative_pairs": g_total,
            "excluded_by_meeting_overlap": g_raw - g_total,
        }
        total_negative += g_total
        total_raw_no_filter += g_raw
        total_excluded += (g_raw - g_total)

    return {
        "total_negative_pairs": total_negative,
        "total_raw_no_meeting_filter": total_raw_no_filter,
        "total_excluded_by_meeting_overlap": total_excluded,
        "by_gender": per_gender_totals,
        "teammate_pair_count": teammate_pairs,
        "stranger_pair_count": stranger_pairs,
        "teammate_pairs_detail": pair_details,
    }


# ----------------------------------------------------------------------
# Cross-check: Enroll sheet should be a subset match of Test's ihm rows
# ----------------------------------------------------------------------

def cross_check_enroll_sheet(wb, test_by_speaker):
    if "Enroll" not in wb.sheetnames:
        print("NOTE: no 'Enroll' sheet found to cross-check against; skipping.", file=sys.stderr)
        return {"checked": False}

    enroll_by_speaker, _, n_enroll_rows = build_speaker_meeting_mic(
        wb, "Enroll", REQUIRED_ENROLL_COLS
    )
    mismatches = []
    for sid, meetings in enroll_by_speaker.items():
        for mtg, counts in meetings.items():
            enroll_ihm = counts["ihm"] + counts["sdm"]  # Enroll sheet should be ihm-only;
            # if it contains any "sdm" rows at all that's already a spec violation
            if counts["sdm"] != 0:
                mismatches.append({
                    "speaker_id": sid, "meeting_id": mtg,
                    "issue": f"Enroll sheet has {counts['sdm']} sdm rows (should be ihm-only)",
                })
                continue
            test_ihm = test_by_speaker.get(sid, {}).get(mtg, {}).get("ihm", 0)
            if enroll_ihm != test_ihm:
                mismatches.append({
                    "speaker_id": sid, "meeting_id": mtg,
                    "issue": f"Enroll ihm count ({enroll_ihm}) != Test ihm count ({test_ihm})",
                })
    return {
        "checked": True,
        "enroll_rows_read": n_enroll_rows,
        "mismatches_found": len(mismatches),
        "mismatches": mismatches[:50],  # cap in case something's very wrong
    }


# ----------------------------------------------------------------------
# Self-test against the hand-verified toy numbers from the writeup
# ----------------------------------------------------------------------

def run_selftest():
    # Matches the toy example: S1 (2 meetings A,B, 150 ihm+150 sdm each),
    # S2 shares meeting A with S1 (teammates) + own meeting C,
    # S3 shares nothing with S1 (stranger, meetings D,E).
    test_by_speaker = {
        "mie001": {"A": {"ihm": 150, "sdm": 150}, "B": {"ihm": 150, "sdm": 150}},
        "mie002": {"A": {"ihm": 150, "sdm": 150}, "C": {"ihm": 150, "sdm": 150}},
        "mie003": {"D": {"ihm": 150, "sdm": 150}, "E": {"ihm": 150, "sdm": 150}},
    }
    gender_by_speaker = {"mie001": "Male", "mie002": "Male", "mie003": "Male"}

    pos = compute_positive_pairs(test_by_speaker)
    neg = compute_negative_pairs(test_by_speaker, gender_by_speaker)

    checks = []
    checks.append(("S1 positive pairs == 90000", pos["per_speaker"]["mie001"]["positive_pairs"] == 90000))
    checks.append(("S1 ihm->ihm component == 45000", pos["per_speaker"]["mie001"]["component_ihm_enroll_to_ihm_test"] == 45000))
    checks.append(("S1 ihm->sdm component == 45000", pos["per_speaker"]["mie001"]["component_ihm_enroll_to_sdm_test"] == 45000))
    checks.append(("total positives == 90000*3", pos["total_positive_pairs"] == 90000 * 3))
    checks.append(("negative total == 270000 (teammates) + 360000 (strangers, x2 pairs)",
                   neg["total_negative_pairs"] == 270000 + 360000 + 360000))
    checks.append(("teammate_pair_count == 1", neg["teammate_pair_count"] == 1))
    checks.append(("stranger_pair_count == 2", neg["stranger_pair_count"] == 2))

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", help="Path to eval_enroll_test.xlsx")
    ap.add_argument("--out", default="exact_trial_counts.json")
    ap.add_argument("--selftest", action="store_true",
                     help="Run built-in correctness check against hand-verified toy numbers, no xlsx needed")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.xlsx:
        print("ERROR: provide the path to eval_enroll_test.xlsx (or run --selftest).", file=sys.stderr)
        sys.exit(1)

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_file():
        print(f"ERROR: {xlsx_path} not found.", file=sys.stderr)
        sys.exit(1)

    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)

    if "Test" not in wb.sheetnames:
        print(f"ERROR: no 'Test' sheet found. Sheets present: {wb.sheetnames}", file=sys.stderr)
        sys.exit(1)

    test_by_speaker, gender_by_speaker, n_test_rows = build_speaker_meeting_mic(
        wb, "Test", REQUIRED_TEST_COLS
    )

    # fill in any missing gender by decoding speaker_id directly, as a fallback
    for sid in test_by_speaker:
        if sid not in gender_by_speaker:
            g = decode_gender_from_speaker_id(sid)
            if g:
                gender_by_speaker[sid] = g

    cross_check = cross_check_enroll_sheet(wb, test_by_speaker)
    wb.close()

    positives = compute_positive_pairs(test_by_speaker)
    negatives = compute_negative_pairs(test_by_speaker, gender_by_speaker)

    grand_total = positives["total_positive_pairs"] + negatives["total_negative_pairs"]

    summary = {
        "input_xlsx": str(xlsx_path.resolve()),
        "test_rows_read": n_test_rows,
        "n_speakers": len(test_by_speaker),
        "enroll_vs_test_cross_check": {
            "checked": cross_check.get("checked"),
            "mismatches_found": cross_check.get("mismatches_found"),
        },
        "positive_pairs": {
            "total": positives["total_positive_pairs"],
            "component_ihm_enroll_to_ihm_test": positives["total_component_ihm_to_ihm"],
            "component_ihm_enroll_to_sdm_test": positives["total_component_ihm_to_sdm"],
        },
        "negative_pairs": {
            "total": negatives["total_negative_pairs"],
            "raw_no_meeting_filter": negatives["total_raw_no_meeting_filter"],
            "excluded_by_meeting_overlap": negatives["total_excluded_by_meeting_overlap"],
            "by_gender": negatives["by_gender"],
            "teammate_pair_count": negatives["teammate_pair_count"],
            "stranger_pair_count": negatives["stranger_pair_count"],
        },
        "grand_total_trial_pairs": grand_total,
        "positive_to_negative_ratio": (
            f"1 : {round(negatives['total_negative_pairs'] / positives['total_positive_pairs'], 1)}"
            if positives["total_positive_pairs"] else None
        ),
    }

    full_output = {
        "summary": summary,
        "positive_pairs_per_speaker": positives["per_speaker"],
        "negative_pairs_teammate_detail": negatives["teammate_pairs_detail"],
        "enroll_vs_test_cross_check_detail": cross_check,
    }

    with open(args.out, "w") as f:
        json.dump(full_output, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFull per-speaker / per-pair detail written to {args.out}")


if __name__ == "__main__":
    main()