#!/usr/bin/env python3
"""
check_h0x_duplication.py — resolves the one open question from the mic-label
audit: are the h0X-named files sitting under sdm/ duplicates of files already
in ihm/, or the only copy of that segment?

Usage:
    python check_h0x_duplication.py "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"
"""
import os, re, sys
from pathlib import Path
from collections import defaultdict

STEM_RE = re.compile(r"^([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(train|dev|eval)_([a-zA-Z0-9]+)_(\d+)-(\d+)$")

def collect_stems(root, mic_config):
    stems = {}
    cfg_dir = root / mic_config
    for split_dir in cfg_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            with os.scandir(speaker_dir) as it:
                for entry in it:
                    if entry.name.endswith(".wav"):
                        stem = entry.name[:-4]
                        stems[stem] = entry.path
    return stems

def main():
    root = Path(sys.argv[1])
    print("Scanning ihm/ ...")
    ihm_stems = collect_stems(root, "ihm")
    print(f"  {len(ihm_stems)} wav files")
    print("Scanning sdm/ ...")
    sdm_stems = collect_stems(root, "sdm")
    print(f"  {len(sdm_stems)} wav files")

    sdm_h0x = {s: p for s, p in sdm_stems.items() if (m := STEM_RE.match(s)) and m.group(4) != "sdm"}
    print(f"\nsdm/ files with an h0X (non-'sdm') mic code: {len(sdm_h0x)}")

    exact_dupe = [s for s in sdm_h0x if s in ihm_stems]
    orphan = [s for s in sdm_h0x if s not in ihm_stems]
    print(f"  -> exact stem also present under ihm/  : {len(exact_dupe)} ({100*len(exact_dupe)/max(1,len(sdm_h0x)):.1f}%)")
    print(f"  -> stem NOT found anywhere under ihm/  : {len(orphan)} ({100*len(orphan)/max(1,len(sdm_h0x)):.1f}%)")

    if exact_dupe:
        print("\nByte-comparing a sample of 'exact stem' matches to confirm they're true duplicates (not just same name)...")
        import filecmp
        sample = exact_dupe[:200]
        identical = 0
        for s in sample:
            try:
                if filecmp.cmp(sdm_h0x[s], ihm_stems[s], shallow=False):
                    identical += 1
            except Exception as e:
                print(f"  couldn't compare {s}: {e}")
        print(f"  {identical}/{len(sample)} sampled pairs are byte-identical")

    if orphan:
        print(f"\nFirst 10 orphan stems (exist ONLY under sdm/, not in ihm/ at all):")
        for s in orphan[:10]:
            print(f"  {s}")

    print("\n--- Verdict ---")
    if len(exact_dupe) / max(1, len(sdm_h0x)) > 0.9:
        print("Mostly duplicates: safe to treat sdm/'s h0X files as redundant copies of ihm/ data.")
    elif len(orphan) / max(1, len(sdm_h0x)) > 0.9:
        print("Mostly orphans: these are real headset segments missing from ihm/ — recovery candidates, not junk.")
    else:
        print("Mixed: handle the two groups differently (see printed counts above).")

if __name__ == "__main__":
    main()