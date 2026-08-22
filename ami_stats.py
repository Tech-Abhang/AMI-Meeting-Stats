#!/usr/bin/env python3
"""
ami_stats.py (v2) — Statistical profile of a local AMI Meeting Corpus copy,
tailored for the P-ECAPA far-field speaker-verification project.

CONFIRMED layout (from actual filenames on the user's machine):

    <root>/audio/
      ihm/
        train/<speaker_dir>/fee041_es2011a_train_ihm_0003427-0003714.wav
                             fee041_es2011a_train_ihm_0003427-0003714.txt
        dev/...
        eval/...
      sdm/
        train/... dev/... eval/...

Filename stem:  {speaker_id}_{meeting_id}_{split}_{mic}_{begin}-{end}
    speaker_id   AMI global speaker id, e.g. fee041 (first letter = gender: f/m)
    meeting_id   AMI meeting id, e.g. es2011a (2-letter prefix = recording series)
    split        train / dev / eval  (should match the folder it's in)
    mic          ihm / sdm           (should match the folder it's in)
    begin, end   7-digit zero-padded integers. Empirically consistent with
                 centiseconds (i.e. begin/100 = start time in seconds) —
                 verified against actual audio duration when --verify is on.

Usage:
    pip install soundfile matplotlib --break-system-packages

    # 1. quick smoke test on a subset first (recommended given this looks
    #    like an external USB drive — full corpus is ~250k+ files)
    python ami_stats.py "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \
        --out ./ami_stats_out --sample 5000 --verify-sample 500

    # 2. full run
    python ami_stats.py "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio" \
        --out ./ami_stats_out --verify-sample 2000

Outputs in --out:
    files.csv        one row per utterance (all parsed + derived fields)
    summary.json      every aggregate number, machine-readable
    summary.md         human-readable report
    charts/*.png        duration histograms, split/mic/gender/site breakdowns,
                         enrollment-feasibility bar chart
"""

import argparse
import json
import os
import random
import re
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

STEM_RE = re.compile(
    r"^(?P<sid>[a-zA-Z0-9]+)_(?P<meeting>[a-zA-Z0-9]+)_"
    r"(?P<split_in_name>train|dev|eval)_(?P<mic_in_name>ihm|sdm)_"
    r"(?P<begin>\d+)-(?P<end>\d+)$"
)
WORD_RE = re.compile(r"[A-Za-z']+")

# Enrollment/test protocol thresholds echoing the paper (Sec 3.1):
# ~5s enrollment utterances, ~2.5s test utterances, 6 enrollment utts/speaker.
ENROLL_MIN_S = 4.5
TEST_LOW_S, TEST_HIGH_S = 1.8, 3.2
ENROLL_NEEDED = 6


def parse_stem(stem: str):
    m = STEM_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["begin_i"] = int(d["begin"])
    d["end_i"] = int(d["end"])
    d["duration_from_name_s"] = round((d["end_i"] - d["begin_i"]) / 100.0, 3)
    d["gender_guess"] = d["sid"][0].lower() if d["sid"] and d["sid"][0].lower() in ("f", "m") else "?"
    d["meeting_series"] = d["meeting"][:2].upper() if len(d["meeting"]) >= 2 else d["meeting"].upper()
    return d


def read_text_stats(txt_path: Path):
    try:
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    words = WORD_RE.findall(raw)
    return {
        "char_count": len(raw),
        "word_count": len(words),
        "unique_words": len(set(w.lower() for w in words)),
    }


def walk_corpus(root: Path, sample_limit=None):
    n = 0
    for mic_config in ("ihm", "sdm"):
        cfg_dir = root / mic_config
        if not cfg_dir.is_dir():
            continue
        for split_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            split = split_dir.name
            for speaker_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                speaker_folder = speaker_dir.name
                by_stem = defaultdict(dict)
                with os.scandir(speaker_dir) as it:
                    for entry in it:
                        suf = Path(entry.name).suffix.lower()
                        if suf in (".wav", ".txt"):
                            by_stem[Path(entry.name).stem][suf] = Path(entry.path)
                for stem, pair in by_stem.items():
                    n += 1
                    if sample_limit and n > sample_limit:
                        return
                    rec = {
                        "mic_config": mic_config,
                        "split": split,
                        "speaker_folder": speaker_folder,
                        "stem": stem,
                        "has_wav": ".wav" in pair,
                        "has_txt": ".txt" in pair,
                    }
                    parsed = parse_stem(stem)
                    rec["parsed"] = parsed is not None
                    if parsed:
                        rec.update(parsed)
                        rec["split_matches_folder"] = parsed["split_in_name"] == split
                        rec["mic_matches_folder"] = parsed["mic_in_name"] == mic_config
                        rec["sid_matches_folder"] = parsed["sid"].lower() == speaker_folder.lower()
                    if rec["has_wav"]:
                        rec["file_size_bytes"] = pair[".wav"].stat().st_size
                        rec["_wav_path"] = str(pair[".wav"])
                    if rec["has_txt"]:
                        tstats = read_text_stats(pair[".txt"])
                        if tstats:
                            rec.update(tstats)
                    yield rec


def verify_audio_sample(rows, n_verify):
    """Open a random subset of wav files to get real duration/sample_rate/
    channels and cross-check against the filename-derived duration."""
    if sf is None or n_verify <= 0:
        return {}
    candidates = [r for r in rows if r.get("_wav_path") and r.get("duration_from_name_s") is not None]
    if not candidates:
        return {}
    sample = random.sample(candidates, min(n_verify, len(candidates)))
    diffs, srs, chans = [], Counter(), Counter()
    for r in sample:
        try:
            info = sf.info(r["_wav_path"])
        except Exception:
            continue
        srs[info.samplerate] += 1
        chans[info.channels] += 1
        diffs.append(abs(info.duration - r["duration_from_name_s"]))
    if not diffs:
        return {}
    return {
        "n_verified": len(diffs),
        "sample_rates": dict(srs),
        "channels": dict(chans),
        "mean_abs_diff_name_vs_actual_s": round(stats.mean(diffs), 4),
        "max_abs_diff_name_vs_actual_s": round(max(diffs), 4),
    }


def pct(x, total):
    return f"{100.0 * x / total:.1f}%" if total else "n/a"


def duration_bucket(d):
    if d < 1:
        return "<1s (likely backchannel)"
    if d < 3:
        return "1-3s"
    if d < 10:
        return "3-10s"
    return ">10s"


def summarize(rows, audio_verify):
    total = len(rows)
    s = {"total_files": total}
    if total == 0:
        return s

    s["missing_wav"] = sum(1 for r in rows if not r["has_wav"])
    s["missing_txt"] = sum(1 for r in rows if not r["has_txt"])
    s["parsed_filename_pct"] = pct(sum(1 for r in rows if r["parsed"]), total)

    parsed_rows = [r for r in rows if r["parsed"]]
    if parsed_rows:
        s["split_mismatch_count"] = sum(1 for r in parsed_rows if not r["split_matches_folder"])
        s["mic_mismatch_count"] = sum(1 for r in parsed_rows if not r["mic_matches_folder"])
        s["sid_mismatch_count"] = sum(1 for r in parsed_rows if not r["sid_matches_folder"])

    for dim in ("mic_config", "split"):
        s[f"count_by_{dim}"] = dict(Counter(r[dim] for r in rows))

    if parsed_rows:
        durations = [r["duration_from_name_s"] for r in parsed_rows]
        s["duration"] = {
            "n": len(durations),
            "total_hours": round(sum(durations) / 3600, 2),
            "mean_s": round(stats.mean(durations), 3),
            "median_s": round(stats.median(durations), 3),
            "stdev_s": round(stats.pstdev(durations), 3) if len(durations) > 1 else 0,
            "min_s": round(min(durations), 3),
            "max_s": round(max(durations), 3),
            "bucket_counts": dict(Counter(duration_bucket(d) for d in durations)),
        }
        by_split_hours = defaultdict(float)
        by_cfg_hours = defaultdict(float)
        for r in parsed_rows:
            by_split_hours[r["split"]] += r["duration_from_name_s"] / 3600
            by_cfg_hours[r["mic_config"]] += r["duration_from_name_s"] / 3600
        s["duration"]["hours_by_split"] = {k: round(v, 2) for k, v in by_split_hours.items()}
        s["duration"]["hours_by_mic_config"] = {k: round(v, 2) for k, v in by_cfg_hours.items()}

        s["speakers"] = {
            "unique_speaker_ids": len(set(r["sid"] for r in parsed_rows)),
            "utterances_per_speaker": dict(Counter(r["sid"] for r in parsed_rows)),
            "unique_meetings": len(set(r["meeting"] for r in parsed_rows)),
            "meeting_series_distribution": dict(Counter(r["meeting_series"] for r in parsed_rows)),
            "gender_distribution_guess": dict(Counter(r["gender_guess"] for r in parsed_rows)),
        }

    words = [r["word_count"] for r in rows if r.get("word_count") is not None]
    if words:
        s["text"] = {
            "n_with_text": len(words),
            "total_words": sum(words),
            "mean_words_per_utt": round(stats.mean(words), 2),
            "median_words_per_utt": round(stats.median(words), 2),
        }

    # split-integrity: speaker overlap across splits
    split_speakers = defaultdict(set)
    for r in rows:
        key = r.get("sid") or r["speaker_folder"]
        split_speakers[r["split"]].add(key)
    overlaps = {}
    splits = list(split_speakers.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            ov = split_speakers[a] & split_speakers[b]
            if ov:
                overlaps[f"{a}_vs_{b}"] = sorted(ov)
    s["speaker_leakage_across_splits"] = overlaps

    # enrollment/test feasibility per paper's protocol (Sec 3.1)
    if parsed_rows:
        by_speaker_split_mic = defaultdict(list)
        for r in parsed_rows:
            by_speaker_split_mic[(r["split"], r["mic_config"], r["sid"])].append(r["duration_from_name_s"])
        feas = {"eligible_speakers": 0, "total_speaker_conditions": 0, "test_utt_pool": 0}
        per_split_eligible = Counter()
        per_split_total = Counter()
        for (split, mic, sid), durs in by_speaker_split_mic.items():
            n_enroll = sum(1 for d in durs if d >= ENROLL_MIN_S)
            n_test = sum(1 for d in durs if TEST_LOW_S <= d <= TEST_HIGH_S)
            feas["total_speaker_conditions"] += 1
            per_split_total[split] += 1
            if n_enroll >= ENROLL_NEEDED:
                feas["eligible_speakers"] += 1
                feas["test_utt_pool"] += n_test
                per_split_eligible[split] += 1
        feas["eligible_by_split"] = dict(per_split_eligible)
        feas["total_by_split"] = dict(per_split_total)
        s["enrollment_feasibility"] = feas

    if audio_verify:
        s["audio_verification_sample"] = audio_verify

    return s


def make_charts(rows, summary, out_dir: Path):
    if plt is None:
        print("matplotlib not installed — skipping charts (pip install matplotlib --break-system-packages)",
              file=sys.stderr)
        return
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    parsed_rows = [r for r in rows if r["parsed"]]
    if not parsed_rows:
        return

    durations = [r["duration_from_name_s"] for r in parsed_rows]

    # 1. duration histogram, IHM vs SDM overlay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mic, color in (("ihm", "#4C72B0"), ("sdm", "#DD8452")):
        d = [r["duration_from_name_s"] for r in parsed_rows if r["mic_config"] == mic]
        if d:
            ax.hist(d, bins=60, range=(0, 15), alpha=0.55, label=mic.upper(), color=color)
    ax.set_xlabel("Utterance duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Utterance duration distribution — IHM vs SDM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "duration_hist_ihm_vs_sdm.png", dpi=150)
    plt.close(fig)

    # 2. hours by split
    hbs = summary.get("duration", {}).get("hours_by_split", {})
    if hbs:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(hbs.keys()), list(hbs.values()), color="#55A868")
        ax.set_ylabel("Hours")
        ax.set_title("Total audio hours per split")
        fig.tight_layout()
        fig.savefig(charts_dir / "hours_by_split.png", dpi=150)
        plt.close(fig)

    # 3. hours by mic config
    hbc = summary.get("duration", {}).get("hours_by_mic_config", {})
    if hbc:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(list(hbc.keys()), list(hbc.values()), color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Hours")
        ax.set_title("Total audio hours — IHM vs SDM")
        fig.tight_layout()
        fig.savefig(charts_dir / "hours_by_mic_config.png", dpi=150)
        plt.close(fig)

    # 4. utterances per speaker histogram
    ups = summary.get("speakers", {}).get("utterances_per_speaker", {})
    if ups:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(list(ups.values()), bins=30, color="#8172B2")
        ax.set_xlabel("Utterances per speaker")
        ax.set_ylabel("Number of speakers")
        ax.set_title("Utterance count per speaker")
        fig.tight_layout()
        fig.savefig(charts_dir / "utterances_per_speaker_hist.png", dpi=150)
        plt.close(fig)

    # 5. gender guess distribution
    gd = summary.get("speakers", {}).get("gender_distribution_guess", {})
    if gd:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(gd.values(), labels=[k.upper() for k in gd.keys()], autopct="%1.0f%%",
               colors=["#DD8452", "#4C72B0", "#999999"])
        ax.set_title("Utterance share by gender (guess: first letter of speaker_id)")
        fig.tight_layout()
        fig.savefig(charts_dir / "gender_distribution.png", dpi=150)
        plt.close(fig)

    # 6. meeting series distribution
    ms = summary.get("speakers", {}).get("meeting_series_distribution", {})
    if ms:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(list(ms.keys()), list(ms.values()), color="#64B5CD")
        ax.set_ylabel("Utterance count")
        ax.set_title("Utterances by meeting-ID prefix (recording series)")
        fig.tight_layout()
        fig.savefig(charts_dir / "meeting_series_distribution.png", dpi=150)
        plt.close(fig)

    # 7. enrollment feasibility
    feas = summary.get("enrollment_feasibility", {})
    if feas.get("eligible_by_split"):
        splits = sorted(feas["total_by_split"].keys())
        eligible = [feas["eligible_by_split"].get(sp, 0) for sp in splits]
        total = [feas["total_by_split"].get(sp, 0) for sp in splits]
        x = range(len(splits))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x, total, label="speaker-conditions (speaker x split x mic)", color="#CCCCCC")
        ax.bar(x, eligible, label=f"eligible (>= {ENROLL_NEEDED} utts >= {ENROLL_MIN_S}s)", color="#55A868")
        ax.set_xticks(list(x))
        ax.set_xticklabels(splits)
        ax.set_ylabel("Count")
        ax.set_title("Enrollment feasibility per protocol in Sec 3.1")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(charts_dir / "enrollment_feasibility.png", dpi=150)
        plt.close(fig)

    # 8. duration buckets
    bc = summary.get("duration", {}).get("bucket_counts", {})
    if bc:
        order = ["<1s (likely backchannel)", "1-3s", "3-10s", ">10s"]
        keys = [k for k in order if k in bc]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(keys, [bc[k] for k in keys], color="#C44E52")
        ax.set_ylabel("Utterance count")
        ax.set_title("Utterance duration buckets")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(charts_dir / "duration_buckets.png", dpi=150)
        plt.close(fig)


def write_markdown(summary, out_path: Path, root: Path):
    lines = [f"# AMI corpus statistics — `{root}`", ""]
    lines.append(f"**Total utterances scanned:** {summary['total_files']}")
    lines.append(f"**Filenames matching expected pattern:** {summary.get('parsed_filename_pct')}")
    lines.append(f"**Missing .wav:** {summary.get('missing_wav')}  |  **Missing .txt:** {summary.get('missing_txt')}")
    if "split_mismatch_count" in summary:
        lines.append(f"**Filename/folder split mismatches:** {summary['split_mismatch_count']}  |  "
                     f"**mic mismatches:** {summary['mic_mismatch_count']}  |  "
                     f"**speaker-id/folder mismatches:** {summary['sid_mismatch_count']}")
    lines.append("")

    if "duration" in summary:
        d = summary["duration"]
        lines.append("## Duration statistics (derived from filename begin/end)")
        lines.append(f"- Total: {d['total_hours']} hours across {d['n']} utterances")
        lines.append(f"- Mean/median/stdev (s): {d['mean_s']} / {d['median_s']} / {d['stdev_s']}")
        lines.append(f"- Min/max (s): {d['min_s']} / {d['max_s']}")
        lines.append(f"- Hours by split: {d.get('hours_by_split')}")
        lines.append(f"- Hours IHM vs SDM: {d.get('hours_by_mic_config')}")
        lines.append(f"- Duration buckets: {d.get('bucket_counts')}")
        lines.append("")

    if "audio_verification_sample" in summary:
        v = summary["audio_verification_sample"]
        lines.append("## Audio header verification (random sample)")
        lines.append(f"- Files opened: {v['n_verified']}")
        lines.append(f"- Sample rates found: {v['sample_rates']}")
        lines.append(f"- Channel counts found: {v['channels']}")
        lines.append(f"- Mean |filename-duration - actual-duration|: {v['mean_abs_diff_name_vs_actual_s']} s "
                     f"(max {v['max_abs_diff_name_vs_actual_s']} s) — near 0 confirms the centisecond assumption")
        lines.append("")

    if "speakers" in summary:
        sp = summary["speakers"]
        lines.append("## Speaker / meeting statistics")
        lines.append(f"- Unique speaker IDs: {sp['unique_speaker_ids']} (paper reports 188 for full AMI)")
        lines.append(f"- Unique meetings: {sp['unique_meetings']}")
        lines.append(f"- Meeting-ID prefix distribution: {sp['meeting_series_distribution']}")
        lines.append(f"- Gender split (guessed from speaker_id first letter — verify against AMI speaker table): "
                     f"{sp['gender_distribution_guess']}")
        lines.append("")

    if "text" in summary:
        t = summary["text"]
        lines.append("## Transcript statistics")
        lines.append(f"- Total words: {t['total_words']}")
        lines.append(f"- Mean/median words per utterance: {t['mean_words_per_utt']} / {t['median_words_per_utt']}")
        lines.append("")

    lines.append("## Split integrity (speaker overlap between train/dev/eval)")
    leak = summary.get("speaker_leakage_across_splits", {})
    if leak:
        for pair, speakers in leak.items():
            lines.append(f"- **{pair}: {len(speakers)} speaker(s) overlap** -> {speakers}")
    else:
        lines.append("- None detected — splits are speaker-disjoint.")
    lines.append("")

    if "enrollment_feasibility" in summary:
        f = summary["enrollment_feasibility"]
        lines.append("## Enrollment/test feasibility (paper protocol: 6 enroll utts >=~5s, test utts ~2.5s)")
        lines.append(f"- Speaker x split x mic conditions with >= {ENROLL_NEEDED} enrollment-length utterances: "
                     f"{f['eligible_speakers']} / {f['total_speaker_conditions']}")
        lines.append(f"- Eligible by split: {f.get('eligible_by_split')}")
        lines.append(f"- Total test-length utterance pool (eligible conditions only): {f['test_utt_pool']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Path to the 'audio' folder (contains ihm/ and sdm/)")
    ap.add_argument("--out", type=Path, default=Path("./ami_stats_out"))
    ap.add_argument("--sample", type=int, default=None, help="Only scan first N files (quick test)")
    ap.add_argument("--verify-sample", type=int, default=1000,
                     help="How many wav files to actually open to verify duration/sample-rate (0 = skip)")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"Root path not found: {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.root} ...")
    rows = list(walk_corpus(args.root, sample_limit=args.sample))
    if not rows:
        sys.exit("No .wav/.txt files found under ihm/ or sdm/. Check that --root points at the 'audio' "
                  "folder that directly contains 'ihm' and 'sdm'.")
    print(f"Found {len(rows)} utterance file-pairs.")

    audio_verify = verify_audio_sample(rows, args.verify_sample)

    import csv
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    fieldnames = sorted({k for r in csv_rows for k in r.keys()})
    with open(args.out / "files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    summary = summarize(rows, audio_verify)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(summary, args.out / "summary.md", args.root)
    make_charts(rows, summary, args.out)

    print(f"Wrote files.csv, summary.json, summary.md, charts/ to {args.out}")


if __name__ == "__main__":
    main()