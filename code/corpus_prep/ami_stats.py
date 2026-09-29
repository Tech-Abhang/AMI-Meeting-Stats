#!/usr/bin/env python3
"""
ami_stats.py (v4) — Statistical profile of a local AMI Meeting Corpus copy,
tailored for the P-ECAPA far-field speaker-verification project.

RESOLVED (v4): earlier versions flagged sdm/ files whose filename carries an
h0X (headset-channel) token instead of literal "sdm" as a possible data
corruption issue. The user spot-checked several such pairs by listening:
audio content correctly matches its FOLDER (ihm/ sounds near-field, sdm/
sounds far-field) regardless of what the miccode token in the filename says.
Conclusion: folder location is the reliable indicator of mic type; the
miccode token in an sdm/ filename is not always "sdm" (most likely it
preserves which reference headset channel supplied the segmentation
boundaries), and this is a naming-convention quirk, not corrupted or
misplaced audio. v4 reports the miccode-token distribution as a plain
descriptive stat instead of a "mismatch to investigate."

CONFIRMED layout (from actual filenames on the user's machine):

    <root>/audio/
      ihm/
        train/<speaker_dir>/fee041_es2011a_train_ihm_0003427-0003714.wav
                             fee041_es2011a_train_ihm_0003427-0003714.txt
        dev/...
        eval/...
      sdm/
        train/... dev/... eval/...

Filename stem:  {speaker_id}_{meeting_id}_{split}_{miccode}_{begin}-{end}
    speaker_id   AMI global speaker id, e.g. fee041
    meeting_id   AMI meeting id, e.g. es2011a (2-letter prefix = recording series,
                 a literal string slice — not an inference)
    split        train / dev / eval  (should match the folder it's in)
    miccode      NOT a fixed ihm/sdm token. Confirmed from real data:
                    - sdm folder  -> literal "sdm"
                    - ihm folder  -> "h00", "h01", "h02", ... (per-speaker headset
                      channel number), NOT the literal string "ihm"
                 Anything else is left as "unknown_mic_code" rather than guessed.
    begin, end   7-digit zero-padded integers. Empirically consistent with
                 centiseconds (i.e. begin/100 = start time in seconds) —
                 confirmed against actual audio duration (0.0s mean error on a
                 2000-file verification sample).

NOTE ON UNCERTAIN FIELDS: this version deliberately does NOT report speaker
gender or any other attribute that would require guessing from the speaker_id
string rather than reading it from a real AMI metadata table. If you need
gender/age/role, pull it from the official AMI corpus metadata
(corpusResources/meetings.xml or the speaker table) and join on speaker_id —
don't infer it from the filename.

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
    charts/*.png        duration histograms, split/mic-token/meeting-series
                         breakdowns, per-speaker and per-meeting histograms,
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
    r"(?P<split_in_name>train|dev|eval)_(?P<miccode>[a-zA-Z0-9]+)_"
    r"(?P<begin>\d+)-(?P<end>\d+)$"
)
# Whole-file / no-segment variant (no begin-end pair). Seen in the rename
# script's 5-part case; kept here so such files are counted, not silently
# dropped, if any survive on disk.
NOSEG_RE = re.compile(
    r"^(?P<sid>[a-zA-Z0-9]+)_(?P<meeting>[a-zA-Z0-9]+)_"
    r"(?P<split_in_name>train|dev|eval)_(?P<miccode>[a-zA-Z0-9]+)$"
)
HEADSET_RE = re.compile(r"^h(?P<channel>\d+)$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z']+")

# Enrollment/test protocol thresholds echoing the paper (Sec 3.1):
# ~5s enrollment utterances, ~2.5s test utterances, 6 enrollment utts/speaker.
ENROLL_MIN_S = 4.5
TEST_LOW_S, TEST_HIGH_S = 1.8, 3.2
ENROLL_NEEDED = 6


def parse_stem(stem: str):
    m = STEM_RE.match(stem)
    if m:
        d = m.groupdict()
        d["has_segment"] = True
        d["begin_i"] = int(d["begin"])
        d["end_i"] = int(d["end"])
        d["duration_from_name_s"] = round((d["end_i"] - d["begin_i"]) / 100.0, 3)
        d["zero_duration"] = d["begin_i"] == d["end_i"]
    else:
        m = NOSEG_RE.match(stem)
        if not m:
            return None
        d = m.groupdict()
        d["has_segment"] = False
        d["begin_i"] = None
        d["end_i"] = None
        d["duration_from_name_s"] = None
        d["zero_duration"] = False

    d["meeting_series"] = d["meeting"][:2].upper() if len(d["meeting"]) >= 2 else d["meeting"].upper()

    hm = HEADSET_RE.match(d["miccode"])
    if d["miccode"].lower() == "sdm":
        d["mic_canonical"] = "sdm"
        d["headset_channel"] = None
    elif hm:
        d["mic_canonical"] = "ihm"
        d["headset_channel"] = int(hm.group("channel"))
    else:
        d["mic_canonical"] = "unknown_mic_code"
        d["headset_channel"] = None
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
                        rec["mic_matches_folder"] = parsed["mic_canonical"] == mic_config
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
        s["sid_mismatch_count"] = sum(1 for r in parsed_rows if not r["sid_matches_folder"])
        s["no_segment_count"] = sum(1 for r in parsed_rows if not r["has_segment"])

    for dim in ("mic_config", "split"):
        s[f"count_by_{dim}"] = dict(Counter(r[dim] for r in rows))

    # Mic-code token: purely descriptive count of what token appears in the
    # filename, broken down by which folder it's actually sitting in. This
    # is the "count the name" ask — NOT a mismatch/error signal. Folder
    # location is the confirmed-reliable indicator of actual mic type.
    if parsed_rows:
        by_folder_token = defaultdict(Counter)
        for r in parsed_rows:
            by_folder_token[r["mic_config"]][r["miccode"]] += 1
        s["mic_code_token_by_folder"] = {folder: dict(counter) for folder, counter in by_folder_token.items()}

    seg_rows = [r for r in parsed_rows if r["has_segment"]]
    if seg_rows:
        s["zero_duration_count"] = sum(1 for r in seg_rows if r["zero_duration"])
        if s["zero_duration_count"]:
            zero_examples = [r["stem"] for r in seg_rows if r["zero_duration"]][:50]
            s["zero_duration_examples"] = zero_examples
            s["zero_duration_by_folder"] = dict(Counter(r["mic_config"] for r in seg_rows if r["zero_duration"]))
            s["zero_duration_by_split"] = dict(Counter(r["split"] for r in seg_rows if r["zero_duration"]))

    if seg_rows:
        durations = [r["duration_from_name_s"] for r in seg_rows]
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
        for r in seg_rows:
            by_split_hours[r["split"]] += r["duration_from_name_s"] / 3600
            by_cfg_hours[r["mic_config"]] += r["duration_from_name_s"] / 3600
        s["duration"]["hours_by_split"] = {k: round(v, 2) for k, v in by_split_hours.items()}
        s["duration"]["hours_by_mic_config"] = {k: round(v, 2) for k, v in by_cfg_hours.items()}

    if parsed_rows:
        s["speakers"] = {
            "unique_speaker_ids": len(set(r["sid"] for r in parsed_rows)),
            "utterances_per_speaker": dict(Counter(r["sid"] for r in parsed_rows)),
        }
        s["meetings"] = {
            "unique_meetings": len(set(r["meeting"] for r in parsed_rows)),
            "utterances_per_meeting": dict(Counter(r["meeting"] for r in parsed_rows)),
            "meeting_series_distribution": dict(Counter(r["meeting_series"] for r in parsed_rows)),
        }
        # ihm/sdm split counted TWO ways, deliberately kept separate:
        #  - by folder (ground truth for actual mic type, per user's audio spot-check)
        #  - by literal filename token (see mic_code_token_by_folder above — descriptive only)
        s["mic_canonical_counts"] = dict(Counter(r["mic_canonical"] for r in parsed_rows))
        unknown = [r["miccode"] for r in parsed_rows if r["mic_canonical"] == "unknown_mic_code"]
        if unknown:
            s["unknown_mic_codes"] = dict(Counter(unknown).most_common(30))

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
    if seg_rows:
        by_speaker_split_mic = defaultdict(list)
        for r in seg_rows:
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
    seg_rows = [r for r in parsed_rows if r["has_segment"]]
    if not parsed_rows:
        return

    # 1. duration histogram, IHM vs SDM overlay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mic, color in (("ihm", "#4C72B0"), ("sdm", "#DD8452")):
        d = [r["duration_from_name_s"] for r in seg_rows if r["mic_config"] == mic]
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

    # 5. mic-code token distribution by folder (purely descriptive — folder
    # location is the confirmed indicator of actual mic type, not this token)
    tok_by_folder = summary.get("mic_code_token_by_folder", {})
    if tok_by_folder:
        all_tokens = sorted({t for counts in tok_by_folder.values() for t in counts})
        folders = sorted(tok_by_folder.keys())
        x = range(len(all_tokens))
        width = 0.8 / max(1, len(folders))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, folder in enumerate(folders):
            vals = [tok_by_folder[folder].get(t, 0) for t in all_tokens]
            ax.bar([xi + i * width for xi in x], vals, width=width, label=folder)
        ax.set_xticks([xi + width * (len(folders) - 1) / 2 for xi in x])
        ax.set_xticklabels(all_tokens, rotation=0)
        ax.set_ylabel("Utterance count")
        ax.set_title("Filename mic-code token, by folder (descriptive — not an error signal)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts_dir / "mic_code_token_by_folder.png", dpi=150)
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

    # 9. utterances per meeting histogram
    upm = summary.get("meetings", {}).get("utterances_per_meeting", {})
    if upm:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(list(upm.values()), bins=30, color="#4C9F70")
        ax.set_xlabel("Utterances per meeting")
        ax.set_ylabel("Number of meetings")
        ax.set_title(f"Utterance count per meeting (n={len(upm)} meetings)")
        fig.tight_layout()
        fig.savefig(charts_dir / "utterances_per_meeting_hist.png", dpi=150)
        plt.close(fig)


def write_markdown(summary, out_path: Path, root: Path):
    lines = [f"# AMI corpus statistics — `{root}`", ""]
    lines.append(f"**Total utterances scanned:** {summary['total_files']}")
    lines.append(f"**Filenames matching expected pattern:** {summary.get('parsed_filename_pct')}")
    lines.append(f"**Missing .wav:** {summary.get('missing_wav')}  |  **Missing .txt:** {summary.get('missing_txt')}")
    if "split_mismatch_count" in summary:
        lines.append(f"**Filename/folder split mismatches:** {summary['split_mismatch_count']}  |  "
                     f"**speaker-id/folder mismatches:** {summary['sid_mismatch_count']}  |  "
                     f"**no-segment (whole-file) entries:** {summary.get('no_segment_count', 0)}")
    lines.append("")

    if summary.get("zero_duration_count"):
        lines.append("## Zero-duration segments (begin == end in filename)")
        lines.append(f"- Count: {summary['zero_duration_count']}")
        lines.append(f"- By folder: {summary.get('zero_duration_by_folder')}")
        lines.append(f"- By split: {summary.get('zero_duration_by_split')}")
        lines.append(f"- Example stems (up to 50): {summary.get('zero_duration_examples')}")
        lines.append("  -> these carry 0.00s duration by construction; check whether they're empty/corrupt")
        lines.append("     files or a segmentation artifact before including them in any duration statistic.")
        lines.append("")
    elif "zero_duration_count" in summary:
        lines.append("## Zero-duration segments (begin == end in filename)")
        lines.append("- None found.")
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
        lines.append("## Speaker statistics")
        lines.append(f"- Unique speaker IDs: {sp['unique_speaker_ids']} (paper reports 188 for full AMI)")
        lines.append(f"- Utterance count per speaker: see files.csv / xlsx 'Per-Speaker' sheet for the full table "
                     f"(min {min(sp['utterances_per_speaker'].values())}, "
                     f"max {max(sp['utterances_per_speaker'].values())})")
        lines.append("")

    if "meetings" in summary:
        mt = summary["meetings"]
        lines.append("## Meeting (session) statistics")
        lines.append(f"- Unique meetings: {mt['unique_meetings']}")
        lines.append(f"- Utterance count per meeting: min {min(mt['utterances_per_meeting'].values())}, "
                     f"max {max(mt['utterances_per_meeting'].values())} "
                     f"(full per-meeting table in files.csv / xlsx)")
        lines.append(f"- Meeting-ID prefix distribution: {mt['meeting_series_distribution']}")
        lines.append("")

    if "mic_code_token_by_folder" in summary:
        lines.append("## Mic-code token in filename, by folder (descriptive — not an error)")
        lines.append("Folder location (ihm/ vs sdm/) is the confirmed-reliable indicator of actual mic type — "
                     "verified by direct listening on paired ihm/sdm files. The token embedded in the filename "
                     "(sdm, h00, h01, h02, h03, ...) does not always say 'sdm' for files in the sdm/ folder; "
                     "this is a naming-convention quirk (most likely: which reference channel supplied the "
                     "segmentation boundaries), not a data-placement error. Counts below are purely descriptive.")
        for folder, counts in summary["mic_code_token_by_folder"].items():
            lines.append(f"- {folder}/: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
        if summary.get("unknown_mic_codes"):
            lines.append(f"- Tokens that are neither 'sdm' nor 'hNN' (top 30): {summary['unknown_mic_codes']}")
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