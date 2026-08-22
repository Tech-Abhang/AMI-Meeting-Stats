import os
from pathlib import Path

# Define the root dataset path
BASE_DIR = Path("/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus")
AUDIO_DIR = BASE_DIR / "audio"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

# SAFETY TOGGLE: Set to False to actually write the .txt files.
DRY_RUN = False

def get_agnostic_key(filename):
    """
    Strips the microphone/environment tag (4th element) from the filename.
    'fee041_es2011a_dev_h00_0003427-0003714.wav' -> 'fee041_es2011a_dev_0003427-0003714'
    """
    parts = filename[:-4].split('_')
    if len(parts) >= 4:
        return "_".join(parts[:3] + parts[4:])
    return filename[:-4]

def generate_transcripts():
    print("--- Generating Transcripts (Ground-Truth File Driven) ---")
    splits = ["dev", "eval", "train"]
    total_txt = {"ihm": 0, "sdm": 0}

    for split in splits:
        print(f"\n================ Processing Split: {split.upper()} ================")
        text_file_path = ANNOTATIONS_DIR / split / "text"
        
        if not text_file_path.exists():
            print(f"Warning: Annotation file not found at {text_file_path}")
            continue

        # 1. Build the Transcript Dictionary in memory using the Agnostic Key
        transcript_dict = {}
        with open(text_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(' ', 1)
                utterance_id = parts[0]
                transcript = parts[1] if len(parts) > 1 else ""

                id_parts = utterance_id.split('_')
                if len(id_parts) < 4:
                    continue 
                
                session = id_parts[1].lower()
                speaker = id_parts[3].lower()
                segments = f"{id_parts[4]}-{id_parts[5]}" if len(id_parts) >= 6 else ""

                # Reconstruct the agnostic key to perfectly match the file generator
                if segments:
                    agnostic_key = f"{speaker}_{session}_{split}_{segments}"
                else:
                    agnostic_key = f"{speaker}_{session}_{split}"
                
                transcript_dict[agnostic_key] = transcript

        print(f"Loaded {len(transcript_dict)} annotations into memory.")

        # 2. Iterate through the actual .wav files and pair them
        for mic_type in ["ihm", "sdm"]:
            target_dir = AUDIO_DIR / mic_type / split
            if not target_dir.exists():
                continue
                
            wav_files = list(target_dir.rglob("*.wav"))
            txt_created = 0

            for wav_path in wav_files:
                # Generate the file's native agnostic key
                file_key = get_agnostic_key(wav_path.name)
                
                # Look it up in our transcript dictionary
                if file_key in transcript_dict:
                    txt_path = wav_path.with_suffix(".txt")
                    
                    if not DRY_RUN:
                        with open(txt_path, "w", encoding="utf-8") as txt_out:
                            txt_out.write(transcript_dict[file_key])
                    
                    txt_created += 1
                    total_txt[mic_type] += 1

            if DRY_RUN:
                print(f"[DRY RUN] {mic_type.upper()}: WOULD generate {txt_created} .txt files (out of {len(wav_files)} .wavs)")
            else:
                print(f"{mic_type.upper()}: Generated {txt_created} .txt files.")

    print("\n================ FINAL SUMMARY ================")
    if not DRY_RUN:
        print(f"IHM Transcripts generated: {total_txt['ihm']}")
        print(f"SDM Transcripts generated: {total_txt['sdm']}")
        if total_txt['ihm'] == total_txt['sdm']:
            print("SUCCESS: Perfect Transcript Parity Confirmed!")
        else:
            print("WARNING: Parity mismatch.")
    else:
        print(f"[DRY RUN] Total IHM files to create: {total_txt['ihm']}")
        print(f"[DRY RUN] Total SDM files to create: {total_txt['sdm']}")
        print("Set DRY_RUN = False to apply changes.")

if __name__ == "__main__":
    generate_transcripts()
