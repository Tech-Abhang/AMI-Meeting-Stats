import os
from pathlib import Path

# Define the root path (Python handles spaces in strings perfectly)
BASE_DIR = "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"

# SAFETY TOGGLE: Set to False to actually rename the files. 
# True will only print what WOULD happen.
DRY_RUN = False

def process_dataset(root_path):
    root = Path(root_path)

    if not root.exists():
        print(f"Error: The path '{root_path}' does not exist.")
        return

    # Find all .wav files in subdirectories recursively
    print(f"Scanning directory: {root_path}...")
    wav_files = list(root.rglob("*.wav"))
    print(f"Found {len(wav_files)} .wav files. Processing...\n")

    renamed_count = 0
    skipped_count = 0

    for file_path in wav_files:
        old_name = file_path.name
        name_without_ext = old_name[:-4]  # Remove .wav
        parts = name_without_ext.split('_')

        # Validate against the expected AMI format
        # Example 1: dev_ami_es2011a_h00_fee041_0003427_0003714 (7 parts)
        # Example 2: eval_ami_en2002a_h00_mee073 (5 parts)
        if len(parts) >= 5 and parts[1] == 'ami':
            type_str = parts[0]     # e.g., dev, eval, train
            # parts[1] is 'ami' (skipped in output)
            session_id = parts[2]   # e.g., es2011a
            env = parts[3]          # e.g., h00
            speaker_id = parts[4]   # e.g., fee041

            if len(parts) == 7:
                # Includes start and end segments
                segment_id = f"{parts[5]}-{parts[6]}"
                new_name = f"{speaker_id}_{session_id}_{type_str}_{env}_{segment_id}.wav"
            elif len(parts) == 5:
                # No segments included
                new_name = f"{speaker_id}_{session_id}_{type_str}_{env}.wav"
            else:
                print(f"Skipping (unexpected part count): {old_name}")
                skipped_count += 1
                continue

            # Generate the new full file path
            new_file_path = file_path.with_name(new_name)

            if DRY_RUN:
                print(f"[DRY RUN] {old_name}  ->  {new_name}")
            else:
                file_path.rename(new_file_path)
                print(f"Renamed: {old_name}  ->  {new_name}")

            renamed_count += 1
        else:
            print(f"Skipping (does not match expected pattern): {old_name}")
            skipped_count += 1

    print("\n--- Summary ---")
    print(f"Total files found: {len(wav_files)}")
    if DRY_RUN:
        print(f"Files that WOULD be renamed: {renamed_count}")
        print("Set DRY_RUN = False in the script to apply changes.")
    else:
        print(f"Successfully renamed: {renamed_count}")
    print(f"Files skipped: {skipped_count}")

if __name__ == "__main__":
    process_dataset(BASE_DIR)
