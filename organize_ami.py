import os
import shutil
from pathlib import Path

# Define the root path to your audio directory
BASE_DIR = "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"

# SAFETY TOGGLE: Set to False to actually move the files and delete empty folders.
# True will only print a preview of what WOULD happen.
DRY_RUN = False

def organize_dataset(root_path):
    root = Path(root_path)

    if not root.exists():
        print(f"Error: The path '{root_path}' does not exist.")
        return

    print(f"Scanning directory: {root_path}...")
    wav_files = list(root.rglob("*.wav"))
    print(f"Found {len(wav_files)} .wav files. Organizing...\n")

    moved_count = 0

    for file_path in wav_files:
        # Determine mic_type (ihm/sdm) and split (dev/eval/train) from the current path
        try:
            rel_path = file_path.relative_to(root)
            mic_type = rel_path.parts[0]      # e.g., 'ihm'
            dataset_split = rel_path.parts[1] # e.g., 'dev'
        except (ValueError, IndexError):
            print(f"Skipping {file_path.name}: not in expected ihm/sdm -> dev/eval/train structure.")
            continue

        filename = file_path.name
        name_without_ext = filename[:-4]
        parts = name_without_ext.split('_')

        # Extract Speaker ID based on the file name format
        if len(parts) >= 5 and parts[1] == 'ami':
            # Format: dev_ami_es2011a_h00_fee041_... (Original AMI format)
            speaker_id = parts[4]
        else:
            # Format: fee041_es2011a_dev_h00_... (Renamed format from previous script)
            speaker_id = parts[0]

        # Construct the new destination directory and file path
        # Example: .../audio/ihm/dev/fee041/
        dest_dir = root / mic_type / dataset_split / speaker_id
        dest_path = dest_dir / filename

        # Skip if the file is already in the correct speaker folder
        if file_path == dest_path:
            continue

        if DRY_RUN:
            print(f"[DRY RUN] Move: {rel_path}  ->  {dest_path.relative_to(root)}")
        else:
            # Create the speaker ID folder if it doesn't exist
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(file_path), str(dest_path))
            print(f"Moved: {filename} -> {mic_type}/{dataset_split}/{speaker_id}/")

        moved_count += 1

    print("\n--- Move Summary ---")
    if DRY_RUN:
        print(f"Files that WOULD be moved: {moved_count}")
        print("Set DRY_RUN = False in the script to apply changes.")
    else:
        print(f"Successfully moved: {moved_count} files.")

    # Clean up empty directories (like the old session ID folders)
    if not DRY_RUN:
        print("\n--- Cleaning up empty folders ---")
        # Sort directories deeply nested first to ensure child folders are deleted before parents
        for dir_path in sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    print(f"Removed empty directory: {dir_path.relative_to(root)}")
                except OSError:
                    pass
        print("Cleanup complete.")

if __name__ == "__main__":
    organize_dataset(BASE_DIR)
