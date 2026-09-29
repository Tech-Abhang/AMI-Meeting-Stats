import os
import subprocess
from pathlib import Path

# Define the root path to your audio directory
BASE_DIR = "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"

def standardize_dataset(root_path):
    root = Path(root_path)

    if not root.exists():
        print(f"Error: The path '{root_path}' does not exist.")
        return

    print(f"Scanning directory: {root_path}...")
    wav_files = list(root.rglob("*.wav"))
    total_files = len(wav_files)
    
    if total_files == 0:
        print("No .wav files found in the specified directory.")
        return

    print(f"Found {total_files} .wav files. Beginning Golden Standard conversion (In-Place)...")
    print("This may take some time depending on your external drive's write speed.\n")

    processed_count = 0
    failed_count = 0

    for i, file_path in enumerate(wav_files, 1):
        # Create a temporary filename in the same directory
        temp_file = file_path.with_name(f"temp_{file_path.name}")

        # Construct the SoX Golden Standard Command
        # -r 16000 : Resample to 16 kHz
        # -c 1     : Mixdown to Mono (1 channel)
        # -b 16    : Set bit-depth to 16-bit
        # -e signed-integer : Set encoding to PCM signed integer
        cmd = [
            "sox", str(file_path),
            "-r", "16000",
            "-c", "1",
            "-b", "16",
            "-e", "signed-integer",
            str(temp_file)
        ]

        try:
            # Execute SoX command. Hide standard output, but raise error if it fails.
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # NO BACKUP: Replace the original file with the newly standardized temporary file
            temp_file.replace(file_path)
            processed_count += 1
            
            # Print progress every 100 files
            if i % 100 == 0 or i == total_files:
                print(f"Progress: [{i}/{total_files}] files standardized...")

        except subprocess.CalledProcessError:
            print(f"Failed to process via SoX: {file_path.name}")
            if temp_file.exists():
                temp_file.unlink() # Cleanup the corrupted temp file
            failed_count += 1
            
        except Exception as e:
            print(f"Unexpected error on {file_path.name}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            failed_count += 1

    print("\n--- Conversion Summary ---")
    print(f"Total files scanned : {total_files}")
    print(f"Successfully updated: {processed_count}")
    print(f"Failed/Skipped      : {failed_count}")

if __name__ == "__main__":
    standardize_dataset(BASE_DIR)
