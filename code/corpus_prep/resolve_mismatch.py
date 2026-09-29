import shutil
from pathlib import Path

# Define the root dataset path
BASE_DIR = Path("/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus")
AUDIO_DIR = BASE_DIR / "audio"

# SAFETY TOGGLE: Set to False to actually move files.
DRY_RUN = False

def get_agnostic_key(rel_path):
    """
    Strips the microphone/environment tag from the filename to create a neutral key.
    Example: 'mtd013pm/mtd013pm_ts3004c_dev_h00_0243804.wav' 
          -> 'mtd013pm/mtd013pm_ts3004c_dev_0243804'
    """
    parts = rel_path.name[:-4].split('_')  # Remove .wav and split by underscore
    
    if len(parts) >= 4:
        # parts[3] is the environment tag (e.g., 'h00' or 'sdm')
        # We skip parts[3] and join the rest back together
        agnostic_name = "_".join(parts[:3] + parts[4:])
        return str(rel_path.parent / agnostic_name)
    
    # Fallback if the filename format is unexpected
    return str(rel_path)

def move_orphans(orphans, src_base, dest_base, name, split):
    moved_count = 0
    for rel_path in orphans:
        src = src_base / rel_path
        dest = dest_base / rel_path
        
        if DRY_RUN:
            if moved_count < 3:
                print(f"[DRY RUN] {name} Move ({split}): {rel_path}")
            elif moved_count == 3:
                print(f"[DRY RUN] ... suppressing further {name} ({split}) output ...")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        
        moved_count += 1
    return moved_count

def enforce_parallel_data():
    print("--- Enforcing Strict (Agnostic) Parallel Parity ---")
    
    splits = ["dev", "eval", "train"]
    total_perfect_pairs = 0
    total_moved = 0
    
    for split in splits:
        print(f"\n================ Processing Split: {split.upper()} ================")
        
        ihm_dir = AUDIO_DIR / "ihm" / split
        sdm_dir = AUDIO_DIR / "sdm" / split
        extra_ihm = AUDIO_DIR / f"extra_ihm_{split}"
        extra_sdm = AUDIO_DIR / f"extra_sdm_{split}"

        if not ihm_dir.exists() or not sdm_dir.exists():
            print(f"Warning: {split} directories missing. Skipping.")
            continue

        print("Scanning directories and generating agnostic fingerprints...")
        
        # Build dictionaries: { agnostic_key : original_relative_path }
        ihm_dict = {get_agnostic_key(f.relative_to(ihm_dir)): f.relative_to(ihm_dir) for f in ihm_dir.rglob("*.wav")}
        sdm_dict = {get_agnostic_key(f.relative_to(sdm_dir)): f.relative_to(sdm_dir) for f in sdm_dir.rglob("*.wav")}

        # Create Sets of just the agnostic keys for fast comparison
        ihm_keys = set(ihm_dict.keys())
        sdm_keys = set(sdm_dict.keys())

        # Calculate intersections and outliers using the neutral keys
        valid_paired_keys = ihm_keys.intersection(sdm_keys)
        ihm_to_move_keys = ihm_keys - valid_paired_keys
        sdm_to_move_keys = sdm_keys - valid_paired_keys

        # Retrieve the original file paths for the files that actually need to move
        ihm_to_move = [ihm_dict[k] for k in ihm_to_move_keys]
        sdm_to_move = [sdm_dict[k] for k in sdm_to_move_keys]

        print(f"Target Paired Files (Intersection) : {len(valid_paired_keys)}")
        print(f"Orphaned IHM files to remove     : {len(ihm_to_move)}")
        print(f"Orphaned SDM files to remove     : {len(sdm_to_move)}\n")

        # Execute the moves
        moved_ihm = move_orphans(ihm_to_move, ihm_dir, extra_ihm, "IHM", split)
        moved_sdm = move_orphans(sdm_to_move, sdm_dir, extra_sdm, "SDM", split)

        if not DRY_RUN:
            print(f"Successfully moved {moved_ihm} unpaired IHM files to {extra_ihm.name}/")
            print(f"Successfully moved {moved_sdm} unpaired SDM files to {extra_sdm.name}/")
            
        total_perfect_pairs += len(valid_paired_keys)
        total_moved += (moved_ihm + moved_sdm)

    print("\n================ FINAL SUMMARY ================")
    if not DRY_RUN:
        print(f"SUCCESS: Your dataset now contains exactly {total_perfect_pairs} perfect parallel pairs.")
        print(f"Total orphaned files isolated: {total_moved}")
    else:
        print("[DRY RUN] No files were actually moved. Set DRY_RUN = False to apply changes.")

if __name__ == "__main__":
    enforce_parallel_data()
