#!/usr/bin/env python3
import os
import torchaudio
import tqdm

SAMPLE_RATE = 16_000

path = "/home/patrick/kaldi/egs/ami/s5/mdm_downloaded/{folder}/audio/{folder}.Array1-01.wav"
new_path = "/home/patrick/ami/audio/sdm"
for split in ["train", "dev", "eval"]:
    new_split_path = os.path.join(new_path, split)
    audio_chunks_path = os.path.join("/home/patrick/ami/annotations/", split, "segments")

    files = {}
    with open(audio_chunks_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            file_name, folder, start_time, end_time = line.strip().split()

            folder = folder.split("_")[1]
            os.system(f"mkdir -p {os.path.join(new_split_path, folder)}")

            if folder not in files:
                files[folder] = []

            files[folder].append((file_name, start_time, end_time))

        for folder, audios in tqdm.tqdm(files.items()):
            orig_file = path.format(folder=folder)
            try:
                waveform, sr = torchaudio.load(orig_file)
            except:
                print(f"File {orig_file} does not exist!")
                continue

#            for file_name, start_time, end_time in audios:
#                chunk = waveform[:, int(SAMPLE_RATE * float(start_time)): int(SAMPLE_RATE * float(end_time))]
#                out_path = f"{split}_{file_name.lower().replace('h00', 'sdm')}.wav"
#                out_path = os.path.join(new_split_path, folder, out_path)
#                torchaudio.save(out_path, chunk, sr)

            abs_folder = os.path.join(new_split_path, folder)
            os.system(f"cd {new_split_path} && tar -czf {folder}.tar.gz {folder}")
