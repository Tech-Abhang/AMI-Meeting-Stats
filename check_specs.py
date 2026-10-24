#!/usr/bin/env python3
"""
check_specs.py

Quick read of what compute_embeddings.py has to work with: GPU (if any),
CPU core count, RAM, and a rough disk-read-speed probe against the AMI
corpus path itself -- since that's a USB external drive, and a slow
disk can bottleneck this exactly as much as a slow CPU can.

Usage:
    python check_specs.py --audio-root "/media/iiitd/My Passport Sachin/PhoneBus/data/AMI_Meeting_Corpus/audio"
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", default=None,
                     help="If given, times reading a few real wav files from here.")
    args = ap.parse_args()

    print("=== Python / OS ===")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")

    print("\n=== CPU / RAM ===")
    try:
        import psutil
        print(f"  Logical CPUs: {psutil.cpu_count(logical=True)}  "
              f"(physical: {psutil.cpu_count(logical=False)})")
        mem = psutil.virtual_memory()
        print(f"  RAM: {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB available")
    except ImportError:
        print(f"  Logical CPUs (os.cpu_count): {os.cpu_count()}")
        print("  (install psutil for RAM info: pip install psutil --break-system-packages)")

    print("\n=== GPU / CUDA (via PyTorch) ===")
    try:
        import torch
        print(f"  torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_ok}")
        if cuda_ok:
            n = torch.cuda.device_count()
            print(f"  GPU count: {n}")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                print(f"    [{i}] {props.name} -- {props.total_memory / 1e9:.1f} GB VRAM, "
                      f"compute capability {props.major}.{props.minor}")
        else:
            print("  No usable CUDA GPU seen by PyTorch. If your machine does have an "
                  "NVIDIA GPU, this usually means PyTorch was installed as the CPU-only "
                  "build -- run `nvidia-smi` separately to check if the GPU and driver "
                  "are visible to the OS at all, which tells us if it's a PyTorch-install "
                  "problem or there's genuinely no usable GPU here.")
    except ImportError:
        print("  torch not importable.")

    print("\n=== `nvidia-smi` (OS-level GPU check, independent of PyTorch) ===")
    ret = os.system("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null")
    if ret != 0:
        print("  nvidia-smi not found or failed -- either no NVIDIA GPU, or no driver installed.")

    if args.audio_root:
        print(f"\n=== Disk read probe against {args.audio_root} ===")
        root = Path(args.audio_root)
        if not root.is_dir():
            print(f"  ERROR: not a directory: {root}")
        else:
            wavs = []
            for p in root.rglob("*.wav"):
                wavs.append(p)
                if len(wavs) >= 30:
                    break
            if not wavs:
                print("  No .wav files found under this path to time.")
            else:
                t0 = time.time()
                total_bytes = 0
                for p in wavs:
                    total_bytes += p.stat().st_size
                    with open(p, "rb") as f:
                        f.read()
                elapsed = time.time() - t0
                print(f"  Read {len(wavs)} files, {total_bytes / 1e6:.1f} MB, "
                      f"in {elapsed:.2f}s -> {len(wavs) / elapsed:.1f} files/s, "
                      f"{total_bytes / 1e6 / elapsed:.1f} MB/s")
                print("  (For reference: a fast NVMe SSD does hundreds of MB/s on this kind "
                      "of read; a USB external HDD often does tens of MB/s or less, "
                      "especially with many small files and random seeks.)")


if __name__ == "__main__":
    main()