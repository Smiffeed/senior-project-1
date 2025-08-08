#!/usr/bin/env python3
"""
📁 Dataset Folder Splitter

This script splits a source directory of audio files and their corresponding
label files into two separate subdirectories (e.g., 'train_val' and 'test')
based on a specified ratio.

It works by grouping files by their base name (stem) to ensure that an
audio file and its label file are always moved together.

The script moves the files, so the source directory will be empty after completion.

Usage:
    python scripts/split_dataset_folder.py ./path/to/source_data ./path/to/output_dir --ratio 0.8
"""
import argparse
from pathlib import Path
import random
import shutil
from tqdm import tqdm
from collections import defaultdict

def split_folder(source_dir: Path, dest_dir: Path, ratio: float):
    """
    Splits files from a source directory into 'train_val' and 'test' subdirectories
    in the destination directory, keeping file pairs (e.g., .wav and .txt) together
    and preserving the subdirectory structure for 'label' and 'single' folders.

    Args:
        source_dir (Path): The directory containing the source files (e.g., 'main').
        dest_dir (Path): The parent directory where 'train_val' and 'test' will be created.
        ratio (float): The proportion of data points to be moved to the 'train_val' folder.
    """
    # --- Setup Paths ---
    train_val_path = dest_dir / "train_val"
    test_path = dest_dir / "test"

    # Create destination directories, including subdirectories
    (train_val_path / "label").mkdir(parents=True, exist_ok=True)
    (train_val_path / "single").mkdir(parents=True, exist_ok=True)
    (test_path / "label").mkdir(parents=True, exist_ok=True)
    (test_path / "single").mkdir(parents=True, exist_ok=True)

    # --- Find and Group Files ---
    print(f"Scanning and grouping files in: {source_dir.resolve()}")
    file_groups = defaultdict(list)

    # 1. Process files with corresponding labels
    labels_dir = source_dir / "label"
    if labels_dir.is_dir():
        print(f"Processing labeled files from '{labels_dir.name}'...")
        for label_file in tqdm(list(labels_dir.glob("*.txt")), desc="Labeled"):
            stem = label_file.stem
            # Assume audio file is in the root of source_dir
            audio_file = source_dir / f"{stem}.wav"
            if audio_file.is_file():
                # The key is the stem, the value is a list of paths
                file_groups[stem].append(audio_file)
                file_groups[stem].append(label_file)
            else:
                print(f"Warning: Label '{label_file.name}' found without matching audio file in root. Skipping.")

    # 2. Process files in the 'single' folder
    single_dir = source_dir / "single"
    if single_dir.is_dir():
        print(f"Processing single-word files from '{single_dir.name}'...")
        for audio_file in tqdm(list(single_dir.glob("*.wav")), desc="Single"):
            # The key is the stem + a prefix to avoid name collisions
            stem = f"single_{audio_file.stem}"
            file_groups[stem].append(audio_file)

    if not file_groups:
        print("Error: No valid file groups found. Please check your directory structure.")
        print(f"Expected structure: audio files in '{source_dir.name}', labels in '{labels_dir.name}', single words in '{single_dir.name}'.")
        return

    # --- Get and Shuffle Stems (Data Points) ---
    unique_stems = list(file_groups.keys())
    print(f"\nFound {len(unique_stems)} unique data points. Shuffling...")
    random.shuffle(unique_stems)

    # --- Calculate Split Point ---
    split_index = int(len(unique_stems) * ratio)
    train_val_stems = unique_stems[:split_index]
    test_stems = unique_stems[split_index:]

    print(f"\nSplitting data points:")
    print(f"  - {len(train_val_stems)} for Training/Validation ({ratio:.0%})")
    print(f"  - {len(test_stems)} for Testing ({1-ratio:.0%})")

    # --- Move Files ---
    def move_files(stems: list, target_path: Path, desc: str):
        print(f"\nMoving files to {target_path}...")
        for stem in tqdm(stems, desc=desc):
            for file_path in file_groups[stem]:
                try:
                    # Determine the correct destination subfolder
                    if file_path.parent.name == 'label':
                        dest_file_path = target_path / 'label' / file_path.name
                    elif file_path.parent.name == 'single':
                        dest_file_path = target_path / 'single' / file_path.name
                    else: # Root audio file
                        dest_file_path = target_path / file_path.name
                    
                    # Ensure parent directory of destination exists
                    dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(dest_file_path))
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")

    move_files(train_val_stems, train_val_path, "Train/Val")
    move_files(test_stems, test_path, "Test")

    print("\n✅ Success! Dataset has been split.")
    print(f"Original source directory '{source_dir}' should now contain only empty subfolders or un-matched files.")

def main():
    parser = argparse.ArgumentParser(
        description="Split a folder of files (audio and labels) into training and testing sets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to the source directory containing all files."
    )
    parser.add_argument(
        "dest_dir",
        type=str,
        help="Path to the destination directory where 'train_val' and 'test' folders will be created."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="The ratio of data points to be included in the training/validation set (default: 0.8 for 80/20 split)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling to ensure reproducibility (default: 42)."
    )
    args = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(args.seed)

    source_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_path.resolve()}'")
        return

    split_folder(source_path, dest_path, args.ratio)

if __name__ == "__main__":
    main()
