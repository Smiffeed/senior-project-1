#!/usr/bin/env python3
"""
📜 Smart Dataset Splitter

This script processes a structured audio dataset, splits the audio files into
training (80%) and evaluation (20%) sets based on specific rules, and then
generates corresponding CSV files with detailed annotations.

Splitting Logic:
1.  It identifies all unique audio files in the source directory.
2.  It prioritizes including files from the 'internet' subdirectory in the 
    20% evaluation set.
3.  If the 'internet' files are more than 20% of the total, the excess is moved
    to the training set.
4.  If the 'internet' files are less than 20%, the script randomly samples from
    all other files to reach the 20% target for the evaluation set.
5.  The remaining 80% of files constitute the training set.

Annotation Logic:
- For multi-word audio, it reads timestamps from corresponding .txt files.
- For single-word audio (in 'single' subfolders), it uses the folder name
  as the label and the full audio duration.

Usage:
    python ./scripts/create_split_dataset.py ./dataset/mix_dataset ./csv
"""
import argparse
import csv
from pathlib import Path
import random
import librosa
from tqdm import tqdm
import collections

def get_all_audio_files(source_dir: Path) -> list:
    """Finds all unique .wav files, avoiding duplicates from label/single folders."""
    all_files = set()
    for p in source_dir.rglob('*.wav'):
        # Use the file stem to identify unique audio files, assuming .wav
        # This avoids adding the same file multiple times if it's in a label-specific folder
        all_files.add(p.stem)
    
    # Reconstruct full paths for unique stems
    unique_paths = []
    for stem in all_files:
        # Find the first occurrence of this wav file
        try:
            unique_paths.append(next(source_dir.rglob(f'{stem}.wav')))
        except StopIteration:
            print(f"Warning: Could not find a path for stem {stem}, skipping.")
            
    return unique_paths

def get_annotations_for_file(audio_path: Path) -> list:
    """
    For a given audio file, finds its annotations.
    Handles both labeled files and single-word files.
    """
    data_rows = []
    
    # --- Case 1: Labeled File (e.g., from 'effect' or 'internet' root) ---
    # Assumes label file is in a 'label' subdirectory of the audio file's parent
    label_path = audio_path.parent / "label" / f"{audio_path.stem}.txt"
    if label_path.exists():
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        start, end, text = parts
                        data_rows.append({
                            "file_path": audio_path.resolve(),
                            "start_time": float(start),
                            "end_time": float(end),
                            "label": text
                        })
            return data_rows
        except Exception as e:
            print(f"Error reading or parsing {label_path.name}: {e}")
            return []

    # --- Case 2: Single Word File (e.g., from 'single' or 'us' directories) ---
    # Assumes the parent directory name is the label
    parent_dir_name = audio_path.parent.name
    # A simple check to see if the parent is a label folder
    if audio_path.parent.parent.name in ['single', 'us']:
        try:
            duration = librosa.get_duration(path=str(audio_path))
            data_rows.append({
                "file_path": audio_path.resolve(),
                "start_time": 0.0,
                "end_time": duration,
                "label": parent_dir_name
            })
            return data_rows
        except Exception as e:
            print(f"Error processing single file {audio_path.name}: {e}")
            return []
            
    # If no specific logic matches, return empty
    return []


def write_csv(file_path: Path, data: list):
    """Writes the given data to a CSV file."""
    if not data:
        print(f"Warning: No data to write for {file_path.name}. Skipping.")
        return
        
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "start_time", "end_time", "label"])
            writer.writeheader()
            writer.writerows(data)
        print(f"✅ Successfully created '{file_path.resolve()}' with {len(data)} rows.")
    except Exception as e:
        print(f"\nError writing to CSV file {file_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/eval CSVs with a constrained split.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source_dir", type=str, help="Path to the root dataset directory (e.g., './dataset/mix_dataset').")
    parser.add_argument("output_dir", type=str, help="Directory to save the train.csv and eval.csv files.")
    parser.add_argument("--eval_size", type=float, default=0.2, help="Proportion of the dataset to use for evaluation (e.g., 0.2 for 20%).")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)
    random.seed(args.random_seed)

    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_path.resolve()}'")
        return

    # 1. Catalog all unique audio files
    all_audio_files = get_all_audio_files(source_path)
    print(f"Found {len(all_audio_files)} unique audio files.")

    # 2. Separate 'internet' files from 'other' files
    internet_files = [f for f in all_audio_files if 'internet' in f.parts]
    other_files = [f for f in all_audio_files if 'internet' not in f.parts]
    print(f"  - {len(internet_files)} files from 'internet' folder.")
    print(f"  - {len(other_files)} other files.")

    # 3. Determine split sizes
    total_files = len(all_audio_files)
    eval_count = int(total_files * args.eval_size)
    train_count = total_files - eval_count
    
    eval_files = []
    train_files = []

    # 4. Create evaluation set with priority for 'internet' files
    if len(internet_files) >= eval_count:
        # 'internet' folder is large enough for the entire eval set
        eval_files = random.sample(internet_files, eval_count)
        # The rest of the internet files go to training
        train_files.extend([f for f in internet_files if f not in eval_files])
        train_files.extend(other_files)
    else:
        # 'internet' folder is smaller than the required eval size
        eval_files.extend(internet_files) # Take all internet files
        needed = eval_count - len(internet_files)
        
        # Randomly sample from 'other' files to fill the rest
        random.shuffle(other_files)
        eval_files.extend(other_files[:needed])
        train_files.extend(other_files[needed:])

    print("\n--- Dataset Split ---")
    print(f"Target: {train_count} training files, {eval_count} evaluation files.")
    print(f"Actual: {len(train_files)} training files, {len(eval_files)} evaluation files.")
    print("-----------------------")

    # 5. Generate annotations and write CSVs
    train_data = []
    for f in tqdm(train_files, desc="Processing Train Set"):
        train_data.extend(get_annotations_for_file(f))
    write_csv(output_path / "train.csv", train_data)

    eval_data = []
    for f in tqdm(eval_files, desc="Processing Eval Set"):
        eval_data.extend(get_annotations_for_file(f))
    write_csv(output_path / "eval.csv", eval_data)


if __name__ == "__main__":
    main()
