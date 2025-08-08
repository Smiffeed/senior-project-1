#!/usr/bin/env python3
"""
🔬 Windowed Dataset Creator

This script generates a windowed dataset from a CSV file with precise start/end timestamps.
It slides a window over each audio file's timeline and assigns a label to each window
based on the amount of profanity it contains.

This is a crucial step for training and evaluating models for profanity localization,
as it allows for systematic experimentation with different window sizes.

Usage:
    python create_windowed_dataset.py \
        --input-csv csv/balanced_main.csv \
        --output-csv csv/balanced_main_windowed_0.3s.csv \
        --window-size 0.3 \
        --hop-size 0.15 \
        --profanity-threshold 0.3

Example with eval data:
    python create_windowed_dataset.py \
        --input-csv csv/eval.csv \
        --output-csv csv/eval_windowed_0.3s.csv \
        --window-size 0.3 \
        --hop-size 0.1
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import torchaudio
from tqdm import tqdm

def get_audio_duration(file_path: str) -> float:
    """Gets the duration of an audio file in seconds."""
    try:
        info = torchaudio.info(file_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"Warning: Could not read duration for {file_path}. Error: {e}")
        return 0.0

def create_windowed_dataset(
    input_csv: str,
    output_csv: str,
    window_size: float,
    hop_size: float,
    profanity_threshold: float,
    label_column: str = 'label',
    none_label: str = 'none'
):
    """
    Generates a windowed dataset from a source CSV with precise timestamps.

    Args:
        input_csv (str): Path to the input CSV file (e.g., 'eval.csv').
        output_csv (str): Path to save the new windowed CSV file.
        window_size (float): The size of each window in seconds.
        hop_size (float): The step size to slide the window in seconds.
        profanity_threshold (float): The minimum fraction of a window's duration that must
                                     contain profanity to be labeled as such.
        label_column (str): The name of the column containing the labels.
        none_label (str): The label for non-profane segments.
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Ensure file paths are correct for the current OS if needed
    # df['file_path'] = df['file_path'].str.replace('\\', '/', regex=False)

    output_records = []
    
    # Group by audio file to process each one individually
    grouped = df.groupby('file_path')
    
    print(f"Processing {len(grouped)} audio files...")
    for file_path, group in tqdm(grouped, desc="Files"):
        
        # Get the total duration of the audio file
        # We use the max end_time from the CSV as a proxy, or read the file directly
        # Reading the file is more accurate if the CSV doesn't cover the full file
        audio_duration = get_audio_duration(file_path)
        if audio_duration == 0.0:
            # Fallback to max time in CSV if reading fails
            audio_duration = group['end_time'].max()
            print(f"Using max end_time {audio_duration:.2f}s for {file_path}")

        if audio_duration == 0.0:
            print(f"Skipping {file_path} due to zero duration.")
            continue

        # Separate profane and non-profane segments for easy lookup
        profane_intervals = group[group[label_column] != none_label]
        
        # Iterate through the audio file with a sliding window
        start_time = 0.0
        while start_time + window_size <= audio_duration:
            end_time = start_time + window_size
            window_label = none_label
            
            # Check for overlap with any profane interval
            max_overlap = 0.0
            best_label = none_label

            for _, row in profane_intervals.iterrows():
                # Calculate the overlap between the window and the profane segment
                overlap_start = max(start_time, row['start_time'])
                overlap_end = min(end_time, row['end_time'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_label = row[label_column]

            # Decide the label for the window
            # If the profanity fills a certain portion of the window, label it
            if max_overlap / window_size >= profanity_threshold:
                window_label = best_label
            
            output_records.append({
                'file_path': file_path,
                'start_time': start_time,
                'end_time': end_time,
                'label': window_label
            })
            
            start_time += hop_size
            
    print(f"Generated {len(output_records)} windowed samples.")
    
    # Create the output DataFrame and save it
    output_df = pd.DataFrame(output_records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Successfully saved windowed dataset to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a windowed dataset for audio classification.")
    parser.add_argument(
        "--input-csv", 
        type=str, 
        required=True, 
        help="Path to the input CSV with precise start/end timestamps."
    )
    parser.add_argument(
        "--output-csv", 
        type=str, 
        required=True, 
        help="Path to save the generated windowed CSV file."
    )
    parser.add_argument(
        "--window-size", 
        type=float, 
        default=0.3, 
        help="Window size in seconds."
    )
    parser.add_argument(
        "--hop-size", 
        type=float, 
        default=0.1, 
        help="Hop (step) size in seconds."
    )
    parser.add_argument(
        "--profanity-threshold", 
        type=float, 
        default=0.3, 
        help="Minimum fraction of window duration to be considered profane (0.0 to 1.0)."
    )
    parser.add_argument(
        "--label-column", 
        type=str, 
        default="label", 
        help="Name of the label column in the input CSV."
    )
    parser.add_argument(
        "--none-label", 
        type=str, 
        default="none", 
        help="The specific label used for non-profane segments."
    )
    
    args = parser.parse_args()
    
    create_windowed_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        window_size=args.window_size,
        hop_size=args.hop_size,
        profanity_threshold=args.profanity_threshold,
        label_column=args.label_column,
        none_label=args.none_label,
    )
