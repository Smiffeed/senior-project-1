#!/usr/bin/env python3
"""
🎯 CREATE WINDOWED CSV DATASET
Create windowed CSV files from evaluation annotations with different overlap configurations.

This script:
- Reads original eval.csv annotations
- Creates 0.5-second windows with different step sizes (overlaps 0.3-0.7s)
- Labels windows based on profanity overlap rule (≥50% = profanity)
- Outputs new CSV files for each overlap configuration

Usage:
    python create_windowed_csv.py --input csv/eval.csv --output-dir windowed_csv
"""

import os
import pandas as pd
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse

class WindowedCSVCreator:
    """Create windowed CSV datasets from annotation files."""
    
    def __init__(self):
        # Thai profanity labels
        self.profanity_labels = {'เย็ด', 'กู', 'มึง', 'เหี้ย', 'ควย', 'สวะ', 'หี', 'แตด'}
    
    def label_window_with_overlap_rule(self, window_start: float, window_end: float, 
                                     annotations: pd.DataFrame, threshold: float = 0.5) -> str:
        """Label window based on profanity overlap rule (≥50% = profanity)."""
        
        window_duration = window_end - window_start
        profanity_duration = 0.0
        
        # Calculate total profanity duration in this window
        for _, annotation in annotations.iterrows():
            if annotation['label'] in self.profanity_labels:
                # Calculate overlap between window and annotation
                overlap_start = max(window_start, annotation['start_time'])
                overlap_end = min(window_end, annotation['end_time'])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    profanity_duration += overlap_duration
        
        # Apply ≥50% rule
        profanity_ratio = profanity_duration / window_duration
        
        if profanity_ratio >= threshold:
            return 'profanity'
        else:
            return 'none'
    
    def create_windowed_csv(self, eval_df: pd.DataFrame, window_size: float, 
                           overlap_seconds: float, output_file: str):
        """Create windowed CSV from evaluation annotations."""
        
        windowed_data = []
        
        # Group by audio file
        audio_groups = eval_df.groupby('file_path')
        
        print(f"🔧 Creating {window_size:.1f}s windowed dataset with {overlap_seconds:.1f}s overlap...")
        print(f"   Step size: {window_size - overlap_seconds:.1f}s")
        
        # Validate configuration
        if overlap_seconds >= window_size:
            print(f"❌ Invalid configuration: overlap ({overlap_seconds}s) >= window size ({window_size}s)")
            return
        
        step_size = window_size - overlap_seconds
        total_windows = 0
        profanity_windows = 0
        clean_windows = 0
        skipped_files = 0
        
        for audio_file, annotations in tqdm(audio_groups, desc="Processing files"):
            if not os.path.exists(audio_file):
                print(f"⚠️ File not found: {audio_file}")
                skipped_files += 1
                continue
            
            try:
                # Get audio duration
                metadata = torchaudio.info(audio_file)
                audio_duration = metadata.num_frames / metadata.sample_rate
                
                # Generate windows covering the entire audio file
                current_time = 0
                while current_time + window_size <= audio_duration:
                    window_start = current_time
                    window_end = current_time + window_size
                    
                    # Determine window label using overlap rule
                    window_label = self.label_window_with_overlap_rule(
                        window_start, window_end, annotations, threshold=0.5
                    )
                    
                    # Add to windowed data
                    windowed_data.append({
                        'file_path': audio_file,
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_duration': window_size,
                        'step_size': step_size,
                        'overlap_seconds': overlap_seconds,
                        'label': window_label
                    })
                    
                    total_windows += 1
                    if window_label == 'profanity':
                        profanity_windows += 1
                    else:
                        clean_windows += 1
                    
                    current_time += step_size
                
                # Handle short audio files (create single window with padding info)
                if audio_duration < window_size:
                    print(f"📝 Audio too short ({audio_duration:.2f}s): {os.path.basename(audio_file)}")
                    
                    # Determine label for this short audio
                    window_label = self.label_window_with_overlap_rule(
                        0, audio_duration, annotations, threshold=0.5
                    )
                    
                    windowed_data.append({
                        'file_path': audio_file,
                        'window_start': 0,
                        'window_end': window_size,  # Will need padding
                        'window_duration': window_size,
                        'step_size': step_size,
                        'overlap_seconds': overlap_seconds,
                        'label': window_label,
                        'needs_padding': True,
                        'original_duration': audio_duration
                    })
                    
                    total_windows += 1
                    if window_label == 'profanity':
                        profanity_windows += 1
                    else:
                        clean_windows += 1
                    
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
                skipped_files += 1
                continue
        
        # Create DataFrame and save to CSV
        windowed_df = pd.DataFrame(windowed_data)
        windowed_df.to_csv(output_file, index=False)
        
        # Print statistics
        print(f"✅ Created windowed CSV: {output_file}")
        print(f"   Total windows: {total_windows}")
        print(f"   Profanity windows: {profanity_windows} ({profanity_windows/total_windows*100:.2f}%)")
        print(f"   Clean windows: {clean_windows} ({clean_windows/total_windows*100:.2f}%)")
        print(f"   Skipped files: {skipped_files}")
        print(f"   Columns: {list(windowed_df.columns)}")
        
        return windowed_df
    
    def create_all_overlap_configurations(self, eval_df: pd.DataFrame, window_size: float,
                                        overlap_seconds_list: List[float], output_dir: str):
        """Create windowed CSV files for all overlap configurations."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 Creating windowed CSV files for {len(overlap_seconds_list)} overlap configurations...")
        print(f"📁 Output directory: {output_dir}")
        
        summary_data = []
        
        for overlap_seconds in overlap_seconds_list:
            # Skip invalid configurations
            if overlap_seconds >= window_size:
                print(f"⚠️ Skipping invalid overlap: {overlap_seconds}s >= {window_size}s")
                continue
            
            # Create output filename
            output_file = os.path.join(output_dir, f"windowed_0.5s_overlap_{overlap_seconds:.1f}s.csv")
            
            print(f"\n🎯 Creating: {os.path.basename(output_file)}")
            
            # Create windowed CSV
            windowed_df = self.create_windowed_csv(eval_df, window_size, overlap_seconds, output_file)
            
            if windowed_df is not None and len(windowed_df) > 0:
                # Add to summary
                profanity_count = sum(windowed_df['label'] == 'profanity')
                clean_count = sum(windowed_df['label'] == 'none')
                
                summary_data.append({
                    'File': os.path.basename(output_file),
                    'Window Size (s)': window_size,
                    'Overlap (s)': overlap_seconds,
                    'Step Size (s)': window_size - overlap_seconds,
                    'Total Windows': len(windowed_df),
                    'Profanity Windows': profanity_count,
                    'Clean Windows': clean_count,
                    'Profanity %': f"{profanity_count/len(windowed_df)*100:.2f}%"
                })
        
        # Create summary report
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(output_dir, "windowed_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            
            print(f"\n📋 Summary Report:")
            print("=" * 120)
            print(summary_df.to_string(index=False))
            print("=" * 120)
            print(f"💾 Summary saved to: {summary_file}")
        
        print(f"\n✅ All windowed CSV files created in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create Windowed CSV Dataset")
    parser.add_argument("--input", type=str, default="csv/eval.csv",
                       help="Input evaluation CSV file")
    parser.add_argument("--output-dir", type=str, default="windowed_csv",
                       help="Output directory for windowed CSV files")
    parser.add_argument("--window-size", type=float, default=0.5,
                       help="Window size in seconds")
    parser.add_argument("--overlap-seconds", nargs='+', type=float,
                       default=[0.1, 0.2, 0.3, 0.4],
                       help="Overlap values in seconds")
    
    args = parser.parse_args()
    
    print(f"🎯 Windowed CSV Creator")
    print(f"📊 Input file: {args.input}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🪟 Window size: {args.window_size}s")
    print(f"🔄 Overlap values: {args.overlap_seconds}")
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Load evaluation data
    try:
        eval_df = pd.read_csv(args.input)
        print(f"✅ Loaded {len(eval_df)} annotations from {len(eval_df['file_path'].unique())} unique files")
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Check required columns
    required_columns = ['file_path', 'start_time', 'end_time', 'label']
    missing_columns = [col for col in required_columns if col not in eval_df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"   Available columns: {list(eval_df.columns)}")
        return
    
    # Create windowed CSV creator
    creator = WindowedCSVCreator()
    
    # Create all windowed CSV files
    creator.create_all_overlap_configurations(
        eval_df, 
        args.window_size, 
        args.overlap_seconds, 
        args.output_dir
    )

if __name__ == "__main__":
    main()
