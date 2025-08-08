#!/usr/bin/env python3
"""
Quick script to check which files in eval.csv actually exist and fix any path issues.
"""

import pandas as pd
import os
from pathlib import Path

def find_matching_files():
    """Find which files from eval.csv actually exist."""
    
    # Load eval.csv
    eval_df = pd.read_csv('csv/eval.csv')
    print(f"📊 Loaded {len(eval_df)} annotations from {len(eval_df['file_path'].unique())} unique files")
    
    # Check each unique file
    unique_files = eval_df['file_path'].unique()
    
    existing_files = []
    missing_files = []
    
    for file_path in unique_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ EXISTS: {os.path.basename(file_path)}")
        else:
            missing_files.append(file_path)
            print(f"❌ MISSING: {os.path.basename(file_path)}")
            
            # Try to find similar files in the expected directory
            expected_dir = os.path.dirname(file_path)
            if os.path.exists(expected_dir):
                filename = os.path.basename(file_path)
                # List files in directory to see what's there
                actual_files = os.listdir(expected_dir)
                print(f"   Directory contains: {len(actual_files)} files")
                
                # Look for similar names
                for actual_file in actual_files:
                    if actual_file.endswith('.wav') and len(actual_file) > 5:
                        # Check if filenames are similar (basic check)
                        if any(char in actual_file for char in filename[:10] if char.isalnum()):
                            print(f"   Possible match: {actual_file}")
    
    print(f"\n📋 Summary:")
    print(f"   ✅ Existing files: {len(existing_files)}")
    print(f"   ❌ Missing files: {len(missing_files)}")
    
    if existing_files:
        # Create a filtered dataset with only existing files
        filtered_df = eval_df[eval_df['file_path'].isin(existing_files)]
        filtered_df.to_csv('csv/eval_filtered.csv', index=False)
        print(f"💾 Created csv/eval_filtered.csv with {len(filtered_df)} annotations from {len(existing_files)} existing files")
        
        return existing_files, missing_files
    else:
        print("❌ No existing files found!")
        return [], missing_files

if __name__ == "__main__":
    find_matching_files()
