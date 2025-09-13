#!/usr/bin/env python3
"""
Preview script to show what files will be processed in batch evaluation
"""

import os
from pathlib import Path

def preview_dataset(dataset_dir):
    """Preview CSV files in a dataset directory"""
    print(f"\n=== DATASET: {dataset_dir} ===")
    
    base_path = Path(dataset_dir)
    
    if not base_path.exists():
        print(f"❌ Directory {dataset_dir} does not exist")
        return 0
    
    csv_files = []
    for csv_file in base_path.rglob("*.csv"):
        parts = csv_file.parts
        if len(parts) >= 2:
            window_dir = parts[-2]  # e.g., "window_0.3s"
            stride_file = parts[-1]  # e.g., "stride_0.125s.csv"
            
            if window_dir.startswith('window_') and stride_file.startswith('stride_'):
                csv_files.append({
                    'window': window_dir,
                    'stride': stride_file.replace('.csv', ''),
                    'path': str(csv_file)
                })
    
    csv_files = sorted(csv_files, key=lambda x: (x['window'], x['stride']))
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Group by window size
    windows = {}
    for csv_file in csv_files:
        window = csv_file['window']
        if window not in windows:
            windows[window] = []
        windows[window].append(csv_file['stride'])
    
    print(f"🪟 Window sizes: {len(windows)} different sizes")
    
    for window, strides in sorted(windows.items()):
        print(f"   {window}: {len(strides)} stride configurations")
        # Show first few strides as examples
        stride_examples = strides[:5]
        if len(strides) > 5:
            stride_examples.append(f"... and {len(strides)-5} more")
        print(f"      {', '.join(stride_examples)}")
    
    return len(csv_files)

def main():
    print("=== BATCH EVALUATION PREVIEW ===")
    print("This shows what files will be processed by the batch evaluator")
    
    datasets = ["csv/eval_by_0.05", "csv/eval_percent"]
    total_files = 0
    
    for dataset in datasets:
        count = preview_dataset(dataset)
        total_files += count
    
    print(f"\n📊 SUMMARY")
    print(f"Total CSV files to process: {total_files}")
    print(f"Evaluation methods per file: 4 (window_eval, word_eval, word_iou_eval, iou_eval)")
    print(f"Total evaluations: {total_files * 4}")
    
    print(f"\n🕒 ESTIMATED TIME")
    # Rough estimation based on our previous runs
    avg_time_per_file = 120  # seconds, based on the 2.0s/2.0s example
    total_time_seconds = total_files * avg_time_per_file
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60
    
    print(f"Estimated time per CSV file: ~{avg_time_per_file} seconds")
    print(f"Total estimated time: {total_time_seconds} seconds ({total_time_minutes:.1f} minutes / {total_time_hours:.1f} hours)")
    
    print(f"\n🚀 TO RUN BATCH EVALUATION:")
    print("# Evaluate both datasets (full evaluation):")
    print("python scripts/batch_evaluation_processor.py")
    print()
    print("# Evaluate specific dataset:")
    print("python scripts/batch_evaluation_processor.py --datasets csv/eval_by_0.05")
    print("python scripts/batch_evaluation_processor.py --datasets csv/eval_percent")
    print()
    print("# Test with limited files (for testing):")
    print("python scripts/batch_evaluation_processor.py --max_configs 5")
    print()
    print("# Skip existing results (resume interrupted evaluation):")
    print("python scripts/batch_evaluation_processor.py --skip_existing")

if __name__ == "__main__":
    main()
