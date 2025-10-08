#!/usr/bin/env python3
"""
Lightweight Dataset Configuration Lister
Lists configurations without requiring heavy ML dependencies
"""

import os
import sys
from pathlib import Path

def find_dataset_configurations(datasets):
    """Find all CSV configurations in dataset directories"""
    configurations = []
    
    for dataset_dir in datasets:
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name
        
        if not dataset_path.exists():
            print(f"⚠️  Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all window directories
        window_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('window_')]
        
        for window_dir in window_dirs:
            window_size = window_dir.name  # e.g., "window_0.5s"
            
            # Find all stride CSV files
            csv_files = list(window_dir.glob("stride_*.csv"))
            
            for csv_file in csv_files:
                stride_name = csv_file.stem  # e.g., "stride_0.25s" or "stride_10.0%"
                
                try:
                    # Extract window size value
                    window_size_value = float(window_size.replace('window_', '').replace('s', ''))
                    
                    # Extract stride value based on format
                    if '%' in stride_name:
                        # Handle percentage format: "stride_10.0%" -> 10.0
                        stride_value = float(stride_name.replace('stride_', '').replace('%', ''))
                    else:
                        # Handle time format: "stride_0.25s" -> 0.25
                        stride_value = float(stride_name.replace('stride_', '').replace('s', ''))
                    
                    configurations.append({
                        'eval_type': eval_type,
                        'window': window_size,
                        'stride': stride_name,
                        'csv_path': str(csv_file),
                        'window_size_value': window_size_value,
                        'stride_value': stride_value
                    })
                except ValueError as e:
                    print(f"⚠️  Skipping invalid file: {csv_file} ({e})")
    
    return sorted(configurations, key=lambda x: (x['eval_type'], x['window_size_value'], x['stride_value']))

def main():
    datasets = ["csv/eval_by_0.05", "csv/eval_percent"]
    
    print("📋 DATASET CONFIGURATIONS")
    print("=" * 80)
    
    # Find all configurations
    all_configs = find_dataset_configurations(datasets)
    
    if not all_configs:
        print("❌ No configurations found!")
        return 1
    
    print(f"Found {len(all_configs)} configurations:")
    print()
    print(f"{'Eval Type':<15} {'Window':<12} {'Stride':<15} {'Path'}")
    print("-" * 80)
    
    for config in all_configs:
        print(f"{config['eval_type']:<15} {config['window']:<12} {config['stride']:<15} {os.path.basename(config['csv_path'])}")
    
    # Group summary
    by_eval_type = {}
    for config in all_configs:
        eval_type = config['eval_type']
        if eval_type not in by_eval_type:
            by_eval_type[eval_type] = []
        by_eval_type[eval_type].append(config)
    
    print(f"\n📊 SUMMARY:")
    for eval_type, configs in by_eval_type.items():
        print(f"   {eval_type}: {len(configs)} configurations")
    
    print(f"\n✅ All configurations parsed successfully!")
    print(f"🎯 The dataset_frame_level_evaluator.py will work with {len(all_configs)} configurations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())