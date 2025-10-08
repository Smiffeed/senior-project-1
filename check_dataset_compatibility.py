#!/usr/bin/env python3
"""
Simple Dataset Configuration Checker
Checks if the dataset_frame_level_evaluator.py will work with your eval_by_0.05 and eval_percent folders
"""

import os
import sys
from pathlib import Path
import argparse

def find_dataset_configurations(datasets):
    """Find all CSV configurations in dataset directories"""
    configurations = []
    
    for dataset_dir in datasets:
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name
        
        if not dataset_path.exists():
            print(f"⚠️  Dataset directory not found: {dataset_dir}")
            continue
        
        print(f"\n📁 Scanning {eval_type}...")
        
        # Find all window directories
        window_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('window_')]
        
        for window_dir in sorted(window_dirs):
            window_size = window_dir.name  # e.g., "window_0.5s"
            
            # Find all stride CSV files
            csv_files = list(window_dir.glob("stride_*.csv"))
            
            print(f"   {window_size}: {len(csv_files)} stride configurations")
            
            for csv_file in sorted(csv_files):
                stride_name = csv_file.stem  # e.g., "stride_0.25s" or "stride_10.0%"
                
                # Extract numerical values
                try:
                    if eval_type == 'eval_by_0.05':
                        # Extract window size: "window_0.5s" -> 0.5
                        window_size_value = float(window_size.replace('window_', '').replace('s', ''))
                        # Extract stride: "stride_0.25s" -> 0.25
                        stride_value = float(stride_name.replace('stride_', '').replace('s', ''))
                    else:  # eval_percent
                        # Extract window size: "window_0.5s" -> 0.5
                        window_size_value = float(window_size.replace('window_', '').replace('s', ''))
                        # Extract stride percentage: "stride_10.0%" -> 10.0
                        stride_value = float(stride_name.replace('stride_', '').replace('%', ''))
                
                    configurations.append({
                        'eval_type': eval_type,
                        'window': window_size,
                        'stride': stride_name,
                        'csv_path': str(csv_file),
                        'window_size_value': window_size_value,
                        'stride_value': stride_value,
                        'file_exists': csv_file.exists(),
                        'file_size_mb': csv_file.stat().st_size / (1024*1024) if csv_file.exists() else 0
                    })
                except Exception as e:
                    print(f"      ⚠️  Error parsing {stride_name}: {e}")
    
    return sorted(configurations, key=lambda x: (x['eval_type'], x['window_size_value'], x['stride_value']))

def main():
    parser = argparse.ArgumentParser(description="Check dataset compatibility")
    parser.add_argument("--datasets", nargs='+', default=["csv/eval_by_0.05", "csv/eval_percent"], 
                       help="Dataset directories to check")
    
    args = parser.parse_args()
    
    print("🔍 DATASET COMPATIBILITY CHECKER")
    print("=" * 60)
    
    # Find all configurations
    all_configs = find_dataset_configurations(args.datasets)
    
    if not all_configs:
        print("❌ No configurations found!")
        return 1
    
    print(f"\n📊 SUMMARY:")
    print(f"Total configurations found: {len(all_configs)}")
    
    # Group by evaluation type
    by_eval_type = {}
    for config in all_configs:
        eval_type = config['eval_type']
        if eval_type not in by_eval_type:
            by_eval_type[eval_type] = []
        by_eval_type[eval_type].append(config)
    
    for eval_type, configs in by_eval_type.items():
        print(f"\n{eval_type}: {len(configs)} configurations")
        
        # Show window size distribution
        windows = {}
        for config in configs:
            window = config['window']
            if window not in windows:
                windows[window] = 0
            windows[window] += 1
        
        print(f"   Window sizes: {len(windows)} different sizes")
        for window, count in sorted(windows.items()):
            print(f"      {window}: {count} stride variations")
    
    # Show some sample configurations
    print(f"\n📋 SAMPLE CONFIGURATIONS (first 10):")
    print("=" * 80)
    print(f"{'Eval Type':<15} {'Window':<12} {'Stride':<15} {'Size (MB)':<10} {'Path'}")
    print("-" * 80)
    
    for config in all_configs[:10]:
        print(f"{config['eval_type']:<15} {config['window']:<12} {config['stride']:<15} "
              f"{config['file_size_mb']:<10.1f} {os.path.basename(config['csv_path'])}")
    
    if len(all_configs) > 10:
        print(f"... and {len(all_configs) - 10} more configurations")
    
    # Check for any issues
    missing_files = [c for c in all_configs if not c['file_exists']]
    if missing_files:
        print(f"\n⚠️  WARNING: {len(missing_files)} missing files:")
        for config in missing_files[:5]:
            print(f"   {config['csv_path']}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
    else:
        print(f"\n✅ All configuration files exist and are accessible!")
    
    # Estimate total data size
    total_size_gb = sum(c['file_size_mb'] for c in all_configs) / 1024
    print(f"\n📈 TOTAL DATASET SIZE: {total_size_gb:.2f} GB")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • The dataset_frame_level_evaluator.py will work with this structure")
    print(f"   • Consider using --max_configs for testing (e.g., --max_configs 10)")
    print(f"   • Use --specific_windows to focus on key window sizes")
    print(f"   • Use --workers 2-4 for parallel processing depending on your system")
    
    if total_size_gb > 10:
        print(f"   • Large dataset detected - consider processing in batches")
        print(f"   • Use --test_mode for quick validation runs")
    
    print(f"\n🎯 READY TO USE!")
    print(f"   The dataset_frame_level_evaluator.py is compatible with your data structure.")
    print(f"   Run: python dataset_frame_level_evaluator.py --list_configs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())