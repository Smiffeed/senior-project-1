#!/usr/bin/env python3
"""
Comprehensive eval_percent batch processor
Processes all window and stride combinations using the clean v3 processor
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import argparse

def get_all_configurations(eval_percent_dir):
    """Get all window and stride combinations from eval_percent folder"""
    configurations = []
    eval_path = Path(eval_percent_dir)
    
    if not eval_path.exists():
        print(f"❌ Error: {eval_percent_dir} does not exist")
        return configurations
    
    # Get all window directories
    window_dirs = [d for d in eval_path.iterdir() if d.is_dir() and d.name.startswith('window_')]
    window_dirs.sort()
    
    for window_dir in window_dirs:
        window_name = window_dir.name
        
        # Get all CSV files in this window directory
        csv_files = [f for f in window_dir.iterdir() if f.suffix == '.csv' and f.name.startswith('stride_')]
        csv_files.sort()
        
        for csv_file in csv_files:
            stride_name = csv_file.stem  # e.g., "stride_10.0%"
            configurations.append({
                'window': window_name,
                'stride': stride_name,
                'csv_path': str(csv_file),
                'relative_path': f"{window_name}/{csv_file.name}"
            })
    
    return configurations

def estimate_dataset_size(csv_file):
    """Estimate dataset size by counting lines in CSV"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        return line_count
    except Exception as e:
        print(f"⚠️  Warning: Could not read {csv_file}: {e}")
        return 0

def process_single_configuration(config, model_path, threshold, output_dir, processor_script):
    """Process a single window/stride configuration"""
    
    print(f"\n{'='*80}")
    print(f"🔄 Processing Configuration:")
    print(f"   Window: {config['window']}")
    print(f"   Stride: {config['stride']}")
    print(f"   CSV: {config['relative_path']}")
    
    # Estimate dataset size
    dataset_size = estimate_dataset_size(config['csv_path'])
    print(f"   Dataset Size: {dataset_size:,} windows")
    
    # Estimate processing time (rough estimate based on previous observations)
    if dataset_size > 100000:
        estimated_time = "20-30 minutes"
        complexity = "🔴 LARGE"
    elif dataset_size > 50000:
        estimated_time = "10-20 minutes"
        complexity = "🟡 MEDIUM"
    else:
        estimated_time = "5-10 minutes"
        complexity = "🟢 SMALL"
    
    print(f"   Complexity: {complexity}")
    print(f"   Est. Time: {estimated_time}")
    print(f"{'='*80}")
    
    # Run the v3 processor
    python_exe = "C:/Users/muldi/Documents/Playground/University/senior-project-1/env/Scripts/python.exe"
    cmd = [
        python_exe,
        processor_script,
        "--csv_file", config['csv_path'],
        "--model_path", model_path,
        "--threshold", str(threshold),
        "--output_dir", output_dir
    ]
    
    start_time = time.time()
    
    try:
        print(f"🚀 Starting evaluation at {datetime.now().strftime('%H:%M:%S')}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed successfully in {duration/60:.1f} minutes")
            print(f"📄 Output: evaluation_results/eval_percent/{config['window']}/{config['stride']}/")
            return True, duration
        else:
            print(f"❌ Failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False, duration
            
    except KeyboardInterrupt:
        print(f"⏹️  Interrupted by user")
        return False, time.time() - start_time
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, time.time() - start_time

def main():
    parser = argparse.ArgumentParser(description="Comprehensive eval_percent batch processor")
    parser.add_argument("--eval_percent_dir", default="./csv/eval_percent", help="Path to eval_percent directory")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output_dir", default="./evaluation_results/eval_percent", help="Output directory")
    parser.add_argument("--processor_script", default="scripts/simple_eval_processor_v3.py", help="Processor script to use")
    parser.add_argument("--window", help="Process only specific window (e.g., 'window_0.3s')")
    parser.add_argument("--stride", help="Process only specific stride (e.g., 'stride_100.0%')")
    parser.add_argument("--start_from", help="Start from specific configuration (format: 'window_0.3s/stride_100.0%')")
    parser.add_argument("--max_configs", type=int, help="Maximum number of configurations to process")
    parser.add_argument("--skip_large", action="store_true", help="Skip large datasets (>100K windows)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without actually running")
    
    args = parser.parse_args()
    
    print(f"🎯 Comprehensive eval_percent batch processor")
    print(f"📁 Eval Percent Dir: {args.eval_percent_dir}")
    print(f"🤖 Model: {args.model_path}")
    print(f"🎯 Threshold: {args.threshold}")
    print(f"📂 Output: {args.output_dir}")
    print(f"🔧 Processor: {args.processor_script}")
    
    # Get all configurations
    print(f"\n🔍 Scanning for configurations...")
    all_configs = get_all_configurations(args.eval_percent_dir)
    
    if not all_configs:
        print(f"❌ No configurations found in {args.eval_percent_dir}")
        sys.exit(1)
    
    print(f"📊 Found {len(all_configs)} total configurations")
    
    # Filter configurations based on arguments
    filtered_configs = []
    start_processing = args.start_from is None
    
    for config in all_configs:
        # Check start_from condition
        if args.start_from and not start_processing:
            if f"{config['window']}/{config['stride']}" == args.start_from:
                start_processing = True
            else:
                continue
        
        # Check window filter
        if args.window and config['window'] != args.window:
            continue
            
        # Check stride filter
        if args.stride and config['stride'] != args.stride:
            continue
        
        # Check dataset size filter
        if args.skip_large:
            dataset_size = estimate_dataset_size(config['csv_path'])
            if dataset_size > 100000:
                print(f"⏭️  Skipping large dataset: {config['relative_path']} ({dataset_size:,} windows)")
                continue
        
        filtered_configs.append(config)
        
        # Check max_configs limit
        if args.max_configs and len(filtered_configs) >= args.max_configs:
            break
    
    if not filtered_configs:
        print(f"❌ No configurations match the specified filters")
        sys.exit(1)
    
    print(f"🎯 Will process {len(filtered_configs)} configurations")
    
    # Show processing plan
    print(f"\n📋 PROCESSING PLAN:")
    total_estimated_windows = 0
    for i, config in enumerate(filtered_configs, 1):
        dataset_size = estimate_dataset_size(config['csv_path'])
        total_estimated_windows += dataset_size
        
        if dataset_size > 100000:
            complexity = "🔴"
        elif dataset_size > 50000:
            complexity = "🟡"
        else:
            complexity = "🟢"
            
        print(f"   {i:2d}. {complexity} {config['relative_path']} ({dataset_size:,} windows)")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Configurations: {len(filtered_configs)}")
    print(f"   Total Windows: {total_estimated_windows:,}")
    print(f"   Estimated Total Time: {total_estimated_windows/1000:.0f}-{total_estimated_windows/500:.0f} minutes")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN - No actual processing performed")
        return
    
    # Confirm before starting
    response = input(f"\n❓ Continue with processing? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print(f"⏹️  Processing cancelled")
        return
    
    # Process configurations
    print(f"\n🚀 Starting batch processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = 0
    failed = 0
    total_time = 0
    
    for i, config in enumerate(filtered_configs, 1):
        print(f"\n📈 Progress: {i}/{len(filtered_configs)} ({i/len(filtered_configs)*100:.1f}%)")
        
        success, duration = process_single_configuration(
            config, 
            args.model_path, 
            args.threshold, 
            args.output_dir,
            args.processor_script
        )
        
        total_time += duration
        
        if success:
            successful += 1
        else:
            failed += 1
            
        print(f"📊 Session Stats: ✅ {successful} | ❌ {failed} | ⏱️  {total_time/60:.1f}min total")
        
        # Show remaining estimate
        remaining = len(filtered_configs) - i
        if remaining > 0 and successful > 0:
            avg_time_per_config = total_time / i
            estimated_remaining = (avg_time_per_config * remaining) / 60
            print(f"⏳ Estimated remaining: {estimated_remaining:.1f} minutes")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎯 BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed > 0:
        print(f"\n⚠️  Some configurations failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All configurations processed successfully!")

if __name__ == "__main__":
    main()
