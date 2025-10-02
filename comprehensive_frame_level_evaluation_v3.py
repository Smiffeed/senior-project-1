#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE FRAME-LEVEL EVALUATION PROCESSOR v3
Following the successful pattern from fixed_smart_parallel_processor.py

This script discovers all CSV configurations and uses subprocess calls to run
frame_level_evaluation_single.py for each configuration, avoiding threading/model loading issues.
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def get_failed_configurations(summary_csv_path: str):
    """Get list of failed configurations from previous evaluation summary"""
    failed_configs = []
    
    if not os.path.exists(summary_csv_path):
        print(f"⚠️  Previous summary file not found: {summary_csv_path}")
        return failed_configs
    
    try:
        df = pd.read_csv(summary_csv_path)
        failed_df = df[df['success'] == False]
        
        print(f"📊 Found {len(failed_df)} failed configurations in previous run:")
        
        # Group failures by error type for summary
        error_counts = failed_df.groupby('error').size().sort_values(ascending=False)
        for error, count in error_counts.head(5).items():
            error_short = error[:50] + "..." if len(error) > 50 else error
            print(f"   • {error_short}: {count} configs")
        
        for _, row in failed_df.iterrows():
            failed_configs.append({
                'eval_type': row['eval_type'],
                'window_size': row['window_size'],
                'stride_value': row['stride_value'],
                'stride_type': row['stride_type'],
                'stride_label': row['stride_label']
            })
        
        print(f"🔄 Will retry {len(failed_configs)} failed configurations")
        return failed_configs
        
    except Exception as e:
        print(f"❌ Error reading previous summary: {e}")
        return failed_configs

def discover_csv_configurations(csv_base_dir: str = "./csv"):
    """Automatically discover all CSV configurations from eval_percent and eval_by_0.05 folders"""
    configurations = []
    csv_path = Path(csv_base_dir)
    
    # Process eval_percent (percentage-based strides)
    eval_percent_dir = csv_path / "eval_percent"
    if eval_percent_dir.exists():
        print(f"🔍 Discovering configurations in {eval_percent_dir}")
        for window_dir in eval_percent_dir.iterdir():
            if window_dir.is_dir() and window_dir.name.startswith('window_'):
                window_size = window_dir.name.replace('window_', '').replace('s', '')
                for csv_file in window_dir.glob('stride_*.csv'):
                    stride_name = csv_file.stem  # e.g., stride_10.0%
                    stride_value = stride_name.replace('stride_', '').replace('%', '')
                    
                    # Skip invalid files
                    try:
                        stride_float = float(stride_value)
                    except ValueError:
                        print(f"⚠️  Skipping invalid file: {csv_file} (stride_value: '{stride_value}')")
                        continue
                    
                    configurations.append({
                        'csv_path': str(csv_file),
                        'eval_type': 'eval_percent',
                        'window_size': float(window_size),
                        'stride_value': stride_float,
                        'stride_type': 'percentage',
                        'stride_label': f"{stride_value}%",
                        'window_name': window_dir.name,
                        'stride_name': stride_name
                    })
    
    # Process eval_by_0.05 (absolute time-based strides)
    eval_by_005_dir = csv_path / "eval_by_0.05"
    if eval_by_005_dir.exists():
        print(f"🔍 Discovering configurations in {eval_by_005_dir}")
        for window_dir in eval_by_005_dir.iterdir():
            if window_dir.is_dir() and window_dir.name.startswith('window_'):
                window_size = window_dir.name.replace('window_', '').replace('s', '')
                for csv_file in window_dir.glob('stride_*.csv'):
                    stride_name = csv_file.stem  # e.g., stride_0.15s
                    
                    # Skip sample files or other non-standard names
                    if 'sample' in stride_name.lower() or not stride_name.startswith('stride_'):
                        print(f"⚠️  Skipping non-standard file: {csv_file}")
                        continue
                    
                    # Extract stride value more carefully
                    stride_value = stride_name.replace('stride_', '')
                    if stride_value.endswith('s'):
                        stride_value = stride_value[:-1]  # Remove only the trailing 's'
                    
                    # Skip invalid files
                    try:
                        stride_float = float(stride_value)
                    except ValueError:
                        print(f"⚠️  Skipping invalid file: {csv_file} (stride_value: '{stride_value}')")
                        continue
                    
                    configurations.append({
                        'csv_path': str(csv_file),
                        'eval_type': 'eval_by_0.05',
                        'window_size': float(window_size),
                        'stride_value': stride_float,
                        'stride_type': 'absolute',
                        'stride_label': f"{stride_value}s",
                        'window_name': window_dir.name,
                        'stride_name': stride_name
                    })
    
    print(f"📊 Discovered {len(configurations)} total configurations")
    
    # Group by evaluation type for summary
    eval_percent_count = len([c for c in configurations if c['eval_type'] == 'eval_percent'])
    eval_by_005_count = len([c for c in configurations if c['eval_type'] == 'eval_by_0.05'])
    
    print(f"   • eval_percent: {eval_percent_count} configurations")
    print(f"   • eval_by_0.05: {eval_by_005_count} configurations")
    
    # Sort configurations by window size (largest first) for better testing
    configurations.sort(key=lambda x: x['window_size'], reverse=True)
    
    return configurations

def run_single_evaluation_subprocess(task_info):
    """Run evaluation using subprocess like fixed_smart_parallel_processor.py"""
    config, model_path, ground_truth, output_base_dir, worker_id = task_info
    
    csv_path = config['csv_path']
    eval_type = config['eval_type']
    window_size = config['window_size']
    stride_value = config['stride_value']
    stride_type = config['stride_type']
    stride_label = config['stride_label']
    
    worker_prefix = f"[W{worker_id}]"
    
    # Get Python executable (prefer venv if available)
    python_executable = sys.executable
    if os.name == 'nt':  # Windows
        venv_python = Path("env/Scripts/python.exe")
        if venv_python.exists():
            python_executable = str(venv_python.absolute())
    
    # Create output directory
    output_dir = Path(output_base_dir) / eval_type / f"window_{window_size}s" / f"stride_{stride_label}"
    
    # Prepare command like fixed_smart_parallel_processor.py
    cmd = [
        python_executable, 
        "frame_level_evaluation_single.py",  # We'll create this single-config processor
        "--csv_file", csv_path,
        "--model_path", model_path,
        "--ground_truth", ground_truth,
        "--output_dir", str(output_dir),
        "--window_size", str(window_size),
        "--stride_value", str(stride_value),
        "--stride_type", stride_type,
        "--eval_type", eval_type
    ]
    
    try:
        start_time = time.time()
        
        print(f"{worker_prefix} 🔍 Processing: window_{window_size}s, stride_{stride_label}")
        
        # Set up environment for proper UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run subprocess without timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd(),
            env=env,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"{worker_prefix} ✅ Completed: window_{window_size}s, stride_{stride_label} ({duration:.1f}s)")
        else:
            error_msg = result.stderr[-200:] if result.stderr else "Unknown error"
            print(f"{worker_prefix} ❌ Failed: window_{window_size}s, stride_{stride_label}")
            print(f"{worker_prefix}    Error: {error_msg}")
        
        # Extract metrics from stdout if available
        metrics = {}
        if success and result.stdout:
            for line in result.stdout.split('\n'):
                if 'Combined Mean IoU:' in line:
                    try:
                        metrics['combined_mean_iou'] = float(line.split(':')[1].strip().replace('%', ''))
                    except:
                        pass
                elif 'Binary F1:' in line:
                    try:
                        metrics['binary_f1'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Multiclass F1:' in line:
                    try:
                        metrics['multiclass_f1'] = float(line.split(':')[1].strip())
                    except:
                        pass
        
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'stride_value': stride_value,
            'stride_type': stride_type,
            'stride_label': stride_label,
            'csv_path': csv_path,
            'success': success,
            'duration': duration,
            'error': result.stderr[-200:] if result.stderr and not success else "",
            'worker_id': worker_id,
            'metrics': metrics,
            'stdout': result.stdout if success else "",
            'stderr': result.stderr if not success else ""
        }
        
    except Exception as e:
        print(f"{worker_prefix} ❌ Exception: window_{window_size}s, stride_{stride_label}: {str(e)}")
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'stride_value': stride_value,
            'stride_type': stride_type,
            'stride_label': stride_label,
            'csv_path': csv_path,
            'success': False,
            'duration': 0,
            'error': str(e),
            'worker_id': worker_id,
            'metrics': {}
        }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Frame-Level Evaluation Processor v3")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--csv_base_dir", default="./csv", help="Base directory containing eval_percent and eval_by_0.05 folders")
    parser.add_argument("--output_dir", default="./frame_level_evaluation_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit_configs", type=int, default=None, help="Limit number of configurations for testing")
    parser.add_argument("--test_mode", action="store_true", help="Run with only 5 configs for testing")
    parser.add_argument("--retry_failed", action="store_true", help="Only process configurations that failed in previous run")
    parser.add_argument("--previous_summary", default="./frame_level_evaluation_results/evaluation_summary.csv", 
                       help="Path to previous evaluation summary CSV to identify failed configurations")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.limit_configs = 5
        print("🧪 TEST MODE: Processing only 5 configurations")
    
    print("🎯 COMPREHENSIVE FRAME-LEVEL EVALUATION PROCESSOR v3")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"CSV Base Dir: {args.csv_base_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    if args.retry_failed:
        print(f"Mode: RETRY FAILED ONLY")
        print(f"Previous Summary: {args.previous_summary}")
    print()
    
    # Check if single processor exists
    single_processor_path = Path("frame_level_evaluation_single.py")
    if not single_processor_path.exists():
        print("❌ Missing frame_level_evaluation_single.py processor!")
        print("   This script should handle single configuration evaluation.")
        return 1
    
    # Discover all CSV configurations
    configurations = discover_csv_configurations(args.csv_base_dir)
    
    if not configurations:
        print("❌ No configurations found!")
        return 1
    
    # Filter to only failed configurations if requested
    if args.retry_failed:
        print("\n🔄 RETRY FAILED MODE: Only processing previously failed configurations")
        failed_configs = get_failed_configurations(args.previous_summary)
        
        if not failed_configs:
            print("✅ No failed configurations found to retry!")
            return 0
        
        # Filter configurations to only include failed ones
        failed_keys = set()
        for fc in failed_configs:
            key = (fc['eval_type'], fc['window_size'], fc['stride_value'], fc['stride_type'])
            failed_keys.add(key)
        
        original_count = len(configurations)
        configurations = [
            config for config in configurations 
            if (config['eval_type'], config['window_size'], config['stride_value'], config['stride_type']) in failed_keys
        ]
        
        print(f"🎯 Filtered {original_count} → {len(configurations)} configurations (failed only)")
        
        if not configurations:
            print("❌ No matching failed configurations found in CSV directories!")
            return 1
    
    # Limit configurations if specified (for testing)
    if args.limit_configs:
        configurations = configurations[:args.limit_configs]
        print(f"🔍 Limited to {len(configurations)} configurations for testing")
    
    # Prepare tasks with worker IDs
    tasks = []
    for i, config in enumerate(configurations):
        worker_id = (i % args.workers) + 1
        tasks.append((config, args.model_path, args.ground_truth, args.output_dir, worker_id))
    
    print(f"🚀 Processing {len(tasks)} configurations...")
    eval_percent_count = len([c for c in configurations if c['eval_type'] == 'eval_percent'])
    eval_by_005_count = len([c for c in configurations if c['eval_type'] == 'eval_by_0.05'])
    print(f"   • eval_percent: {eval_percent_count} configurations")
    print(f"   • eval_by_0.05: {eval_by_005_count} configurations")
    print()
    
    # Run parallel evaluation using ThreadPoolExecutor like fixed_smart_parallel_processor.py
    start_time = time.time()
    results = []
    completed = 0
    
    if args.workers == 1:
        # Sequential processing
        for task in tasks:
            result = run_single_evaluation_subprocess(task)
            results.append(result)
            completed += 1
            success_count = len([r for r in results if r['success']])
            print(f"📈 {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) - ✅ {success_count}")
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(run_single_evaluation_subprocess, task): task 
                            for task in tasks}
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    success_count = len([r for r in results if r['success']])
                    progress = completed / len(tasks) * 100
                    print(f"📈 {completed}/{len(tasks)} ({progress:.1f}%) - ✅ {success_count}")
                except Exception as e:
                    print(f"❌ Task failed with exception: {e}")
                    completed += 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Summary
    success_results = [r for r in results if r['success']]
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"⏱️  Total Time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    
    # Calculate performance metrics
    if results:
        sequential_time = sum(r['duration'] for r in results)
        speedup = sequential_time / total_duration if total_duration > 0 else 1
        efficiency = speedup / args.workers * 100 if args.workers > 0 else 0
        print(f"🚀 Speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
    
    print(f"✅ Success: {len(success_results)}/{len(results)} ({len(success_results)/len(results)*100:.1f}%)")
    
    if success_results:
        # Show metrics summary
        iou_values = [r['metrics'].get('combined_mean_iou') for r in success_results if 'combined_mean_iou' in r['metrics']]
        binary_f1_values = [r['metrics'].get('binary_f1') for r in success_results if 'binary_f1' in r['metrics']]
        
        if iou_values:
            print(f"📊 Mean IoU Range: {min(iou_values):.2f}% - {max(iou_values):.2f}%")
        if binary_f1_values:
            print(f"📊 Binary F1 Range: {min(binary_f1_values):.3f} - {max(binary_f1_values):.3f}")
    
    # Save detailed summary
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare summary data
    summary_data = []
    for result in results:
        row = {
            'eval_type': result['eval_type'],
            'window_size': result['window_size'],
            'stride_value': result['stride_value'],
            'stride_type': result['stride_type'],
            'stride_label': result['stride_label'],
            'success': result['success'],
            'error': result['error'],
            'duration': result['duration'],
            'worker_id': result['worker_id']
        }
        # Add metrics
        row.update(result['metrics'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"📊 Summary saved to: {summary_path}")
    
    # Show failed configurations if any
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n❌ Failed Configurations ({len(failed_results)}):")
        for result in failed_results[:10]:  # Show first 10 failures
            print(f"   {result['eval_type']}: window_{result['window_size']}s, stride_{result['stride_label']} - {result['error'][:100]}")
        if len(failed_results) > 10:
            print(f"   ... and {len(failed_results) - 10} more failures")
    
    return 0 if len(success_results) > 0 else 1

if __name__ == "__main__":
    sys.exit(main())