#!/usr/bin/env python3
"""
Batch VAD Post-Processor for Comprehensive Evaluation Results
Processes all window evaluation results and applies VAD refinement
"""

import os
import sys
import pandas as pd
from pathlib import Path
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

def find_window_evaluation_directories(base_results_dir: str):
    """Find all window evaluation directories in comprehensive results"""
    window_eval_dirs = []
    base_path = Path(base_results_dir)
    
    # Look for window evaluation directories
    for eval_type_dir in base_path.iterdir():
        if eval_type_dir.is_dir() and eval_type_dir.name in ['eval_by_0.05', 'eval_percent']:
            # Look for nested eval_type directory (your structure has double nesting)
            nested_eval_dir = eval_type_dir / eval_type_dir.name
            if nested_eval_dir.exists():
                window_eval_base = nested_eval_dir / 'window_eval'
                if window_eval_base.exists():
                    # Find all window directories
                    for window_dir in window_eval_base.iterdir():
                        if window_dir.is_dir() and window_dir.name.startswith('window_'):
                            # Find all stride directories
                            for stride_dir in window_dir.iterdir():
                                if stride_dir.is_dir() and stride_dir.name.startswith('stride_'):
                                    # Check if raw_predictions.csv exists
                                    raw_predictions_file = stride_dir / 'raw_predictions.csv'
                                    if raw_predictions_file.exists():
                                        window_size = window_dir.name.replace('window_', '').replace('s', '')
                                        stride_info = stride_dir.name.replace('stride_', '')
                                        
                                        window_eval_dirs.append({
                                            'eval_type': eval_type_dir.name,
                                            'window_size': float(window_size),
                                            'stride_info': stride_info,
                                            'window_eval_dir': str(stride_dir),
                                            'window_name': window_dir.name,
                                            'stride_name': stride_dir.name
                                        })
    
    return sorted(window_eval_dirs, key=lambda x: (x['eval_type'], x['window_size'], x['stride_info']))

def process_single_configuration_vad(config_info, ground_truth_file, output_base_dir, worker_id):
    """Process a single configuration with VAD refinement"""
    
    eval_type = config_info['eval_type']
    window_size = config_info['window_size']
    stride_info = config_info['stride_info']
    window_eval_dir = config_info['window_eval_dir']
    window_name = config_info['window_name']
    stride_name = config_info['stride_name']
    
    worker_prefix = f"[W{worker_id}]"
    
    # Create output directory
    output_dir = Path(output_base_dir) / 'vad_refined_results' / eval_type / window_name / stride_name
    
    # Get Python executable
    python_executable = sys.executable
    if os.name == 'nt':  # Windows
        venv_python = Path("env/Scripts/python.exe")
        if venv_python.exists():
            python_executable = str(venv_python.absolute())
    
    cmd = [
        python_executable,
        "vad_post_processor.py",
        "--window_eval_dir", window_eval_dir,
        "--ground_truth", ground_truth_file,
        "--output_dir", str(output_dir),
        "--window_size", str(window_size),
        "--stride_info", stride_info,
        "--eval_type", eval_type
    ]
    
    try:
        start_time = time.time()
        
        print(f"{worker_prefix} Processing: {eval_type}/{window_name}/{stride_name}")
        
        # Set environment for proper encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env, encoding='utf-8', errors='replace')
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        # Initialize metrics
        metrics = {}
        
        if success:
            print(f"{worker_prefix} SUCCESS {eval_type}/{window_name}/{stride_name} ({duration:.1f}s)")
            
            # Extract metrics from output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Combined Mean IoU:' in line:
                        try:
                            metrics['combined_iou'] = float(line.split(':')[1].strip())
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
        else:
            error_msg = result.stderr[-150:] if result.stderr else "Unknown error"
            print(f"{worker_prefix} FAILED {eval_type}/{window_name}/{stride_name} - {error_msg[:100]}")
        
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'window_name': window_name,
            'stride_info': stride_info,
            'stride_name': stride_name,
            'success': success,
            'duration': duration,
            'worker_id': worker_id,
            'error': result.stderr[-200:] if result.stderr and not success else "",
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"{worker_prefix} ❌ EXCEPTION {eval_type}/{window_name}/{stride_name}: {str(e)[:100]}")
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'window_name': window_name,
            'stride_info': stride_info,
            'stride_name': stride_name,
            'success': False,
            'duration': 0,
            'worker_id': worker_id,
            'error': str(e),
            'metrics': {}
        }

def main():
    parser = argparse.ArgumentParser(description="Batch VAD post-processor for comprehensive evaluation results")
    parser.add_argument("--results_dir", default="./fixed_smart_parallel_results", 
                       help="Directory containing comprehensive evaluation results")
    parser.add_argument("--ground_truth", default="./csv/eval_5labels.csv", 
                       help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="./vad_refined_results", 
                       help="Output directory for VAD refined results")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--test_mode", action="store_true", 
                       help="Process only first 5 configurations for testing")
    parser.add_argument("--specific_eval_type", choices=['eval_by_0.05', 'eval_percent'], 
                       help="Process only specific evaluation type")
    parser.add_argument("--max_configs", type=int, 
                       help="Maximum number of configurations to process")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.max_configs = 5
        print("🧪 TEST MODE: Processing only 5 configurations")
    
    print("=== BATCH VAD POST-PROCESSOR ===")
    print(f"Results Directory: {args.results_dir}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print()
    
    # Find all window evaluation directories
    print("🔍 Scanning for window evaluation results...")
    window_eval_configs = find_window_evaluation_directories(args.results_dir)
    
    if args.specific_eval_type:
        window_eval_configs = [config for config in window_eval_configs 
                              if config['eval_type'] == args.specific_eval_type]
        print(f"🎯 Filtered to {args.specific_eval_type}: {len(window_eval_configs)} configurations")
    
    if args.max_configs:
        window_eval_configs = window_eval_configs[:args.max_configs]
        print(f"📊 Limited to {args.max_configs} configurations for processing")
    
    print(f"📁 Found {len(window_eval_configs)} window evaluation configurations:")
    
    # Group by eval_type for summary
    eval_type_counts = {}
    for config in window_eval_configs:
        eval_type = config['eval_type']
        eval_type_counts[eval_type] = eval_type_counts.get(eval_type, 0) + 1
    
    for eval_type, count in eval_type_counts.items():
        print(f"   {eval_type}: {count} configurations")
    
    if len(window_eval_configs) <= 10:
        print("\n📋 Configurations to process:")
        for config in window_eval_configs:
            print(f"   {config['eval_type']}: {config['window_name']} + {config['stride_name']}")
    
    if not window_eval_configs:
        print("❌ No window evaluation configurations found!")
        print("💡 Check that your results directory contains window evaluation results")
        print("💡 Expected structure: results_dir/eval_type/eval_type/window_eval/window_X/stride_Y/raw_predictions.csv")
        return 1
    
    print(f"\n🚀 Starting VAD post-processing with {args.workers} workers...")
    
    start_time = time.time()
    results = []
    completed = 0
    
    # Add worker assignments
    tasks_with_workers = []
    for i, config in enumerate(window_eval_configs):
        worker_id = (i % args.workers) + 1
        tasks_with_workers.append((config, args.ground_truth, args.output_dir, worker_id))
    
    # Process configurations
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(process_single_configuration_vad, task[0], task[1], task[2], task[3]): task 
            for task in tasks_with_workers
        }
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                success_count = len([r for r in results if r['success']])
                progress = completed / len(window_eval_configs) * 100
                print(f"📈 {completed}/{len(window_eval_configs)} ({progress:.1f}%) - ✅ {success_count}")
            except Exception as e:
                print(f"❌ Task failed with exception: {e}")
                completed += 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 VAD POST-PROCESSING COMPLETED!")
    print(f"⏱️  Time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    
    # Calculate performance
    successful = len([r for r in results if r['success']])
    sequential_time = sum(r['duration'] for r in results)
    speedup = sequential_time / total_duration if total_duration > 0 else 1
    efficiency = speedup / args.workers * 100 if args.workers > 0 else 0
    
    print(f"🚀 Speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
    print(f"✅ Success: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Save results summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    for result in results:
        row = {
            'eval_type': result['eval_type'],
            'window_size': result['window_size'],
            'window_name': result['window_name'], 
            'stride_info': result['stride_info'],
            'stride_name': result['stride_name'],
            'success': result['success'],
            'duration_seconds': result['duration'],
            'worker_id': result['worker_id']
        }
        row.update(result.get('metrics', {}))
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "vad_refinement_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"📊 Summary saved to: {summary_path}")
    
    # Show successful results stats
    if successful > 0:
        successful_results = [r for r in results if r['success'] and r.get('metrics')]
        if successful_results:
            print(f"\n📈 VAD REFINEMENT PERFORMANCE:")
            
            all_binary_f1 = [r['metrics'].get('binary_f1', 0) for r in successful_results if 'binary_f1' in r.get('metrics', {})]
            all_multiclass_f1 = [r['metrics'].get('multiclass_f1', 0) for r in successful_results if 'multiclass_f1' in r.get('metrics', {})]
            all_iou = [r['metrics'].get('combined_iou', 0) for r in successful_results if 'combined_iou' in r.get('metrics', {})]
            
            if all_binary_f1:
                print(f"   Binary F1 range: {min(all_binary_f1):.3f} - {max(all_binary_f1):.3f}")
            if all_multiclass_f1:
                print(f"   Multiclass F1 range: {min(all_multiclass_f1):.3f} - {max(all_multiclass_f1):.3f}")
            if all_iou:
                print(f"   Combined IoU range: {min(all_iou):.3f} - {max(all_iou):.3f}")
    
    if args.test_mode:
        print(f"\n✅ Test completed successfully!")
        print(f"💡 To run full processing: python {sys.argv[0]} --workers {args.workers}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())