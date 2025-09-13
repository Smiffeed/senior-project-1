#!/usr/bin/env python3
"""
Parallel comprehensive evaluation processor - runs comprehensive evaluations on multiple configurations
Based on parallel_eval_processor_word_by_0.05.py but using the new comprehensive evaluation system
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
import psutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import subprocess

def get_all_configurations(eval_dir, eval_type="eval_by_0.05"):
    """Get all window and stride combinations from evaluation directory"""
    configurations = []
    eval_path = Path(eval_dir)
    
    if not eval_path.exists():
        print(f"Error: Evaluation directory {eval_path} does not exist!")
        return configurations
    
    # Get all window directories
    window_dirs = [d for d in eval_path.iterdir() if d.is_dir() and d.name.startswith('window_')]
    window_dirs.sort()
    
    for window_dir in window_dirs:
        # Get all stride CSV files
        stride_files = list(window_dir.glob('stride_*.csv'))
        stride_files.sort()
        
        for stride_file in stride_files:
            configurations.append({
                'window': window_dir.name,
                'stride': stride_file.stem,
                'csv_path': str(stride_file),
                'relative_path': f"{window_dir.name}/{stride_file.name}",
                'eval_type': eval_type
            })
    
    return configurations

def estimate_dataset_size(csv_file):
    """Estimate dataset size by counting lines in CSV"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f) - 1  # Subtract header
        return max(0, line_count)
    except Exception as e:
        print(f"Error estimating dataset size for {csv_file}: {e}")
        return 0

def check_if_processed(output_dir, config):
    """Check if configuration has already been processed"""
    methods = ['window_eval', 'word_eval', 'iou_eval', 'iou_only_eval']
    
    for method in methods:
        note_file = Path(output_dir) / config['eval_type'] / method / config['window'] / config['stride'] / "note.txt"
        if not note_file.exists():
            return False
    
    return True

def process_single_configuration_worker(config, model_path, ground_truth_csv, output_dir, processor_script, worker_id):
    """Worker function to process a single configuration"""
    
    # Check if already processed
    if check_if_processed(output_dir, config):
        print(f"Worker {worker_id}: Skipping {config['window']}/{config['stride']} (already processed)")
        return {
            'status': 'skipped',
            'config': config,
            'worker_id': worker_id,
            'reason': 'already_processed'
        }
    
    # Display detailed configuration info
    dataset_size = estimate_dataset_size(config['csv_path'])
    
    print(f"\n{'='*80}")
    print(f"Worker {worker_id}: Processing Configuration:")
    print(f"   Window: {config['window']}")
    print(f"   Stride: {config['stride']}")
    print(f"   CSV: {config['relative_path']}")
    print(f"   Dataset Size: {dataset_size:,} windows")
    print(f"   Eval Type: {config['eval_type']}")
    
    # Estimate processing time
    if dataset_size > 100000:
        complexity = "VERY HIGH"
        estimated_time = "60+ minutes"
    elif dataset_size > 50000:
        complexity = "HIGH"
        estimated_time = "30-60 minutes"
    elif dataset_size > 10000:
        complexity = "MEDIUM"
        estimated_time = "10-30 minutes"
    else:
        complexity = "LOW"
        estimated_time = "5-10 minutes"
    
    print(f"   Complexity: {complexity}")
    print(f"   Est. Time: {estimated_time}")
    print(f"{'='*80}")
    
    # Run the comprehensive processor script
    python_exe = "C:/Users/muldi/Documents/Playground/University/senior-project-1/env/Scripts/python.exe"
    
    cmd = [
        python_exe,
        processor_script,
        "--csv_file", config['csv_path'],
        "--model_path", model_path,
        "--ground_truth", ground_truth_csv,
        "--output_dir", output_dir,
        "--eval_type", config['eval_type']
    ]
    
    start_time = time.time()
    
    try:
        print(f"Worker {worker_id}: Starting evaluation...")
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"Worker {worker_id}: ✅ Completed {config['window']}/{config['stride']} in {duration:.1f}s")
            return {
                'status': 'completed',
                'config': config,
                'worker_id': worker_id,
                'duration': duration,
                'dataset_size': dataset_size
            }
        else:
            print(f"Worker {worker_id}: ❌ Failed {config['window']}/{config['stride']}")
            print(f"Error: {result.stderr}")
            return {
                'status': 'failed',
                'config': config,
                'worker_id': worker_id,
                'error': result.stderr,
                'duration': duration
            }
            
    except subprocess.TimeoutExpired:
        print(f"Worker {worker_id}: ⏰ Timeout {config['window']}/{config['stride']}")
        return {
            'status': 'timeout',
            'config': config,
            'worker_id': worker_id,
            'duration': time.time() - start_time
        }
    except KeyboardInterrupt:
        print(f"Worker {worker_id}: 🛑 Interrupted {config['window']}/{config['stride']}")
        return {
            'status': 'interrupted',
            'config': config,
            'worker_id': worker_id
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"Worker {worker_id}: 💥 Exception {config['window']}/{config['stride']}: {e}")
        return {
            'status': 'exception',
            'config': config,
            'worker_id': worker_id,
            'error': str(e),
            'duration': duration
        }

def get_optimal_worker_count():
    """Auto-detect optimal worker count based on system resources"""
    try:
        import psutil
    except ImportError:
        print("psutil not available, using basic CPU count detection")
        return max(1, mp.cpu_count() - 1)
    
    # Get system specs
    cpu_count = mp.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"\n🖥️  SYSTEM ANALYSIS:")
    print(f"   Physical CPU Cores: {physical_cores}")
    print(f"   Logical CPU Cores: {cpu_count}")
    print(f"   Total RAM: {memory_gb:.1f} GB")
    print(f"   Available RAM: {available_memory_gb:.1f} GB")
    
    # Conservative estimates for comprehensive evaluation workload
    memory_per_worker = 4.0  # GB per worker (increased for comprehensive evaluation)
    
    # Calculate optimal based on different constraints
    memory_limited = int(available_memory_gb / memory_per_worker)
    cpu_limited = physical_cores - 1  # Leave 1 core for OS
    
    # Choose the most limiting factor
    optimal = min(memory_limited, cpu_limited, 6)  # Cap at 6 workers max for comprehensive evaluation
    optimal = max(optimal, 1)  # At least 1 worker
    
    print(f"\n🧮 WORKER CALCULATIONS:")
    print(f"   Memory-limited workers: {memory_limited}")
    print(f"   CPU-limited workers: {cpu_limited}")
    print(f"   Recommended workers: {optimal}")
    
    return optimal

def worker_process(task_queue, result_queue, model_path, ground_truth_csv, output_dir, processor_script, worker_id):
    """Worker process function"""
    while True:
        try:
            config = task_queue.get(timeout=5)
            if config is None:  # Poison pill
                break
            
            result = process_single_configuration_worker(
                config, model_path, ground_truth_csv, output_dir, processor_script, worker_id
            )
            result_queue.put(result)
            
        except Exception as e:
            print(f"Worker {worker_id} exception: {e}")
            result_queue.put({
                'status': 'worker_exception',
                'worker_id': worker_id,
                'error': str(e)
            })
        finally:
            task_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Parallel comprehensive evaluation processor")
    parser.add_argument("--eval_dir", default="./csv/eval_by_0.05", help="Path to evaluation directory")
    parser.add_argument("--eval_type", default="eval_by_0.05", help="Evaluation type name")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--ground_truth", default="./csv/eval.csv", help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="./new_evaluation_results", help="Output directory")
    parser.add_argument("--processor_script", default="scripts/comprehensive_evaluation_processor.py", help="Processor script to use")
    parser.add_argument("--window", help="Process only specific window (e.g., 'window_0.3s')")
    parser.add_argument("--stride", help="Process only specific stride (e.g., 'stride_0.125s')")
    parser.add_argument("--max_configs", type=int, help="Maximum number of configurations to process")
    parser.add_argument("--skip_large", action="store_true", help="Skip large datasets (>100K windows)")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip already processed configurations")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without actually running")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (auto-detect if not specified)")
    parser.add_argument("--auto_workers", action="store_true", default=True, help="Auto-detect optimal worker count")
    parser.add_argument("--sort_by_size", action="store_true", default=True, help="Sort by dataset size (smallest first)")
    
    args = parser.parse_args()
    
    # Auto-detect optimal worker count if not specified
    if args.workers is None:
        args.workers = get_optimal_worker_count()
    elif args.auto_workers:
        suggested_workers = get_optimal_worker_count()
        print(f"Suggested workers: {suggested_workers}, using specified: {args.workers}")
    
    # Determine optimal number of workers
    cpu_count = mp.cpu_count()
    if args.workers > cpu_count:
        print(f"Warning: Requested {args.workers} workers but only {cpu_count} CPUs available")
        args.workers = cpu_count
    
    print(f"PARALLEL COMPREHENSIVE EVALUATION PROCESSOR")
    print(f"Workers: {args.workers} parallel processes")
    print(f"CPU Count: {cpu_count}")
    print(f"Eval Dir: {args.eval_dir}")
    print(f"Eval Type: {args.eval_type}")
    print(f"Model: {args.model_path}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output: {args.output_dir}")
    print(f"Processor: {args.processor_script}")
    print(f"Skip Existing: {args.skip_existing}")
    
    # Get all configurations
    print(f"\nScanning for configurations...")
    all_configs = get_all_configurations(args.eval_dir, args.eval_type)
    
    if not all_configs:
        print("No configurations found!")
        return
    
    print(f"Found {len(all_configs)} total configurations")
    
    # Add dataset sizes to configurations
    for config in all_configs:
        config['dataset_size'] = estimate_dataset_size(config['csv_path'])
    
    # Filter configurations based on arguments
    filtered_configs = []
    skipped_existing = 0
    
    for config in all_configs:
        # Filter by window
        if args.window and config['window'] != args.window:
            continue
        
        # Filter by stride
        if args.stride and config['stride'] != args.stride:
            continue
        
        # Filter large datasets
        if args.skip_large and config['dataset_size'] > 100000:
            continue
        
        # Check if already processed
        if args.skip_existing and check_if_processed(args.output_dir, config):
            skipped_existing += 1
            continue
        
        filtered_configs.append(config)
    
    if skipped_existing > 0:
        print(f"Skipped {skipped_existing} already processed configurations")
    
    if not filtered_configs:
        print("No configurations to process after filtering!")
        return
    
    # Sort by dataset size (smallest first) for efficiency
    if args.sort_by_size:
        filtered_configs.sort(key=lambda x: x['dataset_size'])
    
    # Apply max_configs limit
    if args.max_configs:
        filtered_configs = filtered_configs[:args.max_configs]
    
    print(f"Will process {len(filtered_configs)} configurations")
    
    # Show processing plan
    print(f"\nPROCESSING PLAN:")
    total_estimated_windows = 0
    
    for i, config in enumerate(filtered_configs[:10]):  # Show first 10
        size_str = f"{config['dataset_size']:,}".rjust(8)
        print(f"  {i+1:3d}. {config['window']:12s} {config['stride']:15s} ({size_str} windows)")
        total_estimated_windows += config['dataset_size']
    
    if len(filtered_configs) > 10:
        remaining = len(filtered_configs) - 10
        remaining_windows = sum(config['dataset_size'] for config in filtered_configs[10:])
        total_estimated_windows += remaining_windows
        print(f"  ... and {remaining} more configurations")
    
    print(f"\nTotal estimated windows: {total_estimated_windows:,}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No actual processing will be performed")
        return
    
    # Confirm before starting
    response = input(f"\nStart processing {len(filtered_configs)} configurations with {args.workers} workers? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Set up multiprocessing
    print(f"\n🚀 Starting parallel processing with {args.workers} workers...")
    start_time = time.time()
    
    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add tasks to queue
    for config in filtered_configs:
        task_queue.put(config)
    
    # Start worker processes
    workers = []
    for i in range(args.workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, args.model_path, args.ground_truth, 
                  args.output_dir, args.processor_script, i+1)
        )
        p.start()
        workers.append(p)
    
    # Collect results
    completed = 0
    failed = 0
    results = []
    
    try:
        while completed + failed < len(filtered_configs):
            try:
                result = result_queue.get(timeout=30)
                results.append(result)
                
                if result['status'] == 'completed':
                    completed += 1
                    duration = result.get('duration', 0)
                    dataset_size = result.get('dataset_size', 0)
                    rate = dataset_size / duration if duration > 0 else 0
                    print(f"✅ [{completed}/{len(filtered_configs)}] Completed in {duration:.1f}s ({rate:.0f} windows/s)")
                elif result['status'] == 'skipped':
                    completed += 1
                    print(f"⏩ [{completed}/{len(filtered_configs)}] Skipped (already processed)")
                else:
                    failed += 1
                    print(f"❌ [{failed}] Failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"Error collecting results: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    # Cleanup
    print("\nCleaning up workers...")
    
    # Send poison pills
    for _ in workers:
        task_queue.put(None)
    
    # Wait for workers to finish
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETED")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/(completed+failed)*100:.1f}%" if (completed+failed) > 0 else "N/A")
    
    if completed > 0:
        avg_time_per_config = total_time / completed
        print(f"Average time per config: {avg_time_per_config:.1f} seconds")
    
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
