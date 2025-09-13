#!/usr/bin/env python3
"""
Parallel eval_percent batch processor - runs multiple evaluations simultaneously using evaluate_advanced_models_word.py (Method 2)
"""

import os
import sys
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import argparse
import queue
import threading

def get_all_configurations(eval_percent_dir):
    """Get all window and stride combinations from eval_percent folder"""
    configurations = []
    eval_path = Path(eval_percent_dir)
    
    if not eval_path.exists():
        print(f"Error: {eval_percent_dir} does not exist")
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
        print(f"Warning: Could not read {csv_file}: {e}")
        return 0

def process_single_configuration_worker(config, model_path, threshold, output_dir, processor_script, worker_id):
    """Worker function to process a single configuration"""
    
    # Create specific output directory for this configuration
    specific_output_dir = Path(output_dir) / config['window'] / config['stride']
    
    # Check if already processed
    note_file = specific_output_dir / "note.txt"
    if note_file.exists():
        print(f"Worker {worker_id}: SKIPPING {config['relative_path']} (already exists)")
        return True, 0, config['relative_path'], "SKIPPED"
    
    # Display detailed configuration info like smart_eval_processor
    dataset_size = estimate_dataset_size(config['csv_path'])
    
    print(f"\n{'='*80}")
    print(f"Worker {worker_id}: Processing Configuration:")
    print(f"   Window: {config['window']}")
    print(f"   Stride: {config['stride']}")
    print(f"   CSV: {config['relative_path']}")
    print(f"   Dataset Size: {dataset_size:,} windows")
    
    # Estimate processing time (adjusted based on actual performance data)
    if dataset_size > 100000:
        estimated_time = "60-120 minutes"  # Updated from 20-30
        complexity = "LARGE"
    elif dataset_size > 50000:
        estimated_time = "20-60 minutes"   # Updated from 10-20
        complexity = "MEDIUM"
    elif dataset_size > 10000:
        estimated_time = "5-20 minutes"    # New category
        complexity = "MEDIUM"
    else:
        estimated_time = "1-5 minutes"     # Updated from 5-10
        complexity = "SMALL"
    
    print(f"   Complexity: {complexity}")
    print(f"   Est. Time: {estimated_time}")
    print(f"{'='*80}")
    
    # Create the specific output directory
    specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the processor script with same command structure as smart_eval_processor
    python_exe = "C:/Users/muldi/Documents/Playground/University/senior-project-1/env/Scripts/python.exe"
    
    # Create a unique plots directory for this worker to avoid conflicts
    unique_plots_dir = f"./plots_worker_{worker_id}_{int(time.time())}"
    
    # Set environment variable to use unique plots directory
    env = os.environ.copy()
    env['WORKER_PLOTS_DIR'] = unique_plots_dir
    
    cmd = [
        python_exe,
        processor_script,
        "--csv_file", config['csv_path'],
        "--model_path", model_path,
        "--threshold", str(threshold),
        "--output_dir", output_dir  # Use base output_dir like smart_eval_processor
    ]
    
    start_time = time.time()
    
    try:
        print(f"Worker {worker_id}: Starting evaluation at {datetime.now().strftime('%H:%M:%S')}...")
        
        # Use same execution approach as smart_eval_processor
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=os.getcwd(),
            env=env  # Pass the environment with unique plots directory
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"Worker {worker_id}: Completed successfully in {duration/60:.1f} minutes")
            print(f"Worker {worker_id}: Output: evaluation_results/eval_percent/{config['window']}/{config['stride']}/")
            
            # Clean up the unique plots directory if it exists
            try:
                unique_plots_path = Path(unique_plots_dir)
                if unique_plots_path.exists():
                    import shutil
                    shutil.rmtree(unique_plots_path)
                    print(f"Worker {worker_id}: Cleaned up temporary plots directory")
            except Exception as e:
                print(f"Worker {worker_id}: Warning - could not clean plots directory: {e}")
            
            return True, duration, config['relative_path'], "SUCCESS"
        else:
            print(f"Worker {worker_id}: Failed with return code: {result.returncode}")
            print(f"Worker {worker_id}: Error output: {result.stderr}")
            return False, duration, config['relative_path'], f"FAILED-{result.returncode}"
            
    except KeyboardInterrupt:
        print(f"Worker {worker_id}: Interrupted by user")
        return False, time.time() - start_time, config['relative_path'], "INTERRUPTED"
    except Exception as e:
        print(f"Worker {worker_id}: Unexpected error: {e}")
        return False, time.time() - start_time, config['relative_path'], f"ERROR-{str(e)[:20]}"

def get_optimal_worker_count():
    """Auto-detect optimal worker count based on system resources"""
    try:
        import psutil
    except ImportError:
        print("⚠️  psutil not available, install with: pip install psutil")
        return 4
    
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
    
    # Conservative estimates for evaluation workload (updated based on actual usage)
    memory_per_worker = 3.0  # GB per worker (model + data + processing overhead)
    
    # Calculate optimal based on different constraints
    memory_limited = int(available_memory_gb / memory_per_worker)
    cpu_limited = physical_cores - 1  # Leave 1 core for OS
    
    # Choose the most limiting factor
    optimal = min(memory_limited, cpu_limited, 8)  # Cap at 8 workers max
    optimal = max(optimal, 1)  # At least 1 worker
    
    print(f"\n🧮 WORKER CALCULATIONS:")
    print(f"   Memory-limited workers: {memory_limited}")
    print(f"   CPU-limited workers: {cpu_limited}")
    print(f"   Recommended workers: {optimal}")
    
    return optimal

def worker_process(task_queue, result_queue, model_path, threshold, output_dir, processor_script, worker_id):
    """Worker process function"""
    while True:
        try:
            config = task_queue.get(timeout=1)
            if config is None:  # Shutdown signal
                break
                
            success, duration, path, status = process_single_configuration_worker(
                config, model_path, threshold, output_dir, processor_script, worker_id
            )
            
            result_queue.put((success, duration, path, status))
            # Remove task_done() call since mp.Queue doesn't have it
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Parallel eval_percent batch processor")
    parser.add_argument("--eval_percent_dir", default="./csv/eval_by_0.05", help="Path to eval_percent directory")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output_dir", default="./evaluation_results/eval_by_0.05", help="Output directory")
    parser.add_argument("--processor_script", default="scripts/simple_eval_processor_word.py", help="Processor script to use")
    parser.add_argument("--window", help="Process only specific window (e.g., 'window_0.3s')")
    parser.add_argument("--stride", help="Process only specific stride (e.g., 'stride_100.0%')")
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
        recommended = get_optimal_worker_count()
        if args.workers > recommended:
            print(f"⚠️  Warning: Using {args.workers} workers, but {recommended} is recommended for your system")
    
    # Determine optimal number of workers
    cpu_count = mp.cpu_count()
    if args.workers > cpu_count:
        print(f"Warning: Requested {args.workers} workers but only {cpu_count} CPUs available")
        args.workers = min(args.workers, cpu_count)
    
    print(f"Parallel eval_percent batch processor")
    print(f"Workers: {args.workers} parallel processes")
    print(f"CPU Count: {cpu_count}")
    print(f"Eval Percent Dir: {args.eval_percent_dir}")
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output_dir}")
    print(f"Processor: {args.processor_script}")
    print(f"Skip Existing: {args.skip_existing}")
    
    # Get all configurations
    print(f"\nScanning for configurations...")
    all_configs = get_all_configurations(args.eval_percent_dir)
    
    if not all_configs:
        print(f"No configurations found in {args.eval_percent_dir}")
        sys.exit(1)
    
    print(f"Found {len(all_configs)} total configurations")
    
    # Add dataset sizes to configurations
    for config in all_configs:
        config['dataset_size'] = estimate_dataset_size(config['csv_path'])
    
    # Filter configurations based on arguments
    filtered_configs = []
    skipped_existing = 0
    
    for config in all_configs:
        # Check if already processed
        if args.skip_existing:
            specific_output_dir = Path(args.output_dir) / config['window'] / config['stride']
            note_file = specific_output_dir / "note.txt"
            if note_file.exists():
                skipped_existing += 1
                continue
        
        # Check window filter
        if args.window and config['window'] != args.window:
            continue
            
        # Check stride filter
        if args.stride and config['stride'] != args.stride:
            continue
        
        # Check dataset size filter
        if args.skip_large and config['dataset_size'] > 100000:
            continue
        
        filtered_configs.append(config)
    
    if skipped_existing > 0:
        print(f"Skipped {skipped_existing} already processed configurations")
    
    if not filtered_configs:
        print(f"No configurations to process (all may be completed already)")
        return
    
    # Sort by dataset size (smallest first) for efficiency
    if args.sort_by_size:
        filtered_configs.sort(key=lambda x: x['dataset_size'])
        print(f"Sorted by dataset size (smallest first)")
    
    # Apply max_configs limit
    if args.max_configs:
        filtered_configs = filtered_configs[:args.max_configs]
    
    print(f"Will process {len(filtered_configs)} configurations")
    
    # Show processing plan
    print(f"\nPROCESSING PLAN:")
    total_estimated_windows = 0
    small_count = medium_small_count = medium_count = large_count = 0
    
    for i, config in enumerate(filtered_configs, 1):
        dataset_size = config['dataset_size']
        total_estimated_windows += dataset_size
        
        if dataset_size > 100000:
            complexity = "LARGE"
            large_count += 1
        elif dataset_size > 50000:
            complexity = "MEDIUM"
            medium_count += 1
        elif dataset_size > 10000:
            complexity = "MED-SM"
            medium_small_count += 1
        else:
            complexity = "SMALL"
            small_count += 1
            
        if i <= 10:  # Show first 10
            print(f"   {i:2d}. {complexity:6s} {config['relative_path']} ({dataset_size:,} windows)")
        elif i == 11:
            print(f"   ... ({len(filtered_configs)-10} more configurations)")
    
    print(f"\nSUMMARY:")
    print(f"   Total Configurations: {len(filtered_configs)}")
    print(f"   Small (<10K): {small_count}, Med-Small (10K-50K): {medium_small_count}")
    print(f"   Medium (50K-100K): {medium_count}, Large (>100K): {large_count}")
    print(f"   Total Windows: {total_estimated_windows:,}")
    
    # Calculate time estimates with parallelization (updated based on actual performance)
    # More accurate estimation based on real data: ~4.5 minutes per 10K windows
    estimated_minutes = total_estimated_windows * 4.5 / 10000  # 4.5 min per 10K windows
    estimated_hours = estimated_minutes / 60
    parallel_hours = estimated_hours / args.workers  # Parallel estimate with ~80% efficiency
    
    print(f"   Estimated Total Time: {estimated_hours:.1f} hours (sequential)")
    print(f"   Estimated Parallel Time ({args.workers} workers): {parallel_hours:.1f} hours")
    print(f"   Speed Improvement: {args.workers}x faster")
    
    if args.dry_run:
        print(f"\nDRY RUN - No actual processing performed")
        return
    
    # Confirm before starting
    response = input(f"\nContinue with parallel processing? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print(f"Processing cancelled")
        return
    
    # Start parallel processing
    print(f"\nStarting parallel processing with {args.workers} workers at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add tasks to queue
    for config in filtered_configs:
        task_queue.put(config)
    
    # Start worker processes
    workers = []
    for i in range(args.workers):
        worker = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, args.model_path, args.threshold, 
                  args.output_dir, args.processor_script, i+1)
        )
        worker.start()
        workers.append(worker)
    
    # Monitor progress
    completed = 0
    successful = 0
    failed = 0
    total_time = 0
    start_time = time.time()
    
    try:
        while completed < len(filtered_configs):
            try:
                success, duration, path, status = result_queue.get(timeout=30)
                completed += 1
                total_time += duration
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                elapsed = time.time() - start_time
                remaining = len(filtered_configs) - completed
                
                print(f"\nProgress: {completed}/{len(filtered_configs)} ({completed/len(filtered_configs)*100:.1f}%)")
                print(f"Latest: {status} - {path}")
                print(f"Session Stats: Success {successful} | Failed {failed} | {elapsed/60:.1f}min total")
                
                # Show remaining estimate (same style as smart_eval_processor)
                if completed > 0 and remaining > 0:
                    avg_time_per_config = elapsed / completed
                    estimated_remaining = (avg_time_per_config * remaining) / 60
                    print(f"Estimated remaining: {estimated_remaining:.1f} minutes")
                
            except queue.Empty:
                print("Waiting for workers to complete...")
                continue
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping workers...")
    
    # Stop workers
    for _ in workers:
        task_queue.put(None)  # Shutdown signal
    
    for worker in workers:
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
    
    # Final summary (same style as smart_eval_processor)
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"PARALLEL BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Time: {total_elapsed/60:.1f} minutes")
    print(f"Success Rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "No tasks completed")
    print(f"Results saved to: {args.output_dir}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.workers > 1:
        sequential_estimate = total_elapsed * args.workers
        print(f"Estimated sequential time: {sequential_estimate/60:.1f} minutes")
        print(f"Speed improvement: {sequential_estimate/total_elapsed:.1f}x")
    
    if failed > 0:
        print(f"\nSome configurations failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print(f"\nAll configurations processed successfully!")

if __name__ == "__main__":
    main()
