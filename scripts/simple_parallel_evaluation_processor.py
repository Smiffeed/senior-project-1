#!/usr/bin/env python3
"""
Simple parallel batch evaluation processor
Uses subprocess-based parallel processing to avoid pickle issues
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import multiprocessing
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def find_csv_files(base_dir):
    """Find all CSV files in the dataset directory"""
    csv_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return []
    
    for csv_file in base_path.rglob("*.csv"):
        parts = csv_file.parts
        if len(parts) >= 2:
            window_dir = parts[-2]
            stride_file = parts[-1]
            if window_dir.startswith('window_') and stride_file.startswith('stride_'):
                csv_files.append({
                    'path': str(csv_file),
                    'window': window_dir,
                    'stride': stride_file.replace('.csv', ''),
                    'relative_path': str(csv_file.relative_to(base_path))
                })
    
    return sorted(csv_files, key=lambda x: (x['window'], x['stride']))

def run_single_evaluation_subprocess(args_tuple):
    """Run a single evaluation using subprocess - thread-safe"""
    csv_file_info, model_path, ground_truth, output_base_dir, eval_type, worker_id = args_tuple
    
    csv_path = csv_file_info['path']
    window = csv_file_info['window']
    stride = csv_file_info['stride']
    
    worker_prefix = f"[W{worker_id}]"
    
    # Create output directory
    output_dir = Path(output_base_dir) / eval_type
    
    # Get Python executable
    python_executable = sys.executable
    if os.name == 'nt':  # Windows
        venv_python = Path("env/Scripts/python.exe")
        if venv_python.exists():
            python_executable = str(venv_python.absolute())
    
    cmd = [
        python_executable, 
        "scripts/comprehensive_evaluation_processor.py",
        "--csv_file", csv_path,
        "--model_path", model_path,
        "--ground_truth", ground_truth,
        "--output_dir", str(output_dir),
        "--eval_type", eval_type
    ]
    
    try:
        start_time = time.time()
        
        # Set up environment for GPU sharing
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print(f"{worker_prefix} Starting: {window}/{stride}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"{worker_prefix} ✅ {window}/{stride} ({duration:.1f}s)")
        else:
            error_msg = result.stderr[-150:] if result.stderr else "Unknown error"
            print(f"{worker_prefix} ❌ {window}/{stride} - {error_msg}")
        
        # Extract metrics from output
        metrics = {}
        if success and result.stdout:
            for line in result.stdout.split('\n'):
                if 'Window-level Binary F1:' in line:
                    try:
                        metrics['window_f1'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Word-level Binary F1:' in line:
                    try:
                        metrics['word_f1'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Combined Mean IoU:' in line:
                    try:
                        metrics['combined_iou'] = float(line.split(':')[1].strip())
                    except:
                        pass
        
        return {
            'eval_type': eval_type,
            'window': window,
            'stride': stride,
            'csv_path': csv_path,
            'success': success,
            'duration': duration,
            'error': result.stderr[-200:] if result.stderr and not success else "",
            'worker_id': worker_id,
            'metrics': metrics,
            'stdout_lines': len(result.stdout.split('\n')) if result.stdout else 0
        }
        
    except Exception as e:
        print(f"{worker_prefix} ❌ EXCEPTION {window}/{stride}: {str(e)[:100]}")
        return {
            'eval_type': eval_type,
            'window': window,
            'stride': stride,
            'csv_path': csv_path,
            'success': False,
            'duration': 0,
            'error': str(e),
            'worker_id': worker_id,
            'metrics': {},
            'stdout_lines': 0
        }

def create_progress_tracker():
    """Create a thread-safe progress tracker"""
    class ProgressTracker:
        def __init__(self, total):
            self.total = total
            self.completed = 0
            self.successful = 0
            self.lock = threading.Lock()
        
        def update(self, success):
            with self.lock:
                self.completed += 1
                if success:
                    self.successful += 1
                progress = self.completed / self.total * 100
                print(f"📈 Progress: {self.completed}/{self.total} ({progress:.1f}%) - ✅ {self.successful} successful")
    
    return ProgressTracker

def main():
    parser = argparse.ArgumentParser(description="Simple parallel batch evaluation processor")
    parser.add_argument("--datasets", nargs='+', default=["csv/eval_by_0.05", "csv/eval_percent"])
    parser.add_argument("--model_path", default="./models/4_classes_max_steps")
    parser.add_argument("--ground_truth", default="./csv/eval.csv")
    parser.add_argument("--output_dir", default="./simple_parallel_evaluation_results")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: CPU count // 2)")
    parser.add_argument("--max_configs", type=int, default=None, help="Limit configurations for testing")
    parser.add_argument("--test_mode", action="store_true", help="Run with only 5 configs for testing")
    
    args = parser.parse_args()
    
    # Determine workers
    cpu_count = multiprocessing.cpu_count()
    if args.workers is None:
        args.workers = max(2, cpu_count // 2)
    
    if args.test_mode:
        args.max_configs = 5
        print("🧪 TEST MODE: Processing only 5 configurations")
    
    print("=== SIMPLE PARALLEL EVALUATION PROCESSOR ===")
    print(f"CPU Cores: {cpu_count}")
    print(f"Workers: {args.workers}")
    print(f"Datasets: {args.datasets}")
    print()
    
    # Collect all tasks
    all_tasks = []
    total_files = 0
    
    for dataset_dir in args.datasets:
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name
        
        csv_files = find_csv_files(dataset_dir)
        if not csv_files:
            print(f"❌ No CSV files found in {dataset_dir}")
            continue
        
        print(f"📁 {eval_type}: {len(csv_files)} configurations")
        total_files += len(csv_files)
        
        if args.max_configs:
            csv_files = csv_files[:args.max_configs]
            print(f"   Limited to {len(csv_files)} for testing")
        
        for csv_file_info in csv_files:
            all_tasks.append((csv_file_info, args.model_path, args.ground_truth, 
                            args.output_dir, eval_type))
    
    if not all_tasks:
        print("❌ No tasks to process!")
        return
    
    print(f"\n🚀 Processing {len(all_tasks)} configurations with {args.workers} workers...")
    
    # Add worker IDs to tasks
    tasks_with_workers = []
    for i, task in enumerate(all_tasks):
        worker_id = (i % args.workers) + 1
        tasks_with_workers.append((*task, worker_id))
    
    # Create progress tracker
    ProgressTracker = create_progress_tracker()
    tracker = ProgressTracker(len(tasks_with_workers))
    
    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
    # Each thread runs subprocess, so we still get parallelism without multiprocessing complications
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_evaluation_subprocess, task): task 
                         for task in tasks_with_workers}
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                tracker.update(result['success'])
            except Exception as e:
                print(f"❌ Task failed with exception: {e}")
                # Create a failed result
                task = future_to_task[future]
                results.append({
                    'eval_type': task[4] if len(task) > 4 else 'unknown',
                    'window': task[0]['window'] if len(task) > 0 else 'unknown',
                    'stride': task[0]['stride'] if len(task) > 0 else 'unknown',
                    'csv_path': task[0]['path'] if len(task) > 0 else 'unknown',
                    'success': False,
                    'duration': 0,
                    'error': str(e),
                    'worker_id': task[5] if len(task) > 5 else 0,
                    'metrics': {},
                    'stdout_lines': 0
                })
                tracker.update(False)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 PARALLEL EVALUATION COMPLETED!")
    print(f"⏱️  Total Time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    # Calculate statistics
    successful = len([r for r in results if r['success']])
    failed = len(results) - successful
    
    print(f"📊 Results: {successful} successful, {failed} failed out of {len(results)} total")
    
    if successful > 0:
        # Calculate speedup estimate
        sequential_time = sum(r['duration'] for r in results)
        speedup = sequential_time / total_duration if total_duration > 0 else 1
        efficiency = speedup / args.workers * 100
        print(f"🚀 Estimated speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary CSV
    summary_data = []
    for result in results:
        row = {
            'eval_type': result['eval_type'],
            'window': result['window'],
            'stride': result['stride'],
            'csv_path': result['csv_path'],
            'success': result['success'],
            'duration_seconds': result['duration'],
            'worker_id': result['worker_id'],
            'error': result['error'] if not result['success'] else ""
        }
        # Add metrics
        if 'metrics' in result:
            row.update(result['metrics'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "simple_parallel_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Create text report
    report_path = output_dir / "simple_parallel_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== SIMPLE PARALLEL EVALUATION REPORT ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write(f"Total Time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n\n")
        
        f.write(f"Total Evaluations: {len(results)}\n")
        f.write(f"Successful: {successful} ({successful/len(results)*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/len(results)*100:.1f}%)\n")
        
        if successful > 0:
            f.write(f"Estimated Speedup: {speedup:.1f}x\n")
            f.write(f"Parallel Efficiency: {efficiency:.1f}%\n")
        
        f.write("\n=== FAILED EVALUATIONS ===\n")
        for result in results:
            if not result['success']:
                f.write(f"❌ {result['eval_type']}/{result['window']}/{result['stride']}\n")
                f.write(f"   Error: {result['error'][:100]}...\n\n")
    
    print(f"📊 Summary saved to: {summary_path}")
    print(f"📄 Report saved to: {report_path}")
    
    if args.test_mode:
        print(f"\n✅ Test completed successfully! Ready for full evaluation.")
        print(f"💡 To run full evaluation: python {sys.argv[0]} --workers {args.workers}")

if __name__ == "__main__":
    main()
