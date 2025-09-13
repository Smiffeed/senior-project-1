#!/usr/bin/env python3
"""
Parallel batch evaluation processor for multiple CSV datasets
Processes multiple files simultaneously using multiprocessing
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading

def find_csv_files(base_dir):
    """Find all CSV files in the dataset directory"""
    csv_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return []
    
    # Walk through all subdirectories to find CSV files
    for csv_file in base_path.rglob("*.csv"):
        # Extract window and stride info from path
        parts = csv_file.parts
        if len(parts) >= 2:
            window_dir = parts[-2]  # e.g., "window_0.3s"
            stride_file = parts[-1]  # e.g., "stride_0.125s.csv"
            
            if window_dir.startswith('window_') and stride_file.startswith('stride_'):
                csv_files.append({
                    'path': str(csv_file),
                    'window': window_dir,
                    'stride': stride_file.replace('.csv', ''),
                    'relative_path': str(csv_file.relative_to(base_path))
                })
    
    return sorted(csv_files, key=lambda x: (x['window'], x['stride']))

def run_single_evaluation(task_info):
    """Run evaluation for a single CSV file - designed for multiprocessing"""
    csv_file_info, model_path, ground_truth, output_base_dir, eval_type, worker_id = task_info
    
    csv_path = csv_file_info['path']
    window = csv_file_info['window']
    stride = csv_file_info['stride']
    
    worker_prefix = f"[Worker-{worker_id}]"
    print(f"{worker_prefix} Processing: {window}/{stride}")
    
    # Create output directory for this evaluation type
    output_dir = Path(output_base_dir) / eval_type
    
    # Determine the Python executable - try virtual environment first
    python_executable = sys.executable
    
    # Check if we're on Windows and try to use the virtual environment
    if os.name == 'nt':  # Windows
        venv_python = Path("env/Scripts/python.exe")
        if venv_python.exists():
            python_executable = str(venv_python.absolute())
    else:  # Unix-like
        venv_python = Path("env/bin/python")
        if venv_python.exists():
            python_executable = str(venv_python.absolute())
    
    # Run the comprehensive evaluation processor
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
        
        # Set environment variables to control GPU usage in parallel
        env = os.environ.copy()
        # For CUDA, each process should see all GPUs but we can manage usage
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"{worker_prefix} ✅ SUCCESS - {window}/{stride} ({end_time - start_time:.1f}s)")
            # Extract key metrics from output
            output_lines = result.stdout.split('\n')
            metrics = {}
            for line in output_lines:
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
                'success': True,
                'duration': end_time - start_time,
                'error': "",
                'worker_id': worker_id,
                'metrics': metrics
            }
        else:
            print(f"{worker_prefix} ❌ FAILED - {window}/{stride} ({end_time - start_time:.1f}s)")
            error_msg = result.stderr[-200:] if result.stderr else result.stdout[-200:]
            return {
                'eval_type': eval_type,
                'window': window,
                'stride': stride,
                'csv_path': csv_path,
                'success': False,
                'duration': end_time - start_time,
                'error': error_msg,
                'worker_id': worker_id,
                'metrics': {}
            }
            
    except Exception as e:
        print(f"{worker_prefix} ❌ EXCEPTION - {window}/{stride}: {e}")
        return {
            'eval_type': eval_type,
            'window': window,
            'stride': stride,
            'csv_path': csv_path,
            'success': False,
            'duration': 0,
            'error': str(e),
            'worker_id': worker_id,
            'metrics': {}
        }

def create_summary_report(results, output_dir, parallel_info):
    """Create a summary report of all evaluations"""
    summary_data = []
    
    for result in results:
        row = {
            'eval_type': result['eval_type'],
            'window': result['window'],
            'stride': result['stride'],
            'csv_path': result['csv_path'],
            'success': result['success'],
            'duration_seconds': result['duration'],
            'worker_id': result.get('worker_id', 'unknown'),
            'error': result['error'] if not result['success'] else ""
        }
        # Add metrics if available
        if 'metrics' in result:
            row.update(result['metrics'])
        summary_data.append(row)
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / "parallel_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Create detailed report
    report_path = Path(output_dir) / "parallel_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== PARALLEL BATCH EVALUATION REPORT ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parallel Workers: {parallel_info['num_workers']}\n")
        f.write(f"Total Processing Time: {parallel_info['total_time']:.1f} seconds ({parallel_info['total_time']/60:.1f} minutes)\n\n")
        
        # Summary statistics
        total_evaluations = len(results)
        successful = len([r for r in results if r['success']])
        failed = total_evaluations - successful
        
        f.write("=== SUMMARY STATISTICS ===\n")
        f.write(f"Total Evaluations: {total_evaluations}\n")
        f.write(f"Successful: {successful} ({successful/total_evaluations*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/total_evaluations*100:.1f}%)\n")
        
        if successful > 0:
            # Calculate speedup
            sequential_time = sum(r['duration'] for r in results)
            actual_time = parallel_info['total_time']
            speedup = sequential_time / actual_time if actual_time > 0 else 1
            efficiency = speedup / parallel_info['num_workers'] * 100
            
            f.write(f"Sequential Time Equivalent: {sequential_time:.1f} seconds ({sequential_time/60:.1f} minutes)\n")
            f.write(f"Actual Parallel Time: {actual_time:.1f} seconds ({actual_time/60:.1f} minutes)\n")
            f.write(f"Speedup: {speedup:.1f}x\n")
            f.write(f"Parallel Efficiency: {efficiency:.1f}%\n")
        
        f.write(f"Average Time per Evaluation: {sum(r['duration'] for r in results)/len(results):.1f} seconds\n\n")
        
        # Performance metrics summary
        successful_results = [r for r in results if r['success'] and 'metrics' in r and r['metrics']]
        if successful_results:
            f.write("=== PERFORMANCE METRICS SUMMARY ===\n")
            metrics_summary = {}
            for metric in ['window_f1', 'word_f1', 'combined_iou']:
                values = [r['metrics'].get(metric) for r in successful_results if r['metrics'].get(metric) is not None]
                if values:
                    metrics_summary[metric] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            for metric, stats in metrics_summary.items():
                f.write(f"{metric}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f} (n={stats['count']})\n")
            f.write("\n")
        
        # Worker performance
        f.write("=== WORKER PERFORMANCE ===\n")
        worker_stats = {}
        for result in results:
            worker_id = result.get('worker_id', 'unknown')
            if worker_id not in worker_stats:
                worker_stats[worker_id] = {'total': 0, 'successful': 0, 'total_time': 0}
            worker_stats[worker_id]['total'] += 1
            if result['success']:
                worker_stats[worker_id]['successful'] += 1
            worker_stats[worker_id]['total_time'] += result['duration']
        
        for worker_id, stats in sorted(worker_stats.items()):
            success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"Worker {worker_id}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - Avg: {avg_time:.1f}s\n")
        f.write("\n")
        
        # Group by evaluation type
        eval_types = list(set(r['eval_type'] for r in results))
        for eval_type in eval_types:
            type_results = [r for r in results if r['eval_type'] == eval_type]
            type_successful = len([r for r in type_results if r['success']])
            
            f.write(f"=== {eval_type.upper()} RESULTS ===\n")
            f.write(f"Total: {len(type_results)}\n")
            f.write(f"Successful: {type_successful} ({type_successful/len(type_results)*100:.1f}%)\n")
            f.write(f"Failed: {len(type_results) - type_successful}\n\n")
        
        # Failed evaluations detail
        failed_results = [r for r in results if not r['success']]
        if failed_results:
            f.write("=== FAILED EVALUATIONS DETAILS ===\n")
            for result in failed_results:
                f.write(f"❌ {result['eval_type']}/{result['window']}/{result['stride']} (Worker {result.get('worker_id', 'unknown')})\n")
                f.write(f"   CSV: {result['csv_path']}\n")
                f.write(f"   Error: {result['error'][:100]}...\n\n")
    
    print(f"\n📊 Summary report saved to: {report_path}")
    print(f"📊 Summary CSV saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Parallel batch evaluation for multiple CSV datasets")
    parser.add_argument("--datasets", nargs='+', default=["csv/eval_by_0.05", "csv/eval_percent"], 
                       help="Dataset directories to evaluate")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--ground_truth", default="./csv/eval.csv", help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="./parallel_evaluation_results", help="Output directory")
    parser.add_argument("--max_configs", type=int, default=None, help="Maximum number of configurations to process per dataset (for testing)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip configurations that already have results")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count // 2)")
    parser.add_argument("--gpu_per_worker", action="store_true", help="Allow each worker to use GPU (may cause memory issues)")
    
    args = parser.parse_args()
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    if args.workers is None:
        # Conservative default: use half of CPU cores to avoid overwhelming the system
        args.workers = max(1, cpu_count // 2)
    
    print("=== PARALLEL BATCH EVALUATION PROCESSOR ===")
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model_path}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Parallel Workers: {args.workers} (CPU cores: {cpu_count})")
    if args.max_configs:
        print(f"Max Configurations per Dataset: {args.max_configs}")
    if args.gpu_per_worker:
        print("⚠️  GPU per worker enabled - monitor GPU memory usage!")
    else:
        print("ℹ️  Single GPU mode - workers will share GPU access")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all tasks
    all_tasks = []
    for dataset_dir in args.datasets:
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name  # e.g., "eval_by_0.05" or "eval_percent"
        
        print(f"🔍 Finding CSV files in {dataset_dir}...")
        csv_files = find_csv_files(dataset_dir)
        
        if not csv_files:
            print(f"❌ No CSV files found in {dataset_dir}")
            continue
            
        print(f"📁 Found {len(csv_files)} CSV files")
        
        # Limit number of configurations if specified
        if args.max_configs and len(csv_files) > args.max_configs:
            print(f"🔧 Limiting to first {args.max_configs} configurations for testing")
            csv_files = csv_files[:args.max_configs]
        
        # Add tasks for this dataset
        for csv_file_info in csv_files:
            # Check if results already exist
            if args.skip_existing:
                expected_output = output_dir / eval_type / csv_file_info['window'] / csv_file_info['stride']
                if expected_output.exists() and any(expected_output.iterdir()):
                    continue
            
            all_tasks.append((csv_file_info, args.model_path, args.ground_truth, args.output_dir, eval_type))
    
    if not all_tasks:
        print("❌ No tasks to process!")
        return
    
    print(f"\n🚀 Starting parallel evaluation of {len(all_tasks)} configurations with {args.workers} workers...")
    
    # Add worker IDs to tasks
    tasks_with_workers = []
    for i, task in enumerate(all_tasks):
        worker_id = (i % args.workers) + 1
        tasks_with_workers.append((*task, worker_id))
    
    start_time = time.time()
    results = []
    
    # Progress tracking
    completed = 0
    total_tasks = len(tasks_with_workers)
    
    # Run parallel evaluation
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_evaluation, task): task for task in tasks_with_workers}
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Progress update
            success_count = len([r for r in results if r['success']])
            progress = completed / total_tasks * 100
            print(f"📈 Progress: {completed}/{total_tasks} ({progress:.1f}%) - ✅ {success_count} successful")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 PARALLEL EVALUATION COMPLETED!")
    print(f"⏱️  Wall Clock Time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"📊 Total Evaluations: {len(results)}")
    print(f"✅ Successful: {len([r for r in results if r['success']])}")
    print(f"❌ Failed: {len([r for r in results if not r['success']])}")
    
    # Calculate speedup
    sequential_time = sum(r['duration'] for r in results)
    speedup = sequential_time / total_duration if total_duration > 0 else 1
    print(f"🚀 Speedup: {speedup:.1f}x (vs sequential: {sequential_time/60:.1f} minutes)")
    
    # Create summary report
    parallel_info = {
        'num_workers': args.workers,
        'total_time': total_duration
    }
    create_summary_report(results, output_dir, parallel_info)
    
    print(f"\n📁 All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
