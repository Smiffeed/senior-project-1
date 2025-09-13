#!/usr/bin/env python3
"""
Fixed Smart Parallel Evaluation Processor
Uses ThreadPoolExecutor to avoid multiprocessing pickle issues
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil

def get_gpu_info():
    """Get GPU information using nvidia-smi if available"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpus.append({
                            'name': parts[0],
                            'total_memory': int(parts[1]),
                            'free_memory': int(parts[2])
                        })
            return gpus
    except FileNotFoundError:
        pass
    return []

def determine_optimal_workers():
    """Determine optimal number of workers based on system resources"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    gpus = get_gpu_info()
    
    recommendations = {
        'cpu_workers': cpu_count,
        'gpu_workers': len(gpus) if gpus else 1,
        'memory_per_worker_gb': memory_gb / cpu_count,
        'gpu_memory_total': sum(gpu['total_memory'] for gpu in gpus) if gpus else 0,
        'recommendations': []
    }
    
    # Conservative recommendations
    if memory_gb < 8:
        recommended_workers = max(1, cpu_count // 4)
        recommendations['recommendations'].append(f"Low RAM ({memory_gb:.1f}GB): Use {recommended_workers} workers")
    elif memory_gb < 16:
        recommended_workers = max(2, cpu_count // 3)
        recommendations['recommendations'].append(f"Medium RAM ({memory_gb:.1f}GB): Use {recommended_workers} workers")
    else:
        recommended_workers = max(2, cpu_count // 2)
        recommendations['recommendations'].append(f"High RAM ({memory_gb:.1f}GB): Use {recommended_workers} workers")
    
    if gpus:
        gpu_memory_total = sum(gpu['total_memory'] for gpu in gpus)
        if gpu_memory_total < 8000:
            recommendations['recommendations'].append(f"Limited GPU memory ({gpu_memory_total}MB): Sequential GPU access recommended")
            recommended_workers = min(recommended_workers, 2)
        elif gpu_memory_total < 16000:
            recommendations['recommendations'].append(f"Medium GPU memory ({gpu_memory_total}MB): 2-4 workers recommended")
            recommended_workers = min(recommended_workers, 4)
        else:
            recommendations['recommendations'].append(f"High GPU memory ({gpu_memory_total}MB): Parallel GPU access possible")
    
    recommendations['recommended_workers'] = recommended_workers
    return recommendations

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

def run_single_evaluation_thread_safe(task_info):
    """Run evaluation in a thread-safe manner"""
    csv_file_info, model_path, ground_truth, output_base_dir, eval_type, worker_id, gpu_mode = task_info
    
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
        
        # Set up environment for GPU management
        env = os.environ.copy()
        
        # GPU management based on mode
        if gpu_mode == 'exclusive':
            # Simple round-robin GPU assignment
            gpus = get_gpu_info()
            if gpus:
                gpu_id = worker_id % len(gpus)
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        elif gpu_mode == 'shared':
            # All workers share all GPUs (default)
            pass
        
        # Memory management
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print(f"{worker_prefix} Processing: {window}/{stride}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"{worker_prefix} ✅ {window}/{stride} ({duration:.1f}s)")
        else:
            error_msg = result.stderr[-150:] if result.stderr else "Unknown error"
            print(f"{worker_prefix} ❌ {window}/{stride} - {error_msg}")
        
        # Extract metrics
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
            'metrics': metrics
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
            'metrics': {}
        }

def main():
    parser = argparse.ArgumentParser(description="Fixed smart parallel evaluation processor")
    parser.add_argument("--datasets", nargs='+', default=["csv/eval_by_0.05", "csv/eval_percent"])
    parser.add_argument("--model_path", default="./models/4_classes_max_steps")
    parser.add_argument("--ground_truth", default="./csv/eval.csv")
    parser.add_argument("--output_dir", default="./fixed_smart_parallel_results")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (auto-detect if not specified)")
    parser.add_argument("--gpu_mode", choices=['shared', 'exclusive'], default='shared',
                       help="GPU allocation mode")
    parser.add_argument("--max_configs", type=int, default=None)
    parser.add_argument("--test_mode", action="store_true", help="Run with only 5 configs for testing")
    parser.add_argument("--analyze_system", action="store_true", help="Show system analysis and exit")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.max_configs = 5
        print("🧪 TEST MODE: Processing only 5 configurations")
    
    # Analyze system resources
    system_info = determine_optimal_workers()
    
    if args.analyze_system:
        print("=== SYSTEM RESOURCE ANALYSIS ===")
        print(f"CPU Cores: {system_info['cpu_workers']}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Memory per worker: {system_info['memory_per_worker_gb']:.1f} GB")
        if system_info['gpu_memory_total'] > 0:
            print(f"GPU Memory: {system_info['gpu_memory_total']} MB")
        else:
            print("GPU: Not detected")
        
        print("\nRECOMMENDATIONS:")
        for rec in system_info['recommendations']:
            print(f"  • {rec}")
        print(f"  • Recommended workers: {system_info['recommended_workers']}")
        return
    
    # Determine number of workers
    if args.workers is None:
        args.workers = system_info['recommended_workers']
    
    print("=== FIXED SMART PARALLEL EVALUATION PROCESSOR ===")
    print(f"CPU Cores: {system_info['cpu_workers']}")
    print(f"GPU Mode: {args.gpu_mode}")
    print(f"Workers: {args.workers}")
    print(f"Datasets: {args.datasets}")
    print()
    
    # Display recommendations
    for rec in system_info['recommendations']:
        print(f"💡 {rec}")
    print()
    
    # Collect tasks
    all_tasks = []
    for dataset_dir in args.datasets:
        csv_files = find_csv_files(dataset_dir)
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name
        
        print(f"📁 {eval_type}: {len(csv_files)} configurations")
        
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
    
    # Add worker assignments
    tasks_with_workers = []
    for i, task in enumerate(all_tasks):
        worker_id = (i % args.workers) + 1
        tasks_with_workers.append((*task, worker_id, args.gpu_mode))
    
    start_time = time.time()
    results = []
    completed = 0
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_single_evaluation_thread_safe, task): task 
                         for task in tasks_with_workers}
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                success_count = len([r for r in results if r['success']])
                progress = completed / len(all_tasks) * 100
                print(f"📈 {completed}/{len(all_tasks)} ({progress:.1f}%) - ✅ {success_count}")
            except Exception as e:
                print(f"❌ Task failed with exception: {e}")
                completed += 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 COMPLETED!")
    print(f"⏱️  Time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    
    # Calculate performance
    successful = len([r for r in results if r['success']])
    sequential_time = sum(r['duration'] for r in results)
    speedup = sequential_time / total_duration if total_duration > 0 else 1
    efficiency = speedup / args.workers * 100 if args.workers > 0 else 0
    
    print(f"🚀 Speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
    print(f"✅ Success: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    for result in results:
        row = {
            'eval_type': result['eval_type'],
            'window': result['window'],
            'stride': result['stride'],
            'success': result['success'],
            'duration_seconds': result['duration'],
            'worker_id': result['worker_id']
        }
        row.update(result['metrics'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "fixed_smart_parallel_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"📊 Summary saved to: {summary_path}")
    
    if args.test_mode:
        print(f"\n✅ Test completed successfully!")
        print(f"💡 To run full evaluation: python {sys.argv[0]} --workers {args.workers}")

if __name__ == "__main__":
    main()
