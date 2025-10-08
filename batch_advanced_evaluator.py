#!/usr/bin/env python3
"""
Batch Advanced Frame-Level Evaluator
Processes multiple configurations in parallel using ThreadPoolExecutor
Similar to fixed_smart_parallel_processor.py but for frame-level evaluation
"""

import subprocess
import sys
import os
import psutil
import multiprocessing
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import pandas as pd

def get_system_info():
    """Get system information for optimal worker calculation"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative worker calculation for GPU-intensive tasks
    if memory_gb < 8:
        recommended_workers = max(1, cpu_count // 4)
    elif memory_gb < 16:
        recommended_workers = max(2, cpu_count // 3)
    else:
        recommended_workers = max(2, cpu_count // 2)
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'recommended_workers': recommended_workers
    }

def create_evaluation_configurations():
    """Create different evaluation configurations to test"""
    configurations = [
        # High precision configurations
        {
            'name': 'high_precision_small_window',
            'window_size': 0.3,
            'stride': 0.15,
            'confidence_threshold': 0.7,
            'description': 'High precision with small window (0.3s)'
        },
        {
            'name': 'high_precision_medium_window', 
            'window_size': 0.5,
            'stride': 0.25,
            'confidence_threshold': 0.7,
            'description': 'High precision with medium window (0.5s)'
        },
        
        # Balanced configurations
        {
            'name': 'balanced_small_window',
            'window_size': 0.3,
            'stride': 0.15,
            'confidence_threshold': 0.5,
            'description': 'Balanced precision with small window (0.3s)'
        },
        {
            'name': 'balanced_medium_window',
            'window_size': 0.5,
            'stride': 0.25,
            'confidence_threshold': 0.5,
            'description': 'Balanced precision with medium window (0.5s)'
        },
        {
            'name': 'balanced_large_window',
            'window_size': 1.0,
            'stride': 0.5,
            'confidence_threshold': 0.5,
            'description': 'Balanced precision with large window (1.0s)'
        },
        
        # High coverage configurations
        {
            'name': 'high_coverage_small_window',
            'window_size': 0.3,
            'stride': 0.15,
            'confidence_threshold': 0.3,
            'description': 'High coverage with small window (0.3s)'
        },
        {
            'name': 'high_coverage_medium_window',
            'window_size': 0.5,
            'stride': 0.25,
            'confidence_threshold': 0.3,
            'description': 'High coverage with medium window (0.5s)'
        },
        
        # Speed-optimized configurations
        {
            'name': 'fast_large_window',
            'window_size': 1.0,
            'stride': 0.8,
            'confidence_threshold': 0.5,
            'description': 'Fast evaluation with large window and stride'
        },
        {
            'name': 'fast_very_large_window',
            'window_size': 2.0,
            'stride': 1.5,
            'confidence_threshold': 0.5,
            'description': 'Very fast evaluation with very large window'
        }
    ]
    
    return configurations

def run_single_evaluation_thread_safe(task_info: Tuple) -> Dict:
    """Run single evaluation in thread-safe manner"""
    config, model_dir, ground_truth, audio_dir, output_base_dir, max_files, worker_id = task_info
    
    config_name = config['name']
    worker_prefix = f"[W{worker_id}]"
    
    # Create output directory for this configuration
    output_dir = Path(output_base_dir) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Python executable
    python_executable = sys.executable
    if os.name == 'nt':  # Windows
        venv_python = Path("env/Scripts/python.exe")
        if venv_python.exists():
            python_executable = str(venv_python)
    
    cmd = [
        python_executable,
        "advanced_frame_level_evaluator.py",
        "--model_dir", model_dir,
        "--ground_truth", ground_truth,
        "--audio_dir", audio_dir,
        "--output_dir", str(output_dir),
        "--window_size", str(config['window_size']),
        "--stride", str(config['stride']),
        "--confidence_threshold", str(config['confidence_threshold']),
        "--workers", "1",  # Single worker per configuration to avoid conflicts
    ]
    
    if max_files:
        cmd.extend(["--max_files", str(max_files)])
    
    try:
        start_time = time.time()
        
        print(f"{worker_prefix} Processing: {config_name}")
        print(f"{worker_prefix} Config: window={config['window_size']}s, stride={config['stride']}s, conf={config['confidence_threshold']}")
        
        # Set up environment
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"{worker_prefix} ✅ SUCCESS {config_name} ({duration:.1f}s)")
        else:
            print(f"{worker_prefix} ❌ FAILED {config_name} ({duration:.1f}s)")
            print(f"{worker_prefix} Error: {result.stderr[-200:]}")
        
        # Extract metrics from output files
        metrics = {}
        if success:
            config_file = output_dir / "evaluation_config.json"
            summary_file = output_dir / "evaluation_summary.csv"
            
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                    metrics['total_files'] = file_config.get('total_files', 0)
                    metrics['successful_files'] = file_config.get('successful_files', 0)
                    metrics['failed_files'] = file_config.get('failed_files', 0)
                    metrics['total_time'] = file_config.get('total_time_seconds', 0)
                except Exception as e:
                    print(f"{worker_prefix} ⚠️  Could not read config: {e}")
            
            if summary_file.exists():
                try:
                    summary_df = pd.read_csv(summary_file)
                    successful_rows = summary_df[summary_df.get('success', True) != False]
                    
                    if len(successful_rows) > 0:
                        metrics['mean_binary_f1'] = successful_rows['binary_f1'].mean()
                        metrics['mean_multiclass_f1'] = successful_rows['multiclass_f1'].mean()
                        metrics['mean_iou'] = successful_rows['mean_iou'].mean()
                        metrics['mean_precision'] = successful_rows['binary_precision'].mean()
                        metrics['mean_recall'] = successful_rows['binary_recall'].mean()
                    
                except Exception as e:
                    print(f"{worker_prefix} ⚠️  Could not read summary: {e}")
        
        return {
            'config_name': config_name,
            'config': config,
            'success': success,
            'duration': duration,
            'error': result.stderr[-500:] if result.stderr and not success else "",
            'worker_id': worker_id,
            'metrics': metrics,
            'output_dir': str(output_dir)
        }
        
    except Exception as e:
        print(f"{worker_prefix} ❌ EXCEPTION {config_name}: {str(e)[:100]}")
        return {
            'config_name': config_name,
            'config': config,
            'success': False,
            'duration': 0,
            'error': str(e),
            'worker_id': worker_id,
            'metrics': {},
            'output_dir': str(output_dir)
        }

def analyze_batch_results(results: List[Dict], output_dir: Path):
    """Analyze and compare batch results"""
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n📊 BATCH RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Total Configurations: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if not successful_results:
        print("❌ No successful results to analyze")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for result in successful_results:
        config = result['config']
        metrics = result['metrics']
        
        comparison_data.append({
            'config_name': result['config_name'],
            'description': config['description'],
            'window_size': config['window_size'],
            'stride': config['stride'],
            'confidence_threshold': config['confidence_threshold'],
            'duration_seconds': result['duration'],
            'mean_binary_f1': metrics.get('mean_binary_f1', 0),
            'mean_multiclass_f1': metrics.get('mean_multiclass_f1', 0),
            'mean_iou': metrics.get('mean_iou', 0),
            'mean_precision': metrics.get('mean_precision', 0),
            'mean_recall': metrics.get('mean_recall', 0),
            'total_files': metrics.get('total_files', 0),
            'successful_files': metrics.get('successful_files', 0),
            'success_rate': metrics.get('successful_files', 0) / max(metrics.get('total_files', 1), 1)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_path = output_dir / "batch_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"💾 Comparison saved to: {comparison_path}")
    
    # Show top performers
    if len(comparison_df) > 0:
        print(f"\n🏆 TOP PERFORMERS:")
        
        # Best Binary F1
        if 'mean_binary_f1' in comparison_df.columns:
            best_binary = comparison_df.nlargest(3, 'mean_binary_f1')
            print(f"\nBest Binary F1:")
            for _, row in best_binary.iterrows():
                print(f"   {row['config_name']}: {row['mean_binary_f1']:.3f} (window={row['window_size']}s, conf={row['confidence_threshold']})")
        
        # Best IoU
        if 'mean_iou' in comparison_df.columns:
            best_iou = comparison_df.nlargest(3, 'mean_iou')
            print(f"\nBest Mean IoU:")
            for _, row in best_iou.iterrows():
                print(f"   {row['config_name']}: {row['mean_iou']:.3f} (window={row['window_size']}s, conf={row['confidence_threshold']})")
        
        # Fastest
        fastest = comparison_df.nsmallest(3, 'duration_seconds')
        print(f"\nFastest Evaluation:")
        for _, row in fastest.iterrows():
            print(f"   {row['config_name']}: {row['duration_seconds']:.1f}s (window={row['window_size']}s, stride={row['stride']}s)")
    
    # Summary statistics
    if len(comparison_df) > 0:
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Average Binary F1: {comparison_df['mean_binary_f1'].mean():.3f} ± {comparison_df['mean_binary_f1'].std():.3f}")
        print(f"   Average Multiclass F1: {comparison_df['mean_multiclass_f1'].mean():.3f} ± {comparison_df['mean_multiclass_f1'].std():.3f}")
        print(f"   Average Mean IoU: {comparison_df['mean_iou'].mean():.3f} ± {comparison_df['mean_iou'].std():.3f}")
        print(f"   Average Duration: {comparison_df['duration_seconds'].mean():.1f}s ± {comparison_df['duration_seconds'].std():.1f}s")
        print(f"   Average Success Rate: {comparison_df['success_rate'].mean():.3f}")

def main():
    parser = argparse.ArgumentParser(description="Batch Advanced Frame-Level Evaluator")
    parser.add_argument("--model_dir", default="models/4_classes_max_steps", help="Model directory")
    parser.add_argument("--ground_truth", default="csv/eval_5labels.csv", help="Ground truth CSV")
    parser.add_argument("--audio_dir", default="dataset", help="Audio directory")
    parser.add_argument("--output_dir", default="batch_frame_evaluation_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--max_files", type=int, default=10, help="Max files per configuration (for testing)")
    parser.add_argument("--configs", nargs='+', help="Specific configurations to run")
    parser.add_argument("--list_configs", action="store_true", help="List available configurations")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with fewer files")
    
    args = parser.parse_args()
    
    # Get system info
    system_info = get_system_info()
    if args.workers is None:
        args.workers = min(system_info['recommended_workers'], 3)  # Limit for GPU memory
    
    if args.test_mode:
        args.max_files = 5
        print("🧪 TEST MODE: Processing only 5 files per configuration")
    
    # Get all configurations
    all_configs = create_evaluation_configurations()
    
    if args.list_configs:
        print("📋 AVAILABLE CONFIGURATIONS:")
        print("=" * 60)
        for i, config in enumerate(all_configs, 1):
            print(f"{i:2d}. {config['name']}")
            print(f"    {config['description']}")
            print(f"    Window: {config['window_size']}s, Stride: {config['stride']}s, Confidence: {config['confidence_threshold']}")
            print()
        return 0
    
    # Filter configurations if specified
    if args.configs:
        config_names = set(args.configs)
        configs_to_run = [c for c in all_configs if c['name'] in config_names]
        if not configs_to_run:
            print(f"❌ No matching configurations found for: {args.configs}")
            print("💡 Use --list_configs to see available configurations")
            return 1
    else:
        configs_to_run = all_configs
    
    print("🎯 BATCH ADVANCED FRAME-LEVEL EVALUATOR")
    print("=" * 60)
    print(f"Model Directory: {args.model_dir}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Audio Directory: {args.audio_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Max Files per Config: {args.max_files}")
    print(f"Configurations to run: {len(configs_to_run)}")
    print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_gb']:.1f}GB RAM")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [args.model_dir, args.ground_truth, args.audio_dir]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Required file not found: {file_path}")
            return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show configurations to run
    print("\n🎯 CONFIGURATIONS TO RUN:")
    for i, config in enumerate(configs_to_run, 1):
        print(f"{i:2d}. {config['name']}: {config['description']}")
    
    # Prepare tasks
    tasks = []
    for i, config in enumerate(configs_to_run):
        worker_id = (i % args.workers) + 1
        task = (config, args.model_dir, args.ground_truth, args.audio_dir, 
                args.output_dir, args.max_files, worker_id)
        tasks.append(task)
    
    print(f"\n🚀 Processing {len(tasks)} configurations with {args.workers} workers...")
    
    # Process configurations in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_single_evaluation_thread_safe, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            
            progress = completed / len(tasks) * 100
            print(f"📊 Progress: {completed}/{len(tasks)} ({progress:.1f}%)")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 BATCH EVALUATION COMPLETED!")
    print(f"⏱️  Total Time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    
    # Calculate performance metrics
    successful = len([r for r in results if r['success']])
    sequential_time = sum(r['duration'] for r in results)
    speedup = sequential_time / total_duration if total_duration > 0 else 1
    efficiency = speedup / args.workers * 100 if args.workers > 0 else 0
    
    print(f"🚀 Speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
    print(f"✅ Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Analyze results
    analyze_batch_results(results, output_dir)
    
    # Save batch summary
    batch_summary = {
        'total_configurations': len(results),
        'successful_configurations': successful,
        'failed_configurations': len(results) - successful,
        'total_duration_seconds': total_duration,
        'sequential_duration_seconds': sequential_time,
        'speedup': speedup,
        'efficiency_percent': efficiency,
        'workers_used': args.workers,
        'max_files_per_config': args.max_files,
        'system_info': system_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configurations': [r['config'] for r in results],
        'results_summary': [{
            'config_name': r['config_name'],
            'success': r['success'],
            'duration': r['duration'],
            'metrics': r['metrics']
        } for r in results]
    }
    
    batch_summary_path = output_dir / "batch_summary.json"
    with open(batch_summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Batch summary saved to: {batch_summary_path}")
    
    # Show failed configurations
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n⚠️  FAILED CONFIGURATIONS:")
        for result in failed_results:
            print(f"   {result['config_name']}: {result['error'][:100]}...")
    
    print(f"\n📁 All results saved in: {output_dir}")
    print("🎯 Use batch_comparison.csv to compare configuration performance")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())