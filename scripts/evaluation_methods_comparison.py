#!/usr/bin/env python3
"""
Compare evaluation processing methods: Sequential vs Parallel vs Smart Parallel
"""

import multiprocessing
import psutil
import subprocess
from pathlib import Path

def analyze_system():
    """Analyze system capabilities for parallel processing"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Try to get GPU info
    gpu_info = "Not detected"
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split('\n')
            gpu_info = f"{len(gpu_lines)} GPU(s) detected"
    except FileNotFoundError:
        pass
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'gpu_info': gpu_info
    }

def estimate_processing_times(num_files):
    """Estimate processing times for different methods"""
    # Based on observed ~2 minutes per file
    time_per_file = 120  # seconds
    
    sequential_time = num_files * time_per_file
    
    # Parallel estimates (conservative)
    cpu_count = multiprocessing.cpu_count()
    parallel_workers = max(1, cpu_count // 2)
    parallel_time = sequential_time / parallel_workers
    
    # Smart parallel (slightly better efficiency)
    smart_parallel_time = parallel_time * 0.9  # 10% efficiency improvement
    
    return {
        'sequential': sequential_time,
        'parallel': parallel_time,
        'smart_parallel': smart_parallel_time,
        'workers': parallel_workers
    }

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def main():
    print("=== EVALUATION PROCESSING METHODS COMPARISON ===\n")
    
    # Analyze system
    system = analyze_system()
    print("🖥️  SYSTEM RESOURCES:")
    print(f"   CPU Cores: {system['cpu_count']}")
    print(f"   Memory: {system['memory_gb']:.1f} GB")
    print(f"   GPU: {system['gpu_info']}")
    print()
    
    # Count files in datasets
    datasets = {
        "eval_by_0.05": Path("csv/eval_by_0.05"),
        "eval_percent": Path("csv/eval_percent")
    }
    
    total_files = 0
    print("📁 DATASET ANALYSIS:")
    for name, path in datasets.items():
        if path.exists():
            csv_files = list(path.rglob("*.csv"))
            file_count = len(csv_files)
            total_files += file_count
            print(f"   {name}: {file_count} CSV files")
        else:
            print(f"   {name}: Directory not found")
    
    print(f"   TOTAL: {total_files} files to process")
    print()
    
    # Estimate processing times
    times = estimate_processing_times(total_files)
    
    print("⏱️  ESTIMATED PROCESSING TIMES:")
    print()
    
    # Method 1: Sequential
    print("1️⃣  SEQUENTIAL (Current batch_evaluation_processor.py)")
    print(f"   • Processes 1 file at a time")
    print(f"   • Uses 1 CPU core + 1 GPU")
    print(f"   • Estimated time: {format_time(times['sequential'])} ({times['sequential']/3600:.1f} hours)")
    print(f"   • Memory usage: Low")
    print(f"   • GPU usage: Efficient")
    print()
    
    # Method 2: Basic Parallel  
    print("2️⃣  BASIC PARALLEL (parallel_batch_evaluation_processor.py)")
    print(f"   • Processes {times['workers']} files simultaneously")
    print(f"   • Uses {times['workers']} CPU cores + shared GPU")
    print(f"   • Estimated time: {format_time(times['parallel'])} ({times['parallel']/3600:.1f} hours)")
    print(f"   • Speedup: {times['sequential']/times['parallel']:.1f}x")
    print(f"   • Memory usage: Medium-High")
    print(f"   • GPU usage: Shared (may cause memory conflicts)")
    print()
    
    # Method 3: Smart Parallel
    print("3️⃣  SMART PARALLEL (smart_parallel_evaluation_processor.py)")
    print(f"   • Auto-detects optimal worker count")
    print(f"   • Intelligent GPU memory management")
    print(f"   • Estimated time: {format_time(times['smart_parallel'])} ({times['smart_parallel']/3600:.1f} hours)")
    print(f"   • Speedup: {times['sequential']/times['smart_parallel']:.1f}x")
    print(f"   • Memory usage: Optimized")
    print(f"   • GPU usage: Managed (round-robin or exclusive)")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if system['memory_gb'] < 8:
        print("   • Low RAM: Use Sequential or limit parallel workers to 2")
    elif system['memory_gb'] < 16:
        print("   • Medium RAM: Smart Parallel with 2-4 workers")
    else:
        print("   • High RAM: Smart Parallel with auto-detection")
    
    if "GPU" in system['gpu_info']:
        print("   • GPU detected: Use Smart Parallel with GPU management")
    else:
        print("   • No GPU: Any method will work (CPU-only)")
    
    print()
    print("🚀 QUICK START COMMANDS:")
    print()
    print("   Sequential (Safe, Slow):")
    print("   python scripts/batch_evaluation_processor.py")
    print()
    print("   Basic Parallel (Fast, May have GPU conflicts):")
    print("   python scripts/parallel_batch_evaluation_processor.py --workers 4")
    print()
    print("   Smart Parallel (Recommended):")
    print("   python scripts/smart_parallel_evaluation_processor.py")
    print()
    print("   System Analysis:")
    print("   python scripts/smart_parallel_evaluation_processor.py --analyze_system")

if __name__ == "__main__":
    main()
