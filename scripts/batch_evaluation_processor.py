#!/usr/bin/env python3
"""
Batch evaluation processor for multiple CSV datasets
Processes both eval_by_0.05 and eval_percent datasets
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

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
        # e.g., csv/eval_by_0.05/window_0.3s/stride_0.125s.csv
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

def run_evaluation(csv_file_info, model_path, ground_truth, output_base_dir, eval_type):
    """Run evaluation for a single CSV file"""
    csv_path = csv_file_info['path']
    window = csv_file_info['window']
    stride = csv_file_info['stride']
    
    print(f"\n{'='*80}")
    print(f"Processing: {window}/{stride}")
    print(f"CSV: {csv_path}")
    print(f"{'='*80}")
    
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
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - Completed in {end_time - start_time:.1f}s")
            # Extract key metrics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Window-level Binary F1:' in line or 'Word-level Binary F1:' in line or 'Combined Mean IoU:' in line:
                    print(f"   {line.strip()}")
            return True, end_time - start_time, ""
        else:
            print(f"❌ FAILED - Error in {end_time - start_time:.1f}s")
            print("STDOUT:", result.stdout[-500:] if result.stdout else "None")
            print("STDERR:", result.stderr[-500:] if result.stderr else "None")
            return False, end_time - start_time, result.stderr
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False, 0, str(e)

def create_summary_report(results, output_dir):
    """Create a summary report of all evaluations"""
    summary_data = []
    
    for result in results:
        summary_data.append({
            'eval_type': result['eval_type'],
            'window': result['window'],
            'stride': result['stride'],
            'csv_path': result['csv_path'],
            'success': result['success'],
            'duration_seconds': result['duration'],
            'error': result['error'] if not result['success'] else ""
        })
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / "batch_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Create detailed report
    report_path = Path(output_dir) / "batch_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== BATCH EVALUATION REPORT ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        total_evaluations = len(results)
        successful = len([r for r in results if r['success']])
        failed = total_evaluations - successful
        total_time = sum(r['duration'] for r in results)
        
        f.write("=== SUMMARY STATISTICS ===\n")
        f.write(f"Total Evaluations: {total_evaluations}\n")
        f.write(f"Successful: {successful} ({successful/total_evaluations*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/total_evaluations*100:.1f}%)\n")
        f.write(f"Total Processing Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n")
        f.write(f"Average Time per Evaluation: {total_time/total_evaluations:.1f} seconds\n\n")
        
        # Group by evaluation type
        eval_types = list(set(r['eval_type'] for r in results))
        for eval_type in eval_types:
            type_results = [r for r in results if r['eval_type'] == eval_type]
            type_successful = len([r for r in type_results if r['success']])
            
            f.write(f"=== {eval_type.upper()} RESULTS ===\n")
            f.write(f"Total: {len(type_results)}\n")
            f.write(f"Successful: {type_successful} ({type_successful/len(type_results)*100:.1f}%)\n")
            f.write(f"Failed: {len(type_results) - type_successful}\n\n")
            
            # List all configurations
            for result in type_results:
                status = "✅" if result['success'] else "❌"
                f.write(f"  {status} {result['window']}/{result['stride']} - {result['duration']:.1f}s\n")
                if not result['success'] and result['error']:
                    f.write(f"      Error: {result['error'][:100]}...\n")
            f.write("\n")
        
        # Failed evaluations detail
        failed_results = [r for r in results if not r['success']]
        if failed_results:
            f.write("=== FAILED EVALUATIONS DETAILS ===\n")
            for result in failed_results:
                f.write(f"❌ {result['eval_type']}/{result['window']}/{result['stride']}\n")
                f.write(f"   CSV: {result['csv_path']}\n")
                f.write(f"   Error: {result['error']}\n\n")
    
    print(f"\n📊 Summary report saved to: {report_path}")
    print(f"📊 Summary CSV saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for multiple CSV datasets")
    parser.add_argument("--datasets", nargs='+', default=["csv/eval_by_0.05", "csv/eval_percent"], 
                       help="Dataset directories to evaluate")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--ground_truth", default="./csv/eval.csv", help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="./batch_evaluation_results", help="Output directory")
    parser.add_argument("--max_configs", type=int, default=None, help="Maximum number of configurations to process per dataset (for testing)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip configurations that already have results")
    
    args = parser.parse_args()
    
    print("=== BATCH EVALUATION PROCESSOR ===")
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model_path}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    if args.max_configs:
        print(f"Max Configurations per Dataset: {args.max_configs}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_start_time = time.time()
    
    # Process each dataset
    for dataset_dir in args.datasets:
        # Determine evaluation type from directory name
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name  # e.g., "eval_by_0.05" or "eval_percent"
        
        print(f"\n🔍 Finding CSV files in {dataset_dir}...")
        csv_files = find_csv_files(dataset_dir)
        
        if not csv_files:
            print(f"❌ No CSV files found in {dataset_dir}")
            continue
            
        print(f"📁 Found {len(csv_files)} CSV files")
        
        # Limit number of configurations if specified
        if args.max_configs and len(csv_files) > args.max_configs:
            print(f"🔧 Limiting to first {args.max_configs} configurations for testing")
            csv_files = csv_files[:args.max_configs]
        
        # Process each CSV file
        for i, csv_file_info in enumerate(csv_files):
            print(f"\n[{i+1}/{len(csv_files)}] Processing {eval_type}...")
            
            # Check if results already exist
            if args.skip_existing:
                expected_output = output_dir / eval_type / csv_file_info['window'] / csv_file_info['stride']
                if expected_output.exists() and any(expected_output.iterdir()):
                    print(f"⏭️  Skipping - Results already exist at {expected_output}")
                    all_results.append({
                        'eval_type': eval_type,
                        'window': csv_file_info['window'],
                        'stride': csv_file_info['stride'],
                        'csv_path': csv_file_info['path'],
                        'success': True,
                        'duration': 0,
                        'error': "Skipped - already exists"
                    })
                    continue
            
            # Run evaluation
            success, duration, error = run_evaluation(
                csv_file_info, args.model_path, args.ground_truth, 
                args.output_dir, eval_type
            )
            
            # Record result
            all_results.append({
                'eval_type': eval_type,
                'window': csv_file_info['window'],
                'stride': csv_file_info['stride'],
                'csv_path': csv_file_info['path'],
                'success': success,
                'duration': duration,
                'error': error
            })
            
            # Progress update
            completed = len([r for r in all_results if r['success']])
            total_processed = len(all_results)
            print(f"📈 Progress: {completed}/{total_processed} successful evaluations")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n🎉 BATCH EVALUATION COMPLETED!")
    print(f"⏱️  Total Time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"📊 Total Evaluations: {len(all_results)}")
    print(f"✅ Successful: {len([r for r in all_results if r['success']])}")
    print(f"❌ Failed: {len([r for r in all_results if not r['success']])}")
    
    # Create summary report
    create_summary_report(all_results, output_dir)
    
    print(f"\n📁 All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
