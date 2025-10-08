#!/usr/bin/env python3
"""
Test Script for Dataset-Based Advanced Frame-Level Evaluator
Tests the evaluator with pre-computed windowed datasets from eval_by_0.05 and eval_percent
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json
import pandas as pd

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            # Show last part of stdout for key information
            lines = result.stdout.strip().split('\n')
            print("KEY OUTPUT:")
            for line in lines[-10:]:  # Last 10 lines
                if any(keyword in line.lower() for keyword in ['✅', '❌', '🎉', '📊', 'f1', 'iou', 'success', 'failed']):
                    print(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED (Exit code: {e.returncode})")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-500:])
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-500:])
        return False

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "dataset_frame_level_evaluator.py",
        "models/4_classes_max_steps",  # Model directory
        "csv/eval_5labels.csv",        # Ground truth
        "csv/eval_by_0.05",            # Dataset directory 1
        "csv/eval_percent"             # Dataset directory 2
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def test_list_configurations():
    """Test listing available configurations"""
    cmd = [
        sys.executable, "dataset_frame_level_evaluator.py",
        "--list_configs"
    ]
    
    return run_command(cmd, "List Available Configurations")

def test_basic_evaluation():
    """Test basic evaluation with limited configurations"""
    cmd = [
        sys.executable, "dataset_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv",
        "--datasets", "csv/eval_by_0.05", "csv/eval_percent",
        "--output_dir", "test_results/dataset_basic_evaluation",
        "--max_configs", "3",  # Limit for testing
        "--workers", "2"
    ]
    
    return run_command(cmd, "Basic Dataset Evaluation (3 configs, 2 workers)")

def test_specific_windows():
    """Test evaluation with specific window sizes"""
    cmd = [
        sys.executable, "dataset_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv",
        "--datasets", "csv/eval_by_0.05",
        "--output_dir", "test_results/dataset_specific_windows",
        "--specific_windows", "0.5s", "1.0s",
        "--workers", "2"
    ]
    
    return run_command(cmd, "Specific Windows Evaluation (0.5s, 1.0s)")

def test_high_confidence():
    """Test evaluation with high confidence threshold"""
    cmd = [
        sys.executable, "dataset_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv",
        "--datasets", "csv/eval_percent",
        "--output_dir", "test_results/dataset_high_confidence",
        "--confidence_threshold", "0.7",
        "--max_configs", "2",
        "--workers", "1"
    ]
    
    return run_command(cmd, "High Confidence Evaluation (threshold=0.7)")

def test_mode_evaluation():
    """Test evaluation in test mode (now optimized with window 2.0s)"""
    cmd = [
        sys.executable, "dataset_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv",
        "--datasets", "csv/eval_by_0.05", "csv/eval_percent",
        "--output_dir", "test_results/dataset_test_mode",
        "--test_mode",
        "--workers", "3"
    ]
    
    success = run_command(cmd, "Test Mode Evaluation (optimized with window 2.0s)")
    
    # Check if note.txt files were generated
    test_output_dir = Path("test_results/dataset_test_mode")
    if test_output_dir.exists():
        note_files = list(test_output_dir.glob("*/note.txt"))
        print(f"📝 Found {len(note_files)} note.txt files")
        if note_files:
            print("   Sample note.txt files:")
            for note_file in note_files[:2]:  # Show first 2
                print(f"   - {note_file.name}")
                try:
                    with open(note_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        print(f"     {first_line}")
                except Exception as e:
                    print(f"     Error reading: {e}")
    
    return success

def analyze_results(results_dir):
    """Analyze and display results"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Look for summary and report files
    summary_file = results_path / "batch_evaluation_summary.csv"
    report_file = results_path / "batch_evaluation_report.json"
    
    print(f"\n📊 ANALYZING RESULTS: {results_dir}")
    print("=" * 60)
    
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            print("Processing Statistics:")
            print(f"   Total Configurations: {report.get('total_configurations', 'N/A')}")
            print(f"   Successful: {report.get('successful_configurations', 'N/A')}")
            print(f"   Failed: {report.get('failed_configurations', 'N/A')}")
            print(f"   Total Time: {report.get('total_duration_seconds', 0):.1f}s")
            print(f"   Speedup: {report.get('speedup', 0):.1f}x")
            print(f"   Efficiency: {report.get('efficiency_percent', 0):.1f}%")
            print(f"   Workers Used: {report.get('workers_used', 'N/A')}")
            
            settings = report.get('processing_settings', {})
            print(f"\nSettings:")
            print(f"   Confidence Threshold: {settings.get('confidence_threshold', 'N/A')}")
            print(f"   Model Dir: {settings.get('model_dir', 'N/A')}")
            print(f"   Datasets: {settings.get('datasets', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️  Could not read report file: {e}")
    
    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            
            print(f"\nSummary Statistics:")
            print(f"   Configurations Processed: {len(df)}")
            
            # Check for successful evaluations
            successful_df = df[df.get('success', True) == True]
            if len(successful_df) > 0:
                print(f"   Successful: {len(successful_df)}")
                
                # Calculate average metrics
                if 'mean_iou' in successful_df.columns:
                    mean_iou = successful_df['mean_iou'].mean()
                    print(f"   Average Mean IoU: {mean_iou:.3f}")
                
                if 'binary_f1' in successful_df.columns:
                    binary_f1 = successful_df['binary_f1'].mean()
                    print(f"   Average Binary F1: {binary_f1:.3f}")
                
                if 'multiclass_f1' in successful_df.columns:
                    multiclass_f1 = successful_df['multiclass_f1'].mean()
                    print(f"   Average Multiclass F1: {multiclass_f1:.3f}")
                
                # Show best configurations
                print(f"\n   Top 3 Configurations:")
                
                if 'binary_f1' in successful_df.columns:
                    top_configs = successful_df.nlargest(3, 'binary_f1')
                    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                        print(f"      {i}. {row['eval_type']}/{row['window']}/{row['stride']}: "
                              f"Binary F1={row['binary_f1']:.3f}, IoU={row.get('mean_iou', 0):.3f}")
            else:
                print("   No successful evaluations found")
            
            # Show failed configurations
            failed_df = df[df.get('success', True) == False]
            if len(failed_df) > 0:
                print(f"\n   Failed Configurations: {len(failed_df)}")
                for _, row in failed_df.head(3).iterrows():
                    error_msg = row.get('error', 'Unknown error')[:50]
                    print(f"      {row['eval_type']}/{row['window']}/{row['stride']}: {error_msg}...")
                
        except Exception as e:
            print(f"⚠️  Could not analyze summary file: {e}")
    else:
        print("⚠️  Summary file not found")
    
    # Check for individual configuration results
    config_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith(('eval_by_0.05', 'eval_percent'))]
    if config_dirs:
        print(f"\n   Individual Results: {len(config_dirs)} configuration directories found")
        
        # Show a sample configuration result
        sample_dir = config_dirs[0]
        sample_files = list(sample_dir.glob("*.csv")) + list(sample_dir.glob("*.json"))
        if sample_files:
            print(f"   Sample files in {sample_dir.name}: {[f.name for f in sample_files[:3]]}")

def main():
    parser = argparse.ArgumentParser(description="Test Dataset-Based Advanced Frame-Level Evaluator")
    parser.add_argument("--test", choices=['list', 'basic', 'windows', 'confidence', 'test_mode', 'all'], 
                       default='basic', help="Which test to run")
    parser.add_argument("--analyze_only", action="store_true", 
                       help="Only analyze existing results")
    
    args = parser.parse_args()
    
    print("🎯 DATASET-BASED ADVANCED FRAME-LEVEL EVALUATOR TEST SUITE")
    print("=" * 70)
    
    if not args.analyze_only:
        # Check requirements
        if not check_requirements():
            print("\n❌ Requirements check failed. Please ensure required files exist.")
            return 1
        
        # Run tests
        test_results = {}
        
        if args.test in ['list', 'all']:
            test_results['list'] = test_list_configurations()
        
        if args.test in ['basic', 'all']:
            test_results['basic'] = test_basic_evaluation()
        
        if args.test in ['windows', 'all']:
            test_results['windows'] = test_specific_windows()
        
        if args.test in ['confidence', 'all']:
            test_results['confidence'] = test_high_confidence()
        
        if args.test in ['test_mode', 'all']:
            test_results['test_mode'] = test_mode_evaluation()
        
        # Summary of test results
        print("\n" + "=" * 70)
        print("🏁 TEST SUMMARY")
        print("=" * 70)
        
        for test_name, success in test_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name.capitalize()} Test: {status}")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("📊 RESULTS ANALYSIS")
    print("=" * 70)
    
    if args.test in ['basic', 'all']:
        analyze_results("test_results/dataset_basic_evaluation")
    
    if args.test in ['windows', 'all']:
        analyze_results("test_results/dataset_specific_windows")
    
    if args.test in ['confidence', 'all']:
        analyze_results("test_results/dataset_high_confidence")
    
    if args.test in ['test_mode', 'all']:
        analyze_results("test_results/dataset_test_mode")
    
    print("\n🎉 Test suite completed!")
    print("Check the test_results/ directory for detailed outputs.")
    
    # Show usage examples
    print("\n💡 USAGE EXAMPLES:")
    print("   # List all available configurations:")
    print("   python dataset_frame_level_evaluator.py --list_configs")
    print()
    print("   # Process specific windows:")
    print("   python dataset_frame_level_evaluator.py --specific_windows 0.3s 0.5s --workers 2")
    print()
    print("   # Process eval_by_0.05 only with high confidence:")
    print("   python dataset_frame_level_evaluator.py --datasets csv/eval_by_0.05 --confidence_threshold 0.7")
    print()
    print("   # Test mode (quick run):")
    print("   python dataset_frame_level_evaluator.py --test_mode --workers 3")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())