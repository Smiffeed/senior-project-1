#!/usr/bin/env python3
"""
Test Script for Advanced Frame-Level Evaluator
Demonstrates how to use the advanced evaluation system with different configurations
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-1000:])  # Last 1000 chars
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
        "advanced_frame_level_evaluator.py",
        "models/4_classes_max_steps",  # Model directory
        "csv/eval_5labels.csv",        # Ground truth
        "dataset"                      # Audio directory
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

def test_basic_evaluation():
    """Test basic evaluation with default parameters"""
    cmd = [
        sys.executable, "advanced_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv", 
        "--audio_dir", "dataset",
        "--output_dir", "test_results/basic_evaluation",
        "--max_files", "5",  # Limit for testing
        "--workers", "2"
    ]
    
    return run_command(cmd, "Basic Evaluation Test (5 files, 2 workers)")

def test_high_precision_evaluation():
    """Test evaluation with high precision settings"""
    cmd = [
        sys.executable, "advanced_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv",
        "--audio_dir", "dataset", 
        "--output_dir", "test_results/high_precision_evaluation",
        "--window_size", "0.3",      # Smaller window for precision
        "--stride", "0.15",          # Smaller stride for coverage
        "--confidence_threshold", "0.7",  # Higher confidence
        "--max_files", "3",
        "--workers", "1"
    ]
    
    return run_command(cmd, "High Precision Evaluation (0.3s window, 0.7 confidence)")

def test_fast_evaluation():
    """Test evaluation with fast settings"""
    cmd = [
        sys.executable, "advanced_frame_level_evaluator.py",
        "--model_dir", "models/4_classes_max_steps",
        "--ground_truth", "csv/eval_5labels.csv",
        "--audio_dir", "dataset",
        "--output_dir", "test_results/fast_evaluation", 
        "--window_size", "1.0",      # Larger window for speed
        "--stride", "0.8",           # Larger stride for speed
        "--confidence_threshold", "0.3",  # Lower confidence for coverage
        "--max_files", "5",
        "--workers", "4"             # More workers
    ]
    
    return run_command(cmd, "Fast Evaluation (1.0s window, 0.8s stride, 4 workers)")

def analyze_results(results_dir):
    """Analyze and display results"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Look for summary and config files
    summary_file = results_path / "evaluation_summary.csv"
    config_file = results_path / "evaluation_config.json"
    
    print(f"\n📊 ANALYZING RESULTS: {results_dir}")
    print("=" * 60)
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print("Configuration:")
            print(f"   Window Size: {config.get('window_size', 'N/A')}s")
            print(f"   Stride: {config.get('stride', 'N/A')}s")
            print(f"   Confidence Threshold: {config.get('confidence_threshold', 'N/A')}")
            print(f"   Workers: {config.get('workers', 'N/A')}")
            print(f"   Total Files: {config.get('total_files', 'N/A')}")
            print(f"   Successful: {config.get('successful_files', 'N/A')}")
            print(f"   Failed: {config.get('failed_files', 'N/A')}")
            print(f"   Total Time: {config.get('total_time_seconds', 0):.1f}s")
            print(f"   Avg Time/File: {config.get('average_time_per_file', 0):.1f}s")
            
        except Exception as e:
            print(f"⚠️  Could not read config file: {e}")
    
    if summary_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(summary_file)
            
            print(f"\nSummary Statistics:")
            print(f"   Files Processed: {len(df)}")
            
            # Check for successful evaluations
            successful_df = df[df.get('success', True) != False]
            if len(successful_df) > 0:
                mean_iou = successful_df['mean_iou'].mean() if 'mean_iou' in successful_df.columns else 0
                binary_f1 = successful_df['binary_f1'].mean() if 'binary_f1' in successful_df.columns else 0
                multiclass_f1 = successful_df['multiclass_f1'].mean() if 'multiclass_f1' in successful_df.columns else 0
                
                print(f"   Average Mean IoU: {mean_iou:.3f}")
                print(f"   Average Binary F1: {binary_f1:.3f}")
                print(f"   Average Multiclass F1: {multiclass_f1:.3f}")
                
                # Show top performing files
                if 'binary_f1' in successful_df.columns:
                    top_files = successful_df.nlargest(3, 'binary_f1')
                    print(f"\n   Top 3 Files (Binary F1):")
                    for _, row in top_files.iterrows():
                        print(f"      {row['audio_file']}: {row['binary_f1']:.3f}")
            else:
                print("   No successful evaluations found")
                
        except Exception as e:
            print(f"⚠️  Could not analyze summary file: {e}")
    else:
        print("⚠️  Summary file not found")

def main():
    parser = argparse.ArgumentParser(description="Test Advanced Frame-Level Evaluator")
    parser.add_argument("--test", choices=['basic', 'precision', 'fast', 'all'], 
                       default='basic', help="Which test to run")
    parser.add_argument("--analyze_only", action="store_true", 
                       help="Only analyze existing results")
    
    args = parser.parse_args()
    
    print("🎯 ADVANCED FRAME-LEVEL EVALUATOR TEST SUITE")
    print("=" * 60)
    
    if not args.analyze_only:
        # Check requirements
        if not check_requirements():
            print("\n❌ Requirements check failed. Please ensure required files exist.")
            return 1
        
        # Run tests
        test_results = {}
        
        if args.test in ['basic', 'all']:
            test_results['basic'] = test_basic_evaluation()
        
        if args.test in ['precision', 'all']:
            test_results['precision'] = test_high_precision_evaluation()
        
        if args.test in ['fast', 'all']:
            test_results['fast'] = test_fast_evaluation()
        
        # Summary of test results
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, success in test_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name.capitalize()} Test: {status}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 RESULTS ANALYSIS")
    print("=" * 60)
    
    if args.test in ['basic', 'all']:
        analyze_results("test_results/basic_evaluation")
    
    if args.test in ['precision', 'all']:
        analyze_results("test_results/high_precision_evaluation")
    
    if args.test in ['fast', 'all']:
        analyze_results("test_results/fast_evaluation")
    
    print("\n🎉 Test suite completed!")
    print("Check the test_results/ directory for detailed outputs.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())