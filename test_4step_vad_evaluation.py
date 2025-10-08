#!/usr/bin/env python3
"""
🧪 Test 4-Step Advanced VAD Evaluation System
Demo script showing how to use the enhanced VAD evaluation with IoU threshold analysis
"""

import os
import sys
from pathlib import Path
import argparse

def test_4step_evaluation():
    """Test the 4-step evaluation process"""
    
    # Example configuration - adjust paths as needed
    config = {
        'model_path': 'models/your_trained_model',  # Path to your trained model
        'csv_file': 'csv/test_data.csv',           # Your windowed test data
        'ground_truth': 'dataset/ground_truth.csv', # Ground truth labels
        'output_dir': 'vad_4step_results',         # Output directory
        'window_size': 0.5,                        # 0.5s window (balanced for word detection)
        'stride_value': 50.0,                      # 50% overlap (0.25s stride)
        'stride_type': 'percentage',               # Percentage-based stride
        'eval_type': '4step_advanced'              # Evaluation type identifier
    }
    
    print("🚀 Testing 4-Step Advanced VAD Evaluation")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [config['csv_file'], config['ground_truth']]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("\n🔧 Please ensure you have:")
        print("   - CSV file with windowed test data")
        print("   - Ground truth CSV with start_time, end_time, label columns")
        print("   - Trained model directory")
        return False
    
    # Import and run evaluation
    try:
        from vad_evaluation_advanced import process_single_configuration
        
        print("🎯 Running 4-Step Evaluation Process...")
        print(f"   Step 1: Advanced preprocessing")
        print(f"   Step 2: Window detection (window={config['window_size']}s)")
        print(f"   Step 3: VAD temporal refinement")  
        print(f"   Step 4: IoU threshold analysis (0.1-0.9)")
        print()
        
        success = process_single_configuration(
            model_path=config['model_path'],
            csv_file=config['csv_file'],
            ground_truth_file=config['ground_truth'],
            output_dir=config['output_dir'],
            window_size=config['window_size'],
            stride_value=config['stride_value'],
            stride_type=config['stride_type'],
            eval_type=config['eval_type']
        )
        
        if success:
            print("\n🎉 4-Step Evaluation Completed Successfully!")
            print(f"📁 Results saved to: {config['output_dir']}")
            print("📊 Output files:")
            print("   - note.txt: Comprehensive evaluation report")
            print("   - detailed_results.csv: Per-sample predictions with IoU")
            print("   - iou_threshold_analysis.csv: Threshold performance matrix")
            
            # Show sample results
            results_path = Path(config['output_dir'])
            if (results_path / 'note.txt').exists():
                print("\n📋 Sample Results Preview:")
                with open(results_path / 'note.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Show first 20 lines
                    for line in lines[:20]:
                        print(f"   {line.rstrip()}")
                    if len(lines) > 20:
                        print(f"   ... ({len(lines)-20} more lines)")
        else:
            print("❌ Evaluation failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure vad_evaluation_advanced.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False
    
    return True

def main():
    """Main function with command line support"""
    parser = argparse.ArgumentParser(description="Test 4-Step Advanced VAD Evaluation")
    parser.add_argument("--model_path", help="Path to trained model directory")
    parser.add_argument("--csv_file", help="CSV file with windowed test data")
    parser.add_argument("--ground_truth", help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="vad_4step_test_results", help="Output directory")
    parser.add_argument("--window_size", type=float, default=0.5, help="Window size in seconds")
    parser.add_argument("--stride_value", type=float, default=50.0, help="Stride value")
    parser.add_argument("--stride_type", default="percentage", choices=['percentage', 'absolute'], help="Stride type")
    
    args = parser.parse_args()
    
    if args.model_path and args.csv_file and args.ground_truth:
        # Use command line arguments
        from vad_evaluation_advanced import process_single_configuration
        
        success = process_single_configuration(
            model_path=args.model_path,
            csv_file=args.csv_file,
            ground_truth_file=args.ground_truth,
            output_dir=args.output_dir,
            window_size=args.window_size,
            stride_value=args.stride_value,
            stride_type=args.stride_type,
            eval_type='4step_advanced_cli'
        )
        
        return 0 if success else 1
    else:
        # Run test with default configuration
        success = test_4step_evaluation()
        return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)