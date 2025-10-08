#!/usr/bin/env python3
"""
🧪 Test VAD Evaluation Using Existing Predictions
Quick test to demonstrate the efficient VAD evaluation system
"""

import os
import sys
from pathlib import Path

def test_single_configuration():
    """Test a single configuration to verify the system works"""
    
    print("🧪 Testing VAD Evaluation Using Existing Predictions")
    print("=" * 50)
    
    # Example configuration - using word_eval method
    test_config = {
        'results_dir': 'fixed_smart_parallel_results',
        'eval_type': 'eval_by_0.05',
        'method': 'word_eval',
        'window_size': 0.5,
        'stride_value': 0.25,
        'stride_type': 'absolute',
        'ground_truth': 'csv/eval_5labels.csv',
        'output_dir': 'test_vad_existing_results'
    }
    
    print("📋 Test Configuration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    
    # Check if files exist
    results_path = Path(test_config['results_dir'])
    gt_path = Path(test_config['ground_truth'])
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_path}")
        return False
    
    if not gt_path.exists():
        print(f"❌ Ground truth file not found: {gt_path}")
        return False
    
    # Check specific prediction path
    pred_path = (results_path / test_config['eval_type'] / test_config['eval_type'] / 
                test_config['method'] / f"window_{test_config['window_size']}s" / 
                f"stride_{test_config['stride_value']}s")
    
    print(f"\n🔍 Checking prediction path: {pred_path}")
    
    if not pred_path.exists():
        print(f"❌ Prediction path not found: {pred_path}")
        print("📁 Available paths:")
        base_path = results_path / test_config['eval_type'] / test_config['eval_type'] / test_config['method']
        if base_path.exists():
            for window_dir in base_path.iterdir():
                if window_dir.is_dir():
                    print(f"  {window_dir.name}/")
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir():
                            print(f"    {stride_dir.name}/")
        return False
    
    # Import and run evaluation
    try:
        from vad_evaluation_from_existing import process_vad_evaluation_from_existing
        
        print(f"\n🚀 Running VAD Evaluation...")
        
        success = process_vad_evaluation_from_existing(
            results_dir=test_config['results_dir'],
            eval_type=test_config['eval_type'],
            evaluation_method=test_config['method'],
            window_size=test_config['window_size'],
            stride_value=test_config['stride_value'],
            stride_type=test_config['stride_type'],
            ground_truth_file=test_config['ground_truth'],
            output_dir=test_config['output_dir']
        )
        
        if success:
            print(f"\n🎉 Test Successful!")
            print(f"📁 Results saved to: {test_config['output_dir']}")
            
            # Show output files
            output_path = Path(test_config['output_dir'])
            if output_path.exists():
                print(f"📄 Output files:")
                for file in output_path.iterdir():
                    if file.is_file():
                        print(f"  {file.name}")
            
            return True
        else:
            print(f"❌ Test failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show practical usage examples"""
    
    print("\n📖 USAGE EXAMPLES")
    print("=" * 50)
    
    print("\n1️⃣ Single Configuration (Command Line):")
    print("python vad_evaluation_from_existing.py \\")
    print("  --results_dir fixed_smart_parallel_results \\")
    print("  --eval_type eval_by_0.05 \\")
    print("  --method word_eval \\")
    print("  --window_size 0.5 \\")
    print("  --stride_value 0.25 \\")
    print("  --stride_type absolute \\")
    print("  --ground_truth csv/eval_5labels.csv \\")
    print("  --output_dir vad_results/single_test")
    
    print("\n2️⃣ Batch Processing (Multiple Configurations):")
    print("python batch_vad_from_existing.py \\")
    print("  --eval_types eval_by_0.05 eval_percent \\")
    print("  --methods word_eval \\")
    print("  --max_workers 4 \\")
    print("  --output_dir vad_batch_results")
    
    print("\n3️⃣ Limited Test (First 10 configurations):")
    print("python batch_vad_from_existing.py \\")
    print("  --methods word_eval \\")
    print("  --limit 10 \\")
    print("  --output_dir vad_test_results")
    
    print("\n📊 Available Methods:")
    print("  - word_eval: Word-level evaluation (recommended)")
    print("  - window_eval: Window-level evaluation")
    print("  - iou_eval: IoU-based evaluation")
    print("  - word_iou_eval: Combined word-IoU evaluation")
    
    print("\n📁 Expected Output Structure:")
    print("  vad_results/")
    print("    eval_by_0.05/")
    print("      word/")
    print("        window_0.5s/")
    print("          stride_0.25s/")
    print("            note.txt                    # Complete evaluation report")
    print("            detailed_results.csv       # Per-sample predictions")
    print("            iou_threshold_analysis.csv # Threshold performance")

if __name__ == "__main__":
    print("🎯 VAD Evaluation Using Existing Predictions - Test & Examples")
    print("=" * 60)
    
    # Run test
    success = test_single_configuration()
    
    # Show usage examples
    show_usage_examples()
    
    print(f"\n{'✅ READY TO USE!' if success else '❌ SETUP REQUIRED'}")
    
    sys.exit(0 if success else 1)