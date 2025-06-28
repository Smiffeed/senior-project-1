#!/usr/bin/env python3
"""
Test script for the windowed evaluation functionality.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

def test_enhanced_window_eval():
    """Test the enhanced windowed evaluation."""
    print("=== Testing Enhanced Windowed Evaluation ===")
    
    # Check if model directory exists
    model_dir = "./models/simplified_advanced_audio_train"
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Please train a model first or adjust the model path")
        return False
    
    # Check if evaluation CSV exists
    eval_csv = "./csv/eval.csv"
    if not os.path.exists(eval_csv):
        print(f"Evaluation CSV not found: {eval_csv}")
        print("Please ensure the evaluation data is available")
        return False
    
    try:
        from enhanced_window_eval import EnhancedWindowedEvaluator
        import torch
        
        # Test initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        evaluator = EnhancedWindowedEvaluator(model_dir, device)
        print("✓ Enhanced evaluator initialized successfully")
        
        # Test on a small subset
        output_dir = "./test_enhanced_window_results"
        print(f"Running test evaluation, results will be saved to: {output_dir}")
        
        accuracy, results_df = evaluator.evaluate_dataset(eval_csv, output_dir)
        
        if results_df is not None:
            print(f"✓ Test evaluation completed successfully!")
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ Total windows processed: {len(results_df)}")
            return True
        else:
            print("✗ Test evaluation failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during enhanced window evaluation test: {e}")
        return False

def test_advanced_window_eval():
    """Test the advanced windowed evaluation (if available)."""
    print("\n=== Testing Advanced Windowed Evaluation ===")
    
    try:
        from advanced_window_eval import AdvancedWindowedEvaluator
        import torch
        
        model_dir = "./models/simplified_advanced_audio_train"
        eval_csv = "./csv/eval.csv"
        
        if not os.path.exists(model_dir) or not os.path.exists(eval_csv):
            print("Required files not found, skipping advanced test")
            return False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = AdvancedWindowedEvaluator(model_dir, device)
        print("✓ Advanced evaluator initialized successfully")
        
        # Test on a small subset
        output_dir = "./test_advanced_window_results"
        print(f"Running test evaluation, results will be saved to: {output_dir}")
        
        accuracy, results_df = evaluator.evaluate_dataset(eval_csv, output_dir)
        
        if results_df is not None:
            print(f"✓ Test evaluation completed successfully!")
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ Total windows processed: {len(results_df)}")
            return True
        else:
            print("✗ Test evaluation failed")
            return False
            
    except ImportError as e:
        print(f"Advanced components not available: {e}")
        print("This is expected if ultimate_model_training.py components are not available")
        return False
    except Exception as e:
        print(f"✗ Error during advanced window evaluation test: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting windowed evaluation tests...\n")
    
    # Test enhanced windowed evaluation
    enhanced_success = test_enhanced_window_eval()
    
    # Test advanced windowed evaluation  
    advanced_success = test_advanced_window_eval()
    
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"Enhanced Windowed Evaluation: {'✓ PASS' if enhanced_success else '✗ FAIL'}")
    print(f"Advanced Windowed Evaluation: {'✓ PASS' if advanced_success else '✗ FAIL'}")
    
    if enhanced_success:
        print("\n🎉 At least one windowed evaluation method is working!")
        print("\nUsage examples:")
        print("python enhanced_window_eval.py --model_dir ./models/simplified_advanced_audio_train --eval_csv ./csv/eval.csv")
        if advanced_success:
            print("python advanced_window_eval.py --model_dir ./models/simplified_advanced_audio_train --eval_csv ./csv/eval.csv")
    else:
        print("\n❌ No windowed evaluation methods are working.")
        print("Please check your model and data paths.")

if __name__ == "__main__":
    main()
