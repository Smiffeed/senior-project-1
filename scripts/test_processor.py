#!/usr/bin/env python3
"""
Test script to verify comprehensive evaluation processor functionality
"""

import sys
import os

def test_import():
    """Test if we can import the module"""
    try:
        print("Attempting to import comprehensive_evaluation_processor...")
        import comprehensive_evaluation_processor
        print("✓ Import successful")
        
        print("Attempting to import ComprehensiveEvaluationProcessor class...")
        from comprehensive_evaluation_processor import ComprehensiveEvaluationProcessor
        print("✓ Class import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_instantiation():
    """Test if we can create an instance"""
    try:
        from comprehensive_evaluation_processor import ComprehensiveEvaluationProcessor
        
        print("Creating ComprehensiveEvaluationProcessor instance...")
        processor = ComprehensiveEvaluationProcessor(
            model_path="dummy",
            gt_csv_path="dummy.csv",
            audio_dir="dummy",
            output_dir="dummy"
        )
        print("✓ Instance creation successful")
        return True
    except Exception as e:
        print(f"✗ Instance creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Comprehensive Evaluation Processor ===")
    
    if test_import():
        if test_instantiation():
            print("✓ All basic tests passed")
        else:
            print("✗ Instantiation test failed")
    else:
        print("✗ Import test failed")
