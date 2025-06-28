#!/usr/bin/env python3
"""
Quick test script for the profanity detection models.
"""

import sys
import os
sys.path.append('./scripts')

try:
    from simple_detector import SimpleProfanityDetector
    
    print("Testing Simple Profanity Detector...")
    print("-" * 40)
    
    # Initialize
    detector = SimpleProfanityDetector()
    print("✅ Detector initialized successfully")
    
    # Test with an audio file
    test_file = "./eval/กฤต.wav"
    if os.path.exists(test_file):
        print(f"Testing with: {test_file}")
        
        try:
            result = detector.detect(test_file)
            
            print(f"✅ Prediction successful!")
            print(f"   Is profanity: {result['is_profanity']}")
            print(f"   Class: {result['class']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            
            if result['is_profanity']:
                print(f"   Profanity type: {result['profanity_type']}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    else:
        print(f"❌ Test file not found: {test_file}")
    
    print("\n" + "="*50)
    print("SUCCESS: Your model is ready to use!")
    print("See USAGE_GUIDE.md for detailed instructions.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
