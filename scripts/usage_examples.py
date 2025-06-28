#!/usr/bin/env python3
"""
Simple usage examples for the profanity detection model.
"""

from predict_single_file import ProfanityPredictor
import os

def example_usage():
    """Show example usage of the predictor."""
    
    # Initialize predictor
    predictor = ProfanityPredictor()
    
    # Example 1: Single file prediction with single model
    print("Example 1: Single model prediction")
    print("-" * 40)
    
    # Replace with your actual audio file path
    audio_file = "./eval/กู.wav"  # Example file from your eval folder
    
    if os.path.exists(audio_file):
        result = predictor.predict(audio_file, return_confidence=True)
        
        print(f"File: {audio_file}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Is Profanity: {result['is_profanity']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("\nAll probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.1%}")
    else:
        print(f"Audio file not found: {audio_file}")
    
    print("\n" + "="*50)
    
    # Example 2: Ensemble prediction
    print("Example 2: Ensemble prediction")
    print("-" * 40)
    
    if os.path.exists(audio_file):
        result = predictor.predict(audio_file, use_ensemble=True, return_confidence=True)
        
        print(f"File: {audio_file}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Is Profanity: {result['is_profanity']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Uncertainty: {result['uncertainty']:.4f}")
        print(f"Ensemble Size: {result['ensemble_size']} models")
    
    print("\n" + "="*50)
    
    # Example 3: Batch processing multiple files
    print("Example 3: Batch processing")
    print("-" * 40)
    
    # List some example files from your eval directory
    eval_dir = "./eval"
    if os.path.exists(eval_dir):
        audio_files = [f for f in os.listdir(eval_dir) if f.endswith('.wav')][:5]  # First 5 files
        
        for audio_file in audio_files:
            file_path = os.path.join(eval_dir, audio_file)
            try:
                result = predictor.predict(file_path)
                status = "🚨 PROFANITY" if result['is_profanity'] else "✅ CLEAN"
                print(f"{audio_file}: {status} ({result['predicted_class']}, {result['confidence']:.1%})")
            except Exception as e:
                print(f"{audio_file}: Error - {e}")
    else:
        print("Eval directory not found")

def test_specific_file():
    """Test a specific file if provided via command line."""
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        predictor = ProfanityPredictor()
        
        try:
            # Single model prediction
            result_single = predictor.predict(audio_file, return_confidence=True)
            
            print("Single Model Result:")
            print(f"  Prediction: {result_single['predicted_class']}")
            print(f"  Confidence: {result_single['confidence']:.1%}")
            print(f"  Is Profanity: {result_single['is_profanity']}")
            
            # Ensemble prediction
            result_ensemble = predictor.predict(audio_file, use_ensemble=True, return_confidence=True)
            
            print("\nEnsemble Result:")
            print(f"  Prediction: {result_ensemble['predicted_class']}")
            print(f"  Confidence: {result_ensemble['confidence']:.1%}")
            print(f"  Uncertainty: {result_ensemble['uncertainty']:.4f}")
            print(f"  Is Profanity: {result_ensemble['is_profanity']}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_specific_file()
    else:
        example_usage()
