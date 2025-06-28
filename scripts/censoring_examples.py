"""
Audio Censoring Usage Examples

This script demonstrates how to use the audio censoring functionality
based on the comprehensive evaluation method from comprehensive_evaluation.py
"""

import os
import sys
sys.path.append('.')

from scripts.audio_censoring import AudioCensor
from scripts.simple_censor import SimpleCensor

def example_single_file_censoring():
    """Example of censoring a single audio file."""
    print("=" * 60)
    print("SINGLE FILE CENSORING EXAMPLE")
    print("=" * 60)
    
    # Initialize the censoring system
    model_dir = './models/simplified_advanced_audio_train'
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Please make sure you have trained models in the correct directory.")
        return
    
    # Example input file (you can change this to your audio file)
    input_file = 'test.wav'
    
    if not os.path.exists(input_file):
        print(f"Example audio file not found: {input_file}")
        print("Please specify a valid audio file path.")
        return
    
    try:
        # Method 1: Using SimpleCensor (recommended for single files)
        print("Using SimpleCensor...")
        censor = SimpleCensor(model_dir)
        
        # Test different censoring methods
        methods = ['silence', 'beep', 'noise']
        
        for method in methods:
            print(f"\n--- Censoring with {method} method ---")
            output_file = f'./output/example_censored_{method}.wav'
            
            # Create output directory if it doesn't exist
            os.makedirs('./output', exist_ok=True)
            
            # Process the file
            result_file, detections = censor.detect_and_censor(
                input_file, 
                output_file, 
                method=method, 
                threshold=0.7
            )
            
            print(f"Result saved to: {result_file}")
            print(f"Found {len(detections)} profanity segments")
            
    except Exception as e:
        print(f"Error in single file censoring: {e}")

def example_batch_processing():
    """Example of processing multiple files."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    model_dir = './models/simplified_advanced_audio_train'
    
    try:
        # Method 2: Using AudioCensor for batch processing
        print("Using AudioCensor for batch processing...")
        censor = AudioCensor(model_dir)
        censor.load_model(fold_num=1, stage='best')
        
        # Process all files in eval directory
        input_directory = './eval'
        
        if not os.path.exists(input_directory):
            print(f"Input directory not found: {input_directory}")
            return
        
        # Process with silence censoring
        print(f"Processing all audio files in: {input_directory}")
        results = censor.process_directory(
            input_directory,
            output_dir='./output/batch_censored',
            censor_method='silence'
        )
        
        if results:
            print(f"\nBatch processing complete!")
            print(f"Files processed: {len(results)}")
            total_detections = sum(r['total_detections'] for r in results)
            print(f"Total profanity segments found: {total_detections}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")

def example_custom_settings():
    """Example of using custom settings for censoring."""
    print("\n" + "=" * 60)
    print("CUSTOM SETTINGS EXAMPLE")
    print("=" * 60)
    
    model_dir = './models/simplified_advanced_audio_train'
    input_file = './eval/กูตั้งใจจะเรียนให้จบปีนี้.wav'
    
    if not os.path.exists(input_file):
        print(f"Example audio file not found: {input_file}")
        return
    
    try:
        # Initialize with custom settings
        censor = AudioCensor(model_dir)
        censor.load_model(fold_num=1, stage='best')
        
        # Customize detection parameters
        censor.window_size = 0.3  # Smaller windows for more precise detection
        censor.hop_length = 0.15  # More overlap
        censor.confidence_threshold = 0.8  # Higher confidence requirement
        
        print("Using custom settings:")
        print(f"  Window size: {censor.window_size}s")
        print(f"  Hop length: {censor.hop_length}s")
        print(f"  Confidence threshold: {censor.confidence_threshold}")
        
        # Process with custom settings
        result = censor.process_audio_file(
            input_file,
            output_file='./output/custom_censored.wav',
            censor_method='beep',
            save_report=True,
            merge_detections=True
        )
        
        if result:
            print(f"\nCustom processing complete!")
            print(f"Detections: {result['total_detections']}")
            print(f"Censored duration: {result['total_censored_duration']:.2f}s")
        
    except Exception as e:
        print(f"Error in custom settings example: {e}")

def command_line_example():
    """Show command line usage examples."""
    print("\n" + "=" * 60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 60)
    
    print("You can also use the censoring tools from command line:")
    print()
    print("1. Simple censoring:")
    print("   python scripts/simple_censor.py input.wav")
    print()
    print("2. With custom output file:")
    print("   python scripts/simple_censor.py input.wav -o output_censored.wav")
    print()
    print("3. With beep censoring:")
    print("   python scripts/simple_censor.py input.wav -m beep")
    print()
    print("4. With custom threshold:")
    print("   python scripts/simple_censor.py input.wav -t 0.8")
    print()
    print("5. Interactive mode (no arguments):")
    print("   python scripts/simple_censor.py")

def main():
    """Run all examples."""
    print("Audio Censoring Examples")
    print("Based on comprehensive_evaluation.py detection method")
    print()
    
    # Run examples
    example_single_file_censoring()
    example_batch_processing()
    example_custom_settings()
    command_line_example()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("Check the ./output directory for censored audio files")

if __name__ == "__main__":
    main()
