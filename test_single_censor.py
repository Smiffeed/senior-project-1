"""
Quick Single File Audio Censoring Test

This is the easiest way to test censoring on a single audio file.
Just run this script and follow the prompts!
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime

# Add the scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.simplified_ultimate_training import EnhancedAudioClassifier, SimpleAudioPreprocessor
    from transformers import Wav2Vec2FeatureExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Constants
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

def quick_censor_test(input_file, method='silence', threshold=0.7):
    """Quick test of audio censoring on a single file."""
    
    print(f"🎵 Testing Audio Censoring")
    print(f"📁 Input file: {input_file}")
    print(f"🔇 Method: {method}")
    print(f"📊 Threshold: {threshold}")
    print("-" * 50)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return None
    
    # Model setup
    model_dir = './models/simplified_advanced_audio_train'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    
    print("🤖 Loading model...")
    try:
        # Load feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        
        # Load preprocessor
        preprocessor = SimpleAudioPreprocessor()
        
        # Load trained model
        model_path = os.path.join(model_dir, 'fold_1', 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        model = EnhancedAudioClassifier(model_name, NUM_LABELS)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load audio
    print("🎵 Loading audio...")
    try:
        audio, sr = librosa.load(input_file, sr=16000)
        audio_duration = len(audio) / sr
        print(f"✅ Audio loaded: {audio_duration:.2f} seconds")
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None
    
    # Detection parameters
    window_size = 0.25  # seconds
    hop_length = 0.125  # seconds
    
    print("🔍 Detecting profanities...")
    detections = []
    censored_audio = audio.copy()
    
    # Sliding window detection and censoring
    audio_length = len(audio) / sr
    window_starts = np.arange(0, audio_length - window_size, hop_length)
    
    detection_count = 0
    for window_start in window_starts:
        window_end = window_start + window_size
        
        # Extract segment
        start_sample = int(window_start * sr)
        end_sample = int(window_end * sr)
        audio_segment = audio[start_sample:end_sample]
        
        # Pad if needed
        if len(audio_segment) < int(window_size * sr):
            audio_segment = np.pad(audio_segment, 
                                 (0, int(window_size * sr) - len(audio_segment)), 
                                 'constant')
        
        # Preprocess and predict
        try:
            # Skip preprocessing for now - use raw audio segment
            inputs = feature_extractor(
                audio_segment, sampling_rate=sr, return_tensors="pt", padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(outputs['logits'], dim=-1).item()
                confidence = probs.max().item()
            
            # Check if profanity detected with sufficient confidence
            if prediction != 0 and confidence >= threshold:
                detection_count += 1
                class_name = CLASS_NAMES[prediction]
                
                detections.append({
                    'time': f"{window_start:.2f}s - {window_end:.2f}s",
                    'class': class_name,
                    'confidence': confidence
                })
                
                print(f"🚨 Detection #{detection_count}: {class_name} at {window_start:.2f}s (confidence: {confidence:.3f})")
                
                # Apply censoring
                if method == 'silence':
                    censored_audio[start_sample:end_sample] = 0
                elif method == 'beep':
                    duration = (end_sample - start_sample) / sr
                    t = np.linspace(0, duration, end_sample - start_sample)
                    beep = 0.3 * np.sin(2 * np.pi * 1000 * t)
                    censored_audio[start_sample:end_sample] = beep
                elif method == 'noise':
                    noise_length = end_sample - start_sample
                    noise = 0.1 * np.random.normal(0, 1, noise_length)
                    censored_audio[start_sample:end_sample] = noise
                    
        except Exception as e:
            print(f"⚠️ Warning: Error processing window at {window_start:.2f}s: {e}")
            continue
    
    # Save results
    print(f"\n📊 Results:")
    print(f"   Total detections: {len(detections)}")
    
    if detections:
        print(f"   Detected profanities:")
        for detection in detections:
            print(f"     {detection['time']}: {detection['class']} (confidence: {detection['confidence']:.3f})")
        
        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_censored_{method}.wav"
        
        # Save censored audio
        try:
            sf.write(output_file, censored_audio, sr)
            print(f"✅ Censored audio saved to: {output_file}")
            
            # Also save original for comparison
            comparison_file = f"{base_name}_original.wav"
            sf.write(comparison_file, audio, sr)
            print(f"📄 Original saved to: {comparison_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    else:
        print("✨ No profanities detected - audio is clean!")
        return input_file

def main():
    """Main function for interactive testing."""
    print("=" * 60)
    print("🎵 SINGLE AUDIO FILE CENSORING TEST 🎵")
    print("=" * 60)
    
    # Show available test files
    eval_dir = './eval'
    if os.path.exists(eval_dir):
        print("📂 Available test files in ./eval/:")
        audio_files = [f for f in os.listdir(eval_dir) if f.endswith('.wav')]
        for i, file in enumerate(audio_files[:10], 1):  # Show first 10 files
            print(f"   {i}. {file}")
        if len(audio_files) > 10:
            print(f"   ... and {len(audio_files) - 10} more files")
        print()
    
    # Get user input
    print("🎯 Choose your test:")
    print("1. Quick test with recommended file (has profanity)")
    print("2. Enter custom file path")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        # Use a file that likely contains profanity
        test_file = './eval/ถามจริง กูจะรั่ว.wav'
        if not os.path.exists(test_file):
            # Fallback to first available file
            if audio_files:
                test_file = os.path.join(eval_dir, audio_files[0])
            else:
                print("❌ No test files found in ./eval/")
                return
    else:
        test_file = input("Enter audio file path: ").strip()
    
    # Choose method
    print("\n🔇 Choose censoring method:")
    print("1. Silence (replace with silence)")
    print("2. Beep (replace with beep tone)")
    print("3. Noise (replace with white noise)")
    
    method_choice = input("Enter choice (1, 2, or 3) [1]: ").strip()
    method_map = {'1': 'silence', '2': 'beep', '3': 'noise'}
    method = method_map.get(method_choice, 'silence')
    
    # Choose threshold
    threshold = input("Enter confidence threshold (0.0-1.0) [0.7]: ").strip()
    try:
        threshold = float(threshold) if threshold else 0.7
    except ValueError:
        threshold = 0.7
    
    # Run the test
    print(f"\n🚀 Starting censoring test...")
    result = quick_censor_test(test_file, method, threshold)
    
    if result:
        print(f"\n🎉 Test completed successfully!")
        print(f"🎵 Play the censored audio: {result}")
        print(f"🔍 Compare with original to hear the difference")
    else:
        print(f"\n❌ Test failed - check error messages above")

if __name__ == "__main__":
    main()
