#!/usr/bin/env python3
"""
Investigation script to compare preprocessing differences between 
VAD evaluation system and comprehensive evaluation system
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class SimplePreprocessor:
    """VAD evaluation preprocessing (vad_evaluation_single.py)"""
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def preprocess(self, audio):
        """Main preprocessing function"""
        audio = self.normalize_audio(audio)
        return audio

def advanced_preprocess_audio(audio):
    """Comprehensive evaluation preprocessing (comprehensive_evaluation_processor.py)"""
    try:
        # Apply noise reduction
        audio = librosa.effects.preemphasis(audio)
        
        # Normalize
        if len(audio) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Ensure minimum length
        min_length = int(0.1 * 16000)  # 0.1 seconds minimum
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        return audio
    except Exception as e:
        print(f"Error in advanced preprocessing: {e}")
        return audio

def compare_preprocessing_methods(test_audio_file=None):
    """Compare the two preprocessing approaches"""
    
    print("🔬 PREPROCESSING COMPARISON ANALYSIS")
    print("="*50)
    
    # Create synthetic test audio if no file provided
    if test_audio_file is None:
        print("Creating synthetic test audio...")
        duration = 1.0  # 1 second
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create mixed signal: speech-like + noise
        speech_like = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
        noise = 0.1 * np.random.randn(len(t))
        test_audio = speech_like + noise
        
        # Add some silence at beginning and end
        silence = np.zeros(int(0.1 * sr))
        test_audio = np.concatenate([silence, test_audio, silence])
        
    else:
        print(f"Loading audio from: {test_audio_file}")
        test_audio, sr = librosa.load(test_audio_file, sr=16000)
        test_audio = test_audio[:16000]  # Take first second
    
    print(f"Original audio stats:")
    print(f"  Length: {len(test_audio)} samples ({len(test_audio)/16000:.3f}s)")
    print(f"  Range: [{np.min(test_audio):.4f}, {np.max(test_audio):.4f}]")
    print(f"  RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
    print(f"  Peak: {np.max(np.abs(test_audio)):.4f}")
    
    # Apply both preprocessing methods
    simple_preprocessor = SimplePreprocessor()
    
    simple_result = simple_preprocessor.preprocess(test_audio.copy())
    advanced_result = advanced_preprocess_audio(test_audio.copy())
    
    print("\n📊 PREPROCESSING RESULTS:")
    print("-" * 30)
    
    print("Simple Preprocessing (VAD system):")
    print(f"  Length: {len(simple_result)} samples")
    print(f"  Range: [{np.min(simple_result):.4f}, {np.max(simple_result):.4f}]")
    print(f"  RMS: {np.sqrt(np.mean(simple_result**2)):.4f}")
    print(f"  Peak: {np.max(np.abs(simple_result)):.4f}")
    
    print("\nAdvanced Preprocessing (Comprehensive system):")
    print(f"  Length: {len(advanced_result)} samples")
    print(f"  Range: [{np.min(advanced_result):.4f}, {np.max(advanced_result):.4f}]")
    print(f"  RMS: {np.sqrt(np.mean(advanced_result**2)):.4f}")
    print(f"  Peak: {np.max(np.abs(advanced_result)):.4f}")
    
    # Calculate differences
    print("\n🔍 KEY DIFFERENCES:")
    print("-" * 20)
    
    length_diff = len(advanced_result) - len(simple_result)
    print(f"Length difference: {length_diff} samples")
    
    if len(simple_result) == len(advanced_result):
        # Compare directly if same length
        mse = np.mean((simple_result - advanced_result)**2)
        correlation = np.corrcoef(simple_result, advanced_result)[0,1]
        
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Correlation: {correlation:.6f}")
        
        # Check for pre-emphasis effect
        diff_magnitude = np.mean(np.abs(simple_result - advanced_result))
        print(f"Average absolute difference: {diff_magnitude:.6f}")
        
    else:
        print("Different lengths - cannot directly compare")
    
    # Frequency domain analysis
    print("\n🎵 FREQUENCY DOMAIN ANALYSIS:")
    print("-" * 30)
    
    # Simple FFT analysis
    simple_fft = np.abs(np.fft.fft(simple_result[:min(len(simple_result), 1024)]))
    advanced_fft = np.abs(np.fft.fft(advanced_result[:min(len(advanced_result), 1024)]))
    
    simple_energy = np.sum(simple_fft**2)
    advanced_energy = np.sum(advanced_fft**2)
    
    print(f"Simple system spectral energy: {simple_energy:.2f}")
    print(f"Advanced system spectral energy: {advanced_energy:.2f}")
    print(f"Energy ratio (Advanced/Simple): {advanced_energy/simple_energy:.3f}")
    
    # Check high-frequency emphasis (pre-emphasis effect)
    high_freq_simple = np.sum(simple_fft[len(simple_fft)//2:])
    high_freq_advanced = np.sum(advanced_fft[len(advanced_fft)//2:])
    
    print(f"High-frequency energy ratio: {high_freq_advanced/high_freq_simple:.3f}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Time domain comparison
    plt.subplot(2, 3, 1)
    plt.plot(test_audio[:1000], label='Original', alpha=0.7)
    plt.title('Original Audio (first 1000 samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(simple_result[:1000], label='Simple', color='blue', alpha=0.7)
    plt.title('Simple Preprocessing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(advanced_result[:1000], label='Advanced', color='red', alpha=0.7)
    plt.title('Advanced Preprocessing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency domain comparison
    plt.subplot(2, 3, 4)
    freqs = np.fft.fftfreq(len(simple_fft), 1/16000)[:len(simple_fft)//2]
    plt.semilogy(freqs, simple_fft[:len(simple_fft)//2], label='Simple', alpha=0.7)
    plt.title('Simple System - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    freqs = np.fft.fftfreq(len(advanced_fft), 1/16000)[:len(advanced_fft)//2]
    plt.semilogy(freqs, advanced_fft[:len(advanced_fft)//2], label='Advanced', color='red', alpha=0.7)
    plt.title('Advanced System - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Difference plot
    plt.subplot(2, 3, 6)
    if len(simple_result) == len(advanced_result):
        diff = advanced_result - simple_result
        plt.plot(diff[:1000], label='Difference', color='green', alpha=0.7)
        plt.title('Difference (Advanced - Simple)')
        plt.ylabel('Amplitude Difference')
    else:
        plt.text(0.5, 0.5, 'Different lengths\nCannot compute difference', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Difference Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 ANALYSIS SUMMARY:")
    print("="*50)
    print("The F1 score differences between VAD and merged word evaluation")
    print("are likely caused by:")
    print("")
    print("1. 📈 PRE-EMPHASIS FILTERING:")
    print("   - Advanced system applies pre-emphasis (high-pass filter)")
    print("   - This emphasizes high frequencies and affects model predictions")
    print("   - Can improve consonant detection but may change vowel characteristics")
    print("")
    print("2. 🎯 NORMALIZATION DIFFERENCES:")
    print("   - Advanced system uses epsilon protection (+ 1e-8)")
    print("   - This prevents division by zero and numerical instability")
    print("   - May result in slightly different amplitude scaling")
    print("")
    print("3. ⏱️ PADDING BEHAVIOR:")
    print("   - Advanced system enforces minimum length padding")
    print("   - This can affect very short audio segments")
    print("")
    print("4. 🔊 SPECTRAL CHARACTERISTICS:")
    print(f"   - Energy ratio indicates {'emphasis' if (advanced_energy/simple_energy) > 1.1 else 'similar energy'}")
    print(f"   - High-frequency emphasis: {high_freq_advanced/high_freq_simple:.2f}x")
    print("")
    print("💡 RECOMMENDATION:")
    print("To get truly comparable results, both systems should use")
    print("identical preprocessing pipelines. The differences you're seeing")
    print("are legitimate differences in audio preparation, not evaluation bugs.")

if __name__ == "__main__":
    # Try to find a test audio file
    possible_files = [
        "./test.wav",
        "./test1.wav", 
        "./audio_processing/test.wav",
        "./dataset/test/test.wav"
    ]
    
    test_file = None
    for file_path in possible_files:
        if Path(file_path).exists():
            test_file = file_path
            break
    
    compare_preprocessing_methods(test_file)