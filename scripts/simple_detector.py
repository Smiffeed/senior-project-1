#!/usr/bin/env python3
"""
Simple API-style interface for the profanity detection model.
This provides an easy-to-use class for integrating into other applications.
"""

import torch
import numpy as np
import librosa
import os
import sys
from pathlib import Path

# Add the scripts directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from simplified_ultimate_training import EnhancedAudioClassifier, SimpleAudioPreprocessor
from transformers import Wav2Vec2FeatureExtractor

class SimpleProfanityDetector:
    """
    Simple, easy-to-use profanity detector.
    
    Usage:
        detector = SimpleProfanityDetector()
        result = detector.detect("path/to/audio.wav")
        if result['is_profanity']:
            print(f"Profanity detected: {result['class']} with {result['confidence']:.1%} confidence")
    """
    
    def __init__(self, model_dir="./models/simplified_advanced_audio_train"):
        """Initialize the detector with a trained model."""
        self.model_dir = model_dir
        self.model_name = "airesearch/wav2vec2-large-xlsr-53-th"
        
        # Class names and mapping
        self.classes = ['none', 'เย็ด', 'กู', 'มึง', 'เหี้ย', 'ควย', 'สวะ', 'หี', 'แตด']
        self.profanity_classes = self.classes[1:]  # All except 'none'
        
        # Initialize components
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = SimpleAudioPreprocessor()
        self.model = None
        
        # Load the best model
        self._load_model()
    
    def _load_model(self):
        """Load the best available model."""
        # Try to find the best model from any fold
        best_model_path = None
        for fold in range(1, 6):  # Check folds 1-5
            model_path = os.path.join(self.model_dir, f'fold_{fold}', 'best_model.pt')
            if os.path.exists(model_path):
                best_model_path = model_path
                break
        
        if best_model_path is None:
            raise FileNotFoundError(f"No trained model found in {self.model_dir}")
        
        # Create and load model
        self.model = EnhancedAudioClassifier(self.model_name, len(self.classes))
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded model from {best_model_path}")
    
    def detect(self, audio_path, start_time=None, end_time=None):
        """
        Detect profanity in an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            start_time (float, optional): Start time in seconds
            end_time (float, optional): End time in seconds
        
        Returns:
            dict: Detection result with keys:
                - is_profanity (bool): Whether profanity was detected
                - class (str): Predicted class name
                - confidence (float): Confidence score (0-1)
                - profanity_type (str): Type of profanity if detected, None otherwise
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Preprocess audio
        audio_np = self.preprocessor.preprocess_audio(audio_path, start_time, end_time)
        
        # Extract features
        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=16000
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_values=inputs.input_values,
                attention_mask=inputs.attention_mask
            )
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs
            
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        # Format result
        predicted_class = self.classes[prediction]
        is_profanity = predicted_class != 'none'
        
        result = {
            'is_profanity': is_profanity,
            'class': predicted_class,
            'confidence': confidence,
            'profanity_type': predicted_class if is_profanity else None
        }
        
        return result
    
    def detect_batch(self, audio_paths):
        """
        Detect profanity in multiple audio files.
        
        Args:
            audio_paths (list): List of audio file paths
        
        Returns:
            list: List of detection results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.detect(audio_path)
                result['file'] = audio_path
                results.append(result)
            except Exception as e:
                results.append({
                    'file': audio_path,
                    'error': str(e),
                    'is_profanity': None,
                    'class': None,
                    'confidence': None,
                    'profanity_type': None
                })
        return results
    
    def is_profane(self, audio_path, threshold=0.5):
        """
        Simple boolean check for profanity.
        
        Args:
            audio_path (str): Path to audio file
            threshold (float): Confidence threshold (default: 0.5)
        
        Returns:
            bool: True if profanity detected with confidence > threshold
        """
        result = self.detect(audio_path)
        return result['is_profanity'] and result['confidence'] > threshold
    
    def get_profanity_score(self, audio_path):
        """
        Get a profanity score (0-1) for an audio file.
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            float: Profanity score (0 = clean, 1 = definitely profane)
        """
        result = self.detect(audio_path)
        if result['is_profanity']:
            return result['confidence']
        else:
            return 0.0
    
    def detect_windowed(self, audio_path, window_length=0.5, overlap=0.25):
        """
        Detect profanity using sliding window approach for segment-wise analysis.
        
        Args:
            audio_path (str): Path to the audio file
            window_length (float): Length of each window in seconds (default: 0.5)
            overlap (float): Overlap between windows in seconds (default: 0.25)
        
        Returns:
            dict: Detection results with window-by-window analysis
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and preprocess audio
        audio_np = self.preprocessor.preprocess_audio(audio_path)
        
        # Sliding window parameters
        sr = 16000  # Sample rate
        window_samples = int(window_length * sr)
        hop_samples = int(overlap * sr)
        
        # Generate sliding windows
        windows = []
        audio_length = len(audio_np)
        
        # If audio is shorter than window length, pad it
        if audio_length < window_samples:
            padding = window_samples - audio_length
            audio_np = np.pad(audio_np, (0, padding), mode='constant', constant_values=0)
            audio_length = len(audio_np)
        
        # Extract overlapping windows
        for start_idx in range(0, audio_length - window_samples + 1, hop_samples):
            end_idx = start_idx + window_samples
            window = audio_np[start_idx:end_idx]
            
            # Process this window
            inputs = self.feature_extractor(
                window, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=window_samples
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(
                    input_values=inputs.input_values,
                    attention_mask=inputs.attention_mask
                )
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            # Store window result
            predicted_class = self.classes[prediction]
            is_profanity = predicted_class != 'none'
            
            window_result = {
                'start_time': start_idx / sr,
                'end_time': end_idx / sr,
                'is_profanity': is_profanity,
                'class': predicted_class,
                'confidence': confidence,
                'profanity_type': predicted_class if is_profanity else None
            }
            
            windows.append(window_result)
        
        # Calculate overall statistics
        profanity_windows = [w for w in windows if w['is_profanity']]
        total_windows = len(windows)
        profanity_count = len(profanity_windows)
        
        # Overall prediction (majority vote)
        if profanity_count > total_windows / 2:
            overall_prediction = True
            # Find most common profanity class
            profanity_classes = [w['class'] for w in profanity_windows]
            overall_class = max(set(profanity_classes), key=profanity_classes.count) if profanity_classes else 'none'
            overall_confidence = np.mean([w['confidence'] for w in profanity_windows])
        else:
            overall_prediction = False
            overall_class = 'none'
            clean_windows = [w for w in windows if not w['is_profanity']]
            overall_confidence = np.mean([w['confidence'] for w in clean_windows]) if clean_windows else 0.0
        
        # Compile final result
        result = {
            'is_profanity': overall_prediction,
            'class': overall_class,
            'confidence': overall_confidence,
            'profanity_type': overall_class if overall_prediction else None,
            'total_windows': total_windows,
            'profanity_windows': profanity_count,
            'profanity_ratio': profanity_count / total_windows,
            'window_results': windows
        }
        
        return result

# Example usage
def main():
    """Example usage of the SimpleProfanityDetector."""
    
    # Initialize detector
    detector = SimpleProfanityDetector()
    
    # Example 1: Simple detection
    print("Example 1: Simple profanity detection")
    print("-" * 40)
    
    # Test with a file from your eval directory
    test_file = "./eval/กฤต.wav"  # Replace with actual file
    
    if os.path.exists(test_file):
        result = detector.detect(test_file)
        
        print(f"File: {test_file}")
        print(f"Is profanity: {result['is_profanity']}")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        if result['is_profanity']:
            print(f"Profanity type: {result['profanity_type']}")
    else:
        print(f"Test file not found: {test_file}")
    
    print("\n" + "="*50)
    
    # Example 2: Simple boolean check
    print("Example 2: Simple boolean check")
    print("-" * 40)
    
    if os.path.exists(test_file):
        is_profane = detector.is_profane(test_file, threshold=0.8)
        print(f"Is profane (80% threshold): {is_profane}")
        
        score = detector.get_profanity_score(test_file)
        print(f"Profanity score: {score:.2f}")
    
    print("\n" + "="*50)
    
    # Example 3: Batch processing
    print("Example 3: Batch processing")
    print("-" * 40)
    
    eval_dir = "./eval"
    if os.path.exists(eval_dir):
        # Get first few WAV files
        audio_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) 
                      if f.endswith('.wav')][:5]
        
        results = detector.detect_batch(audio_files)
        
        for result in results:
            if 'error' in result:
                print(f"{os.path.basename(result['file'])}: ERROR - {result['error']}")
            else:
                status = "🚨" if result['is_profanity'] else "✅"
                confidence = result['confidence'] * 100 if result['confidence'] else 0
                print(f"{os.path.basename(result['file'])}: {status} {result['class']} ({confidence:.1f}%)")

if __name__ == "__main__":
    main()
