#!/usr/bin/env python3
"""
Single File Profanity Prediction Script

This script allows you to test your trained model on individual audio files.
Usage:
    python predict_single_file.py path/to/audio/file.wav
    python predict_single_file.py path/to/audio/file.wav --confidence
    python predict_single_file.py path/to/audio/file.wav --ensemble
"""

import torch
import numpy as np
import librosa
import argparse
import os
import sys
from pathlib import Path

# Add the scripts directory to path so we can import our modules
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from simplified_ultimate_training import EnhancedAudioClassifier, SimpleAudioPreprocessor
from transformers import Wav2Vec2FeatureExtractor
from advanced_model_improvements import ModelEnsemble

# Constants
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
CLASS_NAMES = list(LABEL_MAP.keys())
NUM_LABELS = len(LABEL_MAP)

class ProfanityPredictor:
    """Single file profanity prediction class."""
    
    def __init__(self, model_dir="./models/simplified_advanced_audio_train", 
                 model_name="airesearch/wav2vec2-large-xlsr-53-th"):
        self.model_dir = model_dir
        self.model_name = model_name
        
        # Initialize components
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = SimpleAudioPreprocessor()
        
        # Load model(s)
        self.single_model = None
        self.ensemble_models = None
        
    def load_single_model(self, fold_num=1):
        """Load a single best model."""
        model_path = os.path.join(self.model_dir, f'fold_{fold_num}', 'best_model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create and load model
        model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        self.single_model = model
        print(f"Loaded single model from fold {fold_num}")
        
    def load_ensemble_models(self, num_folds=5):
        """Load ensemble of models."""
        models = []
        for fold in range(1, num_folds + 1):
            model_path = os.path.join(self.model_dir, f'fold_{fold}', 'best_model.pt')
            
            if os.path.exists(model_path):
                model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                models.append(model)
        
        if models:
            self.ensemble_models = ModelEnsemble(models)
            print(f"Loaded ensemble with {len(models)} models")
        else:
            raise FileNotFoundError("No models found for ensemble")
    
    def preprocess_audio_file(self, audio_path, start_time=None, end_time=None):
        """Preprocess a single audio file with sliding window approach."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and preprocess audio
        audio_np = self.preprocessor.preprocess_audio(audio_path, start_time, end_time)
        
        # Sliding window parameters
        window_length = 0.5  # 0.5 seconds
        overlap = 0.25  # 0.25 seconds overlap
        sr = 16000  # Sample rate
        
        window_samples = int(window_length * sr)  # 8000 samples
        hop_samples = int(overlap * sr)  # 4000 samples (step size)
        
        # Generate sliding windows
        windows = []
        window_starts = []
        
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
            
            # Extract features for this window
            inputs = self.feature_extractor(
                window, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=window_samples
            )
            
            windows.append({
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'start_time': start_idx / sr,
                'end_time': end_idx / sr
            })
            window_starts.append(start_idx / sr)
        
        return windows, window_starts
    
    def predict_single(self, audio_path, start_time=None, end_time=None, return_confidence=False):
        """Predict using single model with sliding window approach."""
        if self.single_model is None:
            self.load_single_model()
        
        # Get sliding windows
        windows, window_starts = self.preprocess_audio_file(audio_path, start_time, end_time)
        
        results = []
        
        # Process each window
        for i, window_data in enumerate(windows):
            # Prepare batch input
            batch_inputs = {
                'input_values': window_data['input_values'].unsqueeze(0),
                'attention_mask': window_data['attention_mask'].unsqueeze(0)
            }
            
            # Forward pass
            with torch.no_grad():
                outputs = self.single_model(**batch_inputs)
                
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
                window_result = {
                    'window_index': i,
                    'start_time': window_data['start_time'],
                    'end_time': window_data['end_time'],
                    'prediction': prediction,
                    'predicted_class': self.class_names[prediction],
                    'confidence': confidence
                }
                
                if return_confidence:
                    window_result['all_confidences'] = {
                        self.class_names[j]: prob.item() 
                        for j, prob in enumerate(probabilities[0])
                    }
                
                results.append(window_result)
        
        # Calculate overall prediction (majority vote)
        predictions = [r['prediction'] for r in results]
        overall_prediction = max(set(predictions), key=predictions.count)
        overall_confidence = np.mean([r['confidence'] for r in results if r['prediction'] == overall_prediction])
        
        # Count profanity windows
        profanity_windows = sum(1 for r in results if r['prediction'] == 1)  # Assuming 1 = profanity
        
        final_result = {
            'overall_prediction': overall_prediction,
            'overall_class': self.class_names[overall_prediction],
            'overall_confidence': overall_confidence,
            'total_windows': len(results),
            'profanity_windows': profanity_windows,
            'profanity_ratio': profanity_windows / len(results),
            'window_results': results
        }
        
        return final_result
    
    def predict_ensemble(self, audio_path, start_time=None, end_time=None, return_confidence=False):
        """Predict using ensemble model with sliding window approach."""
        if self.ensemble_models is None:
            self.load_ensemble_models()
        
        # Get sliding windows
        windows, window_starts = self.preprocess_audio_file(audio_path, start_time, end_time)
        
        results = []
        
        # Process each window
        for i, window_data in enumerate(windows):
            # Prepare batch input
            batch_inputs = {
                'input_values': window_data['input_values'].unsqueeze(0),
                'attention_mask': window_data['attention_mask'].unsqueeze(0)
            }
            
            # Predict with ensemble
            ensemble_output = self.ensemble_models.predict(batch_inputs)
            probabilities = torch.softmax(ensemble_output['predictions'], dim=-1)
            prediction = torch.argmax(ensemble_output['predictions'], dim=-1).item()
            confidence = probabilities[0][prediction].item()
            uncertainty = ensemble_output['uncertainty'].item()
            
            # Store window result
            window_result = {
                'window_index': i,
                'start_time': window_data['start_time'],
                'end_time': window_data['end_time'],
                'prediction': prediction,
                'predicted_class': self.class_names[prediction],
                'confidence': confidence,
                'uncertainty': uncertainty
            }
            
            if return_confidence:
                window_result['all_confidences'] = {
                    self.class_names[j]: prob.item() 
                    for j, prob in enumerate(probabilities[0])
                }
            
            results.append(window_result)
        
        # Calculate overall prediction (majority vote)
        predictions = [r['prediction'] for r in results]
        overall_prediction = max(set(predictions), key=predictions.count)
        overall_confidence = np.mean([r['confidence'] for r in results if r['prediction'] == overall_prediction])
        overall_uncertainty = np.mean([r['uncertainty'] for r in results])
        
        # Count profanity windows
        profanity_windows = sum(1 for r in results if r['prediction'] == 1)  # Assuming 1 = profanity
        
        final_result = {
            'overall_prediction': overall_prediction,
            'overall_class': self.class_names[overall_prediction],
            'overall_confidence': overall_confidence,
            'overall_uncertainty': overall_uncertainty,
            'total_windows': len(results),
            'profanity_windows': profanity_windows,
            'profanity_ratio': profanity_windows / len(results),
            'window_results': results
        }
        
        return final_result
    
    def predict(self, audio_path, start_time=None, end_time=None, 
                use_ensemble=False, return_confidence=False):
        """Main prediction method."""
        if use_ensemble:
            return self.predict_ensemble(audio_path, start_time, end_time, return_confidence)
        else:
            return self.predict_single(audio_path, start_time, end_time, return_confidence)

def format_time(seconds):
    """Format time in seconds to MM:SS format."""
    if seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def print_result(result, audio_path, start_time=None, end_time=None):
    """Print prediction result with windowed analysis."""
    print("\n" + "="*60)
    print("PROFANITY DETECTION RESULT")
    print("="*60)
    print(f"Audio File: {audio_path}")
    if start_time is not None or end_time is not None:
        print(f"Time Range: {format_time(start_time)} - {format_time(end_time)}")
    print("-"*60)
    
    # Overall prediction
    is_profanity = result['overall_prediction'] != 0
    profanity_status = "🚨 PROFANITY DETECTED" if is_profanity else "✅ CLEAN"
    print(f"Overall Status: {profanity_status}")
    print(f"Overall Class: {result['overall_class']}")
    print(f"Overall Confidence: {result['overall_confidence']:.1%}")
    
    if 'overall_uncertainty' in result:
        print(f"Overall Uncertainty: {result['overall_uncertainty']:.4f}")
    
    # Window analysis summary
    print(f"\nWindow Analysis Summary:")
    print(f"Total Windows: {result['total_windows']}")
    print(f"Profanity Windows: {result['profanity_windows']}")
    print(f"Profanity Ratio: {result['profanity_ratio']:.1%}")
    
    # Show detailed window results
    print(f"\nDetailed Window Results:")
    print("-"*50)
    for window in result['window_results']:
        status_emoji = "🚨" if window['prediction'] != 0 else "✅"
        print(f"Window {window['window_index']:2d}: {format_time(window['start_time'])}-{format_time(window['end_time'])} "
              f"{status_emoji} {window['predicted_class']} ({window['confidence']:.1%})")
        
        # Show all confidences if available
        if 'all_confidences' in window:
            for class_name, conf in window['all_confidences'].items():
                if conf > 0.1:  # Only show classes with >10% confidence
                    print(f"           {class_name}: {conf:.1%}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Predict profanity in audio files')
    parser.add_argument('audio_path', help='Path to the audio file')
    parser.add_argument('--start', type=float, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble prediction')
    parser.add_argument('--confidence', action='store_true', help='Show all class probabilities')
    parser.add_argument('--model-dir', default='./models/simplified_advanced_audio_train',
                       help='Path to model directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        print("Initializing predictor...")
        predictor = ProfanityPredictor(model_dir=args.model_dir)
        
        # Make prediction
        print(f"Processing: {args.audio_path}")
        if args.ensemble:
            print("Loading ensemble models...")
        else:
            print("Loading single model...")
            
        result = predictor.predict(
            args.audio_path, 
            start_time=args.start, 
            end_time=args.end,
            use_ensemble=args.ensemble, 
            return_confidence=args.confidence
        )
        
        # Print result
        print_result(result, args.audio_path, args.start, args.end)
        
        # Return exit code based on result
        return 1 if result['overall_prediction'] != 0 else 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
