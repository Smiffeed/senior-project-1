#!/usr/bin/env python3
"""
Enhanced Windowed Profanity Detection Evaluation Script

This script combines the sliding window approach from window_eval.py with enhanced preprocessing,
but uses a standard Wav2Vec2 model for broader compatibility.

Usage:
    python enhanced_window_eval.py --model_dir ./models/simplified_advanced_audio_train --eval_csv ./csv/eval.csv
"""

import torch
import torchaudio
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the scripts directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Define label mapping
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
CLASS_NAMES = list(LABEL_MAP.keys())
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class EnhancedWindowedEvaluator:
    """
    Enhanced windowed evaluator with better preprocessing but standard model architecture.
    """
    
    def __init__(self, model_dir, device=None):
        """Initialize the evaluator."""
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Window parameters
        self.window_size = 0.5  # seconds
        self.hop_length = 0.25  # seconds
        self.overlap_threshold = 0.25  # minimum overlap ratio
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and feature extractor."""
        print(f"Loading model from {self.model_dir}")
        
        # Check for checkpoint structure first
        checkpoints = list(self.model_dir.glob("checkpoint-*"))
        if checkpoints:
            # Sort by checkpoint number and get the latest
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))
            model_path = str(latest_checkpoint)
            print(f"Using checkpoint: {latest_checkpoint}")
        else:
            # Check for fold-based structure
            fold_dirs = list(self.model_dir.glob("fold_*"))
            if fold_dirs:
                # Use fold_1 by default
                fold_1_path = self.model_dir / "fold_1" / "best_model.pt"
                if fold_1_path.exists():
                    print(f"Found fold-based models, using fold_1: {fold_1_path}")
                    self._load_fold_model(fold_1_path)
                    return
                else:
                    print("Fold directories found but no best_model.pt in fold_1")
            
            model_path = str(self.model_dir)
            print(f"Using model directory directly: {model_path}")
        
        # Load feature extractor
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        except:
            print("Warning: Using fallback feature extractor")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "airesearch/wav2vec2-large-xlsr-53-th"
            )
        
        # Load model
        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(CLASS_NAMES)
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try loading with state dict
            try:
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    "airesearch/wav2vec2-large-xlsr-53-th",
                    num_labels=len(CLASS_NAMES)
                )
                state_dict_path = Path(model_path) / "pytorch_model.bin"
                if state_dict_path.exists():
                    state_dict = torch.load(state_dict_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print("Loaded model with custom state dict")
            except Exception as e2:
                print(f"Failed to load custom model: {e2}")
                raise
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def _load_fold_model(self, model_path):
        """Load a model from the fold-based structure."""
        try:
            # Load feature extractor (use fallback)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "airesearch/wav2vec2-large-xlsr-53-th"
            )
            
            # Create model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "airesearch/wav2vec2-large-xlsr-53-th",
                num_labels=len(CLASS_NAMES)
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Fix key names if they have double nesting (wav2vec2.wav2vec2.*)
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('wav2vec2.wav2vec2.'):
                    # Remove the extra 'wav2vec2.' prefix
                    new_key = key.replace('wav2vec2.wav2vec2.', 'wav2vec2.')
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value
            
            # Load the fixed state dict
            self.model.load_state_dict(fixed_state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Fold model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading fold model: {e}")
            raise
    
    def _enhanced_preprocess_audio_segment(self, file_path, start_time, end_time):
        """Enhanced audio preprocessing with noise reduction and normalization."""
        try:
            # Load audio segment
            metadata = torchaudio.info(file_path)
            sr = metadata.sample_rate
            
            audio, sr = torchaudio.load(
                file_path, 
                frame_offset=int(start_time * sr), 
                num_frames=int((end_time - start_time) * sr)
            )
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
                sr = 16000
            
            # Convert to numpy for processing
            audio = audio.squeeze().numpy()
            
            # Enhanced preprocessing steps
            audio = self._apply_pre_emphasis(audio)
            audio = self._apply_noise_reduction(audio, sr)
            audio = self._apply_normalization(audio)
            
            # Ensure correct length
            target_length = int(self.window_size * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            elif len(audio) > target_length:
                audio = audio[:target_length]
            
            return audio
            
        except Exception as e:
            print(f"Error in enhanced preprocessing: {e}")
            return self._basic_preprocess_audio_segment(file_path, start_time, end_time)
    
    def _apply_pre_emphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        if len(audio) > 1:
            return np.append(audio[0], audio[1:] - coeff * audio[:-1])
        return audio
    
    def _apply_noise_reduction(self, audio, sr):
        """Simple spectral subtraction noise reduction."""
        if len(audio) < sr * 0.1:  # Too short for noise estimation
            return audio
        
        try:
            # Estimate noise from first 0.1 seconds
            noise_sample_length = int(0.1 * sr)
            noise_segment = audio[:noise_sample_length]
            
            # Apply FFT-based noise reduction
            fft_audio = np.fft.fft(audio)
            fft_noise = np.fft.fft(noise_segment)
            
            # Spectral subtraction
            magnitude = np.abs(fft_audio)
            phase = np.angle(fft_audio)
            noise_magnitude = np.abs(fft_noise)
            
            # Ensure same length
            if len(noise_magnitude) != len(magnitude):
                noise_magnitude = np.resize(noise_magnitude, len(magnitude))
            
            # Subtract noise spectrum
            clean_magnitude = magnitude - 0.5 * noise_magnitude
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            clean_fft = clean_magnitude * np.exp(1j * phase)
            clean_audio = np.real(np.fft.ifft(clean_fft))
            
            return clean_audio
        except:
            return audio
    
    def _apply_normalization(self, audio):
        """Apply RMS normalization and windowing."""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / (rms + 1e-8)
        
        # Apply Hamming window
        if len(audio) > 1:
            window = np.hamming(len(audio))
            audio = audio * window
        
        return audio
    
    def _basic_preprocess_audio_segment(self, file_path, start_time, end_time):
        """Fallback basic preprocessing."""
        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        
        audio, sr = torchaudio.load(
            file_path, 
            frame_offset=int(start_time * sr), 
            num_frames=int((end_time - start_time) * sr)
        )
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        return audio.squeeze().numpy()
    
    def _predict_segment(self, audio):
        """Make prediction for an audio segment."""
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        return prediction.item(), probs.squeeze().cpu().numpy()
    
    def evaluate_file(self, file_path, file_df):
        """Evaluate a single file with sliding windows."""
        results = []
        
        try:
            audio_length = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
            
            # Process profanity segments
            for _, row in file_df.iterrows():
                true_start = row['start_time']
                true_end = row['end_time']
                true_label = row['label']
                
                window_starts = np.arange(
                    max(0, true_start - self.window_size),
                    min(audio_length - self.window_size, true_end),
                    self.hop_length
                )
                
                for window_start in window_starts:
                    window_end = window_start + self.window_size
                    
                    overlap_start = max(window_start, true_start)
                    overlap_end = min(window_end, true_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    if overlap_duration > (self.window_size * self.overlap_threshold):
                        audio = self._enhanced_preprocess_audio_segment(
                            file_path, window_start, window_end
                        )
                        pred_class, probabilities = self._predict_segment(audio)
                        
                        results.append({
                            'file': os.path.basename(file_path),
                            'window_start': window_start,
                            'window_end': window_end,
                            'true_start': true_start,
                            'true_end': true_end,
                            'overlap_duration': overlap_duration,
                            'true_label': true_label,
                            'true_label_id': LABEL_MAP[true_label],
                            'predicted_label': CLASS_NAMES[pred_class],
                            'predicted_label_id': pred_class,
                            'confidence': probabilities[pred_class],
                            'all_probabilities': probabilities.tolist()
                        })
            
            # Process non-profanity regions
            all_profanity_times = sorted([
                (row['start_time'], row['end_time']) 
                for _, row in file_df.iterrows()
            ])
            
            current_time = 0
            for prof_start, prof_end in all_profanity_times:
                if current_time < prof_start:
                    for window_start in np.arange(current_time, prof_start - self.window_size, self.hop_length):
                        window_end = window_start + self.window_size
                        
                        audio = self._enhanced_preprocess_audio_segment(
                            file_path, window_start, window_end
                        )
                        pred_class, probabilities = self._predict_segment(audio)
                        
                        results.append({
                            'file': os.path.basename(file_path),
                            'window_start': window_start,
                            'window_end': window_end,
                            'true_start': window_start,
                            'true_end': window_end,
                            'overlap_duration': self.window_size,
                            'true_label': 'none',
                            'true_label_id': LABEL_MAP['none'],
                            'predicted_label': CLASS_NAMES[pred_class],
                            'predicted_label_id': pred_class,
                            'confidence': probabilities[pred_class],
                            'all_probabilities': probabilities.tolist()
                        })
                
                current_time = prof_end
            
            # Process remaining region
            if current_time < audio_length:
                for window_start in np.arange(current_time, audio_length - self.window_size, self.hop_length):
                    window_end = window_start + self.window_size
                    
                    audio = self._enhanced_preprocess_audio_segment(
                        file_path, window_start, window_end
                    )
                    pred_class, probabilities = self._predict_segment(audio)
                    
                    results.append({
                        'file': os.path.basename(file_path),
                        'window_start': window_start,
                        'window_end': window_end,
                        'true_start': window_start,
                        'true_end': window_end,
                        'overlap_duration': self.window_size,
                        'true_label': 'none',
                        'true_label_id': LABEL_MAP['none'],
                        'predicted_label': CLASS_NAMES[pred_class],
                        'predicted_label_id': pred_class,
                        'confidence': probabilities[pred_class],
                        'all_probabilities': probabilities.tolist()
                    })
        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        
        return results
    
    def evaluate_dataset(self, eval_csv, output_dir):
        """Evaluate the entire dataset."""
        print(f"Starting enhanced windowed evaluation...")
        print(f"Window size: {self.window_size}s, Hop length: {self.hop_length}s")
        
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(eval_csv)
        print(f"Loaded {len(df)} evaluation samples")
        
        all_results = []
        unique_files = df['file_path'].unique()
        
        for file_path in tqdm(unique_files, desc="Processing files"):
            file_path = os.path.normpath(file_path).replace('\\', '/')
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            file_df = df[df['file_path'] == file_path]
            file_results = self.evaluate_file(file_path, file_df)
            all_results.extend(file_results)
        
        if not all_results:
            print("No results generated!")
            return None, None
        
        results_df = pd.DataFrame(all_results)
        
        true_labels = results_df['true_label_id'].tolist()
        pred_labels = results_df['predicted_label_id'].tolist()
        
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        class_report = classification_report(
            true_labels, pred_labels, 
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )
        
        print("\nClassification Report:")
        print(classification_report(
            true_labels, pred_labels, 
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0
        ))
        
        # Save results
        results_df.to_csv(os.path.join(output_dir, 'enhanced_window_evaluation_results.csv'), index=False)
        
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', 
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap='Blues'
        )
        plt.title('Enhanced Windowed Evaluation - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'total_windows': len(results_df),
            'window_size': self.window_size,
            'hop_length': self.hop_length,
            'overlap_threshold': self.overlap_threshold
        }
        
        with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        return accuracy, results_df

def main():
    parser = argparse.ArgumentParser(description='Enhanced Windowed Profanity Detection Evaluation')
    parser.add_argument('--model_dir', type=str, default='./models/simplified_advanced_audio_train',
                        help='Directory containing the trained model')
    parser.add_argument('--eval_csv', type=str, default='./csv/eval.csv',
                        help='CSV file with evaluation data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results_enhanced_window',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    evaluator = EnhancedWindowedEvaluator(args.model_dir, device)
    accuracy, results_df = evaluator.evaluate_dataset(args.eval_csv, args.output_dir)
    
    if results_df is not None:
        print(f"\nEvaluation completed successfully!")
        print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
