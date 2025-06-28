#!/usr/bin/env python3
"""
Advanced Windowed Profanity Detection Evaluation Script

This script combines:
- The sliding window approach from window_eval.py (0.5s window, 0.25s hop)
- The advanced preprocessing from ultimate_model_training.py (EnhancedAudioPreprocessor)
- The advanced model architecture from ultimate_model_training.py (MultiModalAudioClassifier)

Usage:
    python advanced_window_eval.py --model_dir ./models/simplified_advanced_audio_train --eval_csv ./csv/eval.csv --output_dir ./evaluation_results
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

# Import the advanced components
try:
    from ultimate_model_training import (
        EnhancedAudioPreprocessor, MultiModalAudioClassifier, 
        LABEL_MAP, CLASS_NAMES, NUM_LABELS
    )
    from transformers import Wav2Vec2FeatureExtractor
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing advanced components: {e}")
    print("Make sure ultimate_model_training.py is in the same directory")
    ADVANCED_COMPONENTS_AVAILABLE = False
    sys.exit(1)

class AdvancedWindowedEvaluator:
    """
    Advanced windowed evaluator that uses enhanced preprocessing and model architecture.
    """
    
    def __init__(self, model_dir, device=None):
        """Initialize the evaluator with advanced components."""
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.preprocessor = EnhancedAudioPreprocessor(sr=16000)
        self.feature_extractor = None
        self.model = None
        
        # Window parameters (from window_eval.py)
        self.window_size = 0.5  # seconds
        self.hop_length = 0.25  # seconds
        self.overlap_threshold = 0.25  # minimum overlap ratio to consider a window
        
        # Load the model and feature extractor
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and feature extractor."""
        print(f"Loading model from {self.model_dir}")
        
        # Find the best checkpoint
        checkpoints = list(self.model_dir.glob("checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.model_dir}")
        
        # Sort by checkpoint number and get the latest
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))
        print(f"Using checkpoint: {latest_checkpoint}")
        
        # Load feature extractor
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(latest_checkpoint))
        except:
            # Fallback to a standard feature extractor
            print("Warning: Using fallback feature extractor")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "airesearch/wav2vec2-large-xlsr-53-th"
            )
        
        # Load model
        try:
            # Try to load as MultiModalAudioClassifier
            self.model = MultiModalAudioClassifier(
                wav2vec_model_name="airesearch/wav2vec2-large-xlsr-53-th",
                num_labels=NUM_LABELS,
                additional_feature_dim=128
            )
            
            # Load state dict
            state_dict_path = latest_checkpoint / "pytorch_model.bin"
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Loaded MultiModalAudioClassifier with state dict")
            else:
                print("Warning: No state dict found, using pre-trained weights only")
                
        except Exception as e:
            print(f"Error loading MultiModalAudioClassifier: {e}")
            print("Falling back to standard Wav2Vec2 model")
            
            from transformers import Wav2Vec2ForSequenceClassification
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                str(latest_checkpoint),
                num_labels=NUM_LABELS
            )
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def _preprocess_audio_segment(self, file_path, start_time, end_time):
        """Preprocess audio segment using enhanced preprocessing."""
        try:
            # Use the enhanced preprocessor
            audio, features = self.preprocessor.preprocess_audio(
                file_path, start_time, end_time, augment=False
            )
            
            # Ensure audio is the right length and format
            target_length = int(self.window_size * 16000)  # 16kHz sampling rate
            
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            elif len(audio) > target_length:
                # Truncate
                audio = audio[:target_length]
            
            return audio, features
            
        except Exception as e:
            print(f"Error in enhanced preprocessing: {e}")
            # Fallback to basic preprocessing
            return self._basic_preprocess_audio_segment(file_path, start_time, end_time)
    
    def _basic_preprocess_audio_segment(self, file_path, start_time, end_time):
        """Fallback basic audio preprocessing."""
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
        
        # Normalize
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        audio = audio.squeeze().numpy()
        
        return audio, None
    
    def _predict_segment(self, audio, features=None):
        """Make prediction for an audio segment using the advanced model."""
        # Prepare Wav2Vec2 inputs
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if isinstance(self.model, MultiModalAudioClassifier):
                # Use advanced model with additional features if available
                additional_features = None
                if features is not None:
                    # Convert features to tensor (simplified)
                    try:
                        # Use mel spectrogram if available
                        if 'mel_spec' in features:
                            mel_spec = torch.tensor(features['mel_spec'], dtype=torch.float32)
                            if mel_spec.dim() == 2:
                                mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
                            additional_features = mel_spec.to(self.device)
                    except Exception as e:
                        print(f"Warning: Could not process additional features: {e}")
                        additional_features = None
                
                outputs = self.model(
                    input_values=inputs['input_values'],
                    attention_mask=inputs.get('attention_mask'),
                    additional_features=additional_features
                )
                logits = outputs['logits']
            else:
                # Use standard model
                outputs = self.model(**inputs)
                logits = outputs.logits
        
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        return prediction.item(), probs.squeeze().cpu().numpy()
    
    def evaluate_file(self, file_path, file_df):
        """Evaluate a single file with sliding windows."""
        results = []
        
        try:
            # Get audio length
            audio_length = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
            
            # Process each profanity segment in the file
            for _, row in file_df.iterrows():
                true_start = row['start_time']
                true_end = row['end_time']
                true_label = row['label']
                
                # Find all windows that overlap with this segment
                window_starts = np.arange(
                    max(0, true_start - self.window_size),
                    min(audio_length - self.window_size, true_end),
                    self.hop_length
                )
                
                for window_start in window_starts:
                    window_end = window_start + self.window_size
                    
                    # Calculate overlap with true segment
                    overlap_start = max(window_start, true_start)
                    overlap_end = min(window_end, true_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    # If overlap is significant
                    if overlap_duration > (self.window_size * self.overlap_threshold):
                        # Process audio segment with enhanced preprocessing
                        audio, features = self._preprocess_audio_segment(
                            file_path, window_start, window_end
                        )
                        pred_class, probabilities = self._predict_segment(audio, features)
                        
                        # Store results
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
                # Process the non-profanity region before this profanity
                if current_time < prof_start:
                    for window_start in np.arange(current_time, prof_start - self.window_size, self.hop_length):
                        window_end = window_start + self.window_size
                        
                        # Process audio segment
                        audio, features = self._preprocess_audio_segment(
                            file_path, window_start, window_end
                        )
                        pred_class, probabilities = self._predict_segment(audio, features)
                        
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
            
            # Process remaining non-profanity region at the end
            if current_time < audio_length:
                for window_start in np.arange(current_time, audio_length - self.window_size, self.hop_length):
                    window_end = window_start + self.window_size
                    
                    # Process audio segment
                    audio, features = self._preprocess_audio_segment(
                        file_path, window_start, window_end
                    )
                    pred_class, probabilities = self._predict_segment(audio, features)
                    
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
        """Evaluate the entire dataset using sliding windows."""
        print(f"Starting advanced windowed evaluation...")
        print(f"Window size: {self.window_size}s, Hop length: {self.hop_length}s")
        print(f"Overlap threshold: {self.overlap_threshold}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read evaluation CSV
        df = pd.read_csv(eval_csv)
        print(f"Loaded {len(df)} evaluation samples from {eval_csv}")
        
        all_results = []
        unique_files = df['file_path'].unique()
        
        print(f"Processing {len(unique_files)} files...")
        
        # Process each file
        for file_path in tqdm(unique_files, desc="Processing files"):
            # Normalize file path
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
        
        # Convert to DataFrame and analyze results
        results_df = pd.DataFrame(all_results)
        
        # Extract labels for metrics
        true_labels = results_df['true_label_id'].tolist()
        pred_labels = results_df['predicted_label_id'].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Generate detailed classification report
        class_report = classification_report(
            true_labels, pred_labels, 
            target_names=CLASS_NAMES,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))
        
        # Plot and save confusion matrix
        self._plot_confusion_matrix(true_labels, pred_labels, output_dir)
        
        # Save results
        results_df.to_csv(os.path.join(output_dir, 'advanced_window_evaluation_results.csv'), index=False)
        
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
        print(f"- detailed results: advanced_window_evaluation_results.csv")
        print(f"- metrics: evaluation_metrics.json")
        print(f"- confusion matrix: confusion_matrix.png")
        
        return accuracy, results_df
    
    def _plot_confusion_matrix(self, true_labels, pred_labels, output_dir):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap='Blues'
        )
        plt.title('Advanced Windowed Evaluation - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Advanced Windowed Profanity Detection Evaluation')
    parser.add_argument('--model_dir', type=str, default='./models/simplified_advanced_audio_train',
                        help='Directory containing the trained model')
    parser.add_argument('--eval_csv', type=str, default='./csv/eval.csv',
                        help='CSV file with evaluation data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results_advanced_window',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create evaluator
    evaluator = AdvancedWindowedEvaluator(args.model_dir, device)
    
    # Run evaluation
    accuracy, results_df = evaluator.evaluate_dataset(args.eval_csv, args.output_dir)
    
    if results_df is not None:
        print(f"\nEvaluation completed successfully!")
        print(f"Final Accuracy: {accuracy:.4f}")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()
