#!/usr/bin/env python3
"""
🚀 ADVANCED MODEL EVALUATION SYSTEM
Evaluation script compatible with advanced model training architectures.

This script can evaluate:
- Enhanced Wav2Vec2 models from advanced_model_training.py
- CNN-LSTM models
- Transformer models
- Ensemble models

Features:
- Windowed evaluation with configurable overlap
- Multiple threshold testing
- Comprehensive metrics and visualization
- Compatible with all advanced model architectures
- Thai language support

Usage:
    python evaluate_advanced_models.py --model-path models/advanced_training --model-type wav2vec2_enhanced
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import pandas as pd
import numpy as np
import os
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.font_manager as fm
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define label mapping (same as in training)
label_map = {
    'none': 0,
    'เย็ด': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4,
    'ควย': 5,
    'สวะ': 6,
    'หี': 7,
    'แตด': 8
}

# Reverse label mapping for output
rev_label_map = {v: k for k, v in label_map.items()}

def setup_thai_font():
    """Setup Thai font for matplotlib"""
    try:
        # Try different Thai fonts
        thai_fonts = [
            'Cordia New',
            'TH Sarabun New',
            'Tahoma',
            'Microsoft Sans Serif'
        ]
        
        for font in thai_fonts:
            try:
                plt.rcParams['font.family'] = font
                break
            except:
                continue
                
    except Exception as e:
        print(f"Warning: Could not setup Thai font: {e}")
        pass

def get_device_info():
    """Get device information."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return device

# Initialize device
device = get_device_info()

class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM architecture for audio classification (same as training)."""
    
    def __init__(self, num_classes: int, config: dict):
        super().__init__()
        self.config = config
        
        # CNN feature extractor
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        cnn_channels = config.get('cnn_channels', [64, 128, 256])
        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(config.get('dropout_rate', 0.3))
                )
            )
            in_channels = out_channels
        
        # LSTM layers
        lstm_hidden = config.get('lstm_hidden_size', 256)
        lstm_layers = config.get('lstm_layers', 2)
        dropout_rate = config.get('dropout_rate', 0.3)
        
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # (batch_size, sequence_length, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        return self.classifier(x)

class TransformerClassifier(nn.Module):
    """Transformer-based audio classifier (same as training)."""
    
    def __init__(self, num_classes: int, config: dict):
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, 512))
        
        # Input projection
        self.input_projection = nn.Linear(1, 512)
        
        # Transformer encoder
        transformer_heads = config.get('transformer_heads', 8)
        transformer_layers = config.get('transformer_layers', 6)
        dropout_rate = config.get('dropout_rate', 0.3)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=transformer_heads,
            dim_feedforward=2048,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.shape
        
        # Project to model dimension
        x = x.unsqueeze(-1)  # Add feature dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Repeat positional encoding if sequence is longer
            pos_enc = self.pos_encoding.repeat(1, (seq_len // 1000) + 1, 1)
            x = x + pos_enc[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        return self.classifier(x)

class AdvancedModelEvaluator:
    """Advanced model evaluator for all architectures."""
    
    def __init__(self, model_path: str, model_type: str = "wav2vec2_enhanced"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        
        # Load config if available
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Load model and feature extractor
        self.model, self.feature_extractor = self._load_model()
        
        print(f"✅ Loaded {model_type} model from {model_path}")
        print(f"🎯 Device: {self.device}")
    
    def _get_default_config(self):
        """Get default configuration if config.json not found."""
        return {
            "dropout_rate": 0.3,
            "cnn_channels": [64, 128, 256],
            "lstm_hidden_size": 256,
            "lstm_layers": 2,
            "transformer_heads": 8,
            "transformer_layers": 6
        }
    
    def _load_model(self):
        """Load model based on type."""
        num_classes = len(label_map)
        
        if self.model_type == "wav2vec2_enhanced":
            return self._load_wav2vec2_model()
        elif self.model_type == "cnn_lstm":
            return self._load_cnn_lstm_model(num_classes)
        elif self.model_type == "transformer":
            return self._load_transformer_model(num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_wav2vec2_model(self):
        """Load Wav2Vec2 model."""
        try:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        except:
            # Fallback to base model if custom model fails
            print("⚠️ Could not load custom model, using base Wav2Vec2...")
            model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        model = model.to(self.device)
        model.eval()
        return model, feature_extractor
    
    def _load_cnn_lstm_model(self, num_classes):
        """Load CNN-LSTM model."""
        model = CNNLSTMClassifier(num_classes, self.config)
        
        # Load model weights
        model_file = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location=self.device))
        else:
            print(f"⚠️ Model weights not found at {model_file}")
        
        model = model.to(self.device)
        model.eval()
        return model, None
    
    def _load_transformer_model(self, num_classes):
        """Load Transformer model."""
        model = TransformerClassifier(num_classes, self.config)
        
        # Load model weights
        model_file = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location=self.device))
        else:
            print(f"⚠️ Model weights not found at {model_file}")
        
        model = model.to(self.device)
        model.eval()
        return model, None
    
    def preprocess_audio(self, file_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Load and preprocess audio segment to match training."""
        try:
            file_path = file_path.replace('\\', '/')
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            metadata = torchaudio.info(file_path)
            sr = metadata.sample_rate
            audio_length_sec = metadata.num_frames / sr

            # Add padding for context
            padding = 0.2
            start_time = max(0, start_time - padding)
            end_time = min(end_time + padding, audio_length_sec)

            if end_time <= start_time:
                return None

            audio, sr = torchaudio.load(
                file_path,
                frame_offset=int(start_time * sr),
                num_frames=int((end_time - start_time) * sr)
            )

            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample to 16kHz
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)

            audio_np = audio.squeeze().numpy()

            if len(audio_np) == 0:
                return None

            # Enhanced preprocessing (matching training)
            # Apply Hamming window
            audio_np = audio_np * np.hamming(len(audio_np))
            
            # Pre-emphasis
            audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)

            # Simple noise reduction
            noise_threshold = 0.005
            audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)

            # RMS normalization
            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms > 0:
                audio_np = audio_np / rms * 0.1

            # Z-score normalization
            audio_np = (audio_np - audio_np.mean()) / (audio_np.std() + 1e-8)

            return audio_np
            
        except Exception as e:
            print(f"Error preprocessing {file_path} ({start_time}-{end_time}): {e}")
            return None
    
    def evaluate_window(self, file_path: str, start_time: float, end_time: float, 
                       threshold: float = 0.5) -> Tuple[str, float]:
        """Evaluate a single window."""
        try:
            # Process audio
            audio = self.preprocess_audio(file_path, start_time, end_time)
            if audio is None:
                return "error", 0.0
            
            # Get prediction based on model type
            if self.model_type == "wav2vec2_enhanced":
                return self._evaluate_wav2vec2_window(audio, threshold)
            else:
                return self._evaluate_pytorch_window(audio, threshold)
                
        except Exception as e:
            print(f"Error evaluating window {file_path} ({start_time}-{end_time}): {str(e)}")
            return "error", 0.0
    
    def _evaluate_wav2vec2_window(self, audio: np.ndarray, threshold: float) -> Tuple[str, float]:
        """Evaluate window using Wav2Vec2 model."""
        # Apply feature extractor
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=16000
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predictions = torch.softmax(logits, dim=-1)
            
            # Apply threshold for binary classification
            none_prob = predictions[0][0].item()  # 'none' class probability
            
            if none_prob >= threshold:
                predicted_label = "none"
                confidence = none_prob
            else:
                # Find the most likely profanity class
                profanity_probs = predictions[0][1:]  # Exclude 'none' class
                profanity_label_id = torch.argmax(profanity_probs).item() + 1  # +1 because we excluded 'none'
                predicted_label = rev_label_map[profanity_label_id]
                confidence = profanity_probs[profanity_label_id - 1].item()
                
        return predicted_label, confidence
    
    def _evaluate_pytorch_window(self, audio: np.ndarray, threshold: float) -> Tuple[str, float]:
        """Evaluate window using PyTorch model (CNN-LSTM or Transformer)."""
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(audio_tensor)
            predictions = torch.softmax(logits, dim=-1)
            
            # Apply threshold for binary classification
            none_prob = predictions[0][0].item()  # 'none' class probability
            
            if none_prob >= threshold:
                predicted_label = "none"
                confidence = none_prob
            else:
                # Find the most likely profanity class
                profanity_probs = predictions[0][1:]  # Exclude 'none' class
                profanity_label_id = torch.argmax(profanity_probs).item() + 1  # +1 because we excluded 'none'
                predicted_label = rev_label_map[profanity_label_id]
                confidence = profanity_probs[profanity_label_id - 1].item()
                
        return predicted_label, confidence

def plot_confusion_matrix(true_labels, pred_labels, labels, output_path="plots/confusion_matrix_advanced.png"):
    """Plot confusion matrix with Thai font support."""
    setup_thai_font()
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='YlOrRd')
    
    plt.title('Advanced Model Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"📊 Confusion matrix saved to: {output_path}")

def plot_binary_confusion_matrix(true_binary, pred_binary, output_path="plots/binary_confusion_matrix.png"):
    """Plot binary confusion matrix for None vs Profanity classification."""
    setup_thai_font()
    
    # Create binary labels
    binary_labels = ['None', 'Profanity']
    
    # Generate confusion matrix
    cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with percentage annotations
    total_samples = cm.sum()
    cm_percent = cm / total_samples * 100
    
    # Create annotations with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            row.append(f'{count}\n({percent:.1f}%)')
        annotations.append(row)
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', 
                xticklabels=binary_labels, 
                yticklabels=binary_labels,
                cmap='Blues', cbar_kws={'label': 'Count'})
    
    plt.title('Binary Classification: None vs Profanity', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add performance metrics as text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Add metrics text box
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}\nSpecificity: {specificity:.3f}'
    plt.text(2.2, 1, metrics_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"📊 Binary confusion matrix saved to: {output_path}")
    
    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 
        'f1': f1, 'specificity': specificity
    }

def evaluate_windowed_dataset(evaluator: AdvancedModelEvaluator, eval_csv: str, 
                            thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
                            output_dir: str = "evaluation_results"):
    """Evaluate windowed dataset with multiple thresholds."""
    
    # Load windowed data
    if not os.path.exists(eval_csv):
        print(f"❌ Evaluation CSV not found: {eval_csv}")
        return
    
    df = pd.read_csv(eval_csv)
    print(f"📊 Loaded {len(df)} windowed samples from {eval_csv}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate for each threshold
    for threshold in thresholds:
        print(f"\n🎯 Evaluating with threshold: {threshold}")
        
        results = []
        
        # Process each window
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Threshold {threshold}"):
            predicted_label, confidence = evaluator.evaluate_window(
                row['file_path'],
                row['window_start'],
                row['window_end'],
                threshold
            )
            
            results.append({
                'file_path': row['file_path'],
                'window_start': row['window_start'],
                'window_end': row['window_end'],
                'true_label': row['label'],
                'predicted_label': predicted_label,
                'confidence': confidence,
                'threshold': threshold
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        threshold_str = str(threshold).replace('.', '_')
        results_file = os.path.join(output_dir, f"eval_results_threshold_{threshold_str}.csv")
        results_df.to_csv(results_file, index=False)
        
        # Calculate metrics
        calculate_metrics(results_df, threshold, output_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")

def calculate_metrics(results_df: pd.DataFrame, threshold: float, output_dir: str):
    """Calculate and save comprehensive metrics."""
    
    threshold_str = str(threshold).replace('.', '_')
    
    # Binary classification metrics
    results_df['is_true_profanity'] = results_df['true_label'] != 'none'
    results_df['is_predicted_profanity'] = results_df['predicted_label'] != 'none'
    results_df['is_correct'] = results_df['true_label'] == results_df['predicted_label']
    
    # Calculate confusion matrix values
    tp = ((results_df['is_true_profanity']) & 
          (results_df['is_predicted_profanity']) & 
          (results_df['is_correct'])).sum()
    
    fp = ((results_df['is_predicted_profanity']) & 
          (~results_df['is_correct'])).sum()
    
    tn = ((~results_df['is_true_profanity']) & 
          (~results_df['is_predicted_profanity'])).sum()
    
    fn = (results_df['is_true_profanity'] & 
          (~results_df['is_predicted_profanity'] | 
           ~results_df['is_correct'])).sum()
    
    # Calculate metrics
    accuracy = (tp + tn) / len(results_df) if len(results_df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Create metrics report
    metrics_report = f"""
🎯 ADVANCED MODEL EVALUATION REPORT
Threshold: {threshold}
================================

📊 BINARY CLASSIFICATION METRICS (None vs Profanity):
Accuracy: {accuracy:.4f}
Balanced Accuracy: {balanced_accuracy:.4f}
Precision: {precision:.4f}
Recall (Sensitivity): {recall:.4f}
Specificity: {specificity:.4f}
F1-Score: {f1:.4f}

📈 CONFUSION MATRIX VALUES:
True Positives (Profanity → Profanity): {tp}
False Positives (None → Profanity): {fp}
True Negatives (None → None): {tn}
False Negatives (Profanity → None): {fn}
Total Samples: {len(results_df)}

📋 CLASS DISTRIBUTION:
Profanity samples: {tp + fn} ({(tp + fn)/len(results_df):.2%})
None samples: {tn + fp} ({(tn + fp)/len(results_df):.2%})
"""
    
    # Save metrics report
    metrics_file = os.path.join(output_dir, f"metrics_threshold_{threshold_str}.txt")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(metrics_report)
    
    print(f"📋 Metrics for threshold {threshold}:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    
    # Generate binary confusion matrix (None vs Profanity)
    binary_true = (results_df['true_label'] != 'none').astype(int)  # 0 = None, 1 = Profanity
    binary_pred = (results_df['predicted_label'] != 'none').astype(int)  # 0 = None, 1 = Profanity
    
    binary_cm_file = os.path.join(output_dir, f"binary_confusion_matrix_threshold_{threshold_str}.png")
    binary_metrics = plot_binary_confusion_matrix(binary_true, binary_pred, binary_cm_file)
    
    # Add binary metrics to the report
    metrics_report += f"""
🎯 BINARY CONFUSION MATRIX BREAKDOWN:
True Negatives (None → None): {binary_metrics['tn']}
False Positives (None → Profanity): {binary_metrics['fp']}  
False Negatives (Profanity → None): {binary_metrics['fn']}
True Positives (Profanity → Profanity): {binary_metrics['tp']}

📊 DERIVED METRICS:
Accuracy: {binary_metrics['accuracy']:.4f}
Precision: {binary_metrics['precision']:.4f}
Recall/Sensitivity: {binary_metrics['recall']:.4f}
F1-Score: {binary_metrics['f1']:.4f}
Specificity: {binary_metrics['specificity']:.4f}
"""
    
    # Update the metrics file with binary metrics
    metrics_file = os.path.join(output_dir, f"metrics_threshold_{threshold_str}.txt")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(metrics_report)
    
    # Generate detailed confusion matrix for profanity classes (if requested)
    if tp + fn > 0:  # Only if there are profanity samples
        profanity_results = results_df[results_df['true_label'] != 'none'].copy()
        
        # Replace 'none' predictions with 'missed_profanity' for visualization
        profanity_results.loc[profanity_results['predicted_label'] == 'none', 'predicted_label'] = 'missed_profanity'
        
        # Get profanity labels that actually exist in the data
        actual_profanity_labels = profanity_results['true_label'].unique().tolist()
        predicted_labels = profanity_results['predicted_label'].unique().tolist()
        all_labels = list(set(actual_profanity_labels + predicted_labels))
        
        # Only create confusion matrix if we have valid labels
        if len(all_labels) > 0 and len(profanity_results) > 0:
            try:
                # Plot detailed profanity confusion matrix
                detailed_cm_file = os.path.join(output_dir, f"detailed_profanity_confusion_matrix_threshold_{threshold_str}.png")
                plot_confusion_matrix(
                    profanity_results['true_label'].values,
                    profanity_results['predicted_label'].values,
                    all_labels,
                    detailed_cm_file
                )
            except Exception as e:
                print(f"⚠️ Could not create detailed profanity confusion matrix: {e}")
        else:
            print(f"⚠️ No profanity predictions found for threshold {threshold}")
            print("   Consider lowering the threshold to improve detection")

def quick_binary_evaluation(model_path: str, eval_csv: str, model_type: str = "wav2vec2_enhanced", 
                           threshold: float = 0.5, output_dir: str = "quick_eval_results"):
    """Quick evaluation focused on binary classification only."""
    
    print(f"🚀 Quick Binary Evaluation")
    print(f"📁 Model: {model_path}")
    print(f"📊 Data: {eval_csv}")
    print(f"🎯 Threshold: {threshold}")
    
    # Initialize evaluator
    evaluator = AdvancedModelEvaluator(model_path, model_type)
    
    # Load data
    df = pd.read_csv(eval_csv)
    print(f"📋 Loaded {len(df)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate samples
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        predicted_label, confidence = evaluator.evaluate_window(
            row['file_path'],
            row['window_start'],
            row['window_end'],
            threshold
        )
        
        results.append({
            'true_label': row['label'],
            'predicted_label': predicted_label,
            'confidence': confidence
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create binary labels
    binary_true = (results_df['true_label'] != 'none').astype(int)
    binary_pred = (results_df['predicted_label'] != 'none').astype(int)
    
    # Generate binary confusion matrix
    binary_cm_file = os.path.join(output_dir, "binary_confusion_matrix.png")
    binary_metrics = plot_binary_confusion_matrix(binary_true, binary_pred, binary_cm_file)
    
    # Print results
    print(f"\n📊 BINARY CLASSIFICATION RESULTS:")
    print(f"   Accuracy: {binary_metrics['accuracy']:.4f}")
    print(f"   Precision: {binary_metrics['precision']:.4f}")
    print(f"   Recall: {binary_metrics['recall']:.4f}")
    print(f"   F1-Score: {binary_metrics['f1']:.4f}")
    print(f"   Specificity: {binary_metrics['specificity']:.4f}")
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    return binary_metrics

def main():
    parser = argparse.ArgumentParser(description="Advanced Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--model-type", type=str, 
                       choices=["wav2vec2_enhanced", "cnn_lstm", "transformer"],
                       default="wav2vec2_enhanced",
                       help="Type of model architecture")
    parser.add_argument("--eval-data", type=str, 
                       help="Path to windowed evaluation CSV file")
    parser.add_argument("--thresholds", nargs='+', type=float,
                       default=[0.3, 0.4, 0.5, 0.6, 0.7],
                       help="Thresholds to test")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_advanced",
                       help="Output directory for results")
    parser.add_argument("--single-threshold", type=float,
                       help="Evaluate with single threshold only")
    parser.add_argument("--quick-binary", action="store_true",
                       help="Quick binary evaluation (None vs Profanity only)")
    
    args = parser.parse_args()
    
    print(f"🚀 Advanced Model Evaluation")
    print(f"📁 Model path: {args.model_path}")
    print(f"🤖 Model type: {args.model_type}")
    print(f"🎯 Device: {device}")
    
    # Initialize evaluator
    try:
        evaluator = AdvancedModelEvaluator(args.model_path, args.model_type)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Determine evaluation data
    if args.eval_data:
        eval_csv = args.eval_data
    else:
        # Try to find windowed CSV files
        windowed_files = [
            "windowed_csv/windowed_0.5s_overlap_0.3s.csv",
            "windowed_csv/windowed_0.5s_overlap_0.4s.csv",
            "csv/eval_windowed_0.5s.csv",
            "csv/eval_windowed.csv"
        ]
        
        eval_csv = None
        for file in windowed_files:
            if os.path.exists(file):
                eval_csv = file
                break
        
        if eval_csv is None:
            print("❌ No windowed evaluation data found!")
            print("   Please specify --eval-data or create windowed data first")
            return
    
    print(f"📊 Evaluation data: {eval_csv}")
    
    # Determine thresholds
    if args.single_threshold:
        thresholds = [args.single_threshold]
    else:
        thresholds = args.thresholds
    
    print(f"🎯 Thresholds: {thresholds}")
    
    # Run evaluation
    evaluate_windowed_dataset(
        evaluator, 
        eval_csv, 
        thresholds, 
        args.output_dir
    )

if __name__ == "__main__":
    main()
