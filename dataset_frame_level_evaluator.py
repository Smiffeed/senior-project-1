#!/usr/bin/env python3
"""
🎯 DATASET-BASED ADVANCED FRAME-LEVEL EVALUATOR
Uses pre-computed windowed datasets from eval_by_0.05 and eval_percent directories
Combines frame-level detection with advanced preprocessing and VAD refinement
"""

import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
import json
import argparse
import os
import sys
import psutil
import multiprocessing
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_consistent_classification_report(y_true, y_pred, target_names=None, labels=None):
    """Create a classification report that shows both accuracy and micro avg consistently"""
    from sklearn.metrics import classification_report, accuracy_score
    import numpy as np
    
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return "No data available for classification report"
    
    try:
        # Handle case where target_names is provided but not all classes are present
        if target_names is not None and labels is None:
            # Get unique classes in predictions
            unique_classes = sorted(list(set(y_true + y_pred)))
            
            # If we have fewer unique classes than target names, adjust
            if len(unique_classes) < len(target_names):
                # Only use target names for classes that actually appear
                labels = unique_classes
                target_names = [target_names[i] if i < len(target_names) else f'class_{i}' 
                              for i in range(len(unique_classes))]
        
        # Get the base report
        report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, 
                                     zero_division=0, output_dict=False)
        
        # Calculate accuracy manually
        accuracy = accuracy_score(y_true, y_pred)
        
    except Exception as e:
        # Fallback: create a simple report without target names
        try:
            report = classification_report(y_true, y_pred, zero_division=0, output_dict=False)
            accuracy = accuracy_score(y_true, y_pred)
        except Exception as e2:
            return f"Unable to generate classification report: {str(e2)}"
    
    # For binary classification, add micro avg line
    # For multiclass classification, add accuracy line
    lines = report.strip().split('\n')
    
    # Find where to insert the accuracy/micro avg line
    insert_idx = -1
    for i, line in enumerate(lines):
        if 'macro avg' in line or 'weighted avg' in line:
            insert_idx = i
            break
    
    if insert_idx == -1:
        insert_idx = len(lines)
    
    # Check if we need to add accuracy line (for multiclass) or micro avg (for binary)
    has_accuracy = any('accuracy' in line and 'balanced' not in line for line in lines)
    has_micro_avg = any('micro avg' in line for line in lines)
    
    new_lines = lines[:]
    
    # Only add lines if we have meaningful data
    if len(y_true) > 0 and len(set(y_true + y_pred)) > 0:
        if not has_accuracy:
            # Add accuracy line for multiclass
            accuracy_line = f"    accuracy                           {accuracy:.2f}     {len(y_true)}"
            new_lines.insert(insert_idx, accuracy_line)
            insert_idx += 1
        
        if not has_micro_avg and len(set(y_true + y_pred)) <= 2:
            # Add micro avg line for binary (micro avg = accuracy in multiclass)
            micro_line = f"   micro avg       {accuracy:.2f}      {accuracy:.2f}      {accuracy:.2f}     {len(y_true)}"
            new_lines.insert(insert_idx, micro_line)
    
    return '\n'.join(new_lines)

def save_evaluation_note(output_dir: str, config: Dict, evaluation_results: Dict, 
                        y_true_binary: List[int], y_pred_binary: List[int],
                        y_true_multiclass: List[int], y_pred_multiclass: List[int],
                        processing_stats: Dict):
    """Save comprehensive evaluation note matching comprehensive_evaluation_processor.py format"""
    
    note_path = Path(output_dir) / "note.txt"
    
    try:
        with open(note_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("🎯 DATASET-BASED ADVANCED FRAME-LEVEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration
            f.write("📋 CONFIGURATION:\n")
            f.write(f"   • Evaluation Type: {config.get('eval_type', 'Unknown')}\n")
            f.write(f"   • Window Size: {config.get('window', 'Unknown')}\n")
            f.write(f"   • Stride: {config.get('stride', 'Unknown')}\n")
            f.write(f"   • Confidence Threshold: {config.get('confidence_threshold', 'Unknown')}\n")
            f.write(f"   • Model: Dataset-based evaluation with ensemble predictions\n")
            f.write(f"   • VAD Refinement: Enabled\n")
            f.write(f"   • Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Processing Statistics
            f.write("📊 PROCESSING STATISTICS:\n")
            f.write(f"   • Raw Windows: {processing_stats.get('raw_windows', 0):,}\n")
            f.write(f"   • Profanity Windows: {processing_stats.get('profanity_windows', 0):,}\n")
            f.write(f"   • Merged Predictions: {processing_stats.get('merged_predictions', 0):,}\n")
            f.write(f"   • Refined Predictions: {processing_stats.get('refined_predictions', 0):,}\n")
            f.write(f"   • Processing Time: {processing_stats.get('processing_time_seconds', 0):.2f}s\n\n")
            
            # Binary Classification Metrics
            f.write("🎯 BINARY CLASSIFICATION METRICS (Profanity vs Clean):\n")
            f.write("-" * 60 + "\n")
            
            f.write(f"Accuracy:         {evaluation_results.get('binary_accuracy', 0):.4f}\n")
            f.write(f"Precision:        {evaluation_results.get('binary_precision', 0):.4f}\n")
            f.write(f"Recall:           {evaluation_results.get('binary_recall', 0):.4f}\n")
            f.write(f"F1-Score:         {evaluation_results.get('binary_f1', 0):.4f}\n\n")
            
            # Binary Classification Report
            f.write("📝 BINARY CLASSIFICATION REPORT:\n")
            f.write("-" * 60 + "\n")
            if len(y_true_binary) > 0 and len(y_pred_binary) > 0:
                binary_report = create_consistent_classification_report(
                    y_true_binary, y_pred_binary, 
                    target_names=['Clean', 'Profanity'], 
                    labels=[0, 1]
                )
                f.write(binary_report + "\n\n")
            else:
                f.write("No binary classification data available\n\n")
            
            # Multiclass Classification Metrics
            f.write("🎯 MULTICLASS CLASSIFICATION METRICS (All Classes):\n")
            f.write("-" * 60 + "\n")
            
            f.write(f"Accuracy:         {evaluation_results.get('multiclass_accuracy', 0):.4f}\n")
            f.write(f"Macro Precision:  {evaluation_results.get('multiclass_precision', 0):.4f}\n")
            f.write(f"Macro Recall:     {evaluation_results.get('multiclass_recall', 0):.4f}\n")
            f.write(f"Macro F1-Score:   {evaluation_results.get('multiclass_f1', 0):.4f}\n\n")
            
            # Multiclass Classification Report
            f.write("📝 MULTICLASS CLASSIFICATION REPORT (All Classes):\n")
            f.write("-" * 60 + "\n")
            if len(y_true_multiclass) > 0 and len(y_pred_multiclass) > 0:
                multiclass_report = create_consistent_classification_report(
                    y_true_multiclass, y_pred_multiclass,
                    target_names=CLASS_NAMES,
                    labels=list(range(NUM_LABELS))
                )
                f.write(multiclass_report + "\n\n")
            else:
                f.write("No multiclass classification data available\n\n")
            
            # Profanity-Only Classification (if applicable)
            profanity_only = evaluation_results.get('profanity_only_classification', {})
            if profanity_only:
                f.write("🎯 PROFANITY-ONLY CLASSIFICATION METRICS:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Accuracy:         {profanity_only.get('accuracy', 0):.4f}\n")
                f.write(f"Balanced Accuracy: {profanity_only.get('balanced_accuracy', 0):.4f}\n")
                f.write(f"Macro Precision:  {profanity_only.get('macro_precision', 0):.4f}\n")
                f.write(f"Macro Recall:     {profanity_only.get('macro_recall', 0):.4f}\n")
                f.write(f"Macro F1-Score:   {profanity_only.get('macro_f1', 0):.4f}\n")
                f.write(f"Weighted F1-Score: {profanity_only.get('weighted_f1', 0):.4f}\n\n")
            
            # Mean IoU Metrics
            f.write("🎯 INTERSECTION OVER UNION (IoU) METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean IoU:         {evaluation_results.get('mean_iou', 0):.4f}\n")
            iou_by_class = evaluation_results.get('iou_by_class', {})
            for class_name, iou_val in iou_by_class.items():
                f.write(f"IoU {class_name:>12}: {iou_val:.4f}\n")
            f.write("\n")
            
            # Dataset Statistics
            f.write("📈 DATASET STATISTICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Windows:    {len(y_true_binary) if y_true_binary else 0:,}\n")
            
            if y_true_binary:
                clean_count = y_true_binary.count(0)
                profanity_count = y_true_binary.count(1)
                f.write(f"Clean Windows:    {clean_count:,} ({clean_count/len(y_true_binary)*100:.1f}%)\n")
                f.write(f"Profanity Windows: {profanity_count:,} ({profanity_count/len(y_true_binary)*100:.1f}%)\n")
            
            if y_true_multiclass:
                f.write("\nClass Distribution:\n")
                for class_idx, class_name in enumerate(CLASS_NAMES):
                    count = y_true_multiclass.count(class_idx)
                    if len(y_true_multiclass) > 0:
                        percentage = count / len(y_true_multiclass) * 100
                        f.write(f"  {class_name:>12}: {count:,} ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("🎯 End of Dataset-Based Advanced Frame-Level Evaluation Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print(f"📝 Saved evaluation note: {note_path}")
        
    except Exception as e:
        print(f"⚠️  Error saving evaluation note: {e}")
        # Create a minimal note file
        try:
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuration: {config}\n")
                f.write(f"Error generating full report: {e}\n")
        except:
            pass

class AdvancedAudioPreprocessor:
    """Advanced audio preprocessor from comprehensive_evaluation_processor.py"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def advanced_preprocess_audio(self, file_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Advanced audio preprocessing matching training pipeline"""
        try:
            # Load audio with precise timing
            duration = end_time - start_time
            audio, sr = librosa.load(file_path, sr=self.sample_rate, offset=start_time, duration=duration)
            
            # Apply noise reduction and preemphasis
            audio = librosa.effects.preemphasis(audio, coef=0.97)
            
            # Normalize to [-1, 1] range
            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Ensure minimum length (0.1 seconds)
            min_length = int(0.1 * self.sample_rate)
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
            
            return audio
            
        except Exception as e:
            print(f"Error preprocessing audio {file_path}[{start_time:.2f}-{end_time:.2f}]: {e}")
            # Return 0.1 second of silence as fallback
            return np.zeros(int(0.1 * self.sample_rate))

class EnhancedModelEnsemble:
    """Enhanced model ensemble for improved predictions"""
    
    def __init__(self, model_dir: str, model_name: str = "airesearch/wav2vec2-large-xlsr-53-th"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.models = []
        self.feature_extractor = None
        self.device = device
        
        # Load feature extractor and models
        self._load_feature_extractor()
        self._load_ensemble_models()
    
    def _load_feature_extractor(self):
        """Load feature extractor"""
        try:
            # Try to load from model directory first
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor_config.json')
            if os.path.exists(preprocessor_path):
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_dir, return_attention_mask=True, do_normalize=True
                )
                print(f"✅ Loaded feature extractor from: {self.model_dir}")
            else:
                # Fallback to base model
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_name, return_attention_mask=True, do_normalize=True
                )
                print(f"✅ Loaded feature extractor from base model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading feature extractor: {e}")
            raise
    
    def _load_ensemble_models(self, num_folds=5):
        """Load ensemble of models or single model"""
        model_path = Path(self.model_dir)
        
        # Check if this is a standard HuggingFace model directory
        hf_files = ['config.json', 'model.safetensors', 'preprocessor_config.json']
        has_hf_files = all((model_path / file).exists() for file in hf_files)
        
        if has_hf_files:
            # Load single HuggingFace model
            try:
                model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    self.model_dir, 
                    num_labels=NUM_LABELS,
                    ignore_mismatched_sizes=True
                ).to(self.device)
                model.eval()
                self.models.append(model)
                print(f"✅ Loaded HuggingFace model from: {self.model_dir}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load HuggingFace model: {e}")
        
        # Try to load ensemble from folds
        for fold in range(1, num_folds + 1):
            fold_dir = model_path / f"fold_{fold}"
            if fold_dir.exists() and (fold_dir / "config.json").exists():
                try:
                    model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        str(fold_dir), 
                        num_labels=NUM_LABELS,
                        ignore_mismatched_sizes=True
                    ).to(self.device)
                    model.eval()
                    self.models.append(model)
                    print(f"✅ Loaded fold {fold} model")
                except Exception as e:
                    print(f"⚠️  Failed to load fold {fold}: {e}")
        
        # Try other common paths if no folds found
        if not self.models:
            pytorch_files = list(model_path.glob("*.pth")) + list(model_path.glob("*.pt"))
            for model_file in pytorch_files[:3]:  # Limit to 3 models
                try:
                    # Load base model and then load state dict
                    model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        self.model_name, 
                        num_labels=NUM_LABELS,
                        ignore_mismatched_sizes=True
                    ).to(self.device)
                    
                    # Load custom weights
                    state_dict = torch.load(model_file, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models.append(model)
                    print(f"✅ Loaded PyTorch model: {model_file.name}")
                except Exception as e:
                    print(f"⚠️  Failed to load {model_file.name}: {e}")
        
        if not self.models:
            raise RuntimeError(f"❌ No models could be loaded from {self.model_dir}")
        
        print(f"🎯 Loaded {len(self.models)} model(s) for ensemble")
    
    def predict_window(self, audio: np.ndarray) -> Dict:
        """Make ensemble prediction for audio window"""
        try:
            # Extract features
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=16000  # 1 second max
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions from all models
            all_logits = []
            all_probabilities = []
            
            for model in self.models:
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    all_logits.append(logits)
                    all_probabilities.append(probabilities)
            
            # Ensemble averaging
            if len(all_probabilities) > 1:
                ensemble_probabilities = torch.stack(all_probabilities).mean(dim=0)
                uncertainty = torch.stack(all_probabilities).std(dim=0).mean().item()
            else:
                ensemble_probabilities = all_probabilities[0]
                uncertainty = 0.0
            
            # Get final prediction
            predicted_class = torch.argmax(ensemble_probabilities, dim=-1).item()
            confidence_scores = ensemble_probabilities.cpu().numpy()[0]
            max_confidence = confidence_scores[predicted_class]
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': REV_LABEL_MAP[predicted_class],
                'confidence_scores': confidence_scores,
                'max_confidence': max_confidence,
                'uncertainty': uncertainty
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {
                'predicted_class': 0,
                'predicted_label': 'none',
                'confidence_scores': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                'max_confidence': 1.0,
                'uncertainty': 1.0
            }

class VADRefinementProcessor:
    """VAD refinement processor from vad_post_processor.py"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def merge_overlapping_predictions(self, predictions_df: pd.DataFrame, overlap_threshold: float = 0.1) -> pd.DataFrame:
        """Merge overlapping predictions of the same label"""
        if len(predictions_df) == 0:
            return predictions_df
        
        # Sort by file and start time
        predictions_df = predictions_df.sort_values(['file_path', 'start_time']).reset_index(drop=True)
        merged_predictions = []
        
        # Group by file_path
        for file_path, file_group in predictions_df.groupby('file_path'):
            file_predictions = []
            
            # Group by predicted label (only merge same labels)
            for label, label_group in file_group.groupby('predicted_label'):
                if label == 'none':
                    continue  # Skip none predictions for merging
                
                label_predictions = label_group.sort_values('start_time').reset_index(drop=True)
                
                if len(label_predictions) == 0:
                    continue
                
                # Start with first prediction
                current_pred = {
                    'file_path': file_path,
                    'start_time': label_predictions.iloc[0]['start_time'],
                    'end_time': label_predictions.iloc[0]['end_time'],
                    'predicted_label': label,
                    'max_confidence': label_predictions.iloc[0]['max_confidence'],
                    'uncertainty': label_predictions.iloc[0].get('uncertainty', 0.0)
                }
                
                # Merge overlapping predictions
                for i in range(1, len(label_predictions)):
                    next_pred = label_predictions.iloc[i]
                    
                    # Check if overlapping or close enough
                    if next_pred['start_time'] <= current_pred['end_time'] + overlap_threshold:
                        # Merge: extend end time and update confidence
                        current_pred['end_time'] = max(current_pred['end_time'], next_pred['end_time'])
                        current_pred['max_confidence'] = max(current_pred['max_confidence'], next_pred['max_confidence'])
                        current_pred['uncertainty'] = min(current_pred['uncertainty'], next_pred.get('uncertainty', 0.0))
                    else:
                        # No overlap: save current and start new
                        file_predictions.append(current_pred.copy())
                        current_pred = {
                            'file_path': file_path,
                            'start_time': next_pred['start_time'],
                            'end_time': next_pred['end_time'],
                            'predicted_label': label,
                            'max_confidence': next_pred['max_confidence'],
                            'uncertainty': next_pred.get('uncertainty', 0.0)
                        }
                
                # Add final prediction
                file_predictions.append(current_pred)
            
            merged_predictions.extend(file_predictions)
        
        return pd.DataFrame(merged_predictions)
    
    def apply_vad_refinement(self, merged_predictions: pd.DataFrame, vad_top_db: int = 20) -> pd.DataFrame:
        """Apply VAD refinement to merged prediction boundaries"""
        refined_predictions = []
        
        print(f"🎯 Applying VAD refinement to {len(merged_predictions)} merged predictions...")
        
        for idx, pred in merged_predictions.iterrows():
            if idx % 50 == 0:
                print(f"  Processing prediction {idx+1}/{len(merged_predictions)}")
            
            try:
                # Load audio segment with padding
                file_path = pred['file_path']
                start_time = pred['start_time']
                end_time = pred['end_time']
                
                # Add padding around the prediction
                padding = 0.1  # 100ms padding
                padded_start = max(0, start_time - padding)
                padded_duration = end_time - padded_start + padding
                
                # Load audio
                audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                       offset=padded_start, duration=padded_duration)
                
                if len(audio) == 0:
                    # Keep original if no audio
                    refined_pred = pred.to_dict().copy()
                    refined_pred['vad_refined'] = False
                    refined_predictions.append(refined_pred)
                    continue
                
                # Apply VAD to find speech segments
                speech_intervals = librosa.effects.split(audio, top_db=vad_top_db)
                
                if len(speech_intervals) == 0:
                    # Keep original if no speech detected
                    refined_pred = pred.to_dict().copy()
                    refined_pred['vad_refined'] = False
                    refined_predictions.append(refined_pred)
                    continue
                
                # Convert speech intervals to absolute time
                speech_segments = []
                for interval_start, interval_end in speech_intervals:
                    abs_start = padded_start + (interval_start / sr)
                    abs_end = padded_start + (interval_end / sr)
                    
                    # Only keep segments that overlap with original prediction
                    if abs_end > start_time and abs_start < end_time:
                        # Constrain to original boundaries
                        refined_start = max(abs_start, start_time)
                        refined_end = min(abs_end, end_time)
                        if refined_end > refined_start:
                            speech_segments.append((refined_start, refined_end))
                
                if not speech_segments:
                    # Keep original if no overlapping speech
                    refined_pred = pred.to_dict().copy()
                    refined_pred['vad_refined'] = False
                    refined_predictions.append(refined_pred)
                    continue
                
                # Merge close speech segments
                speech_segments.sort()
                merged_segments = []
                current_start, current_end = speech_segments[0]
                
                for seg_start, seg_end in speech_segments[1:]:
                    if seg_start <= current_end + 0.1:  # 100ms gap tolerance
                        current_end = max(current_end, seg_end)
                    else:
                        merged_segments.append((current_start, current_end))
                        current_start, current_end = seg_start, seg_end
                
                merged_segments.append((current_start, current_end))
                
                # Create refined prediction(s)
                for i, (refined_start, refined_end) in enumerate(merged_segments):
                    refined_pred = pred.to_dict().copy()
                    refined_pred['start_time'] = refined_start
                    refined_pred['end_time'] = refined_end
                    refined_pred['vad_refined'] = True
                    refined_pred['segment_id'] = i if len(merged_segments) > 1 else 0
                    refined_predictions.append(refined_pred)
            
            except Exception as e:
                print(f"  WARNING: VAD refinement failed for {pred['file_path']} [{start_time:.2f}-{end_time:.2f}]: {e}")
                # Keep original prediction if VAD fails
                refined_pred = pred.to_dict().copy()
                refined_pred['vad_refined'] = False
                refined_predictions.append(refined_pred)
        
        print(f"✨ VAD refinement complete: {len(refined_predictions)} refined predictions")
        return pd.DataFrame(refined_predictions)

def find_dataset_configurations(datasets: List[str]) -> List[Dict]:
    """Find all CSV configurations in dataset directories"""
    configurations = []
    
    for dataset_dir in datasets:
        dataset_path = Path(dataset_dir)
        eval_type = dataset_path.name
        
        if not dataset_path.exists():
            print(f"⚠️  Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all window directories
        window_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('window_')]
        
        for window_dir in window_dirs:
            window_size = window_dir.name  # e.g., "window_0.5s"
            
            # Find all stride CSV files
            csv_files = list(window_dir.glob("stride_*.csv"))
            
            for csv_file in csv_files:
                stride_name = csv_file.stem  # e.g., "stride_0.25s"
                
                # Extract window size value
                window_size_value = float(window_size.replace('window_', '').replace('s', ''))
                
                # Extract stride value based on format
                try:
                    if '%' in stride_name:
                        # Handle percentage format: "stride_10.0%" -> 10.0
                        stride_value = float(stride_name.replace('stride_', '').replace('%', ''))
                    else:
                        # Handle time format: "stride_0.25s" -> 0.25
                        stride_value = float(stride_name.replace('stride_', '').replace('s', ''))
                except ValueError as e:
                    print(f"⚠️  Skipping invalid file: {csv_file} (could not convert string to float: '{stride_name.replace('stride_', '').replace('s', '')}')")
                    continue
                
                configurations.append({
                    'eval_type': eval_type,
                    'window': window_size,
                    'stride': stride_name,
                    'csv_path': str(csv_file),
                    'window_size_value': window_size_value,
                    'stride_value': stride_value
                })
    
    return sorted(configurations, key=lambda x: (x['eval_type'], x['window_size_value'], x['stride_value']))

def process_single_configuration_thread_safe(task_info: Tuple) -> Dict:
    """Process a single configuration in thread-safe manner"""
    config, model_dir, ground_truth_df, confidence_threshold, model_name, output_base_dir, worker_id = task_info
    
    csv_path = config['csv_path']
    eval_type = config['eval_type']
    window = config['window']
    stride = config['stride']
    
    worker_prefix = f"[W{worker_id}]"
    
    try:
        start_time = time.time()
        
        print(f"{worker_prefix} Processing: {eval_type}/{window}/{stride}")
        
        # Load windowed data
        df = pd.read_csv(csv_path)
        print(f"{worker_prefix} Loaded {len(df)} windows from {csv_path}")
        
        # Initialize components for this thread
        preprocessor = AdvancedAudioPreprocessor()
        model_ensemble = EnhancedModelEnsemble(model_dir, model_name)
        vad_processor = VADRefinementProcessor()
        
        # Process all windows with model predictions
        predictions = []
        
        print(f"{worker_prefix} Making model predictions...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Worker {worker_id}", leave=False):
            try:
                # Preprocess audio window
                audio_window = preprocessor.advanced_preprocess_audio(
                    row['file_path'], row['start_time'], row['end_time']
                )
                
                # Get prediction
                prediction = model_ensemble.predict_window(audio_window)
                
                # Add window information
                prediction.update({
                    'file_path': row['file_path'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'true_label': row['label'],
                    'window_id': idx
                })
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"{worker_prefix} ⚠️  Error processing window {idx}: {e}")
                # Add fallback prediction
                predictions.append({
                    'file_path': row['file_path'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'true_label': row['label'],
                    'predicted_class': 0,
                    'predicted_label': 'none',
                    'max_confidence': 0.0,
                    'uncertainty': 1.0,
                    'window_id': idx
                })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Filter high-confidence profanity predictions
        profanity_predictions = predictions_df[
            (predictions_df['predicted_label'] != 'none') & 
            (predictions_df['max_confidence'] >= confidence_threshold)
        ].copy()
        
        print(f"{worker_prefix} High-confidence profanity windows: {len(profanity_predictions)}")
        
        # Step 1: Merge overlapping predictions
        merged_predictions = vad_processor.merge_overlapping_predictions(profanity_predictions)
        print(f"{worker_prefix} After merging: {len(merged_predictions)}")
        
        # Step 2: Apply VAD refinement
        refined_predictions = vad_processor.apply_vad_refinement(merged_predictions)
        print(f"{worker_prefix} After VAD refinement: {len(refined_predictions)}")
        
        # Step 3: Evaluate against ground truth
        evaluation_results = evaluate_predictions(refined_predictions, ground_truth_df, predictions_df)
        
        # Create output for this configuration
        config_output = {
            'config': config,
            'raw_windows': len(predictions_df),
            'profanity_windows': len(profanity_predictions),
            'merged_predictions': len(merged_predictions),
            'refined_predictions': len(refined_predictions),
            'evaluation_results': evaluation_results,
            'predictions_df': predictions_df,
            'merged_predictions_df': merged_predictions,
            'refined_predictions_df': refined_predictions
        }
        
        # Save results to output directory
        config_name = f"{eval_type}_{window}_{stride}"
        config_output_dir = Path(output_base_dir) / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV files
        predictions_df.to_csv(config_output_dir / 'raw_predictions.csv', index=False)
        merged_predictions.to_csv(config_output_dir / 'merged_predictions.csv', index=False)
        refined_predictions.to_csv(config_output_dir / 'refined_predictions.csv', index=False)
        
        # Save evaluation summary
        processing_stats = {
            'raw_windows': len(predictions_df),
            'profanity_windows': len(profanity_predictions),
            'merged_predictions': len(merged_predictions),
            'refined_predictions': len(refined_predictions),
            'processing_time_seconds': time.time() - start_time
        }
        
        with open(config_output_dir / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
            json.dump({
                'configuration': config,
                'processing_stats': processing_stats,
                'evaluation_metrics': evaluation_results,
                'processing_time_seconds': time.time() - start_time
            }, f, indent=2, ensure_ascii=False)
        
        # Generate comprehensive evaluation note
        try:
            # Extract detailed matches from evaluation results for note generation
            detailed_matches = evaluation_results.get('detailed_matches', [])
            
            y_true_binary = []
            y_pred_binary = []
            y_true_multiclass = []
            y_pred_multiclass = []
            
            # Convert detailed matches to lists for note generation
            for match in detailed_matches:
                gt_label = match['gt_label']
                pred_label = match['pred_label']
                
                # Convert to numeric labels
                true_label_num = LABEL_MAP.get(gt_label, 0)
                pred_label_num = LABEL_MAP.get(pred_label, 0)
                
                # Binary classification (profanity vs clean)
                y_true_binary.append(1 if true_label_num > 0 else 0)
                y_pred_binary.append(1 if pred_label_num > 0 else 0)
                
                # Multiclass classification
                y_true_multiclass.append(true_label_num)
                y_pred_multiclass.append(pred_label_num)
            
            # Save comprehensive note
            save_evaluation_note(
                str(config_output_dir), 
                config, 
                evaluation_results,
                y_true_binary, 
                y_pred_binary,
                y_true_multiclass, 
                y_pred_multiclass,
                processing_stats
            )
        except Exception as e:
            print(f"{worker_prefix} ⚠️  Error generating evaluation note: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{worker_prefix} ✅ SUCCESS {eval_type}/{window}/{stride} ({duration:.1f}s)")
        
        return {
            'config': config,
            'success': True,
            'duration': duration,
            'metrics': evaluation_results,
            'output_dir': str(config_output_dir),
            'worker_id': worker_id
        }
        
    except Exception as e:
        print(f"{worker_prefix} ❌ ERROR {eval_type}/{window}/{stride}: {e}")
        return {
            'config': config,
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e),
            'worker_id': worker_id
        }

def evaluate_predictions(refined_predictions: pd.DataFrame, ground_truth_df: pd.DataFrame, raw_predictions_df: pd.DataFrame) -> Dict:
    """Evaluate predictions against ground truth with comprehensive metrics"""
    
    if len(refined_predictions) == 0:
        print("⚠️  No refined predictions to evaluate")
        return {
            'mean_iou': 0.0,
            'binary_f1': 0.0,
            'multiclass_f1': 0.0,
            'total_predictions': 0,
            'total_ground_truth': len(ground_truth_df)
        }
    
    # Calculate IoU for each ground truth word against all refined predictions
    detailed_matches = []
    
    for _, gt_row in ground_truth_df.iterrows():
        gt_start = gt_row['start_time']
        gt_end = gt_row['end_time']
        gt_label = gt_row['label']
        gt_file = gt_row.get('file_path', gt_row.get('audio_file', ''))
        
        # Find best matching prediction for this ground truth
        best_iou = 0.0
        best_pred_label = 'none'
        best_confidence = 0.0
        
        # Check refined predictions from the same file
        file_refined_preds = refined_predictions[refined_predictions['file_path'] == gt_file]
        
        for _, pred_row in file_refined_preds.iterrows():
            pred_start = pred_row['start_time']
            pred_end = pred_row['end_time']
            pred_label = pred_row['predicted_label']
            pred_confidence = pred_row['max_confidence']
            
            # Calculate IoU
            iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)
            
            if iou > best_iou:
                best_iou = iou
                best_pred_label = pred_label
                best_confidence = pred_confidence
        
        detailed_matches.append({
            'gt_start': gt_start,
            'gt_end': gt_end,
            'gt_label': gt_label,
            'pred_label': best_pred_label,
            'iou': best_iou,
            'confidence': best_confidence
        })
    
    # Calculate metrics
    gt_labels = [match['gt_label'] for match in detailed_matches]
    pred_labels = [match['pred_label'] for match in detailed_matches]
    ious = [match['iou'] for match in detailed_matches]
    
    # Binary metrics
    gt_binary = [1 if label != 'none' else 0 for label in gt_labels]
    pred_binary = [1 if label != 'none' else 0 for label in pred_labels]
    
    binary_accuracy = accuracy_score(gt_binary, pred_binary)
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        gt_binary, pred_binary, average='binary', zero_division=0
    )
    
    # Multiclass metrics (profanity only)
    profanity_gt = [label for label in gt_labels if label != 'none']
    profanity_pred = [pred_labels[i] for i, label in enumerate(gt_labels) if label != 'none']
    
    if profanity_gt and profanity_pred:
        multiclass_accuracy = accuracy_score(profanity_gt, profanity_pred)
        multiclass_precision, multiclass_recall, multiclass_f1, _ = precision_recall_fscore_support(
            profanity_gt, profanity_pred, average='weighted', zero_division=0
        )
    else:
        multiclass_accuracy = multiclass_precision = multiclass_recall = multiclass_f1 = 0.0
    
    # IoU metrics
    mean_iou = np.mean(ious) if ious else 0.0
    
    # IoU by class
    iou_by_class = {}
    for label in ['เย็ด', 'กู', 'มึง', 'เหี้ย']:
        class_ious = [match['iou'] for match in detailed_matches if match['gt_label'] == label]
        iou_by_class[label] = np.mean(class_ious) if class_ious else 0.0
    
    return {
        'total_ground_truth': len(detailed_matches),
        'total_predictions': len(refined_predictions),
        'mean_iou': mean_iou,
        'iou_by_class': iou_by_class,
        'binary_accuracy': binary_accuracy,
        'binary_precision': binary_precision,
        'binary_recall': binary_recall,
        'binary_f1': binary_f1,
        'multiclass_accuracy': multiclass_accuracy,
        'multiclass_precision': multiclass_precision,
        'multiclass_recall': multiclass_recall,
        'multiclass_f1': multiclass_f1,
        'detailed_matches': detailed_matches
    }

def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Calculate IoU between prediction and ground truth segments"""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    return intersection / union if union > 0 else 0.0

def get_system_info():
    """Get system information for optimal worker calculation"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative worker calculation for GPU-intensive tasks
    if memory_gb < 8:
        recommended_workers = max(1, cpu_count // 4)
    elif memory_gb < 16:
        recommended_workers = max(2, cpu_count // 3)
    else:
        recommended_workers = max(2, cpu_count // 2)
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'recommended_workers': recommended_workers
    }

def main():
    parser = argparse.ArgumentParser(description="Dataset-based Advanced Frame-Level Evaluator")
    parser.add_argument("--datasets", nargs='+', default=["csv/eval_by_0.05", "csv/eval_percent"], 
                       help="Dataset directories to process")
    parser.add_argument("--model_dir", default="models/4_classes_max_steps", help="Model directory")
    parser.add_argument("--ground_truth", default="csv/eval_5labels.csv", help="Ground truth CSV")
    parser.add_argument("--output_dir", default="dataset_frame_evaluation_results", help="Output directory")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--model_name", default="airesearch/wav2vec2-large-xlsr-53-th", help="Base model name")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--max_configs", type=int, default=None, help="Maximum configurations to process")
    parser.add_argument("--specific_windows", nargs='+', help="Specific windows to process (e.g., 0.3s 0.5s)")
    parser.add_argument("--specific_strides", nargs='+', help="Specific strides to process (e.g., 0.25s 0.5s)")
    parser.add_argument("--list_configs", action="store_true", help="List available configurations")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with limited configs")
    
    args = parser.parse_args()
    
    # Get system info
    system_info = get_system_info()
    if args.workers is None:
        args.workers = min(system_info['recommended_workers'], 3)  # Limit for GPU memory
    
    # Find all configurations
    all_configs = find_dataset_configurations(args.datasets)
    
    if args.list_configs:
        print("📋 AVAILABLE CONFIGURATIONS:")
        print("=" * 80)
        for config in all_configs:
            print(f"{config['eval_type']:<15} {config['window']:<12} {config['stride']:<12} {config['csv_path']}")
        print(f"\nTotal: {len(all_configs)} configurations")
        return 0
    
    if not all_configs:
        print("❌ No configurations found!")
        print("💡 Check that dataset directories exist and contain window_*/stride_*.csv files")
        return 1
    
    # Apply filters
    filtered_configs = all_configs.copy()
    
    if args.specific_windows:
        window_set = set(f"window_{w}" if not w.startswith('window_') else w for w in args.specific_windows)
        filtered_configs = [c for c in filtered_configs if c['window'] in window_set]
        print(f"🔍 Filtered by windows {args.specific_windows}: {len(filtered_configs)} configs")
    
    if args.specific_strides:
        stride_set = set(f"stride_{s}" if not s.startswith('stride_') else s for s in args.specific_strides)
        filtered_configs = [c for c in filtered_configs if c['stride'] in stride_set]
        print(f"🔍 Filtered by strides {args.specific_strides}: {len(filtered_configs)} configs")
    
    if args.test_mode:
        # Filter for window 2.0s for faster evaluation
        test_configs = [c for c in filtered_configs if c['window'] == 'window_2.0s']
        if not test_configs:
            # Fallback to any available configs if no 2.0s window found
            test_configs = filtered_configs[:5]
            print(f"🧪 TEST MODE: No window_2.0s found, using first {len(test_configs)} configurations")
        else:
            # Limit to reasonable number of 2.0s window configs
            test_configs = test_configs[:3]
            print(f"🧪 TEST MODE: Using window_2.0s for faster evaluation - {len(test_configs)} configurations")
        filtered_configs = test_configs
    
    if args.max_configs:
        filtered_configs = filtered_configs[:args.max_configs]
        print(f"📊 Limited to {len(filtered_configs)} configurations")
    
    configs_to_run = filtered_configs
    
    print("🎯 DATASET-BASED ADVANCED FRAME-LEVEL EVALUATOR")
    print("=" * 80)
    print(f"Model Directory: {args.model_dir}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Workers: {args.workers}")
    print(f"Configurations to process: {len(configs_to_run)}")
    print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_gb']:.1f}GB RAM")
    print("=" * 80)
    
    # Check required files
    required_files = [args.model_dir, args.ground_truth]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Required file not found: {file_path}")
            return 1
    
    # Load ground truth
    try:
        ground_truth_df = pd.read_csv(args.ground_truth)
        print(f"✅ Loaded ground truth: {len(ground_truth_df)} entries")
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show configurations to process
    if len(configs_to_run) <= 10:
        print(f"\n🎯 CONFIGURATIONS TO PROCESS:")
        for i, config in enumerate(configs_to_run, 1):
            print(f"{i:2d}. {config['eval_type']}/{config['window']}/{config['stride']}")
    else:
        print(f"\n🎯 Processing {len(configs_to_run)} configurations (showing first 5):")
        for i, config in enumerate(configs_to_run[:5], 1):
            print(f"{i:2d}. {config['eval_type']}/{config['window']}/{config['stride']}")
        print(f"    ... and {len(configs_to_run) - 5} more")
    
    # Prepare tasks
    tasks = []
    for i, config in enumerate(configs_to_run):
        worker_id = (i % args.workers) + 1
        task = (config, args.model_dir, ground_truth_df, args.confidence_threshold, 
                args.model_name, args.output_dir, worker_id)
        tasks.append(task)
    
    print(f"\n🚀 Processing {len(tasks)} configurations with {args.workers} workers...")
    
    # Process configurations in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(process_single_configuration_thread_safe, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            
            progress = completed / len(tasks) * 100
            print(f"📊 Progress: {completed}/{len(tasks)} ({progress:.1f}%)")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"⏱️  Total Time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    
    # Calculate summary statistics
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed_results)}")
    
    if successful_results:
        # Calculate performance metrics
        sequential_time = sum(r['duration'] for r in results)
        speedup = sequential_time / total_duration if total_duration > 0 else 1
        efficiency = speedup / args.workers * 100 if args.workers > 0 else 0
        
        print(f"🚀 Speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
        
        # Calculate aggregate metrics
        all_binary_f1 = [r['metrics']['binary_f1'] for r in successful_results if 'metrics' in r]
        all_multiclass_f1 = [r['metrics']['multiclass_f1'] for r in successful_results if 'metrics' in r]
        all_mean_iou = [r['metrics']['mean_iou'] for r in successful_results if 'metrics' in r]
        
        if all_binary_f1:
            print(f"📊 Average Binary F1: {np.mean(all_binary_f1):.3f} ± {np.std(all_binary_f1):.3f}")
        if all_multiclass_f1:
            print(f"📊 Average Multiclass F1: {np.mean(all_multiclass_f1):.3f} ± {np.std(all_multiclass_f1):.3f}")
        if all_mean_iou:
            print(f"📊 Average Mean IoU: {np.mean(all_mean_iou):.3f} ± {np.std(all_mean_iou):.3f}")
    
    # Save batch summary
    summary_data = []
    for result in results:
        if result.get('success', False):
            config = result['config']
            metrics = result.get('metrics', {})
            summary_data.append({
                'eval_type': config['eval_type'],
                'window': config['window'],
                'stride': config['stride'],
                'window_size_value': config['window_size_value'],
                'stride_value': config['stride_value'],
                'success': True,
                'duration_seconds': result['duration'],
                'mean_iou': metrics.get('mean_iou', 0),
                'binary_f1': metrics.get('binary_f1', 0),
                'binary_precision': metrics.get('binary_precision', 0),
                'binary_recall': metrics.get('binary_recall', 0),
                'multiclass_f1': metrics.get('multiclass_f1', 0),
                'multiclass_precision': metrics.get('multiclass_precision', 0),
                'multiclass_recall': metrics.get('multiclass_recall', 0),
                'total_predictions': metrics.get('total_predictions', 0),
                'total_ground_truth': metrics.get('total_ground_truth', 0),
                'output_dir': result.get('output_dir', '')
            })
        else:
            config = result['config']
            summary_data.append({
                'eval_type': config['eval_type'],
                'window': config['window'],
                'stride': config['stride'],
                'window_size_value': config['window_size_value'],
                'stride_value': config['stride_value'],
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'duration_seconds': result.get('duration', 0)
            })
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "batch_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed batch report
    batch_report = {
        'total_configurations': len(results),
        'successful_configurations': len(successful_results),
        'failed_configurations': len(failed_results),
        'total_duration_seconds': total_duration,
        'sequential_duration_seconds': sum(r['duration'] for r in results),
        'speedup': sequential_time / total_duration if total_duration > 0 else 1,
        'efficiency_percent': speedup / args.workers * 100 if args.workers > 0 else 0,
        'workers_used': args.workers,
        'system_info': system_info,
        'processing_settings': {
            'confidence_threshold': args.confidence_threshold,
            'model_dir': args.model_dir,
            'ground_truth': args.ground_truth,
            'datasets': args.datasets
        },
        'timestamp': datetime.now().isoformat()
    }
    
    batch_report_path = output_dir / "batch_evaluation_report.json"
    with open(batch_report_path, 'w', encoding='utf-8') as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   Summary CSV: {summary_path}")
    print(f"   Batch Report: {batch_report_path}")
    print(f"   Individual Results: {output_dir}/")
    
    # Show failed configurations
    if failed_results:
        print(f"\n⚠️  FAILED CONFIGURATIONS:")
        for result in failed_results[:5]:  # Show first 5 failures
            config = result['config']
            print(f"   {config['eval_type']}/{config['window']}/{config['stride']}: {result.get('error', 'Unknown error')[:100]}")
        if len(failed_results) > 5:
            print(f"   ... and {len(failed_results) - 5} more")
    
    # Show top performers
    successful_df = summary_df[summary_df['success'] == True]
    if len(successful_df) > 0:
        print(f"\n🏆 TOP PERFORMERS:")
        
        # Best Binary F1
        if 'binary_f1' in successful_df.columns:
            best_binary = successful_df.nlargest(3, 'binary_f1')
            print(f"\nBest Binary F1:")
            for _, row in best_binary.iterrows():
                print(f"   {row['eval_type']}/{row['window']}/{row['stride']}: {row['binary_f1']:.3f}")
        
        # Best Mean IoU
        if 'mean_iou' in successful_df.columns:
            best_iou = successful_df.nlargest(3, 'mean_iou')
            print(f"\nBest Mean IoU:")
            for _, row in best_iou.iterrows():
                print(f"   {row['eval_type']}/{row['window']}/{row['stride']}: {row['mean_iou']:.3f}")
    
    print(f"\n🎯 Evaluation complete! Check {output_dir} for detailed results.")
    return 0

if __name__ == "__main__":
    sys.exit(main())