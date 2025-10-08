#!/usr/bin/env python3
"""
🎯 ADVANCED FRAME-LEVEL EVALUATION SYSTEM
Combines the best aspects of frame_level_censor.py, comprehensive_evaluation_processor.py, 
fixed_smart_parallel_processor.py, and vad_post_processor.py

Features:
- Frame-level detection using sliding windows (like frame_level_censor.py)
- Advanced audio preprocessing (from comprehensive_evaluation_processor.py)
- Parallel processing for efficiency (from fixed_smart_parallel_processor.py)
- VAD refinement for precise boundaries (from vad_post_processor.py)
- Ensemble model support for better accuracy
- Comprehensive evaluation with multiple metrics
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

class AdvancedAudioPreprocessor:
    """Advanced audio preprocessor combining best practices from comprehensive_evaluation_processor.py"""
    
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

class AdvancedFrameLevelEvaluator:
    """
    Advanced frame-level evaluator combining best practices from all reference files
    """
    
    def __init__(self, model_dir: str, window_size: float = 0.5, stride: float = 0.25, 
                 confidence_threshold: float = 0.5, model_name: str = "airesearch/wav2vec2-large-xlsr-53-th"):
        """
        Initialize the advanced evaluator
        
        Args:
            model_dir: Directory containing trained models
            window_size: Window size in seconds
            stride: Stride in seconds  
            confidence_threshold: Minimum confidence for detection
            model_name: Base model name
        """
        self.model_dir = model_dir
        self.window_size = window_size
        self.stride = stride
        self.sample_rate = 16000
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Initialize components
        self.preprocessor = AdvancedAudioPreprocessor(sample_rate=self.sample_rate)
        self.model_ensemble = EnhancedModelEnsemble(model_dir, model_name)
        self.vad_processor = VADRefinementProcessor(sample_rate=self.sample_rate)
        
        print(f"🎯 Initialized AdvancedFrameLevelEvaluator")
        print(f"   Window size: {window_size}s, Stride: {stride}s")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Models loaded: {len(self.model_ensemble.models)}")
    
    def evaluate_audio_file(self, audio_file_path: str, ground_truth_df: pd.DataFrame) -> Dict:
        """
        Evaluate a single audio file using frame-level detection
        
        Args:
            audio_file_path: Path to audio file
            ground_truth_df: Ground truth dataframe for this file
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"🎵 Evaluating: {os.path.basename(audio_file_path)}")
        
        # Load audio to get duration
        try:
            audio_info = librosa.get_duration(path=audio_file_path)
            audio_duration = audio_info
        except Exception as e:
            print(f"⚠️  Could not get audio duration: {e}")
            return {'error': f'Could not load audio file: {e}'}
        
        # Generate sliding windows
        window_predictions = []
        window_samples = int(self.window_size * self.sample_rate)
        stride_samples = int(self.stride * self.sample_rate)
        
        num_windows = int((audio_duration * self.sample_rate - window_samples) / stride_samples) + 1
        
        print(f"   Duration: {audio_duration:.2f}s, Windows: {num_windows}")
        
        # Process each window
        for i in tqdm(range(num_windows), desc="Processing windows"):
            start_time = i * self.stride
            end_time = start_time + self.window_size
            
            # Skip if window extends beyond audio
            if end_time > audio_duration:
                end_time = audio_duration
                if end_time - start_time < 0.1:  # Skip very short windows
                    continue
            
            # Preprocess audio window
            audio_window = self.preprocessor.advanced_preprocess_audio(
                audio_file_path, start_time, end_time
            )
            
            # Get prediction
            prediction = self.model_ensemble.predict_window(audio_window)
            
            # Add timing information
            prediction.update({
                'file_path': audio_file_path,
                'start_time': start_time,
                'end_time': end_time,
                'window_id': i
            })
            
            window_predictions.append(prediction)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(window_predictions)
        
        # Filter high-confidence profanity predictions
        profanity_predictions = predictions_df[
            (predictions_df['predicted_label'] != 'none') & 
            (predictions_df['max_confidence'] >= self.confidence_threshold)
        ].copy()
        
        print(f"   Raw windows: {len(predictions_df)}")
        print(f"   High-confidence profanity: {len(profanity_predictions)}")
        
        # Step 1: Merge overlapping predictions
        merged_predictions = self.vad_processor.merge_overlapping_predictions(profanity_predictions)
        print(f"   After merging: {len(merged_predictions)}")
        
        # Step 2: Apply VAD refinement
        refined_predictions = self.vad_processor.apply_vad_refinement(merged_predictions)
        print(f"   After VAD refinement: {len(refined_predictions)}")
        
        # Step 3: Evaluate against ground truth
        evaluation_results = self._evaluate_predictions(refined_predictions, ground_truth_df, audio_file_path)
        
        return {
            'audio_file': audio_file_path,
            'raw_windows': len(predictions_df),
            'profanity_windows': len(profanity_predictions), 
            'merged_predictions': len(merged_predictions),
            'refined_predictions': len(refined_predictions),
            'evaluation_results': evaluation_results,
            'raw_predictions_df': predictions_df,
            'merged_predictions_df': merged_predictions,
            'refined_predictions_df': refined_predictions
        }
    
    def _evaluate_predictions(self, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, audio_file: str) -> Dict:
        """Evaluate predictions against ground truth"""
        
        # Filter ground truth for this file
        file_column = 'file_path' if 'file_path' in ground_truth_df.columns else 'audio_file'
        file_gt = ground_truth_df[ground_truth_df[file_column] == audio_file].copy()
        
        if len(file_gt) == 0:
            print(f"⚠️  No ground truth found for {audio_file}")
            return {'error': 'No ground truth found'}
        
        # Calculate IoU for each ground truth word
        detailed_matches = []
        
        for _, gt_row in file_gt.iterrows():
            gt_start = gt_row['start_time']
            gt_end = gt_row['end_time']
            gt_label = gt_row['label']
            
            # Find best matching prediction
            best_iou = 0.0
            best_pred_label = 'none'
            best_confidence = 0.0
            
            for _, pred_row in predictions_df.iterrows():
                pred_start = pred_row['start_time']
                pred_end = pred_row['end_time']
                pred_label = pred_row['predicted_label']
                pred_confidence = pred_row['max_confidence']
                
                # Calculate IoU
                iou = self._calculate_iou(pred_start, pred_end, gt_start, gt_end)
                
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
            'total_predictions': len(predictions_df),
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
    
    def _calculate_iou(self, pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
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
    
    # Conservative worker calculation
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

def process_single_audio_file(task_info: Tuple) -> Dict:
    """Process a single audio file (thread-safe)"""
    audio_file, model_dir, ground_truth_df, window_size, stride, confidence_threshold, model_name = task_info
    
    try:
        # Initialize evaluator for this thread
        evaluator = AdvancedFrameLevelEvaluator(
            model_dir=model_dir,
            window_size=window_size,
            stride=stride,
            confidence_threshold=confidence_threshold,
            model_name=model_name
        )
        
        # Evaluate the audio file
        results = evaluator.evaluate_audio_file(audio_file, ground_truth_df)
        results['success'] = True
        return results
        
    except Exception as e:
        print(f"❌ Error processing {audio_file}: {e}")
        return {
            'audio_file': audio_file,
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Advanced Frame-Level Audio Evaluation System")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained models")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output_dir", default="./advanced_frame_evaluation_results", help="Output directory")
    parser.add_argument("--window_size", type=float, default=0.5, help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=0.25, help="Stride in seconds")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--model_name", default="airesearch/wav2vec2-large-xlsr-53-th", help="Base model name")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--file_pattern", default="*.wav", help="Audio file pattern")
    
    args = parser.parse_args()
    
    # Get system info
    system_info = get_system_info()
    if args.workers is None:
        args.workers = system_info['recommended_workers']
    
    print("=" * 60)
    print("🎯 ADVANCED FRAME-LEVEL EVALUATION SYSTEM")
    print("=" * 60)
    print(f"Model Directory: {args.model_dir}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Audio Directory: {args.audio_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Window Size: {args.window_size}s")
    print(f"Stride: {args.stride}s")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Workers: {args.workers}")
    print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_gb']:.1f}GB RAM")
    print("=" * 60)
    
    # Load ground truth
    try:
        ground_truth_df = pd.read_csv(args.ground_truth)
        print(f"✅ Loaded ground truth: {len(ground_truth_df)} entries")
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return 1
    
    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = list(audio_dir.glob(args.file_pattern))
    
    if args.max_files:
        audio_files = audio_files[:args.max_files]
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("❌ No audio files found!")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for audio_file in audio_files:
        task = (
            str(audio_file), args.model_dir, ground_truth_df, 
            args.window_size, args.stride, args.confidence_threshold, args.model_name
        )
        tasks.append(task)
    
    print(f"\n🚀 Processing {len(tasks)} files with {args.workers} workers...")
    
    # Process files in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(process_single_audio_file, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Evaluating"):
            result = future.result()
            results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate summary statistics
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"✅ Successful: {len(successful_results)}/{len(results)}")
    print(f"❌ Failed: {len(failed_results)}")
    
    if successful_results:
        # Calculate aggregate metrics
        all_binary_f1 = [r['evaluation_results']['binary_f1'] for r in successful_results 
                        if 'evaluation_results' in r and 'binary_f1' in r['evaluation_results']]
        all_multiclass_f1 = [r['evaluation_results']['multiclass_f1'] for r in successful_results 
                           if 'evaluation_results' in r and 'multiclass_f1' in r['evaluation_results']]
        all_mean_iou = [r['evaluation_results']['mean_iou'] for r in successful_results 
                       if 'evaluation_results' in r and 'mean_iou' in r['evaluation_results']]
        
        if all_binary_f1:
            print(f"📊 Average Binary F1: {np.mean(all_binary_f1):.3f}")
        if all_multiclass_f1:
            print(f"📊 Average Multiclass F1: {np.mean(all_multiclass_f1):.3f}")
        if all_mean_iou:
            print(f"📊 Average Mean IoU: {np.mean(all_mean_iou):.3f}")
    
    # Save detailed results
    summary_data = []
    for result in results:
        if result.get('success', False) and 'evaluation_results' in result:
            eval_results = result['evaluation_results']
            summary_data.append({
                'audio_file': os.path.basename(result['audio_file']),
                'raw_windows': result.get('raw_windows', 0),
                'profanity_windows': result.get('profanity_windows', 0),
                'merged_predictions': result.get('merged_predictions', 0),
                'refined_predictions': result.get('refined_predictions', 0),
                'mean_iou': eval_results.get('mean_iou', 0.0),
                'binary_f1': eval_results.get('binary_f1', 0.0),
                'binary_precision': eval_results.get('binary_precision', 0.0),
                'binary_recall': eval_results.get('binary_recall', 0.0),
                'multiclass_f1': eval_results.get('multiclass_f1', 0.0),
                'multiclass_precision': eval_results.get('multiclass_precision', 0.0),
                'multiclass_recall': eval_results.get('multiclass_recall', 0.0),
                'total_ground_truth': eval_results.get('total_ground_truth', 0),
                'total_predictions': eval_results.get('total_predictions', 0)
            })
        else:
            summary_data.append({
                'audio_file': os.path.basename(result.get('audio_file', 'unknown')),
                'error': result.get('error', 'Unknown error'),
                'success': False
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Save configuration
    config_info = {
        'model_dir': args.model_dir,
        'ground_truth': args.ground_truth,
        'audio_dir': args.audio_dir,
        'window_size': args.window_size,
        'stride': args.stride,
        'confidence_threshold': args.confidence_threshold,
        'model_name': args.model_name,
        'workers': args.workers,
        'total_files': len(audio_files),
        'successful_files': len(successful_results),
        'failed_files': len(failed_results),
        'total_time_seconds': total_time,
        'average_time_per_file': total_time / len(audio_files) if audio_files else 0,
        'system_info': system_info,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = output_dir / "evaluation_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   Summary: {summary_path}")
    print(f"   Config: {config_path}")
    
    if failed_results:
        print(f"\n⚠️  Failed files:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"   {result.get('audio_file', 'unknown')}: {result.get('error', 'Unknown error')}")
        if len(failed_results) > 5:
            print(f"   ... and {len(failed_results) - 5} more")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())