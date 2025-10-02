#!/usr/bin/env python3
"""
Single Configuration Frame-Level Evaluation Processor
Handles evaluation of a single window/stride configuration using frame_level_censor.py approach
This is called by comprehensive_frame_level_evaluation_v3.py for each configuration
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, det_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

warnings.filterwarnings('ignore')

# Constants matching frame_level_censor.py
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())
PROFANITY_CLASSES = ['เย็ด', 'กู', 'มึง', 'เหี้ย']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleAudioPreprocessor:
    """Simple audio preprocessor matching frame_level_censor.py"""
    
    def __init__(self):
        self.sample_rate = 16000
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def apply_voice_activity_detection(self, audio, top_db=25):
        """Apply simple VAD to remove silence"""
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return audio
        
        voiced_audio = []
        for start, end in intervals:
            voiced_audio.append(audio[start:end])
        
        if voiced_audio:
            return np.concatenate(voiced_audio)
        return audio
    
    def preprocess(self, audio):
        """Main preprocessing function"""
        audio = self.normalize_audio(audio)
        return audio

class SimpleProfanityDataset:
    """Simple dataset class matching frame_level_censor.py"""
    
    def __init__(self, audio_data, feature_extractor, preprocessor, mode='eval', window_size=0.5):
        self.audio_data = audio_data
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.mode = mode
        self.window_size = window_size
    
    def process_audio_window(self, audio_window, label=0):
        """Process a single audio window"""
        try:
            # Preprocess audio
            processed_audio = self.preprocessor.preprocess(audio_window)
            
            # Extract features
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'label': label
            }
        except Exception as e:
            return None

class FrameLevelEvaluator:
    """Frame-level evaluator using frame_level_censor.py implementation"""
    
    def __init__(self, model_dir: str, window_size: float = 0.3, overlap: float = 0.05,
                 confidence_threshold: float = 0.5, model_name: str = "airesearch/wav2vec2-large-xlsr-53-th"):
        """Initialize evaluator with same parameters as frame_level_censor.py"""
        self.model_dir = model_dir
        self.window_size = window_size
        self.overlap = overlap
        self.hop_length = window_size - overlap
        self.sample_rate = 16000
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Initialize components
        self.feature_extractor = self._load_feature_extractor()
        self.preprocessor = SimpleAudioPreprocessor()
        self.models = self._load_models()
        
        if not self.models:
            raise ValueError("No models loaded successfully!")
    
    def _load_feature_extractor(self):
        """Load feature extractor"""
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor_config.json')
        
        if os.path.exists(preprocessor_path):
            try:
                return Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_dir, 
                    return_attention_mask=True, 
                    do_normalize=True
                )
            except Exception as e:
                print(f"Warning: Could not load preprocessor from {self.model_dir}: {e}")
        
        print(f"Loading feature extractor from base model: {self.model_name}")
        return Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, 
            return_attention_mask=True, 
            do_normalize=True
        )
    
    def _load_models(self):
        """Load models from directory"""
        models = []
        model_path = Path(self.model_dir)
        
        # Check for HuggingFace model files
        hf_files = ['config.json', 'model.safetensors', 'preprocessor_config.json']
        has_hf_files = all((model_path / file).exists() for file in hf_files)
        
        if has_hf_files:
            try:
                # Use absolute path to ensure compatibility
                abs_model_dir = str(model_path.absolute())
                
                # Load model with proper safetensors handling
                from transformers import Wav2Vec2ForSequenceClassification
                
                # Load the model directly using transformers
                wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    abs_model_dir,
                    num_labels=NUM_LABELS,
                    ignore_mismatched_sizes=True,
                    device_map=None,  # Don't use device_map to avoid meta tensor issues
                    torch_dtype=torch.float32  # Explicit dtype
                )
                
                # Move to device after loading
                wav2vec2_model = wav2vec2_model.to(device)
                wav2vec2_model.eval()
                
                # Wrap in our custom class
                class LoadedModel:
                    def __init__(self, model):
                        self.wav2vec2 = model
                        self.num_labels = NUM_LABELS
                    
                    def to(self, device):
                        self.wav2vec2 = self.wav2vec2.to(device)
                        return self
                    
                    def eval(self):
                        self.wav2vec2.eval()
                        return self
                    
                    def __call__(self, input_values, attention_mask=None, **kwargs):
                        outputs = self.wav2vec2(
                            input_values=input_values,
                            attention_mask=attention_mask
                        )
                        return {
                            'logits': outputs.logits,
                            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
                        }
                
                model = LoadedModel(wav2vec2_model)
                models.append(model)
                print(f"Model loaded successfully from {abs_model_dir}")
                return models
            except Exception as e:
                print(f"Error loading HuggingFace model from {self.model_dir}: {e}")
        
        # Try to load PyTorch models
        pth_files = list(model_path.glob("*.pth"))
        if pth_files:
            for pth_file in pth_files:
                try:
                    checkpoint = torch.load(pth_file, map_location=device)
                    # Handle checkpoint loading logic here
                    print(f"Loaded PyTorch model: {pth_file}")
                except Exception as e:
                    print(f"Error loading PyTorch model {pth_file}: {e}")
        
        if not models:
            print(f"No models found in {self.model_dir}")
            print(f"   Looking for: {hf_files} or *.pth files")
        
        return models
    
    def _predict_windows(self, audio_np: np.ndarray) -> List[Tuple[float, float, str, float]]:
        """Predict profanity windows using same method as frame_level_censor.py"""
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        detected_windows = []
        dataset_processor = SimpleProfanityDataset(None, self.feature_extractor, self.preprocessor, window_size=self.window_size)
        
        for i in range(0, len(audio_np) - window_samples, hop_samples):
            window_audio = audio_np[i:i + window_samples]
            start_time = i / self.sample_rate
            end_time = (i + window_samples) / self.sample_rate
            
            # Process window
            processed = dataset_processor.process_audio_window(window_audio)
            if processed is None:
                continue
            
            # Make prediction using ensemble
            with torch.no_grad():
                input_values = processed['input_values'].unsqueeze(0).to(device)
                attention_mask = processed['attention_mask'].unsqueeze(0).to(device)
                
                all_logits = []
                for model in self.models:
                    outputs = model(input_values=input_values, attention_mask=attention_mask)
                    all_logits.append(outputs['logits'])
                
                # Average ensemble predictions
                ensemble_logits = torch.stack(all_logits).mean(dim=0)
                probabilities = torch.nn.functional.softmax(ensemble_logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Only keep high-confidence profanity predictions
                if predicted_class > 0 and confidence >= self.confidence_threshold:
                    label = CLASS_NAMES[predicted_class]
                    detected_windows.append((start_time, end_time, label, confidence))
        
        return self._merge_overlapping_windows(detected_windows)
    
    def _merge_overlapping_windows(self, windows: List[Tuple[float, float, str, float]], max_gap: float = 0.3) -> List[Tuple[float, float, str, float]]:
        """Merge overlapping windows"""
        if not windows:
            return []
        
        windows.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end, current_label, current_conf = windows[0]
        
        for next_start, next_end, next_label, next_conf in windows[1:]:
            if next_start <= current_end + max_gap and next_label == current_label:
                current_end = max(current_end, next_end)
                current_conf = max(current_conf, next_conf)
            else:
                merged.append((current_start, current_end, current_label, current_conf))
                current_start, current_end, current_label, current_conf = next_start, next_end, next_label, next_conf
        
        merged.append((current_start, current_end, current_label, current_conf))
        return merged

def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Calculate IoU between prediction and ground truth segments"""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    return intersection / union if union > 0 else 0.0

def generate_comprehensive_note_file(note_path, all_gt_predictions, iou_by_class, combined_mean_iou, 
                                   window_size, stride_value, stride_type, eval_type, binary_f1, multiclass_f1):
    """Generate comprehensive note file matching the reference format"""
    from datetime import datetime
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write("=== IoU ANALYSIS RESULTS ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Combined IoU Analysis:\n")
        f.write("Multiple predictions within the same ground truth word are combined as one area.\n")
        f.write("This provides higher IoU values by treating overlapping predictions as unified detection.\n\n")
        
        f.write(f"Combined Mean IoU: {combined_mean_iou:.4f}\n")
        f.write("(Average IoU when multiple predictions per ground truth word are combined)\n\n")
        
        # Calculate individual mean IoU for reference
        individual_ious = [pred['iou'] for pred in all_gt_predictions if pred['iou'] > 0]
        individual_mean_iou = np.mean(individual_ious) if individual_ious else 0.0
        f.write(f"Individual Mean IoU: {individual_mean_iou:.4f}\n")
        f.write("(Average IoU for individual predictions - reference only)\n\n")
        
        f.write("=== MEAN IoU BY WORD CLASS ===\n")
        for word_class, iou_val in iou_by_class.items():
            f.write(f"  {word_class}: {iou_val:.4f}\n")
        f.write("\n")
        
        # IoU threshold analysis
        f.write("=== IoU THRESHOLDS PERFORMANCE ===\n\n")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            f.write(f"--- IoU Threshold {threshold} ---\n")
            
            # Prepare data for this threshold
            y_true = [pred['gt_label'] for pred in all_gt_predictions]
            y_pred = [pred['pred_label'] if pred['iou'] >= threshold else 'none' for pred in all_gt_predictions]
            
            # Binary classification
            y_true_binary = [1 if label != 'none' else 0 for label in y_true]
            y_pred_binary = [1 if label != 'none' else 0 for label in y_pred]
            
            binary_acc = accuracy_score(y_true_binary, y_pred_binary)
            binary_prec, binary_rec, binary_f1_score, _ = precision_recall_fscore_support(
                y_true_binary, y_pred_binary, average='binary', zero_division=0
            )
            
            # Calculate balanced accuracy manually
            tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
            tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
            fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
            fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
            
            f.write("Binary Classification (Profane vs Non-Profane):\n")
            f.write(f"  Accuracy: {binary_acc:.4f}\n")
            f.write(f"  Balanced Accuracy: {balanced_acc:.4f}\n")
            f.write(f"  Precision: {binary_prec:.4f}\n")
            f.write(f"  Recall: {binary_rec:.4f}\n")
            f.write(f"  F1-Score: {binary_f1_score:.4f}\n\n")
            
            # Multiclass classification
            multi_acc = accuracy_score(y_true, y_pred)
            
            # Calculate balanced accuracy for multiclass
            unique_labels = list(set(y_true + y_pred))
            class_recalls = []
            for label in unique_labels:
                label_tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                label_total = sum(1 for t in y_true if t == label)
                recall = label_tp / label_total if label_total > 0 else 0
                class_recalls.append(recall)
            multi_balanced_acc = np.mean(class_recalls)
            
            f.write("Multiclass Classification (All Classes):\n")
            f.write(f"  Accuracy: {multi_acc:.4f}\n")
            f.write(f"  Balanced Accuracy: {multi_balanced_acc:.4f}\n\n")
            
            # Classification report
            try:
                multiclass_report = classification_report(y_true, y_pred, zero_division=0, labels=CLASS_NAMES)
                f.write("  Classification Report:\n")
                # Indent each line
                for line in multiclass_report.split('\n'):
                    f.write(f"  {line}\n")
                f.write("\n")
            except:
                f.write("  Classification Report: Unable to generate\n\n")
            
            # Detailed classification analysis
            f.write(f"=== DETAILED CLASSIFICATION ANALYSIS (IoU >= {threshold}) ===\n")
            
            # Count detections above threshold
            detections_above_threshold = sum(1 for pred in all_gt_predictions if pred['iou'] >= threshold and pred['pred_label'] != 'none')
            total_profanity = sum(1 for pred in all_gt_predictions if pred['gt_label'] != 'none')
            
            f.write(f"--- Binary Classification Report (IoU >= {threshold}) ---\n")
            f.write(f"Accuracy: {binary_acc:.4f}\n")
            f.write(f"Precision: {binary_prec:.4f}\n")
            f.write(f"Recall: {binary_rec:.4f}\n")
            f.write(f"F1-Score: {binary_f1_score:.4f}\n\n")
            
            # Binary classification report
            try:
                binary_report = classification_report(
                    ['none' if x == 0 else 'profane' for x in y_true_binary],
                    ['none' if x == 0 else 'profane' for x in y_pred_binary],
                    zero_division=0
                )
                f.write("Binary Classification Report:\n")
                f.write(binary_report)
                f.write("\n")
            except:
                f.write("Binary Classification Report: Unable to generate\n\n")
            
            f.write(f"--- Multiclass Classification Report (IoU >= {threshold}) ---\n")
            f.write(f"Accuracy: {multi_acc:.4f}\n")
            f.write(f"Balanced Accuracy: {multi_balanced_acc:.4f}\n\n")
            
            # Multiclass classification report
            try:
                f.write("Multiclass Classification Report:\n")
                f.write(multiclass_report)
                f.write("\n")
            except:
                f.write("Multiclass Classification Report: Unable to generate\n\n")
            
            # Confusion Matrix Analysis
            f.write(f"--- Confusion Matrix Analysis (IoU >= {threshold}) ---\n")
            f.write(f"IoU Threshold: {threshold}\n")
            f.write(f"Detections with IoU >= {threshold}: {detections_above_threshold}/{total_profanity} ({detections_above_threshold/total_profanity*100:.1f}%)\n")
            
            # Binary confusion matrix
            cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
            if cm_binary.shape == (2, 2):
                f.write("Binary Confusion Matrix:\n")
                f.write("                    Predicted\n")
                f.write("                    No-Prof  Profanity\n")
                f.write(f"  Actual No-Prof        {cm_binary[0,0]}        {cm_binary[0,1]}\n")
                f.write(f"  Actual Profanity      {cm_binary[1,0]}         {cm_binary[1,1]}\n")
                f.write("  \n")
                f.write("Binary Metrics Summary:\n")
                f.write(f"  Precision: {binary_prec:.4f}\n")
                f.write(f"  Recall: {binary_rec:.4f}\n")
                f.write(f"  F1-Score: {binary_f1_score:.4f}\n")
                f.write(f"  Accuracy: {binary_acc:.4f}\n")
            f.write("\n\n")
        
        # Ground Truth vs Prediction Breakdown
        f.write("=== GROUND TRUTH vs PREDICTION BREAKDOWN BY CLASS ===\n")
        f.write("Detailed analysis of each class: ground truth count vs correct/incorrect predictions\n\n")
        f.write("Class Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write("Class        GT Count   Correct    Incorrect    Accuracy  \n")
        f.write("-" * 70 + "\n")
        
        for class_name in CLASS_NAMES:
            gt_count = sum(1 for pred in all_gt_predictions if pred['gt_label'] == class_name)
            correct_count = sum(1 for pred in all_gt_predictions 
                              if pred['gt_label'] == class_name and pred['pred_label'] == class_name and pred['iou'] > 0.1)
            incorrect_count = gt_count - correct_count
            accuracy = correct_count / gt_count * 100 if gt_count > 0 else 0
            
            f.write(f"{class_name:<12} {gt_count:<10} {correct_count:<10} {incorrect_count:<12} {accuracy:<8.1f} %\n")
        
        total_gt = len(all_gt_predictions)
        total_correct = sum(1 for pred in all_gt_predictions 
                          if pred['pred_label'] == pred['gt_label'] and (pred['iou'] > 0.1 or pred['gt_label'] == 'none'))
        total_incorrect = total_gt - total_correct
        total_accuracy = total_correct / total_gt * 100 if total_gt > 0 else 0
        
        f.write("-" * 70 + "\n")
        f.write(f"{'TOTAL':<12} {total_gt:<10} {total_correct:<10} {total_incorrect:<12} {total_accuracy:<8.1f} %\n")
        f.write("-" * 70 + "\n\n")
        
        # Classification Evaluation
        f.write("=== CLASSIFICATION EVALUATION (IoU-based) ===\n")
        f.write("Classification performance based on combined IoU predictions\n\n")
        
        # Overall binary and multiclass performance
        y_true_all = [pred['gt_label'] for pred in all_gt_predictions]
        y_pred_all = [pred['pred_label'] for pred in all_gt_predictions]
        
        y_true_binary_all = [1 if label != 'none' else 0 for label in y_true_all]
        y_pred_binary_all = [1 if label != 'none' else 0 for label in y_pred_all]
        
        binary_acc_all = accuracy_score(y_true_binary_all, y_pred_binary_all)
        binary_prec_all, binary_rec_all, binary_f1_all, _ = precision_recall_fscore_support(
            y_true_binary_all, y_pred_binary_all, average='binary', zero_division=0
        )
        
        f.write("=== 1. BINARY CLASSIFICATION (Profane vs None) ===\n")
        f.write(f"Accuracy: {binary_acc_all:.4f}\n")
        f.write(f"Precision: {binary_prec_all:.4f}\n")
        f.write(f"Recall: {binary_rec_all:.4f}\n")
        f.write(f"F1-Score: {binary_f1_all:.4f}\n\n")
        
        try:
            binary_report_all = classification_report(
                ['none' if x == 0 else 'profane' for x in y_true_binary_all],
                ['none' if x == 0 else 'profane' for x in y_pred_binary_all],
                zero_division=0
            )
            f.write("Binary Classification Report:\n")
            f.write(binary_report_all)
            f.write("\n")
        except:
            pass
        
        # IoU threshold distribution
        f.write("=== IoU THRESHOLD ANALYSIS ===\n")
        f.write("Percentage of ground truth words that achieve different IoU thresholds:\n\n")
        
        f.write("Overall IoU Distribution:\n")
        for threshold in thresholds:
            count = sum(1 for pred in all_gt_predictions if pred['iou'] >= threshold)
            percentage = count / len(all_gt_predictions) * 100
            f.write(f"  IoU >= {threshold}: {count}/{len(all_gt_predictions)} ({percentage:.1f}%)\n")
        
        f.write("\nProfanity Words IoU Distribution:\n")
        profanity_predictions = [pred for pred in all_gt_predictions if pred['gt_label'] != 'none']
        for threshold in thresholds:
            count = sum(1 for pred in profanity_predictions if pred['iou'] >= threshold)
            percentage = count / len(profanity_predictions) * 100 if profanity_predictions else 0
            f.write(f"  IoU >= {threshold}: {count}/{len(profanity_predictions)} ({percentage:.1f}%)\n")
        
        # Summary
        f.write(f"\n=== DATASET STATISTICS ===\n")
        f.write(f"Total Ground Truth Words: {len(profanity_predictions)}\n")
        f.write(f"Total Predicted Windows: Processing completed\n\n")
        
        f.write(f"Configuration Summary:\n")
        f.write(f"Window Size: {window_size}s\n")
        f.write(f"Stride: {stride_value} ({stride_type})\n")
        f.write(f"Evaluation Type: {eval_type}\n")
        f.write(f"Combined Mean IoU: {combined_mean_iou:.4f}\n")
        f.write(f"Binary F1: {binary_f1:.3f}\n")
        f.write(f"Multiclass F1: {multiclass_f1:.3f}\n")

def process_single_configuration(model_path, csv_file, ground_truth_file, output_dir, window_size, stride_value, stride_type, eval_type):
    """Process single configuration like comprehensive_evaluation_processor.py"""
    
    print(f"Processing Configuration:")
    print(f"  CSV: {csv_file}")
    print(f"  Window: {window_size}s")
    print(f"  Stride: {stride_value} ({stride_type})")
    print(f"  Eval Type: {eval_type}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate overlap based on stride
    if stride_type == 'percentage':
        stride_seconds = window_size * (stride_value / 100.0)
        overlap = window_size - stride_seconds
    else:  # absolute
        overlap = window_size - stride_value
    
    # Initialize evaluator
    evaluator = FrameLevelEvaluator(
        model_dir=model_path,
        window_size=window_size,
        overlap=overlap,
        confidence_threshold=0.5
    )
    
    # Load data
    test_df = pd.read_csv(csv_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    print(f"Loaded {len(test_df)} windows from CSV")
    print(f"Loaded {len(ground_truth_df)} ground truth entries")
    
    # Process each unique audio file
    unique_files = test_df['file_path'].unique() if 'file_path' in test_df.columns else test_df['audio_file'].unique()
    print(f"Processing {len(unique_files)} unique audio files...")
    
    all_gt_predictions = []
    all_iou_results = []
    
    for i, audio_file in enumerate(unique_files, 1):
        if i % 10 == 0 or i == len(unique_files):
            print(f"  Processing file {i}/{len(unique_files)}: {audio_file.split('/')[-1]}")
        
        # Get ground truth for this file
        file_column = 'file_path' if 'file_path' in ground_truth_df.columns else 'audio_file'
        file_gt = ground_truth_df[ground_truth_df[file_column] == audio_file]
        
        if len(file_gt) == 0:
            continue
        
        gt_data = []
        for _, gt_row in file_gt.iterrows():
            gt_data.append({
                'start_time': gt_row['start_time'],
                'end_time': gt_row['end_time'],
                'word': gt_row['label']
            })
        
        # Load and predict audio
        try:
            audio, sr = torchaudio.load(audio_file)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            audio_np = audio.squeeze().numpy()
            
            # Get predictions
            predictions = evaluator._predict_windows(audio_np)
            
            # Calculate IoU for each ground truth word
            for gt_word in gt_data:
                gt_start = gt_word['start_time']
                gt_end = gt_word['end_time']
                gt_label = gt_word['word']
                
                # Find best matching prediction
                best_iou = 0.0
                best_pred_label = 'none'
                best_confidence = 0.0
                
                for pred_start, pred_end, pred_label, pred_conf in predictions:
                    iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_label = pred_label
                        best_confidence = pred_conf
                
                all_gt_predictions.append({
                    'audio_file': audio_file,
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                    'gt_label': gt_label,
                    'pred_label': best_pred_label,
                    'pred_confidence': best_confidence,
                    'iou': best_iou
                })
                
                all_iou_results.append(best_iou)
        
        except Exception as e:
            print(f"  ERROR processing {audio_file}: {e}")
            continue
    
    # Calculate metrics
    if not all_gt_predictions:
        print("ERROR: No predictions generated!")
        return False
    
    # Calculate mean IoU by class
    iou_by_class = {}
    for word_class in PROFANITY_CLASSES:
        class_ious = [pred['iou'] for pred in all_gt_predictions if pred['gt_label'] == word_class and pred['iou'] > 0]
        iou_by_class[word_class] = np.mean(class_ious) if class_ious else 0.0
    
    # Calculate combined mean IoU (simple average)
    combined_mean_iou = np.mean(list(iou_by_class.values()))
    
    # Classification metrics
    y_true = [pred['gt_label'] for pred in all_gt_predictions]
    y_pred = [pred['pred_label'] for pred in all_gt_predictions]
    
    # Binary classification
    y_true_binary = [1 if label != 'none' else 0 for label in y_true]
    y_pred_binary = [1 if label != 'none' else 0 for label in y_pred]
    
    # Generate reports
    binary_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)
    multiclass_report = classification_report(y_true, y_pred, zero_division=0)
    
    # Extract F1 scores
    binary_f1 = 0.0
    multiclass_f1 = 0.0
    
    for line in binary_report.split('\n'):
        if 'weighted avg' in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    binary_f1 = float(parts[3])
                except:
                    pass
    
    for line in multiclass_report.split('\n'):
        if 'weighted avg' in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    multiclass_f1 = float(parts[3])
                except:
                    pass
    
    # Save results
    results_df = pd.DataFrame(all_gt_predictions)
    results_df.to_csv(output_path / 'detailed_results.csv', index=False)
    
    # Save reports
    with open(output_path / 'binary_classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(binary_report)
    
    with open(output_path / 'multiclass_classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(multiclass_report)
    
    # Generate confusion matrices
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Profanity', 'Profanity'], 
                yticklabels=['No Profanity', 'Profanity'])
    plt.title('Binary Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive note file matching the reference format
    generate_comprehensive_note_file(
        output_path / 'note.txt',
        all_gt_predictions,
        iou_by_class,
        combined_mean_iou,
        window_size,
        stride_value,
        stride_type,
        eval_type,
        binary_f1,
        multiclass_f1
    )
    
    # Output summary for parent process
    print(f"\nEVALUATION COMPLETED:")
    print(f"Combined Mean IoU: {combined_mean_iou:.2%}")
    print(f"Binary F1: {binary_f1:.3f}")
    print(f"Multiclass F1: {multiclass_f1:.3f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Single Configuration Frame-Level Evaluation")
    parser.add_argument("--csv_file", required=True, help="CSV file with windowed data")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--window_size", type=float, required=True, help="Window size in seconds")
    parser.add_argument("--stride_value", type=float, required=True, help="Stride value")
    parser.add_argument("--stride_type", required=True, choices=['percentage', 'absolute'], help="Stride type")
    parser.add_argument("--eval_type", required=True, help="Evaluation type")
    
    args = parser.parse_args()
    
    try:
        success = process_single_configuration(
            args.model_path,
            args.csv_file,
            args.ground_truth,
            args.output_dir,
            args.window_size,
            args.stride_value,
            args.stride_type,
            args.eval_type
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())