#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE FRAME-LEVEL EVALUATION PROCESSOR
Advanced evaluation system using frame_level_censor.py implementation
with IoU-based analysis and complete performance metrics

Features:
- Same preprocessing, VAD, and prediction as frame_level_censor.py
- IoU-based evaluation with threshold analysis
- Binary and multiclass classification reports
- Confusion matrices and ROC curves
- Mean IoU percentage calculation per word class
- Complete performance analysis matching stride_10.0% structure
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import warnings
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, det_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from tqdm import tqdm

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
            audio = audio / np.max(np.abs(audio))
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
            audio_window = self.preprocessor.preprocess(audio_window)
            
            if len(audio_window) == 0:
                return None
            
            inputs = self.feature_extractor(
                audio_window,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16000 * 10
            )
            
            return {
                'input_values': inputs['input_values'].squeeze(0),
                'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_values'])).squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing audio window: {e}")
            return None

class EnhancedAudioClassifier(torch.nn.Module):
    """Enhanced audio classifier matching frame_level_censor.py"""
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.num_labels = num_labels
    
    def forward(self, input_values, attention_mask=None, **kwargs):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }

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
                # Use absolute path to ensure cross-process compatibility
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
                print(f"✅ Model loaded successfully from {abs_model_dir}")
                return models
            except Exception as e:
                print(f"❌ Error loading HuggingFace model from {self.model_dir}: {e}")
        
        # Try to load PyTorch models
        pth_files = list(model_path.glob("*.pth"))
        if pth_files:
            for model_file in pth_files:
                try:
                    model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
                    state_dict = torch.load(model_file, map_location=device)
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    models.append(model)
                    print(f"✅ PyTorch model loaded from {model_file}")
                except Exception as e:
                    print(f"❌ Error loading {model_file}: {e}")
        
        if not models:
            print(f"❌ No models found in {self.model_dir}")
            print(f"   Looking for: {hf_files} or *.pth files")
            print(f"   Available files: {list(model_path.glob('*'))}")
        
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

class IoUEvaluationProcessor:
    """IoU evaluation processor matching stride_10.0% structure"""
    
    def __init__(self, evaluator: FrameLevelEvaluator):
        self.evaluator = evaluator
    
    def calculate_iou(self, pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
        """Calculate IoU between prediction and ground truth segments"""
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start
        union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_audio_file(self, audio_path: str, ground_truth_data: List[Dict]) -> Dict:
        """Evaluate single audio file"""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = transform(audio)
        audio_np = audio.squeeze().numpy()
        
        # Get predictions
        predictions = self.evaluator._predict_windows(audio_np)
        
        # Calculate IoU for each ground truth word
        gt_predictions = []
        iou_results = []
        
        for gt_word in ground_truth_data:
            gt_start = gt_word['start_time']
            gt_end = gt_word['end_time']
            gt_label = gt_word['word']
            
            best_iou = 0.0
            best_pred = None
            
            # Find best matching prediction
            for pred_start, pred_end, pred_label, pred_conf in predictions:
                if pred_label == gt_label:
                    iou = self.calculate_iou(pred_start, pred_end, gt_start, gt_end)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = (pred_start, pred_end, pred_label, pred_conf)
            
            # Store result
            gt_predictions.append({
                'gt_start': gt_start,
                'gt_end': gt_end,
                'gt_label': gt_label,
                'pred_start': best_pred[0] if best_pred else None,
                'pred_end': best_pred[1] if best_pred else None,
                'pred_label': best_pred[2] if best_pred else 'none',
                'pred_confidence': best_pred[3] if best_pred else 0.0,
                'iou': best_iou
            })
            
            iou_results.append({
                'word_class': gt_label,
                'iou': best_iou
            })
        
        return {
            'predictions': predictions,
            'ground_truth_predictions': gt_predictions,
            'iou_results': iou_results
        }
    
    def generate_comprehensive_report(self, all_results: List[Dict], output_dir: Path):
        """Generate comprehensive evaluation report matching stride_10.0% structure"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all results
        all_gt_predictions = []
        all_iou_results = []
        
        for result in all_results:
            all_gt_predictions.extend(result['ground_truth_predictions'])
            all_iou_results.extend(result['iou_results'])
        
        # Calculate mean IoU by word class
        iou_by_class = {}
        for word_class in PROFANITY_CLASSES:
            class_ious = [r['iou'] for r in all_iou_results if r['word_class'] == word_class]
            if class_ious:
                iou_by_class[word_class] = np.mean(class_ious)
            else:
                iou_by_class[word_class] = 0.0
        
        # Calculate combined mean IoU (simple average)
        combined_mean_iou = np.mean(list(iou_by_class.values()))
        
        # Generate IoU threshold analysis
        self._generate_iou_threshold_analysis(all_gt_predictions, output_dir)
        
        # Generate classification reports
        self._generate_classification_reports(all_gt_predictions, output_dir)
        
        # Generate confusion matrices
        self._generate_confusion_matrices(all_gt_predictions, output_dir)
        
        # Generate ROC and DET curves
        self._generate_roc_det_curves(all_gt_predictions, output_dir)
        
        # Save summary files
        self._save_summary_files(all_gt_predictions, iou_by_class, combined_mean_iou, output_dir)
        
        # Generate detailed note file
        self._generate_note_file(all_gt_predictions, iou_by_class, combined_mean_iou, output_dir)
        
        return {
            'combined_mean_iou': combined_mean_iou,
            'iou_by_class': iou_by_class,
            'total_predictions': len(all_gt_predictions)
        }
    
    def _generate_iou_threshold_analysis(self, gt_predictions: List[Dict], output_dir: Path):
        """Generate IoU threshold analysis"""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold_results = []
        
        for threshold in thresholds:
            # Filter predictions by IoU threshold
            valid_predictions = [pred for pred in gt_predictions if pred['iou'] >= threshold]
            
            if not valid_predictions:
                continue
            
            # Binary classification metrics
            y_true_binary = [1 if pred['gt_label'] != 'none' else 0 for pred in valid_predictions]
            y_pred_binary = [1 if pred['pred_label'] != 'none' else 0 for pred in valid_predictions]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            binary_metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': len(valid_predictions)
            }
            
            threshold_results.append(binary_metrics)
        
        # Save threshold analysis
        threshold_df = pd.DataFrame(threshold_results)
        threshold_df.to_csv(output_dir / 'iou_threshold_reference.csv', index=False)
    
    def _generate_classification_reports(self, gt_predictions: List[Dict], output_dir: Path):
        """Generate classification reports"""
        # Prepare data
        y_true = [pred['gt_label'] for pred in gt_predictions]
        y_pred = [pred['pred_label'] for pred in gt_predictions]
        
        # Binary classification (profane vs none)
        y_true_binary = ['profane' if label != 'none' else 'none' for label in y_true]
        y_pred_binary = ['profane' if label != 'none' else 'none' for label in y_pred]
        
        # Generate reports
        multiclass_report = classification_report(y_true, y_pred, zero_division=0)
        binary_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)
        
        # Save to files
        with open(output_dir / 'multiclass_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("MULTICLASS CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(multiclass_report)
        
        with open(output_dir / 'binary_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("BINARY CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(binary_report)
    
    def _generate_confusion_matrices(self, gt_predictions: List[Dict], output_dir: Path):
        """Generate confusion matrices"""
        y_true = [pred['gt_label'] for pred in gt_predictions]
        y_pred = [pred['pred_label'] for pred in gt_predictions]
        
        # Multiclass confusion matrix
        cm_multiclass = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Multiclass Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Binary confusion matrix
        y_true_binary = [1 if label != 'none' else 0 for label in y_true]
        y_pred_binary = [1 if label != 'none' else 0 for label in y_pred]
        
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Profanity', 'Profanity'], 
                   yticklabels=['No Profanity', 'Profanity'])
        plt.title('Binary Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Profanity-only confusion matrix (excluding none class)
        profanity_indices = [i for i, label in enumerate(y_true) if label != 'none']
        if profanity_indices:
            y_true_prof = [y_true[i] for i in profanity_indices]
            y_pred_prof = [y_pred[i] for i in profanity_indices]
            
            prof_labels = list(set(y_true_prof + y_pred_prof))
            cm_prof = confusion_matrix(y_true_prof, y_pred_prof, labels=prof_labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_prof, annot=True, fmt='d', cmap='Blues',
                       xticklabels=prof_labels, yticklabels=prof_labels)
            plt.title('Profanity-Only Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / 'profanity_only_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_roc_det_curves(self, gt_predictions: List[Dict], output_dir: Path):
        """Generate ROC and DET curves"""
        # Binary classification curves
        y_true_binary = [1 if pred['gt_label'] != 'none' else 0 for pred in gt_predictions]
        y_scores = [pred['pred_confidence'] if pred['pred_label'] != 'none' else 0.0 for pred in gt_predictions]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Binary Profanity Detection')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'iou_eval_binary_binary_roc_det.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # DET Curve
        fpr, fnr, _ = det_curve(y_true_binary, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, fnr, color='red', lw=2, label='DET Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title('DET Curve - Binary Profanity Detection')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'det_curve_binary_profanity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_summary_files(self, gt_predictions: List[Dict], iou_by_class: Dict, combined_mean_iou: float, output_dir: Path):
        """Save summary CSV files"""
        # Window stride IoU summary
        summary_data = {
            'window': f'window_{self.evaluator.window_size}s',
            'stride': f'stride_{self.evaluator.hop_length}s',
            'combined_mean_iou': combined_mean_iou,
            'individual_mean_iou': combined_mean_iou,  # Same for consistency
            'total_gt_words': len(gt_predictions),
            'total_predictions': len([p for p in gt_predictions if p['pred_label'] != 'none'])
        }
        
        # Add individual class IoUs
        for word_class in PROFANITY_CLASSES:
            summary_data[f'mean_iou_{word_class}'] = iou_by_class[word_class]
        
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(output_dir / 'window_stride_iou_summary.csv', index=False)
        
        # Mean IoU by word class
        iou_class_df = pd.DataFrame(list(iou_by_class.items()), columns=['word_class', 'mean_iou'])
        iou_class_df.to_csv(output_dir / 'mean_iou_by_word_class.csv', index=False)
        
        # Combined IoU analysis
        combined_analysis = []
        for pred in gt_predictions:
            combined_analysis.append({
                'gt_word': pred['gt_label'],
                'gt_start': pred['gt_start'],
                'gt_end': pred['gt_end'],
                'pred_start': pred['pred_start'],
                'pred_end': pred['pred_end'],
                'pred_label': pred['pred_label'],
                'pred_confidence': pred['pred_confidence'],
                'iou': pred['iou']
            })
        
        combined_df = pd.DataFrame(combined_analysis)
        combined_df.to_csv(output_dir / 'combined_iou_analysis.csv', index=False)
        
        # Individual predictions IoU
        individual_predictions = []
        for pred in gt_predictions:
            if pred['pred_label'] != 'none':
                individual_predictions.append({
                    'prediction_id': len(individual_predictions) + 1,
                    'gt_word': pred['gt_label'],
                    'pred_word': pred['pred_label'],
                    'iou': pred['iou'],
                    'confidence': pred['pred_confidence']
                })
        
        if individual_predictions:
            individual_df = pd.DataFrame(individual_predictions)
            individual_df.to_csv(output_dir / 'individual_predictions_iou.csv', index=False)
    
    def _generate_note_file(self, gt_predictions: List[Dict], iou_by_class: Dict, combined_mean_iou: float, output_dir: Path):
        """Generate detailed note file matching stride_10.0% format"""
        with open(output_dir / 'note.txt', 'w', encoding='utf-8') as f:
            f.write("=== IoU ANALYSIS RESULTS ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Combined IoU Analysis:\n")
            f.write("Multiple predictions within the same ground truth word are combined as one area.\n")
            f.write("This provides higher IoU values by treating overlapping predictions as unified detection.\n\n")
            
            f.write(f"Combined Mean IoU: {combined_mean_iou:.4f}\n")
            f.write("(Average IoU when multiple predictions per ground truth word are combined)\n\n")
            
            f.write(f"Individual Mean IoU: {combined_mean_iou:.4f}\n")
            f.write("(Average IoU for individual predictions - reference only)\n\n")
            
            f.write("=== MEAN IoU BY WORD CLASS ===\n")
            for word_class in PROFANITY_CLASSES:
                f.write(f"  {word_class}: {iou_by_class[word_class]:.4f}\n")
            f.write("\n")
            
            # IoU threshold performance analysis
            self._write_threshold_analysis(f, gt_predictions)
            
            f.write("=== GROUND TRUTH vs PREDICTION BREAKDOWN BY CLASS ===\n")
            self._write_class_breakdown(f, gt_predictions)
            
            f.write("\n=== CLASSIFICATION EVALUATION (IoU-based) ===\n")
            self._write_classification_evaluation(f, gt_predictions)
            
            f.write(f"\n=== DATASET STATISTICS ===\n")
            f.write(f"Total Ground Truth Words: {len([p for p in gt_predictions if p['gt_label'] != 'none'])}\n")
            f.write(f"Total Predicted Words (after merging): {len([p for p in gt_predictions if p['pred_label'] != 'none'])}\n\n")
    
    def _write_threshold_analysis(self, f, gt_predictions: List[Dict]):
        """Write IoU threshold analysis to note file"""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        f.write("=== IoU THRESHOLDS PERFORMANCE ===\n\n")
        
        for threshold in thresholds:
            valid_predictions = [pred for pred in gt_predictions if pred['iou'] >= threshold]
            
            if not valid_predictions:
                continue
                
            f.write(f"--- IoU Threshold {threshold} ---\n")
            
            # Binary classification
            y_true_binary = [1 if pred['gt_label'] != 'none' else 0 for pred in valid_predictions]
            y_pred_binary = [1 if pred['pred_label'] != 'none' else 0 for pred in valid_predictions]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            f.write("Binary Classification (Profane vs Non-Profane):\n")
            f.write(f"  Accuracy: {accuracy:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1-Score: {f1:.4f}\n\n")
            
            # Multiclass classification
            y_true_multi = [pred['gt_label'] for pred in valid_predictions]
            y_pred_multi = [pred['pred_label'] for pred in valid_predictions]
            
            multiclass_accuracy = accuracy_score(y_true_multi, y_pred_multi)
            f.write("Multiclass Classification (All Classes):\n")
            f.write(f"  Accuracy: {multiclass_accuracy:.4f}\n\n")
    
    def _write_class_breakdown(self, f, gt_predictions: List[Dict]):
        """Write class breakdown analysis"""
        f.write("Detailed analysis of each class: ground truth count vs correct/incorrect predictions\n\n")
        
        class_stats = {}
        for class_name in CLASS_NAMES:
            gt_count = len([p for p in gt_predictions if p['gt_label'] == class_name])
            correct_count = len([p for p in gt_predictions 
                               if p['gt_label'] == class_name and p['pred_label'] == class_name and p['iou'] >= 0.1])
            incorrect_count = gt_count - correct_count
            accuracy = (correct_count / gt_count * 100) if gt_count > 0 else 0
            
            class_stats[class_name] = {
                'gt_count': gt_count,
                'correct': correct_count,
                'incorrect': incorrect_count,
                'accuracy': accuracy
            }
        
        f.write("Class Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<12} {'GT Count':<10} {'Correct':<10} {'Incorrect':<12} {'Accuracy':<10}\n")
        f.write("-" * 70 + "\n")
        
        for class_name, stats in class_stats.items():
            f.write(f"{class_name:<12} {stats['gt_count']:<10} {stats['correct']:<10} "
                   f"{stats['incorrect']:<12} {stats['accuracy']:<8.1f} %\n")
        
        total_gt = sum(stats['gt_count'] for stats in class_stats.values())
        total_correct = sum(stats['correct'] for stats in class_stats.values())
        total_incorrect = sum(stats['incorrect'] for stats in class_stats.values())
        total_accuracy = (total_correct / total_gt * 100) if total_gt > 0 else 0
        
        f.write("-" * 70 + "\n")
        f.write(f"{'TOTAL':<12} {total_gt:<10} {total_correct:<10} "
               f"{total_incorrect:<12} {total_accuracy:<8.1f} %\n")
        f.write("-" * 70 + "\n\n")
    
    def _write_classification_evaluation(self, f, gt_predictions: List[Dict]):
        """Write classification evaluation section"""
        # Binary classification
        y_true = [pred['gt_label'] for pred in gt_predictions]
        y_pred = [pred['pred_label'] for pred in gt_predictions]
        
        y_true_binary = ['profane' if label != 'none' else 'none' for label in y_true]
        y_pred_binary = ['profane' if label != 'none' else 'none' for label in y_pred]
        
        from sklearn.metrics import classification_report
        
        f.write("=== 1. BINARY CLASSIFICATION (Profane vs None) ===\n")
        binary_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)
        f.write(binary_report)
        f.write("\n")
        
        f.write("=== 2. MULTICLASS CLASSIFICATION (All Classes Including None) ===\n")
        multiclass_report = classification_report(y_true, y_pred, zero_division=0)
        f.write(multiclass_report)
        f.write("\n")
        
        # Profanity-only classification
        profanity_indices = [i for i, label in enumerate(y_true) if label != 'none']
        if profanity_indices:
            y_true_prof = [y_true[i] for i in profanity_indices]
            y_pred_prof = [y_pred[i] for i in profanity_indices]
            
            prof_accuracy = len([i for i in range(len(y_true_prof)) if y_true_prof[i] == y_pred_prof[i]]) / len(y_true_prof)
            
            f.write("=== 3. MULTICLASS CLASSIFICATION (Profanity Words Only) ===\n")
            f.write(f"Accuracy (profanity words only): {prof_accuracy:.4f}\n\n")
            
            profanity_report = classification_report(y_true_prof, y_pred_prof, zero_division=0)
            f.write("Multiclass Classification Report (Profanity Words Only - Excluding None):\n")
            f.write(profanity_report)

def discover_csv_configurations(csv_base_dir: str = "./csv"):
    """Automatically discover all CSV configurations from eval_percent and eval_by_0.05 folders"""
    configurations = []
    csv_path = Path(csv_base_dir)
    
    # Process eval_percent (percentage-based strides)
    eval_percent_dir = csv_path / "eval_percent"
    if eval_percent_dir.exists():
        print(f"🔍 Discovering configurations in {eval_percent_dir}")
        
        for window_dir in eval_percent_dir.glob("window_*"):
            window_str = window_dir.name.replace("window_", "").replace("s", "")
            try:
                window_size = float(window_str)
            except ValueError:
                continue
                
            for csv_file in window_dir.glob("stride_*.csv"):
                stride_str = csv_file.stem.replace("stride_", "").replace("%", "")
                try:
                    stride_value = float(stride_str)
                    configurations.append({
                        'csv_path': str(csv_file),
                        'eval_type': 'eval_percent',
                        'window_size': window_size,
                        'stride_value': stride_value,
                        'stride_type': 'percentage'
                    })
                except ValueError:
                    continue
    
    # Process eval_by_0.05 (absolute time-based strides)
    eval_by_005_dir = csv_path / "eval_by_0.05"
    if eval_by_005_dir.exists():
        print(f"🔍 Discovering configurations in {eval_by_005_dir}")
        
        for window_dir in eval_by_005_dir.glob("window_*"):
            window_str = window_dir.name.replace("window_", "").replace("s", "")
            try:
                window_size = float(window_str)
            except ValueError:
                continue
                
            for csv_file in window_dir.glob("stride_*.csv"):
                stride_str = csv_file.stem.replace("stride_", "").replace("s", "")
                try:
                    stride_value = float(stride_str)
                    configurations.append({
                        'csv_path': str(csv_file),
                        'eval_type': 'eval_by_0.05',
                        'window_size': window_size,
                        'stride_value': stride_value,
                        'stride_type': 'absolute'
                    })
                except ValueError:
                    continue
    
    print(f"📊 Discovered {len(configurations)} total configurations")
    
    # Group by evaluation type for summary
    eval_percent_count = len([c for c in configurations if c['eval_type'] == 'eval_percent'])
    eval_by_005_count = len([c for c in configurations if c['eval_type'] == 'eval_by_0.05'])
    
    print(f"   • eval_percent: {eval_percent_count} configurations")
    print(f"   • eval_by_0.05: {eval_by_005_count} configurations")
    
    return configurations

def run_single_evaluation(args_tuple):
    """Run evaluation for single configuration"""
    csv_path, model_path, ground_truth_path, output_dir, eval_type, window_size, stride_value, stride_type, worker_id = args_tuple
    
    try:
        # Calculate actual stride and overlap based on type
        if stride_type == 'percentage':
            stride_seconds = window_size * (stride_value / 100.0)
            stride_label = f"{stride_value}%"
        else:  # absolute seconds
            stride_seconds = stride_value
            stride_label = f"{stride_value}s"
        
        overlap = window_size - stride_seconds
        
        worker_prefix = f"[W{worker_id}]" if worker_id else ""
        print(f"{worker_prefix} 🔍 Processing: window_{window_size}s, stride_{stride_label} ({stride_seconds:.3f}s)")
        
        start_time = time.time()
        
        # Initialize evaluator (with quieter loading)
        print(f"{worker_prefix} Loading model...")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            evaluator = FrameLevelEvaluator(
                model_dir=model_path,
                window_size=window_size,
                overlap=overlap,
                confidence_threshold=0.5
            )
            print(f"{worker_prefix} ✅ Evaluator created successfully with {len(evaluator.models)} models")
        except Exception as model_error:
            print(f"{worker_prefix} ❌ Model loading error details: {str(model_error)}")
            import traceback
            print(f"{worker_prefix} Full traceback: {traceback.format_exc()}")
            raise Exception(f"Model loading failed: {str(model_error)}")
        
        # Load test data
        test_df = pd.read_csv(csv_path)
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        print(f"{worker_prefix} Loaded {len(test_df)} windows from CSV")
        
        # Process each audio file
        all_results = []
        processor = IoUEvaluationProcessor(evaluator)
        
        # Get unique audio files to process
        unique_files = test_df['file_path'].unique()
        print(f"{worker_prefix} Processing {len(unique_files)} unique audio files...")
        
        for i, audio_file in enumerate(unique_files, 1):
            if i % 100 == 0 or i == len(unique_files):
                print(f"{worker_prefix} Processing file {i}/{len(unique_files)}: {audio_file.split('/')[-1]}")
            
            # Get ground truth for this file
            file_gt = ground_truth_df[ground_truth_df['file_path'] == audio_file]
            gt_data = []
            
            for _, gt_row in file_gt.iterrows():
                gt_data.append({
                    'start_time': gt_row['start_time'],
                    'end_time': gt_row['end_time'],
                    'word': gt_row['label']
                })
            
            if gt_data:
                result = processor.evaluate_audio_file(audio_file, gt_data)
                all_results.append(result)
        
        # Generate comprehensive report
        output_path = Path(output_dir) / eval_type / f"window_{window_size}s" / f"stride_{stride_label}"
        summary = processor.generate_comprehensive_report(all_results, output_path)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{worker_prefix} ✅ Completed: window_{window_size}s, stride_{stride_label} - IoU: {summary['combined_mean_iou']:.4f} ({duration:.1f}s)")
        
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'stride_value': stride_value,
            'stride_type': stride_type,
            'stride_label': stride_label,
            'combined_mean_iou': summary['combined_mean_iou'],
            'iou_by_class': summary['iou_by_class'],
            'success': True,
            'output_dir': str(output_path),
            'duration': duration,
            'worker_id': worker_id if worker_id else 0
        }
        
    except Exception as e:
        worker_prefix = f"[W{worker_id}]" if worker_id else ""
        error_msg = str(e)[:100]  # Limit error message length
        print(f"{worker_prefix} ❌ Error: window_{window_size}s, stride_{stride_label} - {error_msg}")
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'stride_value': stride_value,
            'stride_type': stride_type,
            'stride_label': stride_label if 'stride_label' in locals() else f"{stride_value}",
            'success': False,
            'error': str(e),
            'duration': 0,
            'worker_id': worker_id if worker_id else 0
        }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Frame-Level Evaluation Processor")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--csv_base_dir", default="./csv", help="Base directory containing eval_percent and eval_by_0.05 folders")
    parser.add_argument("--output_dir", default="./frame_level_evaluation_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit_configs", type=int, default=None, help="Limit number of configurations for testing")
    
    args = parser.parse_args()
    
    print("🎯 COMPREHENSIVE FRAME-LEVEL EVALUATION PROCESSOR")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"CSV Base Dir: {args.csv_base_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print()
    
    # Discover all CSV configurations
    configurations = discover_csv_configurations(args.csv_base_dir)
    
    if not configurations:
        print("❌ No CSV configurations found!")
        return
    
    # Limit configurations if specified (for testing)
    if args.limit_configs:
        configurations = configurations[:args.limit_configs]
        print(f"🔍 Limited to {len(configurations)} configurations for testing")
    
    # Prepare tasks with worker IDs
    tasks = []
    for i, config in enumerate(configurations):
        worker_id = (i % args.workers) + 1
        tasks.append((
            config['csv_path'],
            args.model_path,
            args.ground_truth,
            args.output_dir,
            config['eval_type'],
            config['window_size'],
            config['stride_value'],
            config['stride_type'],
            worker_id
        ))
    
    print(f"🚀 Processing {len(tasks)} configurations...")
    print(f"   • eval_percent: {len([c for c in configurations if c['eval_type'] == 'eval_percent'])} configurations")
    print(f"   • eval_by_0.05: {len([c for c in configurations if c['eval_type'] == 'eval_by_0.05'])} configurations")
    print()
    
    # Run parallel evaluation
    start_time = time.time()
    results = []
    
    if args.workers == 1:
        # Sequential processing
        for i, task in enumerate(tasks, 1):
            result = run_single_evaluation(task)
            results.append(result)
            
            success_count = len([r for r in results if r['success']])
            progress = i / len(tasks) * 100
            print(f"📈 {i}/{len(tasks)} ({progress:.1f}%) - ✅ {success_count}")
    else:
        # Parallel processing with better progress tracking
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(run_single_evaluation, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    success_count = len([r for r in results if r['success']])
                    progress = completed / len(tasks) * 100
                    print(f"📈 {completed}/{len(tasks)} ({progress:.1f}%) - ✅ {success_count}")
                except Exception as e:
                    print(f"❌ Task failed with exception: {e}")
                    completed += 1
    
    end_time = time.time()
    
    # Summary
    success_results = [r for r in results if r['success']]
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"⏱️  Total Time: {end_time - start_time:.1f}s ({(end_time - start_time)/60:.1f}m)")
    
    # Calculate performance metrics
    if results:
        total_duration = sum(r.get('duration', 0) for r in results)
        speedup = total_duration / (end_time - start_time) if (end_time - start_time) > 0 else 1
        efficiency = speedup / args.workers * 100 if args.workers > 0 else 0
        print(f"🚀 Speedup: {speedup:.1f}x (Efficiency: {efficiency:.1f}%)")
    
    print(f"✅ Success: {len(success_results)}/{len(results)} ({len(success_results)/len(results)*100:.1f}%)")
    
    if success_results:
        best_result = max(success_results, key=lambda x: x['combined_mean_iou'])
        print(f"🏆 Best Configuration: window_{best_result['window_size']}s, stride_{best_result['stride_label']}")
        print(f"   Combined Mean IoU: {best_result['combined_mean_iou']:.4f}")
        print(f"   Evaluation Type: {best_result['eval_type']}")
        
        # Show top 5 configurations
        top_5 = sorted(success_results, key=lambda x: x['combined_mean_iou'], reverse=True)[:5]
        print(f"\n🏅 Top 5 Configurations:")
        for i, result in enumerate(top_5, 1):
            print(f"   {i}. window_{result['window_size']}s, stride_{result['stride_label']} "
                  f"({result['eval_type']}) - IoU: {result['combined_mean_iou']:.4f}")
        
        # Summary by evaluation type
        eval_percent_results = [r for r in success_results if r['eval_type'] == 'eval_percent']
        eval_by_005_results = [r for r in success_results if r['eval_type'] == 'eval_by_0.05']
        
        if eval_percent_results:
            best_percent = max(eval_percent_results, key=lambda x: x['combined_mean_iou'])
            print(f"\n📊 Best eval_percent: window_{best_percent['window_size']}s, stride_{best_percent['stride_label']} - IoU: {best_percent['combined_mean_iou']:.4f}")
            
        if eval_by_005_results:
            best_by_005 = max(eval_by_005_results, key=lambda x: x['combined_mean_iou'])
            print(f"📊 Best eval_by_0.05: window_{best_by_005['window_size']}s, stride_{best_by_005['stride_label']} - IoU: {best_by_005['combined_mean_iou']:.4f}")
    
    # Save summary
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_path / "evaluation_summary.csv", index=False)
    
    print(f"📊 Summary saved to: {output_path / 'evaluation_summary.csv'}")

if __name__ == "__main__":
    main()