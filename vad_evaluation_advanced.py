#!/usr/bin/env python3
"""
🎯 4-STEP ADVANCED VAD EVALUATION SYSTEM
Based on frame_level_censor.py methodology with comprehensive IoU analysis

🔄 EVALUATION PROCESS:
Step 1: Advanced Preprocessing
  - Pre-emphasis filtering for spectral balance
  - Epsilon-protected normalization for numerical stability  
  - Minimum length padding for temporal consistency

Step 2: Window/Merged Word Detection
  - Sliding window approach with ensemble models
  - Merge overlapping predictions into coherent regions
  - High-confidence profanity detection

Step 3: VAD for Temporal Precision
  - Voice Activity Detection to refine boundaries
  - Librosa split-based speech segment detection
  - Preserve largest speech segments for accuracy

Step 4: IoU Threshold Analysis (0.1-0.9)
  - Multi-threshold evaluation for precision analysis
  - True/False positive determination based on IoU overlap
  - Comprehensive metrics including mean IoU percentages

🎨 OUTPUT METRICS:
- Overall Mean IoU percentage
- Mean IoU by word class
- IoU threshold performance matrix (0.1-0.9)  
- Traditional F1 scores for comparison
- Detailed CSV results with temporal boundaries
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
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

class AdvancedAudioPreprocessor:
    """Advanced audio preprocessor matching comprehensive_evaluation_processor.py"""
    
    def __init__(self):
        self.sample_rate = 16000
    
    def advanced_preprocess_audio(self, audio):
        """Advanced audio preprocessing matching training pipeline"""
        try:
            # Apply noise reduction (pre-emphasis filtering)
            audio = librosa.effects.preemphasis(audio)
            
            # Normalize with epsilon protection
            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Ensure minimum length
            min_length = int(0.1 * self.sample_rate)  # 0.1 seconds minimum
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            print(f"Error in advanced preprocessing: {e}")
            return audio
    
    def preprocess(self, audio):
        """Main preprocessing function - now uses advanced preprocessing"""
        return self.advanced_preprocess_audio(audio)

class SimpleProfanityDataset:
    """Dataset class using advanced preprocessing"""
    
    def __init__(self, audio_data, feature_extractor, preprocessor, mode='eval', window_size=0.5):
        self.audio_data = audio_data
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.mode = mode
        self.window_size = window_size
    
    def process_audio_window(self, audio_window, label=0):
        """Process a single audio window"""
        try:
            # Use advanced preprocessing
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

class FrameLevelEvaluatorAdvanced:
    """Frame-level evaluator using advanced preprocessing"""
    
    def __init__(self, model_dir: str, window_size: float = 0.3, overlap: float = 0.05,
                 confidence_threshold: float = 0.5, model_name: str = "airesearch/wav2vec2-large-xlsr-53-th"):
        """Initialize evaluator with advanced preprocessing"""
        self.model_dir = model_dir
        self.window_size = window_size
        self.overlap = overlap
        self.hop_length = window_size - overlap
        self.sample_rate = 16000
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Initialize components with advanced preprocessing
        self.feature_extractor = self._load_feature_extractor()
        self.preprocessor = AdvancedAudioPreprocessor()  # Using advanced preprocessor
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
                abs_model_dir = str(model_path.absolute())
                
                wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    abs_model_dir,
                    num_labels=NUM_LABELS,
                    ignore_mismatched_sizes=True,
                    device_map=None,
                    torch_dtype=torch.float32
                )
                
                wav2vec2_model = wav2vec2_model.to(device)
                wav2vec2_model.eval()
                
                class LoadedModel:
                    def __init__(self, model):
                        self.model = model
                    
                    def __call__(self, input_values, attention_mask=None):
                        return self.model(input_values=input_values, attention_mask=attention_mask)
                
                model = LoadedModel(wav2vec2_model)
                models.append(model)
                print(f"✅ Model loaded successfully from {abs_model_dir}")
                return models
            except Exception as e:
                print(f"❌ Error loading HuggingFace model from {self.model_dir}: {e}")
        
        if not models:
            print(f"❌ No models found in {self.model_dir}")
        
        return models
    
    def _predict_windows(self, audio_np: np.ndarray) -> List[Tuple[float, float, str, float]]:
        """
        Step 1: Using advanced preprocessing
        Step 2: Using result from window detection (word-level method)
        """
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        detected_windows = []
        dataset_processor = SimpleProfanityDataset(None, self.feature_extractor, self.preprocessor, window_size=self.window_size)
        
        print(f"🔍 Step 1-2: Scanning audio with advanced preprocessing and word-level detection...")
        print(f"   Window size: {self.window_size}s, Hop: {self.hop_length}s")
        
        for i in range(0, len(audio_np) - window_samples, hop_samples):
            window_audio = audio_np[i:i + window_samples]
            start_time = i / self.sample_rate
            end_time = (i + window_samples) / self.sample_rate
            
            # Step 1: Process window with advanced preprocessing
            processed = dataset_processor.process_audio_window(window_audio)
            if processed is None:
                continue
            
            # Step 2: Make prediction using ensemble (word-level method)
            with torch.no_grad():
                input_values = processed['input_values'].unsqueeze(0).to(device)
                attention_mask = processed['attention_mask'].unsqueeze(0).to(device)
                
                all_logits = []
                for model in self.models:
                    outputs = model(input_values, attention_mask=attention_mask)
                    all_logits.append(outputs.logits)
                
                # Average ensemble predictions
                ensemble_logits = torch.stack(all_logits).mean(dim=0)
                probabilities = torch.nn.functional.softmax(ensemble_logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Only keep high-confidence profanity predictions
                if predicted_class > 0 and confidence >= self.confidence_threshold:
                    label = list(LABEL_MAP.keys())[predicted_class]
                    detected_windows.append((start_time, end_time, label, confidence))
        
        # Merge overlapping windows before VAD refinement
        merged_windows = self._merge_overlapping_windows(detected_windows)
        print(f"   Found {len(detected_windows)} windows → {len(merged_windows)} merged regions")
        
        # Step 3: Apply VAD for temporal precision
        print("🎯 Step 3: Applying VAD for temporal precision...")
        refined_windows = []
        for window in merged_windows:
            refined = refine_boundaries_with_vad(audio_np, window, self.sample_rate)
            refined_windows.append(refined)
        
        print(f"   Refined {len(merged_windows)} regions with VAD")
        return refined_windows
    
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

def refine_boundaries_with_vad(audio_np: np.ndarray, region: Tuple[float, float, str, float], sample_rate: int = 16000) -> Tuple[float, float, str, float]:
    """
    Refine boundaries using Voice Activity Detection for temporal precision
    Step 3: Using VAD for temporal precision
    """
    start_time, end_time, label, confidence = region
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    region_audio = audio_np[start_sample:end_sample]
    
    try:
        # Apply VAD to find speech segments with more sensitive threshold
        speech_intervals = librosa.effects.split(region_audio, top_db=20)
        
        if len(speech_intervals) == 0:
            return region  # Return original if no speech detected
        
        # Take the largest speech segment for temporal precision
        largest_interval = max(speech_intervals, key=lambda x: x[1] - x[0])
        speech_start, speech_end = largest_interval
        
        # Convert back to absolute time
        refined_start = start_time + (speech_start / sample_rate)
        refined_end = start_time + (speech_end / sample_rate)
        
        # Ensure minimum duration (100ms)
        if refined_end - refined_start < 0.1:
            return region  # Return original if too short
        
        return (refined_start, refined_end, label, confidence)
        
    except Exception as e:
        print(f"⚠️  VAD refinement failed: {e}")
        return region

def process_single_configuration(model_path, csv_file, ground_truth_file, output_dir, window_size, stride_value, stride_type, eval_type):
    """
    Process single configuration with 4-step evaluation process:
    1. Advanced preprocessing
    2. Window/merged word detection  
    3. VAD for temporal precision
    4. IoU threshold analysis (0.1-0.9)
    """
    
    print(f"🔧 Processing Configuration with 4-Step Advanced Evaluation:")
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
    
    # Initialize evaluator with advanced preprocessing
    evaluator = FrameLevelEvaluatorAdvanced(
        model_dir=model_path,
        window_size=window_size,
        overlap=overlap,
        confidence_threshold=0.5
    )
    
    # Load data
    test_df = pd.read_csv(csv_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    print(f"📊 Loaded {len(test_df)} windows from CSV")
    print(f"📊 Loaded {len(ground_truth_df)} ground truth entries")
    
    # Process each unique audio file
    unique_files = test_df['file_path'].unique() if 'file_path' in test_df.columns else test_df['audio_file'].unique()
    print(f"🎵 Processing {len(unique_files)} unique audio files...")
    
    all_gt_predictions = []
    
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
            
            # Steps 1-3: Get predictions using advanced preprocessing + VAD refinement
            predictions = evaluator._predict_windows(audio_np)
            
            # Step 4: Calculate IoU for each ground truth word with threshold analysis
            for gt_word in gt_data:
                gt_start = gt_word['start_time']
                gt_end = gt_word['end_time']
                gt_label = gt_word['word']
                
                # Find best matching prediction
                best_iou = 0.0
                best_pred_label = 'none'
                best_confidence = 0.0
                best_pred_start = 0.0
                best_pred_end = 0.0
                
                for pred_start, pred_end, pred_label, pred_conf in predictions:
                    iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_label = pred_label
                        best_confidence = pred_conf
                        best_pred_start = pred_start
                        best_pred_end = pred_end
                
                all_gt_predictions.append({
                    'gt_label': gt_label,
                    'pred_label': best_pred_label,
                    'iou': best_iou,
                    'confidence': best_confidence,
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                    'pred_start': best_pred_start,
                    'pred_end': best_pred_end,
                    'audio_file': audio_file
                })
        
        except Exception as e:
            print(f"  ❌ ERROR processing {audio_file}: {e}")
            continue
    
    # Step 4: IoU threshold analysis (0.1-0.9) and comprehensive metrics
    if not all_gt_predictions:
        print("❌ ERROR: No predictions generated!")
        return False
    
    print("\n🎯 Step 4: IoU threshold analysis (0.1-0.9)...")
    
    # Calculate metrics for different IoU thresholds
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = {}
    
    for threshold in iou_thresholds:
        # Determine true/false positives based on IoU threshold
        tp_count = 0
        fp_count = 0
        tn_count = 0
        fn_count = 0
        
        y_true_threshold = []
        y_pred_threshold = []
        
        for pred in all_gt_predictions:
            gt_is_profanity = pred['gt_label'] != 'none'
            pred_is_profanity = pred['pred_label'] != 'none'
            iou_meets_threshold = pred['iou'] >= threshold
            
            if gt_is_profanity:
                if pred_is_profanity and iou_meets_threshold:
                    tp_count += 1
                    y_true_threshold.append(1)
                    y_pred_threshold.append(1)
                else:
                    fn_count += 1
                    y_true_threshold.append(1)
                    y_pred_threshold.append(0)
            else:
                if pred_is_profanity:
                    fp_count += 1
                    y_true_threshold.append(0)
                    y_pred_threshold.append(1)
                else:
                    tn_count += 1
                    y_true_threshold.append(0)
                    y_pred_threshold.append(0)
        
        # Calculate metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        threshold_results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp_count,
            'fp': fp_count,
            'tn': tn_count,
            'fn': fn_count
        }
    
    # Calculate mean IoU by class and overall
    iou_by_class = {}
    for word_class in PROFANITY_CLASSES:
        class_ious = [pred['iou'] for pred in all_gt_predictions if pred['gt_label'] == word_class]
        iou_by_class[word_class] = np.mean(class_ious) if class_ious else 0.0
    
    # Overall mean IoU (including all predictions)
    all_ious = [pred['iou'] for pred in all_gt_predictions]
    overall_mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Traditional classification metrics (best IoU matching regardless of threshold)
    y_true = [pred['gt_label'] for pred in all_gt_predictions]
    y_pred = [pred['pred_label'] for pred in all_gt_predictions]
    
    # Binary classification
    y_true_binary = [1 if label != 'none' else 0 for label in y_true]
    y_pred_binary = [1 if label != 'none' else 0 for label in y_pred]
    
    # Generate reports
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    binary_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)
    multiclass_report = classification_report(y_true, y_pred, zero_division=0)
    
    # Extract F1 scores
    _, _, binary_f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    _, _, multiclass_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Save detailed results
    results_df = pd.DataFrame(all_gt_predictions)
    results_df.to_csv(output_path / 'detailed_results.csv', index=False)
    
    # Save IoU threshold analysis
    threshold_df = pd.DataFrame(threshold_results).T
    threshold_df.to_csv(output_path / 'iou_threshold_analysis.csv')
    
    # Save comprehensive note file
    with open(output_path / 'note.txt', 'w', encoding='utf-8') as f:
        f.write("=== 4-STEP ADVANCED VAD EVALUATION ===\n")
        f.write("Step 1: Advanced preprocessing (pre-emphasis, epsilon normalization, padding)\n")
        f.write("Step 2: Window/merged word detection with ensemble models\n")
        f.write("Step 3: VAD for temporal precision\n")
        f.write("Step 4: IoU threshold analysis (0.1-0.9)\n\n")
        
        f.write(f"Configuration: Window {window_size}s, Stride {stride_value} ({stride_type})\n")
        f.write(f"Evaluation Type: {eval_type}\n\n")
        
        f.write(f"Overall Mean IoU: {overall_mean_iou:.4f} ({overall_mean_iou:.1%})\n")
        f.write(f"Traditional Binary F1: {binary_f1:.3f}\n")
        f.write(f"Traditional Multiclass F1: {multiclass_f1:.3f}\n\n")
        
        f.write("=== MEAN IoU BY WORD CLASS ===\n")
        for word_class, iou_val in iou_by_class.items():
            f.write(f"  {word_class}: {iou_val:.4f} ({iou_val:.1%})\n")
        f.write("\n")
        
        f.write("=== IoU THRESHOLD ANALYSIS ===\n")
        f.write("Threshold | Precision | Recall | F1-Score | TP | FP | TN | FN\n")
        f.write("----------|-----------|--------|----------|----|----|----|----|n")
        for threshold in iou_thresholds:
            r = threshold_results[threshold]
            f.write(f"   {threshold:.1f}    |   {r['precision']:.3f}   | {r['recall']:.3f} |  {r['f1']:.3f}  |{r['tp']:3d} |{r['fp']:3d} |{r['tn']:3d} |{r['fn']:3d} |\n")
        f.write("\n")
        
        f.write("=== BINARY CLASSIFICATION REPORT (Traditional) ===\n")
        f.write(binary_report)
        f.write("\n\n")
        
        f.write("=== MULTICLASS CLASSIFICATION REPORT (Traditional) ===\n")
        f.write(multiclass_report)
    
    # Output summary
    print(f"\n✅ 4-STEP EVALUATION COMPLETED:")
    print(f"Overall Mean IoU: {overall_mean_iou:.1%}")
    print(f"Traditional Binary F1: {binary_f1:.3f}")
    print(f"Traditional Multiclass F1: {multiclass_f1:.3f}")
    print("\nBest IoU Threshold Performance:")
    best_threshold = max(iou_thresholds, key=lambda t: threshold_results[t]['f1'])
    best_f1 = threshold_results[best_threshold]['f1']
    print(f"  Threshold {best_threshold}: F1={best_f1:.3f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="VAD Evaluation with Advanced Preprocessing")
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
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())