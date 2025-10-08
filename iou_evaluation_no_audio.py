#!/usr/bin/env python3
"""
🎯 VAD EVALUATION WITHOUT AUDIO PROCESSING
Version that works with existing predictions without requiring librosa/torchaudio
Focuses on using existing word_eval results and applying IoU analysis
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import classification_report, precision_recall_fscore_support

warnings.filterwarnings('ignore')

# Constants
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())
PROFANITY_CLASSES = ['เย็ด', 'กู', 'มึง', 'เหี้ย']

def load_existing_predictions(results_dir: str, eval_type: str, evaluation_method: str, 
                            window_size: float, stride_value: float, stride_type: str) -> Optional[pd.DataFrame]:
    """Load existing predictions from fixed_smart_parallel_results"""
    
    if stride_type == 'absolute':
        stride_str = f"stride_{stride_value}s"
    else:  # percentage
        stride_str = f"stride_{stride_value}%"
    
    prediction_path = Path(results_dir) / eval_type / eval_type / evaluation_method / f"window_{window_size}s" / stride_str
    
    print(f"🔍 Looking for existing predictions in: {prediction_path}")
    
    possible_files = ['merged_predictions.csv', 'detailed_results.csv', 'predictions.csv', 'results.csv']
    
    for filename in possible_files:
        file_path = prediction_path / filename
        if file_path.exists():
            print(f"✅ Found existing predictions: {file_path}")
            try:
                df = pd.read_csv(file_path)
                print(f"   Loaded {len(df)} predictions")
                return df
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
    
    print(f"❌ No prediction files found in {prediction_path}")
    return None

def parse_existing_predictions(predictions_df: pd.DataFrame) -> List[Tuple[str, List[Tuple[float, float, str, float]]]]:
    """Parse existing predictions into format needed for evaluation"""
    
    print("🔄 Step 1-2: Parsing existing predictions...")
    
    predictions_by_file = {}
    
    for _, row in predictions_df.iterrows():
        # Extract audio file path
        audio_file = None
        if 'audio_file' in row and pd.notna(row['audio_file']):
            audio_file = row['audio_file']
        elif 'file_path' in row and pd.notna(row['file_path']):
            audio_file = row['file_path']
        elif 'filename' in row and pd.notna(row['filename']):
            audio_file = row['filename']
        
        if not audio_file:
            continue
            
        # Extract prediction details
        start_time = row.get('start_time', row.get('pred_start', 0.0))
        end_time = row.get('end_time', row.get('pred_end', 0.0))
        
        # Extract predicted label and confidence
        pred_label = 'none'
        confidence = 0.5
        
        if 'predicted_label' in row and pd.notna(row['predicted_label']):
            pred_label = row['predicted_label']
        elif 'pred_label' in row and pd.notna(row['pred_label']):
            pred_label = row['pred_label']
        elif 'label' in row and pd.notna(row['label']):
            pred_label = row['label']
            
        if 'max_confidence' in row and pd.notna(row['max_confidence']):
            confidence = float(row['max_confidence'])
        elif 'confidence' in row and pd.notna(row['confidence']):
            confidence = float(row['confidence'])
        elif 'pred_confidence' in row and pd.notna(row['pred_confidence']):
            confidence = float(row['pred_confidence'])
        
        # Only keep profanity predictions
        if pred_label != 'none':
            if audio_file not in predictions_by_file:
                predictions_by_file[audio_file] = []
            
            predictions_by_file[audio_file].append((
                float(start_time), float(end_time), pred_label, float(confidence)
            ))
    
    result = [(audio_file, predictions) for audio_file, predictions in predictions_by_file.items()]
    
    total_predictions = sum(len(predictions) for _, predictions in result)
    print(f"   Parsed {total_predictions} profanity predictions from {len(result)} audio files")
    
    return result

def merge_overlapping_windows(windows: List[Tuple[float, float, str, float]], max_gap: float = 0.3) -> List[Tuple[float, float, str, float]]:
    """Merge overlapping or nearby profanity windows"""
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

def process_evaluation_without_audio(results_dir: str, eval_type: str, evaluation_method: str,
                                   window_size: float, stride_value: float, stride_type: str,
                                   ground_truth_file: str, output_dir: str):
    """
    Main function: IoU evaluation using existing predictions (no audio processing)
    """
    
    print(f"🚀 IoU Evaluation Using Existing Predictions (No Audio Processing)")
    print("=" * 70)
    print(f"  Results Dir: {results_dir}")
    print(f"  Eval Type: {eval_type}")
    print(f"  Method: {evaluation_method}")
    print(f"  Window: {window_size}s")
    print(f"  Stride: {stride_value} ({stride_type})")
    print(f"  Ground Truth: {ground_truth_file}")
    print(f"  Output: {output_dir}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing predictions
    predictions_df = load_existing_predictions(results_dir, eval_type, evaluation_method, 
                                             window_size, stride_value, stride_type)
    
    if predictions_df is None:
        print("❌ ERROR: Could not load existing predictions!")
        return False
    
    # Parse predictions
    file_predictions = parse_existing_predictions(predictions_df)
    
    if not file_predictions:
        print("❌ ERROR: No valid predictions found!")
        return False
    
    # Load ground truth
    ground_truth_df = pd.read_csv(ground_truth_file)
    print(f"📊 Loaded {len(ground_truth_df)} ground truth entries")
    
    # Process each audio file
    all_gt_predictions = []
    
    print(f"\n🎯 Processing {len(file_predictions)} audio files...")
    
    for i, (audio_file, raw_predictions) in enumerate(file_predictions, 1):
        if i % 10 == 0 or i == len(file_predictions):
            print(f"  Processing file {i}/{len(file_predictions)}: {Path(audio_file).name}")
        
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
        
        # Merge overlapping predictions (no VAD - just merge)
        merged_predictions = merge_overlapping_windows(raw_predictions)
        
        # Calculate IoU for each ground truth word
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
            
            for pred_start, pred_end, pred_label, pred_conf in merged_predictions:
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
    
    # IoU threshold analysis and metrics calculation
    if not all_gt_predictions:
        print("❌ ERROR: No predictions generated!")
        return False
    
    print(f"\n🎯 IoU threshold analysis (0.1-0.9)...")
    
    # Calculate metrics for different IoU thresholds
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = {}
    
    for threshold in iou_thresholds:
        tp_count = 0
        fp_count = 0
        tn_count = 0
        fn_count = 0
        
        for pred in all_gt_predictions:
            gt_is_profanity = pred['gt_label'] != 'none'
            pred_is_profanity = pred['pred_label'] != 'none'
            iou_meets_threshold = pred['iou'] >= threshold
            
            if gt_is_profanity:
                if pred_is_profanity and iou_meets_threshold:
                    tp_count += 1
                else:
                    fn_count += 1
            else:
                if pred_is_profanity:
                    fp_count += 1
                else:
                    tn_count += 1
        
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
    
    # Overall mean IoU
    all_ious = [pred['iou'] for pred in all_gt_predictions]
    overall_mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Traditional classification metrics
    y_true = [pred['gt_label'] for pred in all_gt_predictions]
    y_pred = [pred['pred_label'] for pred in all_gt_predictions]
    
    y_true_binary = [1 if label != 'none' else 0 for label in y_true]
    y_pred_binary = [1 if label != 'none' else 0 for label in y_pred]
    
    binary_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)
    multiclass_report = classification_report(y_true, y_pred, zero_division=0)
    
    _, _, binary_f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
    _, _, multiclass_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Save results
    results_df = pd.DataFrame(all_gt_predictions)
    results_df.to_csv(output_path / 'detailed_results.csv', index=False)
    
    threshold_df = pd.DataFrame(threshold_results).T
    threshold_df.to_csv(output_path / 'iou_threshold_analysis.csv')
    
    # Save comprehensive note file
    with open(output_path / 'note.txt', 'w', encoding='utf-8') as f:
        f.write("=== IoU EVALUATION USING EXISTING PREDICTIONS (NO AUDIO PROCESSING) ===\n")
        f.write("Using pre-computed predictions from fixed_smart_parallel_results\n")
        f.write("Step 1-2: Load and parse existing predictions\n")
        f.write("Step 3: Merge overlapping predictions (no VAD audio processing)\n")
        f.write("Step 4: IoU threshold analysis (0.1-0.9)\n\n")
        
        f.write(f"Source: {results_dir}/{eval_type}/{eval_type}/{evaluation_method}/\n")
        f.write(f"Configuration: Window {window_size}s, Stride {stride_value} ({stride_type})\n")
        f.write(f"Ground Truth: {ground_truth_file}\n\n")
        
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
    print(f"\n✅ IoU EVALUATION COMPLETED (Using Existing Predictions):")
    print(f"Overall Mean IoU: {overall_mean_iou:.1%}")
    print(f"Traditional Binary F1: {binary_f1:.3f}")
    print(f"Traditional Multiclass F1: {multiclass_f1:.3f}")
    
    best_threshold = max(iou_thresholds, key=lambda t: threshold_results[t]['f1'])
    best_f1 = threshold_results[best_threshold]['f1']
    print(f"Best IoU Threshold: {best_threshold} (F1={best_f1:.3f})")
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    return True

def main():
    """Main function with command line support"""
    parser = argparse.ArgumentParser(description="IoU Evaluation Using Existing Predictions (No Audio Processing)")
    parser.add_argument("--results_dir", default="fixed_smart_parallel_results", 
                       help="Directory with existing results")
    parser.add_argument("--eval_type", required=True, choices=['eval_by_0.05', 'eval_percent'],
                       help="Evaluation type")
    parser.add_argument("--method", required=True, 
                       choices=['window_eval', 'word_eval', 'iou_eval', 'word_iou_eval'],
                       help="Evaluation method")
    parser.add_argument("--window_size", type=float, required=True, help="Window size in seconds")
    parser.add_argument("--stride_value", type=float, required=True, help="Stride value")
    parser.add_argument("--stride_type", required=True, choices=['absolute', 'percentage'],
                       help="Stride type")
    parser.add_argument("--ground_truth", default="csv/eval_5labels.csv", 
                       help="Ground truth CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    try:
        success = process_evaluation_without_audio(
            results_dir=args.results_dir,
            eval_type=args.eval_type,
            evaluation_method=args.method,
            window_size=args.window_size,
            stride_value=args.stride_value,
            stride_type=args.stride_type,
            ground_truth_file=args.ground_truth,
            output_dir=args.output_dir
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())