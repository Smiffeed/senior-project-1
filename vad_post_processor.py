#!/usr/bin/env python3
"""
Post-Processing VAD Refinement for Comprehensive Evaluation Results
Takes window evaluation results, merges overlapping predictions, and applies VAD refinement
at specific timestamps without re-running the entire evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import torchaudio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

class VADRefinementProcessor:
    """Post-process window evaluation results with VAD refinement"""
    
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
                    continue  # Skip none predictions
                
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
                    'true_label': label_predictions.iloc[0]['true_label']
                }
                
                # Merge overlapping predictions
                for i in range(1, len(label_predictions)):
                    next_pred = label_predictions.iloc[i]
                    
                    # Check if overlapping or close enough
                    if next_pred['start_time'] <= current_pred['end_time'] + overlap_threshold:
                        # Merge: extend end time and update confidence
                        current_pred['end_time'] = max(current_pred['end_time'], next_pred['end_time'])
                        current_pred['max_confidence'] = max(current_pred['max_confidence'], next_pred['max_confidence'])
                    else:
                        # No overlap: save current and start new
                        file_predictions.append(current_pred.copy())
                        current_pred = {
                            'file_path': file_path,
                            'start_time': next_pred['start_time'],
                            'end_time': next_pred['end_time'],
                            'predicted_label': label,
                            'max_confidence': next_pred['max_confidence'],
                            'true_label': next_pred['true_label']
                        }
                
                # Add final prediction
                file_predictions.append(current_pred)
            
            merged_predictions.extend(file_predictions)
        
        return pd.DataFrame(merged_predictions)
    
    def apply_vad_refinement(self, merged_predictions: pd.DataFrame, vad_top_db: int = 20) -> pd.DataFrame:
        """Apply VAD refinement to merged prediction boundaries"""
        refined_predictions = []
        
        print(f"Applying VAD refinement to {len(merged_predictions)} merged predictions...")
        
        for idx, pred in merged_predictions.iterrows():
            if idx % 50 == 0:
                print(f"  Processing prediction {idx+1}/{len(merged_predictions)}")
            
            try:
                # Load audio segment
                file_path = pred['file_path']
                start_time = pred['start_time']
                end_time = pred['end_time']
                
                # Add some padding around the prediction
                padding = 0.1  # 100ms padding
                padded_start = max(0, start_time - padding)
                padded_duration = end_time - padded_start + padding
                
                # Load audio
                audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                       offset=padded_start, duration=padded_duration)
                
                if len(audio) == 0:
                    # Keep original if audio loading failed
                    refined_predictions.append(pred.to_dict())
                    continue
                
                # Apply VAD to find speech segments
                speech_intervals = librosa.effects.split(audio, top_db=vad_top_db)
                
                if len(speech_intervals) == 0:
                    # No speech detected - keep original
                    refined_predictions.append(pred.to_dict())
                    continue
                
                # Convert speech intervals to absolute time
                speech_segments = []
                for interval_start, interval_end in speech_intervals:
                    abs_start = padded_start + (interval_start / self.sample_rate)
                    abs_end = padded_start + (interval_end / self.sample_rate)
                    
                    # Only keep segments that overlap with original prediction
                    if abs_end > start_time and abs_start < end_time:
                        # Trim to original prediction boundaries
                        refined_start = max(abs_start, start_time)
                        refined_end = min(abs_end, end_time)
                        
                        if refined_end > refined_start:
                            speech_segments.append((refined_start, refined_end))
                
                if not speech_segments:
                    # No speech found in prediction region - keep original
                    refined_predictions.append(pred.to_dict())
                    continue
                
                # Merge close speech segments
                speech_segments.sort()
                merged_segments = []
                current_start, current_end = speech_segments[0]
                
                for seg_start, seg_end in speech_segments[1:]:
                    if seg_start <= current_end + 0.05:  # 50ms gap tolerance
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
                    refined_pred['original_start'] = start_time
                    refined_pred['original_end'] = end_time
                    
                    # If multiple segments, add segment index
                    if len(merged_segments) > 1:
                        refined_pred['segment_id'] = i
                    
                    refined_predictions.append(refined_pred)
            
            except Exception as e:
                print(f"  WARNING: VAD refinement failed for {pred['file_path']} [{start_time:.2f}-{end_time:.2f}]: {e}")
                # Keep original prediction if VAD fails
                refined_pred = pred.to_dict().copy()
                refined_pred['vad_refined'] = False
                refined_predictions.append(refined_pred)
        
        print(f"VAD refinement complete: {len(refined_predictions)} refined predictions")
        return pd.DataFrame(refined_predictions)
    
    def calculate_iou(self, pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
        """Calculate IoU between prediction and ground truth segments"""
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start
        union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_refined_predictions(self, refined_predictions: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        """Evaluate refined predictions against ground truth"""
        print(f"Evaluating {len(refined_predictions)} refined predictions...")
        
        # Get unique audio files
        audio_files = refined_predictions['file_path'].unique()
        all_gt_predictions = []
        
        for audio_file in audio_files:
            # Get predictions for this file
            file_predictions = refined_predictions[refined_predictions['file_path'] == audio_file]
            
            # Get ground truth for this file
            file_column = 'file_path' if 'file_path' in ground_truth_df.columns else 'audio_file'
            file_gt = ground_truth_df[ground_truth_df[file_column] == audio_file]
            
            if len(file_gt) == 0:
                continue
            
            # For each ground truth word, find best matching prediction
            for _, gt_row in file_gt.iterrows():
                gt_start = gt_row['start_time']
                gt_end = gt_row['end_time']
                gt_label = gt_row['label']
                
                # Find best matching prediction
                best_iou = 0.0
                best_pred_label = 'none'
                best_confidence = 0.0
                
                for _, pred_row in file_predictions.iterrows():
                    pred_start = pred_row['start_time']
                    pred_end = pred_row['end_time']
                    pred_label = pred_row['predicted_label']
                    pred_conf = pred_row['max_confidence']
                    
                    iou = self.calculate_iou(pred_start, pred_end, gt_start, gt_end)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_label = pred_label
                        best_confidence = pred_conf
                
                all_gt_predictions.append({
                    'gt_label': gt_label,
                    'pred_label': best_pred_label,
                    'iou': best_iou,
                    'confidence': best_confidence,
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                    'audio_file': audio_file
                })
        
        # Calculate metrics
        profanity_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
        
        # IoU by class
        iou_by_class = {}
        for word_class in profanity_classes:
            class_ious = [pred['iou'] for pred in all_gt_predictions 
                         if pred['gt_label'] == word_class and pred['iou'] > 0]
            iou_by_class[word_class] = np.mean(class_ious) if class_ious else 0.0
        
        combined_mean_iou = np.mean(list(iou_by_class.values()))
        
        # Classification metrics
        y_true = [pred['gt_label'] for pred in all_gt_predictions]
        y_pred = [pred['pred_label'] for pred in all_gt_predictions]
        
        # Binary F1
        y_true_binary = [1 if label != 'none' else 0 for label in y_true]
        y_pred_binary = [1 if label != 'none' else 0 for label in y_pred]
        _, _, binary_f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        
        # Multiclass F1
        _, _, multiclass_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'combined_mean_iou': combined_mean_iou,
            'iou_by_class': iou_by_class,
            'binary_f1': binary_f1,
            'multiclass_f1': multiclass_f1,
            'total_predictions': len(all_gt_predictions),
            'total_refined_segments': len(refined_predictions), 
            'detailed_results': all_gt_predictions
        }

def process_window_evaluation_with_vad(window_eval_dir: str, ground_truth_file: str, output_dir: str, 
                                     window_size: float, stride_info: str, eval_type: str):
    """Process window evaluation results with VAD refinement"""
    
    print(f"Processing Window Evaluation with VAD Refinement:")
    print(f"  Window Eval Dir: {window_eval_dir}")
    print(f"  Window Size: {window_size}s")
    print(f"  Stride: {stride_info}")
    print(f"  Eval Type: {eval_type}")
    
    # Load window evaluation results
    raw_predictions_file = Path(window_eval_dir) / 'raw_predictions.csv'
    if not raw_predictions_file.exists():
        raise FileNotFoundError(f"Raw predictions file not found: {raw_predictions_file}")
    
    print(f"Loading window evaluation results...")
    window_predictions = pd.read_csv(raw_predictions_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    print(f"  Loaded {len(window_predictions)} window predictions")
    print(f"  Loaded {len(ground_truth_df)} ground truth entries")
    
    # Filter to only profanity predictions for merging
    profanity_predictions = window_predictions[window_predictions['predicted_label'] != 'none'].copy()
    print(f"  Found {len(profanity_predictions)} profanity predictions to merge")
    
    # Initialize VAD processor
    vad_processor = VADRefinementProcessor()
    
    # Step 1: Merge overlapping predictions
    print(f"\n🔄 Step 1: Merging overlapping predictions...")
    merged_predictions = vad_processor.merge_overlapping_predictions(profanity_predictions)
    print(f"  Merged {len(profanity_predictions)} predictions → {len(merged_predictions)} merged predictions")
    
    # Step 2: Apply VAD refinement to merged predictions
    print(f"\n🎯 Step 2: Applying VAD refinement...")
    refined_predictions = vad_processor.apply_vad_refinement(merged_predictions)
    print(f"  Refined {len(merged_predictions)} merged predictions → {len(refined_predictions)} refined predictions")
    
    # Step 3: Evaluate refined predictions
    print(f"\n📊 Step 3: Evaluating refined predictions...")
    evaluation_results = vad_processor.evaluate_refined_predictions(refined_predictions, ground_truth_df)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\n💾 Saving results to {output_path}...")
    
    # Save merged predictions
    merged_predictions.to_csv(output_path / 'merged_predictions.csv', index=False)
    
    # Save refined predictions
    refined_predictions.to_csv(output_path / 'vad_refined_predictions.csv', index=False)
    
    # Save detailed evaluation results
    pd.DataFrame(evaluation_results['detailed_results']).to_csv(output_path / 'detailed_evaluation_results.csv', index=False)
    
    # Save comprehensive note
    with open(output_path / 'vad_refinement_note.txt', 'w', encoding='utf-8') as f:
        f.write("=== WINDOW EVALUATION + VAD REFINEMENT RESULTS ===\n")
        f.write(f"Processing Date: {pd.Timestamp.now()}\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Window Size: {window_size}s\n")
        f.write(f"  Stride: {stride_info}\n")
        f.write(f"  Evaluation Type: {eval_type}\n\n")
        
        f.write(f"Processing Pipeline:\n")
        f.write(f"  1. Window Evaluation: {len(window_predictions)} window predictions\n")
        f.write(f"  2. Profanity Filtering: {len(profanity_predictions)} profanity predictions\n")
        f.write(f"  3. Overlapping Merge: {len(merged_predictions)} merged predictions\n")
        f.write(f"  4. VAD Refinement: {len(refined_predictions)} refined predictions\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"  Combined Mean IoU: {evaluation_results['combined_mean_iou']:.4f}\n")
        f.write(f"  Binary F1: {evaluation_results['binary_f1']:.3f}\n")
        f.write(f"  Multiclass F1: {evaluation_results['multiclass_f1']:.3f}\n\n")
        
        f.write(f"IoU by Word Class:\n")
        for word_class, iou_val in evaluation_results['iou_by_class'].items():
            f.write(f"  {word_class}: {iou_val:.4f}\n")
        f.write("\n")
        
        f.write(f"VAD Refinement Statistics:\n")
        vad_refined_count = len(refined_predictions[refined_predictions.get('vad_refined', True) == True])
        f.write(f"  Successfully refined: {vad_refined_count}/{len(refined_predictions)}\n")
        f.write(f"  Refinement success rate: {vad_refined_count/len(refined_predictions)*100:.1f}%\n")
    
    # Print summary
    print(f"\nVAD REFINEMENT COMPLETED!")
    print(f"Results Summary:")
    print(f"   Combined Mean IoU: {evaluation_results['combined_mean_iou']:.4f}")
    print(f"   Binary F1: {evaluation_results['binary_f1']:.3f}")
    print(f"   Multiclass F1: {evaluation_results['multiclass_f1']:.3f}")
    print(f"   Total Refined Predictions: {len(refined_predictions)}")
    print(f"Results saved to: {output_path}")
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Post-process window evaluation results with VAD refinement")
    parser.add_argument("--window_eval_dir", required=True, help="Directory containing window evaluation results")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory for refined results")
    parser.add_argument("--window_size", type=float, required=True, help="Window size in seconds")
    parser.add_argument("--stride_info", required=True, help="Stride information (e.g., '0.05s' or '90%')")
    parser.add_argument("--eval_type", required=True, help="Evaluation type (e.g., 'eval_by_0.05')")
    parser.add_argument("--vad_top_db", type=int, default=20, help="VAD top_db parameter")
    
    args = parser.parse_args()
    
    try:
        results = process_window_evaluation_with_vad(
            args.window_eval_dir,
            args.ground_truth, 
            args.output_dir,
            args.window_size,
            args.stride_info,
            args.eval_type
        )
        
        print(f"\n🎉 Processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())