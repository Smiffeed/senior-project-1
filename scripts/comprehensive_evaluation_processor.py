#!/usr/bin/env python3
"""
Comprehensive evaluation processor - performs one-time prediction with multiple evaluation methods
Computes: Window-level, Word-level merged, IoU-based, and IoU-only evaluations
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import librosa
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def generate_roc_det_curves(y_true, y_scores, output_dir, method_name, binary_labels=None):
    """
    Generate ROC and DET curves for evaluation methods
    
    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities  
        output_dir: Directory to save curves
        method_name: Name of the evaluation method
        binary_labels: For binary classification (None for multiclass)
    """
    try:
        from sklearn.metrics import roc_curve, auc, det_curve
        import matplotlib.pyplot as plt
        import numpy as np
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Binary ROC/DET
        if binary_labels is not None:
            # Binary classification ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(12, 5))
            
            # ROC Curve
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{method_name} - ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # DET Curve
            plt.subplot(1, 2, 2)
            try:
                fpr_det, fnr_det, _ = det_curve(y_true, y_scores, pos_label=1)
                plt.plot(fpr_det * 100, fnr_det * 100, color='darkred', lw=2)
                plt.xlabel('False Positive Rate (%)')
                plt.ylabel('False Negative Rate (%)')
                plt.title(f'{method_name} - DET Curve')
                plt.grid(True, alpha=0.3)
                plt.xscale('log')
                plt.yscale('log')
            except Exception as e:
                plt.text(0.5, 0.5, f'DET curve error: {str(e)}', 
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title(f'{method_name} - DET Curve (Error)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{method_name}_binary_roc_det.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            return roc_auc
            
        else:
            # Multiclass ROC (One-vs-Rest)
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            from itertools import cycle
            
            # Get unique classes
            classes = np.unique(y_true)
            n_classes = len(classes)
            
            if n_classes < 2:
                return None
                
            # Binarize the output
            if y_scores.ndim == 1:
                # If y_scores is 1D, create dummy scores for multiclass
                y_scores_multi = np.zeros((len(y_true), n_classes))
                for i, class_val in enumerate(classes):
                    y_scores_multi[y_true == class_val, i] = 1.0
            else:
                y_scores_multi = y_scores
                
            y_true_bin = label_binarize(y_true, classes=classes)
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            plt.figure(figsize=(15, 5))
            
            # ROC curves
            plt.subplot(1, 2, 1)
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkred', 'darkgreen'])
            
            for i, color in zip(range(min(n_classes, y_scores_multi.shape[1])), colors):
                if i < y_true_bin.shape[1] and i < y_scores_multi.shape[1]:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_multi[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    class_name = classes[i] if i < len(classes) else f'Class_{i}'
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                            label=f'{class_name} (AUC = {roc_auc[i]:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{method_name} - Multiclass ROC Curves')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Precision-Recall curves as substitute for DET in multiclass
            plt.subplot(1, 2, 2)
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            for i, color in zip(range(min(n_classes, y_scores_multi.shape[1])), colors):
                if i < y_true_bin.shape[1] and i < y_scores_multi.shape[1]:
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores_multi[:, i])
                    avg_precision = average_precision_score(y_true_bin[:, i], y_scores_multi[:, i])
                    class_name = classes[i] if i < len(classes) else f'Class_{i}'
                    plt.plot(recall, precision, color=color, lw=2,
                            label=f'{class_name} (AP = {avg_precision:.4f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{method_name} - Precision-Recall Curves')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{method_name}_multiclass_roc_pr.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Return average AUC
            return np.mean(list(roc_auc.values())) if roc_auc else None
            
    except Exception as e:
        print(f"Error generating ROC/DET curves for {method_name}: {e}")
        return None

# Define label mapping
label_map = {
    'none': 0,
    'เย็ด': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4
}

rev_label_map = {v: k for k, v in label_map.items()}

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

def setup_thai_font():
    """Setup Thai font for matplotlib"""
    try:
        plt.rcParams['font.family'] = ['Tahoma', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['font.size'] = 10
    except:
        pass

def advanced_preprocess_audio(file_path, start_time, end_time):
    """Preprocess audio slice using the same steps as the training pipeline.

    Steps:
    - Slice-load with torchaudio at source sample rate using frame offsets
    - Convert to mono
    - Pre-emphasis (librosa.effects.preemphasis, coef=0.97)
    - Gentle noise gate based on low-energy percentile
    - Light edge windowing (Hamming blend)
    - Mild dynamic range compression
    - Resample to 16kHz if needed
    - Z-score normalize, then scale to 0.5
    - Pad/truncate to the exact window length in samples
    """
    try:
        # Determine desired output length from the requested window duration
        duration = max(1e-6, float(end_time) - float(start_time))
        target_sr = 16000
        max_length = int(round(duration * target_sr)) or target_sr

        # Load specific slice using torchaudio for precise offsets
        try:
            metadata = torchaudio.info(file_path)
            sr = metadata.sample_rate
            frame_offset = int(round(start_time * sr))
            num_frames = max(1, int(round((end_time - start_time) * sr)))
            audio, sr = torchaudio.load(
                file_path,
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        except Exception:
            # Fallback to librosa if torchaudio slice-load fails
            audio_np, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
            audio = torch.from_numpy(audio_np).unsqueeze(0)

        # Mono
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio_np = audio.squeeze().numpy().astype(np.float32, copy=False)

        # Pre-emphasis (preserve high-frequency content)
        try:
            audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
        except Exception:
            pass

        # Gentle noise gate using 20th percentile energy
        if audio_np.size:
            energy = np.abs(audio_np)
            noise_threshold = np.percentile(energy, 20)
            noise_gate_threshold = float(noise_threshold) * 2.5
            mask = energy < noise_gate_threshold
            # Attenuate quiet parts but keep signal
            audio_np = np.where(mask, audio_np * 0.1, audio_np)

        # Light edge windowing to reduce boundary artifacts
        L = len(audio_np)
        if L > 160:
            w = np.hamming(L).astype(np.float32)
            w = 0.85 + 0.15 * w  # keep mostly original amplitude
            audio_np = audio_np * w

        # Mild dynamic range compression
        rms = float(np.sqrt(np.mean(audio_np**2)) + 1e-8)
        # Soft clipping around 2*rms
        audio_np = np.tanh(audio_np / (2.0 * rms)) * (2.0 * rms)

        # Convert back to tensor for resampling if needed
        audio_t = torch.from_numpy(audio_np).unsqueeze(0)
        if sr != target_sr:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                audio_t = resampler(audio_t)
            except Exception:
                # Fallback to librosa resample
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
                audio_t = torch.from_numpy(audio_np).unsqueeze(0)
        
        # Normalize: z-score then scale
        mean = audio_t.mean()
        std = audio_t.std()
        if float(std) > 1e-8:
            audio_t = (audio_t - mean) / std
        else:
            audio_t = audio_t - mean
        audio_t = audio_t * 0.5

        # Pad or truncate to exact window length
        if audio_t.shape[1] < max_length:
            pad = max_length - audio_t.shape[1]
            audio_t = torch.nn.functional.pad(audio_t, (0, pad))
        elif audio_t.shape[1] > max_length:
            audio_t = audio_t[:, :max_length]

        return audio_t.squeeze().numpy().astype(np.float32, copy=False)
    except Exception as e:
        print(f"Error preprocessing audio {file_path}: {e}")
        # Fallback to silence of the requested duration (at least 0.1s)
        fallback_len = max(int(round((end_time - start_time) * 16000)), 1600)
        return np.zeros(fallback_len, dtype=np.float32)

def evaluate_window(model, feature_extractor, file_path, start_time, end_time, device):
    """Evaluate a single window and return predictions"""
    try:
        # Preprocess audio
        audio = advanced_preprocess_audio(file_path, start_time, end_time)
        
        # Extract features
        inputs = feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence_scores = probabilities.cpu().numpy()[0]
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': rev_label_map[predicted_class],
            'confidence_scores': confidence_scores,
            'max_confidence': confidence_scores[predicted_class]
        }
    except Exception as e:
        print(f"Error evaluating window {file_path} [{start_time}-{end_time}]: {e}")
        return {
            'predicted_class': 0,
            'predicted_label': 'none',
            'confidence_scores': np.zeros(5),
            'max_confidence': 0.0
        }

def merge_overlapping_predictions(predictions_df, overlap_threshold=0.1):
    """Merge overlapping or close predictions of the same label"""
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
                # Don't merge 'none' predictions
                for _, row in label_group.iterrows():
                    file_predictions.append(row.to_dict())
                continue
            
            # Sort by start time
            label_group = label_group.sort_values('start_time').reset_index(drop=True)
            
            if len(label_group) == 1:
                file_predictions.append(label_group.iloc[0].to_dict())
                continue
            
            # Merge overlapping/close predictions
            current_prediction = label_group.iloc[0].to_dict()
            
            for i in range(1, len(label_group)):
                next_pred = label_group.iloc[i]
                
                # Check if predictions are close enough to merge
                gap = next_pred['start_time'] - current_prediction['end_time']
                
                if gap <= overlap_threshold:  # Merge if gap is small
                    # Extend current prediction
                    current_prediction['end_time'] = max(current_prediction['end_time'], next_pred['end_time'])
                    current_prediction['max_confidence'] = max(current_prediction['max_confidence'], next_pred['max_confidence'])
                else:
                    # Save current and start new
                    file_predictions.append(current_prediction)
                    current_prediction = next_pred.to_dict()
            
            # Add the last prediction
            file_predictions.append(current_prediction)
        
        merged_predictions.extend(file_predictions)
    
    return pd.DataFrame(merged_predictions)

def calculate_iou(pred_start, pred_end, true_start, true_end):
    """Calculate Intersection over Union for two time intervals"""
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection_duration = intersection_end - intersection_start
    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union_duration = union_end - union_start
    
    if union_duration == 0:
        return 0.0
    
    return intersection_duration / union_duration

def evaluate_combined_iou(merged_predictions, ground_truth_df):
    """
    IoU evaluation that combines multiple predictions within the same ground truth word area
    This method treats all predictions within a ground truth word as one combined area
    
    Traditional IoU approach: Calculate IoU for profanity ground truth vs predictions
    Classification metrics: Include ALL ground truth (including none) for complete evaluation
    """
    # Separate ground truth: profanity for IoU calculation, all for classification
    gt_profanity = ground_truth_df[ground_truth_df['label'] != 'none'].copy()
    gt_all = ground_truth_df.copy()  # Include none for classification
    
    detailed_matches = []
    word_ious = []  # IoU for each profanity ground truth word
    
    for _, gt_row in gt_profanity.iterrows():
        gt_start, gt_end = gt_row['start_time'], gt_row['end_time']
        gt_label = gt_row['label']
        gt_file = gt_row['file_path']
        
        # Find all predictions that overlap with this ground truth word
        file_predictions = merged_predictions[merged_predictions['file_path'] == gt_file]
        overlapping_preds = []
        
        for _, pred_row in file_predictions.iterrows():
            pred_start, pred_end = pred_row['start_time'], pred_row['end_time']
            
            # Check if there's any overlap
            if pred_start < gt_end and pred_end > gt_start:
                overlapping_preds.append({
                    'start': pred_start,
                    'end': pred_end,
                    'label': pred_row['predicted_label'],
                    'confidence': pred_row['max_confidence']
                })
        
        if not overlapping_preds:
            # No predictions for this ground truth word
            word_ious.append(0.0)
            detailed_matches.append({
                'gt_file': gt_file,
                'gt_start': gt_start,
                'gt_end': gt_end,
                'gt_label': gt_label,
                'combined_pred_start': None,
                'combined_pred_end': None,
                'num_predictions': 0,
                'predicted_labels': [],
                'iou': 0.0,
                'has_correct_label': False
            })
            continue
        
        # Combine all overlapping predictions into one area
        combined_start = min(pred['start'] for pred in overlapping_preds)
        combined_end = max(pred['end'] for pred in overlapping_preds)
        
        # Calculate IoU between combined prediction area and ground truth
        intersection_start = max(combined_start, gt_start)
        intersection_end = min(combined_end, gt_end)
        
        if intersection_start >= intersection_end:
            iou = 0.0
        else:
            intersection_duration = intersection_end - intersection_start
            union_start = min(combined_start, gt_start)
            union_end = max(combined_end, gt_end)
            union_duration = union_end - union_start
            iou = intersection_duration / union_duration if union_duration > 0 else 0.0
        
        # Check if any prediction has the correct label
        predicted_labels = [pred['label'] for pred in overlapping_preds]
        has_correct_label = gt_label in predicted_labels
        
        word_ious.append(iou)
        detailed_matches.append({
            'gt_file': gt_file,
            'gt_start': gt_start,
            'gt_end': gt_end,
            'gt_label': gt_label,
            'combined_pred_start': combined_start,
            'combined_pred_end': combined_end,
            'num_predictions': len(overlapping_preds),
            'predicted_labels': predicted_labels,
            'iou': iou,
            'has_correct_label': has_correct_label
        })
    
    # Add none ground truth for complete classification evaluation
    gt_none = gt_all[gt_all['label'] == 'none'].copy()
    for _, gt_row in gt_none.iterrows():
        gt_start, gt_end = gt_row['start_time'], gt_row['end_time']
        gt_label = gt_row['label']
        gt_file = gt_row['file_path']
        
        # Check if any profanity predictions overlap with this none region
        file_predictions = merged_predictions[
            (merged_predictions['file_path'] == gt_file) & 
            (merged_predictions['predicted_label'] != 'none')
        ]
        
        overlapping_labels = []
        for _, pred_row in file_predictions.iterrows():
            pred_start, pred_end = pred_row['start_time'], pred_row['end_time']
            if pred_start < gt_end and pred_end > gt_start:
                overlapping_labels.append(pred_row['predicted_label'])
        
        # Add none ground truth to matches (for classification, IoU = 0 since not meaningful)
        detailed_matches.append({
            'gt_file': gt_file,
            'gt_start': gt_start,
            'gt_end': gt_end,
            'gt_label': gt_label,
            'combined_pred_start': None,
            'combined_pred_end': None,
            'num_predictions': len(overlapping_labels),
            'predicted_labels': overlapping_labels,
            'iou': 0.0,  # IoU not meaningful for none regions
            'has_correct_label': len(overlapping_labels) == 0  # Correct if no profanity overlap
        })
    
    # Calculate mean IoU (only from profanity words)
    mean_iou = sum(word_ious) / len(word_ious) if word_ious else 0.0
    
    # Calculate mean IoU per word class (only profanity classes have meaningful IoU)
    word_class_ious = {}
    for match in detailed_matches:
        label = match['gt_label']
        if label != 'none' and match['iou'] > 0:  # Only profanity with actual IoU
            if label not in word_class_ious:
                word_class_ious[label] = []
            word_class_ious[label].append(match['iou'])
    
    mean_iou_per_class = {
        label: sum(ious) / len(ious) if ious else 0.0 
        for label, ious in word_class_ious.items()
    }
    
    return detailed_matches, mean_iou, word_ious, mean_iou_per_class

def evaluate_with_iou(merged_predictions, ground_truth_df, iou_thresholds=None):
    """Evaluate predictions using IoU-based matching with complete ground truth coverage"""
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Use only profanity predictions for IoU calculation (none predictions have no temporal region)
    profanity_predictions = merged_predictions[merged_predictions['predicted_label'] != 'none'].copy()
    # Use ALL ground truth (including none) to match other evaluation methods
    all_ground_truth = ground_truth_df.copy()
    profanity_ground_truth = ground_truth_df[ground_truth_df['label'] != 'none'].copy()
    
    results = {}
    detailed_matches = []
    
    # Process each ground truth word (including none) to create complete evaluation
    for _, gt_row in all_ground_truth.iterrows():
        best_pred_match = None
        best_iou = 0.0
        
        if gt_row['label'] == 'none':
            # For non-profanity ground truth, check if any profanity predictions overlap
            file_preds = profanity_predictions[
                profanity_predictions['file_path'] == gt_row['file_path']
            ]
            
            # Check for any overlapping profanity predictions
            has_overlapping_profanity = False
            for _, pred_row in file_preds.iterrows():
                iou = calculate_iou(
                    pred_row['start_time'], pred_row['end_time'],
                    gt_row['start_time'], gt_row['end_time']
                )
                if iou > 0:  # Any overlap means incorrect prediction on none region
                    has_overlapping_profanity = True
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_match = pred_row
            
            # For none ground truth: correct if no overlapping profanity predictions
            detailed_matches.append({
                'gt_file': gt_row['file_path'],
                'gt_start': gt_row['start_time'],
                'gt_end': gt_row['end_time'],
                'gt_label': gt_row['label'],
                'pred_file': best_pred_match['file_path'] if best_pred_match is not None else '',
                'pred_start': best_pred_match['start_time'] if best_pred_match is not None else 0,
                'pred_end': best_pred_match['end_time'] if best_pred_match is not None else 0,
                'pred_label': best_pred_match['predicted_label'] if best_pred_match is not None else 'none',
                'pred_confidence': best_pred_match['max_confidence'] if best_pred_match is not None else 0,
                'iou': best_iou,
                'is_correct_label': not has_overlapping_profanity  # Correct if no profanity overlap
            })
        else:
            # For profanity ground truth, find best matching profanity prediction
            file_preds = profanity_predictions[
                profanity_predictions['file_path'] == gt_row['file_path']
            ]
            
            for _, pred_row in file_preds.iterrows():
                iou = calculate_iou(
                    pred_row['start_time'], pred_row['end_time'],
                    gt_row['start_time'], gt_row['end_time']
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_match = pred_row
            
            # For profanity ground truth
            detailed_matches.append({
                'gt_file': gt_row['file_path'],
                'gt_start': gt_row['start_time'],
                'gt_end': gt_row['end_time'],
                'gt_label': gt_row['label'],
                'pred_file': best_pred_match['file_path'] if best_pred_match is not None else '',
                'pred_start': best_pred_match['start_time'] if best_pred_match is not None else 0,
                'pred_end': best_pred_match['end_time'] if best_pred_match is not None else 0,
                'pred_label': best_pred_match['predicted_label'] if best_pred_match is not None else 'none',
                'pred_confidence': best_pred_match['max_confidence'] if best_pred_match is not None else 0,
                'iou': best_iou,
                'is_correct_label': (gt_row['label'] == best_pred_match['predicted_label']) if best_pred_match is not None else False
            })
    
    # Calculate metrics for each IoU threshold
    for threshold in iou_thresholds:
        # Ground-truth-centered evaluation
        # TP: Correctly classified ground truth at this IoU threshold
        tp = 0
        fp = 0
        fn = 0
        
        for match in detailed_matches:
            if match['gt_label'] == 'none':
                # For none ground truth: TP if no profanity predictions overlap
                if match['is_correct_label']:  # No overlapping profanity
                    tp += 1
                else:  # Has overlapping profanity predictions
                    fn += 1
            else:
                # For profanity ground truth: TP if IoU >= threshold AND correct label
                if match['iou'] >= threshold and match['is_correct_label']:
                    tp += 1
                else:
                    fn += 1
        
        # FP: Count unmatched profanity predictions that don't correspond to correct ground truth
        # This is complex in ground-truth-centered view, so we'll calculate it differently
        total_ground_truth = len(all_ground_truth)
        fp = len(profanity_predictions) - tp  # Simplified: predictions not resulting in TP
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate accuracy and balanced accuracy for IoU threshold
        y_true = []
        y_pred = []
        
        # Add all matches (TP and FP)
        for match in detailed_matches:
            if match['iou'] >= threshold:
                y_true.append(match['gt_label'])
                y_pred.append(match['pred_label'])
            else:
                y_true.append('none')
                y_pred.append(match['pred_label'])
        
        # Add false negatives (unmatched ground truth)
        matched_gt_count = tp
        for _ in range(len(profanity_ground_truth) - matched_gt_count):
            y_true.append('profane')  # Any profanity class
            y_pred.append('none')
        
        # Calculate accuracy metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        balanced_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        
        # Store multiclass data for classification reporting
        multiclass_y_true = []
        multiclass_y_pred = []
        
        for match in detailed_matches:
            multiclass_y_true.append(match['gt_label'])
            if match['iou'] >= threshold and match['is_correct_label']:
                multiclass_y_pred.append(match['gt_label'])  # Correct prediction
            elif match['iou'] >= threshold and not match['is_correct_label']:
                multiclass_y_pred.append(match['pred_label'])  # Wrong prediction but sufficient IoU
            else:
                multiclass_y_pred.append('none')  # Insufficient IoU = missed detection
        
        # Calculate multiclass metrics
        multiclass_accuracy = accuracy_score(multiclass_y_true, multiclass_y_pred) if len(multiclass_y_true) > 0 else 0.0
        multiclass_balanced_acc = balanced_accuracy_score(multiclass_y_true, multiclass_y_pred) if len(multiclass_y_true) > 0 else 0.0
        
        results[f'iou_{threshold}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'threshold': threshold,
            'multiclass_accuracy': multiclass_accuracy,
            'multiclass_balanced_accuracy': multiclass_balanced_acc,
            'multiclass_y_true': multiclass_y_true,
            'multiclass_y_pred': multiclass_y_pred
        }
    
    # Calculate mean IoU
    mean_iou = np.mean([match['iou'] for match in detailed_matches]) if detailed_matches else 0.0
    
    return results, detailed_matches, mean_iou

def create_binary_confusion_matrix(true_labels, pred_labels, title, save_path):
    """Create binary confusion matrix"""
    setup_thai_font()
    
    # Convert to binary (profane vs non-profane)
    true_binary = ['Profane' if label != 'none' else 'Non-Profane' for label in true_labels]
    pred_binary = ['Profane' if label != 'none' else 'Non-Profane' for label in pred_labels]
    
    cm = confusion_matrix(true_binary, pred_binary, labels=['Non-Profane', 'Profane'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Profane', 'Profane'],
                yticklabels=['Non-Profane', 'Profane'])
    plt.title(f'Binary Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_multiclass_confusion_matrix(true_labels, pred_labels, title, save_path):
    """Create multiclass confusion matrix"""
    setup_thai_font()
    
    # Filter to only profanity classes
    profanity_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
    
    # Filter data to only include profanity predictions and ground truth
    filtered_true = []
    filtered_pred = []
    
    for true, pred in zip(true_labels, pred_labels):
        if true in profanity_labels:  # Only include ground truth profanity
            filtered_true.append(true)
            filtered_pred.append(pred if pred in profanity_labels else 'none')
    
    if len(filtered_true) == 0:
        # Create empty plot if no profanity data
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No profanity data available', ha='center', va='center')
        plt.title(f'Multiclass Confusion Matrix - {title}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Include 'none' in labels for misclassifications
    all_labels = profanity_labels + ['none']
    cm = confusion_matrix(filtered_true, filtered_pred, labels=all_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=profanity_labels)
    plt.title(f'Multiclass Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_profanity_only_confusion_matrix(true_labels, pred_labels, title, save_path):
    """Create multiclass confusion matrix for profanity words only (no 'none' label)"""
    setup_thai_font()
    
    # Filter to only profanity classes
    profanity_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
    
    # Filter data to only include cases where both true and predicted are profanity
    filtered_true = []
    filtered_pred = []
    
    for true, pred in zip(true_labels, pred_labels):
        if true in profanity_labels and pred in profanity_labels:
            filtered_true.append(true)
            filtered_pred.append(pred)
    
    if len(filtered_true) == 0:
        # Create empty plot if no profanity data
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No profanity-to-profanity predictions available', ha='center', va='center')
        plt.title(f'Profanity-Only Confusion Matrix - {title}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Only profanity labels
    cm = confusion_matrix(filtered_true, filtered_pred, labels=profanity_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=profanity_labels, yticklabels=profanity_labels)
    plt.title(f'Profanity-Only Confusion Matrix - {title}\n(Excludes \'none\' predictions)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_binary_metrics(true_labels, pred_labels):
    """Calculate binary classification metrics"""
    # Convert to binary
    true_binary = [1 if label != 'none' else 0 for label in true_labels]
    pred_binary = [1 if label != 'none' else 0 for label in pred_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    balanced_acc = balanced_accuracy_score(true_binary, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_multiclass_metrics(true_labels, pred_labels):
    """Calculate multiclass metrics for profanity only"""
    profanity_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
    
    # Filter to only cases where ground truth is profanity
    filtered_true = []
    filtered_pred = []
    
    for true, pred in zip(true_labels, pred_labels):
        if true in profanity_labels:
            filtered_true.append(true)
            filtered_pred.append(pred if pred in profanity_labels else 'none')
    
    if len(filtered_true) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Calculate metrics
    accuracy = accuracy_score(filtered_true, filtered_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_true, filtered_pred, labels=profanity_labels, 
        average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_window_eval_note(results_dict, output_path, window_true_labels, window_pred_labels):
    """Save window-level evaluation results to note.txt"""
    from sklearn.metrics import classification_report
    
    note_path = output_path / "note.txt"
    
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write("=== WINDOW-LEVEL EVALUATION RESULTS ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Window-level evaluation
        window_results = results_dict['window_evaluation']
        f.write("=== BINARY CLASSIFICATION (Profane vs Non-Profane) ===\n")
        f.write(f"  Accuracy: {window_results['binary']['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy: {window_results['binary']['balanced_accuracy']:.4f}\n")
        f.write(f"  Precision: {window_results['binary']['precision']:.4f}\n")
        f.write(f"  Recall: {window_results['binary']['recall']:.4f}\n")
        f.write(f"  F1-Score: {window_results['binary']['f1']:.4f}\n\n")
        
        # Window-level binary classification report
        f.write("Binary Classification Report:\n")
        window_binary_labels = ['profane' if label != 'none' else 'none' for label in window_true_labels]
        window_binary_pred = ['profane' if label != 'none' else 'none' for label in window_pred_labels]
        window_binary_report = create_consistent_classification_report(window_binary_labels, window_binary_pred, 
                                                                         target_names=['none', 'profane'])
        f.write(window_binary_report + "\n\n")
        
        f.write("=== MULTICLASS CLASSIFICATION (All Classes Including None) ===\n")
        # Calculate all-classes metrics
        all_classes_accuracy = sum(1 for t, p in zip(window_true_labels, window_pred_labels) if t == p) / len(window_true_labels) if window_true_labels else 0.0
        profanity_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
        all_correct_profane = sum(1 for t, p in zip(window_true_labels, window_pred_labels) if t == p and t in profanity_classes)
        all_total_profane_true = sum(1 for t in window_true_labels if t in profanity_classes)
        all_total_profane_pred = sum(1 for p in window_pred_labels if p in profanity_classes)
        
        all_precision = all_correct_profane / all_total_profane_pred if all_total_profane_pred > 0 else 0.0
        all_recall = all_correct_profane / all_total_profane_true if all_total_profane_true > 0 else 0.0
        all_f1 = 2 * (all_precision * all_recall) / (all_precision + all_recall) if (all_precision + all_recall) > 0 else 0.0
        
        f.write(f"  Accuracy: {all_classes_accuracy:.4f}\n")
        f.write(f"  Precision: {all_precision:.4f}\n")
        f.write(f"  Recall: {all_recall:.4f}\n")
        f.write(f"  F1-Score: {all_f1:.4f}\n\n")
        
        # Window-level multiclass classification report (all classes)
        f.write("Multiclass Classification Report (All Classes):\n")
        all_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย', 'none']
        window_all_multiclass_report = create_consistent_classification_report(window_true_labels, window_pred_labels, 
                                                                                 labels=all_labels)
        f.write(window_all_multiclass_report + "\n\n")
        
        f.write("=== MULTICLASS CLASSIFICATION (Profanity Words Only) ===\n")
        f.write(f"  Accuracy: {window_results['multiclass']['accuracy']:.4f}\n")
        f.write(f"  Precision: {window_results['multiclass']['precision']:.4f}\n")
        f.write(f"  Recall: {window_results['multiclass']['recall']:.4f}\n")
        f.write(f"  F1-Score: {window_results['multiclass']['f1']:.4f}\n\n")
        
        # Window-level multiclass classification report (profanity only)
        f.write("Multiclass Classification Report (Profanity Only):\n")
        # Filter to only profanity words
        profanity_true = [t for t in window_true_labels if t != 'none']
        profanity_pred = [p for i, p in enumerate(window_pred_labels) if window_true_labels[i] != 'none']
        profanity_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
        
        if profanity_true and profanity_pred:
            window_profanity_multiclass_report = create_consistent_classification_report(profanity_true, profanity_pred, 
                                                                                          labels=profanity_labels)
            f.write(window_profanity_multiclass_report + "\n\n")
        else:
            f.write("No profanity words available for classification.\n\n")
        
        # Statistics
        f.write("=== DATASET STATISTICS ===\n")
        stats = results_dict['statistics']
        f.write(f"Total Windows: {stats['total_windows']}\n")
        f.write(f"Profanity Windows: {stats['profanity_windows']}\n")
        f.write(f"Non-Profanity Windows: {stats['nonprofanity_windows']}\n\n")
        
        # Generate ROC and DET curves
        f.write("=== ROC AND DET CURVE ANALYSIS ===\n")
        
        # Binary ROC/DET
        try:
            # Get binary scores (use probabilities if available, otherwise binary predictions)
            binary_true = [1 if label != 'none' else 0 for label in window_true_labels]
            binary_pred_scores = [1 if label != 'none' else 0 for label in window_pred_labels]  # Use binary predictions as scores
            
            binary_auc = generate_roc_det_curves(binary_true, binary_pred_scores, output_path, 
                                               "window_eval_binary", binary_labels=['none', 'profane'])
            if binary_auc is not None:
                f.write(f"Binary ROC AUC: {binary_auc:.4f}\n")
            else:
                f.write("Binary ROC AUC: Could not calculate\n")
        except Exception as e:
            f.write(f"Binary ROC/DET generation failed: {e}\n")
        
        # Multiclass ROC
        try:
            # Convert labels to numeric for multiclass ROC
            label_to_num = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4}
            multiclass_true = [label_to_num.get(label, 0) for label in window_true_labels]
            multiclass_pred_scores = [label_to_num.get(label, 0) for label in window_pred_labels]
            
            multiclass_auc = generate_roc_det_curves(multiclass_true, multiclass_pred_scores, output_path, 
                                                   "window_eval_multiclass")
            if multiclass_auc is not None:
                f.write(f"Multiclass Average ROC AUC: {multiclass_auc:.4f}\n")
            else:
                f.write("Multiclass ROC AUC: Could not calculate\n")
        except Exception as e:
            f.write(f"Multiclass ROC generation failed: {e}\n")
            
        f.write("\nROC and DET curve plots saved as PNG files in the output directory.\n")

def save_word_eval_note(results_dict, output_path, word_true_labels, word_pred_labels):
    """Save word-level evaluation results to note.txt"""
    from sklearn.metrics import classification_report
    
    note_path = output_path / "note.txt"
    
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write("=== WORD-LEVEL MERGED EVALUATION RESULTS ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Word-level evaluation
        word_results = results_dict['word_evaluation']
        f.write("=== BINARY CLASSIFICATION (Profane vs Non-Profane) ===\n")
        f.write(f"  Accuracy: {word_results['binary']['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy: {word_results['binary']['balanced_accuracy']:.4f}\n")
        f.write(f"  Precision: {word_results['binary']['precision']:.4f}\n")
        f.write(f"  Recall: {word_results['binary']['recall']:.4f}\n")
        f.write(f"  F1-Score: {word_results['binary']['f1']:.4f}\n\n")
        
        # Word-level binary classification report
        f.write("Binary Classification Report:\n")
        word_binary_labels = ['profane' if label != 'none' else 'none' for label in word_true_labels]
        word_binary_pred = ['profane' if label != 'none' else 'none' for label in word_pred_labels]
        word_binary_report = create_consistent_classification_report(word_binary_labels, word_binary_pred, 
                                                                       target_names=['none', 'profane'])
        f.write(word_binary_report + "\n\n")
        
        f.write("=== MULTICLASS CLASSIFICATION (All Classes Including None) ===\n")
        # Calculate all-classes metrics
        all_classes_accuracy = sum(1 for t, p in zip(word_true_labels, word_pred_labels) if t == p) / len(word_true_labels) if word_true_labels else 0.0
        profanity_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
        all_correct_profane = sum(1 for t, p in zip(word_true_labels, word_pred_labels) if t == p and t in profanity_classes)
        all_total_profane_true = sum(1 for t in word_true_labels if t in profanity_classes)
        all_total_profane_pred = sum(1 for p in word_pred_labels if p in profanity_classes)
        
        all_precision = all_correct_profane / all_total_profane_pred if all_total_profane_pred > 0 else 0.0
        all_recall = all_correct_profane / all_total_profane_true if all_total_profane_true > 0 else 0.0
        all_f1 = 2 * (all_precision * all_recall) / (all_precision + all_recall) if (all_precision + all_recall) > 0 else 0.0
        
        f.write(f"  Accuracy: {all_classes_accuracy:.4f}\n")
        f.write(f"  Precision: {all_precision:.4f}\n")
        f.write(f"  Recall: {all_recall:.4f}\n")
        f.write(f"  F1-Score: {all_f1:.4f}\n\n")
        
        # Word-level multiclass classification report (all classes)
        f.write("Multiclass Classification Report (All Classes):\n")
        all_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย', 'none']
        word_all_multiclass_report = create_consistent_classification_report(word_true_labels, word_pred_labels, 
                                                                               labels=all_labels)
        f.write(word_all_multiclass_report + "\n\n")
        
        f.write("=== MULTICLASS CLASSIFICATION (Profanity Words Only) ===\n")
        f.write(f"  Accuracy: {word_results['multiclass']['accuracy']:.4f}\n")
        f.write(f"  Precision: {word_results['multiclass']['precision']:.4f}\n")
        f.write(f"  Recall: {word_results['multiclass']['recall']:.4f}\n")
        f.write(f"  F1-Score: {word_results['multiclass']['f1']:.4f}\n\n")
        
        # Word-level multiclass classification report (profanity only)
        f.write("Multiclass Classification Report (Profanity Only):\n")
        # Filter to only profanity words
        profanity_true = [t for t in word_true_labels if t != 'none']
        profanity_pred = [p for i, p in enumerate(word_pred_labels) if word_true_labels[i] != 'none']
        profanity_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
        
        if profanity_true and profanity_pred:
            word_profanity_multiclass_report = create_consistent_classification_report(profanity_true, profanity_pred, 
                                                                                        labels=profanity_labels)
            f.write(word_profanity_multiclass_report + "\n\n")
        else:
            f.write("No profanity words available for classification.\n\n")
        
        # Statistics
        f.write("=== DATASET STATISTICS ===\n")
        stats = results_dict['statistics']
        f.write(f"Total Ground Truth Words: {stats['total_gt_words']}\n")
        f.write(f"Total Predicted Words (after merging): {stats['total_pred_words']}\n\n")
        
        # Generate ROC and DET curves
        f.write("=== ROC AND DET CURVE ANALYSIS ===\n")
        
        # Binary ROC/DET
        try:
            # Get binary scores
            binary_true = [1 if label != 'none' else 0 for label in word_true_labels]
            binary_pred_scores = [1 if label != 'none' else 0 for label in word_pred_labels]
            
            binary_auc = generate_roc_det_curves(binary_true, binary_pred_scores, output_path, 
                                               "word_eval_binary", binary_labels=['none', 'profane'])
            if binary_auc is not None:
                f.write(f"Binary ROC AUC: {binary_auc:.4f}\n")
            else:
                f.write("Binary ROC AUC: Could not calculate\n")
        except Exception as e:
            f.write(f"Binary ROC/DET generation failed: {e}\n")
        
        # Multiclass ROC
        try:
            # Convert labels to numeric for multiclass ROC
            label_to_num = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4}
            multiclass_true = [label_to_num.get(label, 0) for label in word_true_labels]
            multiclass_pred_scores = [label_to_num.get(label, 0) for label in word_pred_labels]
            
            multiclass_auc = generate_roc_det_curves(multiclass_true, multiclass_pred_scores, output_path, 
                                                   "word_eval_multiclass")
            if multiclass_auc is not None:
                f.write(f"Multiclass Average ROC AUC: {multiclass_auc:.4f}\n")
            else:
                f.write("Multiclass ROC AUC: Could not calculate\n")
        except Exception as e:
            f.write(f"Multiclass ROC generation failed: {e}\n")
            
        f.write("\nROC and DET curve plots saved as PNG files in the output directory.\n")

def evaluate_word_iou(merged_predictions, ground_truth_df, iou_thresholds=None):
    """Word-level evaluation using IoU thresholds for classification like word_eval but with IoU criteria"""
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Create word-level labels using IoU-based matching for each threshold
    results = {}
    
    # Use ALL ground truth words (including 'none') to match word_eval behavior
    gt_all = ground_truth_df.copy()
    
    for threshold in iou_thresholds:
        word_true_labels = []
        word_pred_labels = []
        
        # For each ground truth word, find best matching merged prediction using IoU
        for _, gt_row in gt_all.iterrows():
            best_iou = 0
            best_match = None
            
            # Find overlapping merged predictions in same file
            file_predictions = merged_predictions[
                merged_predictions['file_path'] == gt_row['file_path']
            ]
            
            for _, pred_row in file_predictions.iterrows():
                iou = calculate_iou(
                    pred_row['start_time'], pred_row['end_time'],
                    gt_row['start_time'], gt_row['end_time']
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_row
            
            word_true_labels.append(gt_row['label'])
            
            # For 'none' ground truth words, prediction should always be 'none'
            if gt_row['label'] == 'none':
                word_pred_labels.append('none')
            else:
                # For profanity words, use IoU threshold to determine if prediction is valid
                if best_match is not None and best_iou >= threshold:
                    word_pred_labels.append(best_match['predicted_label'])
                else:
                    word_pred_labels.append('none')
        
        # Calculate binary metrics for this threshold
        binary_metrics = calculate_binary_metrics(word_true_labels, word_pred_labels)
        multiclass_metrics = calculate_multiclass_metrics(word_true_labels, word_pred_labels)
        
        results[f'iou_{threshold}'] = {
            'threshold': threshold,
            'binary': binary_metrics,
            'multiclass': multiclass_metrics,
            'word_true_labels': word_true_labels,
            'word_pred_labels': word_pred_labels
        }
    
    return results

def save_word_iou_eval_note(results_dict, output_path):
    """Save word-level IoU evaluation results to note.txt"""
    from sklearn.metrics import classification_report
    
    note_path = output_path / "note.txt"
    
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write("=== WORD-LEVEL IoU EVALUATION RESULTS ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Word-level evaluation using IoU thresholds for classification.\n")
        f.write("Same as word_eval but uses IoU criteria to determine valid predictions.\n\n")
        
        # Word IoU evaluation results
        word_iou_results = results_dict['word_iou_evaluation']
        f.write("=== IoU THRESHOLDS PERFORMANCE ===\n")
        for threshold_key, metrics in word_iou_results.items():
            if threshold_key.startswith('iou_'):
                threshold = metrics['threshold']
                f.write(f"\n--- IoU Threshold {threshold} ---\n")
                
                # Binary classification
                binary = metrics['binary']
                f.write(f"Binary Classification (Profane vs Non-Profane):\n")
                f.write(f"  Accuracy: {binary['accuracy']:.4f}\n")
                f.write(f"  Balanced Accuracy: {binary['balanced_accuracy']:.4f}\n")
                f.write(f"  Precision: {binary['precision']:.4f}\n")
                f.write(f"  Recall: {binary['recall']:.4f}\n")
                f.write(f"  F1-Score: {binary['f1']:.4f}\n\n")
                
                # Multiclass classification
                multiclass = metrics['multiclass']
                f.write(f"Multiclass Classification (Profanity Words Only):\n")
                f.write(f"  Accuracy: {multiclass['accuracy']:.4f}\n")
                f.write(f"  Precision: {multiclass['precision']:.4f}\n")
                f.write(f"  Recall: {multiclass['recall']:.4f}\n")
                f.write(f"  F1-Score: {multiclass['f1']:.4f}\n\n")
                
                # Three-tier classification reporting
                word_true_labels = metrics['word_true_labels']
                word_pred_labels = metrics['word_pred_labels']
                
                # 1. Binary Classification Report (profane vs none)
                f.write("=== 1. BINARY CLASSIFICATION (Profane vs None) ===\n")
                word_binary_labels = ['profane' if label != 'none' else 'none' for label in word_true_labels]
                word_binary_pred = ['profane' if label != 'none' else 'none' for label in word_pred_labels]
                
                if len(set(word_binary_labels)) > 1 or len(set(word_binary_pred)) > 1:
                    word_binary_report = create_consistent_classification_report(word_binary_labels, word_binary_pred, 
                                                                               target_names=['none', 'profane'])
                    f.write("Binary Classification Report:\n")
                    f.write(word_binary_report + "\n\n")
                else:
                    f.write("Insufficient class diversity for binary classification report.\n\n")
                
                # 2. Multiclass Classification Report (all classes including none)
                f.write("=== 2. MULTICLASS CLASSIFICATION (All Classes Including None) ===\n")
                profanity_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย', 'none']
                
                # Calculate all-class metrics
                all_accuracy = sum(1 for t, p in zip(word_true_labels, word_pred_labels) if t == p) / len(word_true_labels) if word_true_labels else 0.0
                
                # Calculate precision, recall, F1 for profanity classes only
                profanity_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
                correct_profane = sum(1 for t, p in zip(word_true_labels, word_pred_labels) if t == p and t in profanity_classes)
                total_profane_true = sum(1 for t in word_true_labels if t in profanity_classes)
                total_profane_pred = sum(1 for p in word_pred_labels if p in profanity_classes)
                
                all_precision = correct_profane / total_profane_pred if total_profane_pred > 0 else 0.0
                all_recall = correct_profane / total_profane_true if total_profane_true > 0 else 0.0
                all_f1 = 2 * (all_precision * all_recall) / (all_precision + all_recall) if (all_precision + all_recall) > 0 else 0.0
                
                f.write(f"Accuracy: {all_accuracy:.4f}\n")
                f.write(f"Precision (profanity): {all_precision:.4f}\n")
                f.write(f"Recall (profanity): {all_recall:.4f}\n")
                f.write(f"F1-Score (profanity): {all_f1:.4f}\n\n")
                
                if len(set(word_true_labels)) > 1 or len(set(word_pred_labels)) > 1:
                    word_multiclass_report = create_consistent_classification_report(word_true_labels, word_pred_labels, 
                                                                                    labels=profanity_labels)
                    f.write("Multiclass Classification Report (All Classes Including None):\n")
                    f.write(word_multiclass_report + "\n\n")
                else:
                    f.write("Insufficient class diversity for multiclass classification report.\n\n")
                
                # 3. Multiclass Classification Report (profanity words only - exclude none)
                f.write("=== 3. MULTICLASS CLASSIFICATION (Profanity Words Only) ===\n")
                
                # Filter to only profanity words
                profanity_only_true = []
                profanity_only_pred = []
                for i, (true_label, pred_label) in enumerate(zip(word_true_labels, word_pred_labels)):
                    if true_label != 'none':  # Only include ground truth profanity words
                        profanity_only_true.append(true_label)
                        profanity_only_pred.append(pred_label)
                
                if profanity_only_true and profanity_only_pred:
                    profanity_accuracy = sum(1 for t, p in zip(profanity_only_true, profanity_only_pred) if t == p) / len(profanity_only_true)
                    f.write(f"Accuracy (profanity words only): {profanity_accuracy:.4f}\n\n")
                    
                    profanity_report = create_consistent_classification_report(profanity_only_true, profanity_only_pred, 
                                                                             labels=['เย็ด', 'กู', 'มึง', 'เหี้ย'])
                    f.write("Multiclass Classification Report (Profanity Words Only - Excluding None):\n")
                    f.write(profanity_report + "\n\n")
                else:
                    f.write("No profanity words in ground truth for profanity-only classification.\n\n")
        
        # Statistics
        f.write("=== DATASET STATISTICS ===\n")
        stats = results_dict['statistics']
        f.write(f"Total Ground Truth Words: {stats['total_gt_words']}\n")
        f.write(f"Total Predicted Words (after merging): {stats['total_pred_words']}\n\n")
        
        # Generate ROC and DET curves for each IoU threshold
        f.write("=== ROC AND DET CURVE ANALYSIS ===\n")
        
        # Generate curves for multiple thresholds
        thresholds_to_analyze = [0.1, 0.3, 0.5, 0.7, 0.9]  # Representative subset
        
        for threshold in thresholds_to_analyze:
            threshold_key = f'iou_{threshold}'
            if threshold_key in word_iou_results:
                metrics = word_iou_results[threshold_key]
                word_true_labels = metrics['word_true_labels']
                word_pred_labels = metrics['word_pred_labels']
                
                f.write(f"\n--- ROC/DET Analysis for IoU Threshold {threshold} ---\n")
                
                # Binary ROC/DET
                try:
                    binary_true = [1 if label != 'none' else 0 for label in word_true_labels]
                    binary_pred_scores = [1 if label != 'none' else 0 for label in word_pred_labels]
                    
                    binary_auc = generate_roc_det_curves(binary_true, binary_pred_scores, output_path, 
                                                       f"word_iou_eval_binary_iou{threshold}", 
                                                       binary_labels=['none', 'profane'])
                    if binary_auc is not None:
                        f.write(f"Binary ROC AUC (IoU {threshold}): {binary_auc:.4f}\n")
                    else:
                        f.write(f"Binary ROC AUC (IoU {threshold}): Could not calculate\n")
                except Exception as e:
                    f.write(f"Binary ROC/DET generation failed for IoU {threshold}: {e}\n")
                
                # Multiclass ROC
                try:
                    label_to_num = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4}
                    multiclass_true = [label_to_num.get(label, 0) for label in word_true_labels]
                    multiclass_pred_scores = [label_to_num.get(label, 0) for label in word_pred_labels]
                    
                    multiclass_auc = generate_roc_det_curves(multiclass_true, multiclass_pred_scores, output_path, 
                                                           f"word_iou_eval_multiclass_iou{threshold}")
                    if multiclass_auc is not None:
                        f.write(f"Multiclass Average ROC AUC (IoU {threshold}): {multiclass_auc:.4f}\n")
                    else:
                        f.write(f"Multiclass ROC AUC (IoU {threshold}): Could not calculate\n")
                except Exception as e:
                    f.write(f"Multiclass ROC generation failed for IoU {threshold}: {e}\n")
        
        f.write("\nROC and DET curve plots saved as PNG files in the output directory.\n")

def save_iou_eval_note(results_dict, output_path, combined_matches=None):
    """Save combined IoU analysis results to note.txt with classification reports and confusion matrices"""
    from sklearn.metrics import classification_report
    
    note_path = output_path / "note.txt"
    
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write("=== IoU ANALYSIS RESULTS ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Combined IoU Analysis:\n")
        f.write("Multiple predictions within the same ground truth word are combined as one area.\n")
        f.write("This provides higher IoU values by treating overlapping predictions as unified detection.\n\n")
        
        f.write(f"Combined Mean IoU: {results_dict['combined_mean_iou']:.4f}\n")
        f.write("(Average IoU when multiple predictions per ground truth word are combined)\n\n")
        
        f.write(f"Individual Mean IoU: {results_dict['individual_mean_iou']:.4f}\n")
        f.write("(Average IoU for individual predictions - reference only)\n\n")
        
        # Mean IoU per word class
        f.write("=== MEAN IoU BY WORD CLASS ===\n")
        for word_class, mean_iou in results_dict['mean_iou_per_class'].items():
            f.write(f"  {word_class}: {mean_iou:.4f}\n")
        f.write("\n")
        
        # IoU threshold-based classifications (like word_iou_eval)
        iou_results = results_dict['iou_evaluation']
        f.write("=== IoU THRESHOLDS PERFORMANCE ===\n")
        for threshold_key, metrics in iou_results.items():
            if threshold_key.startswith('iou_'):
                threshold = metrics['threshold']
                f.write(f"\n--- IoU Threshold {threshold} ---\n")
                
                # Binary classification metrics
                f.write(f"Binary Classification (Profane vs Non-Profane):\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1']:.4f}\n\n")
                
                # Multiclass classification metrics
                if 'multiclass_accuracy' in metrics:
                    f.write(f"Multiclass Classification (All Classes):\n")
                    f.write(f"  Accuracy: {metrics['multiclass_accuracy']:.4f}\n")
                    f.write(f"  Balanced Accuracy: {metrics['multiclass_balanced_accuracy']:.4f}\n")
                    
                    # Generate multiclass classification report if data is available
                    if 'multiclass_y_true' in metrics and 'multiclass_y_pred' in metrics:
                        y_true = metrics['multiclass_y_true']
                        y_pred = metrics['multiclass_y_pred']
                        
                        if len(set(y_true)) > 1 or len(set(y_pred)) > 1:
                            try:
                                multiclass_report = create_consistent_classification_report(
                                    y_true, y_pred, 
                                    labels=['เย็ด', 'กู', 'มึง', 'เหี้ย', 'none']
                                )
                                f.write(f"\n  Classification Report:\n")
                                # Indent the classification report
                                for line in multiclass_report.split('\n'):
                                    f.write(f"  {line}\n")
                            except Exception as e:
                                f.write(f"  Classification report generation failed: {e}\n")
                        else:
                            f.write(f"  Insufficient class diversity for classification report.\n")
                    f.write("\n")
                
                # Add detailed classification analysis for this IoU threshold
                if combined_matches:
                    f.write(f"=== DETAILED CLASSIFICATION ANALYSIS (IoU >= {threshold}) ===\n")
                    
                    # Prepare ground truth and predictions for this threshold
                    threshold_gt_labels = []
                    threshold_pred_labels = []
                    
                    # Calculate binary classification based on IoU threshold
                    binary_tp = 0  # Profanity correctly detected with sufficient IoU
                    binary_fp = 0  # Profanity detected with sufficient IoU but wrong/none GT
                    binary_tn = 0  # None correctly identified (no profanity detection with sufficient IoU)
                    binary_fn = 0  # Profanity missed (insufficient IoU)
                    
                    iou_sufficient_count = 0
                    total_profanity_gt = 0
                    total_none_gt = 0
                    
                    for match in combined_matches:
                        if match['gt_label'] == 'none':
                            total_none_gt += 1
                            threshold_gt_labels.append('none')
                            # For none GT: any detection with sufficient IoU is FP, no detection is TN
                            if match['num_predictions'] > 0:
                                # Check if any prediction would meet IoU threshold (for none, any profanity overlap is bad)
                                if match['num_predictions'] > 0:  # Any profanity prediction in none region
                                    binary_fp += 1  # False positive: detected profanity in none region
                                    threshold_pred_labels.append('profane')  # For binary
                                else:
                                    binary_tn += 1  # True negative: no profanity detected in none region
                                    threshold_pred_labels.append('none')
                            else:
                                binary_tn += 1  # True negative: no profanity detected in none region
                                threshold_pred_labels.append('none')
                        else:
                            total_profanity_gt += 1
                            threshold_gt_labels.append(match['gt_label'])
                            if match['iou'] >= threshold:
                                iou_sufficient_count += 1
                                if match['has_correct_label']:
                                    binary_tp += 1  # True positive: profanity detected with correct label and sufficient IoU
                                    threshold_pred_labels.append(match['gt_label'])  # Correct prediction
                                else:
                                    binary_tp += 1  # Still TP for binary: profanity detected (regardless of exact label)
                                    # Use the actual predicted label if available, otherwise 'profane'
                                    threshold_pred_labels.append(match.get('pred_label', 'profane'))
                            else:
                                binary_fn += 1  # False negative: profanity missed (insufficient IoU)
                                threshold_pred_labels.append('none')  # No sufficient detection
                    
                    # Calculate binary metrics
                    binary_precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0.0
                    binary_recall = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0.0  
                    binary_f1 = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0.0
                    binary_accuracy = (binary_tp + binary_tn) / (binary_tp + binary_fp + binary_tn + binary_fn) if (binary_tp + binary_fp + binary_tn + binary_fn) > 0 else 0.0
                    
                    # Binary Classification Report using sklearn
                    from sklearn.metrics import classification_report, confusion_matrix
                    
                    # Convert to binary labels for binary classification report
                    binary_gt = ['none' if label == 'none' else 'profane' for label in threshold_gt_labels]
                    binary_pred = ['none' if label == 'none' else 'profane' for label in threshold_pred_labels]
                    
                    f.write(f"--- Binary Classification Report (IoU >= {threshold}) ---\n")
                    f.write("Accuracy: {:.4f}\n".format(binary_accuracy))
                    f.write("Precision: {:.4f}\n".format(binary_precision))
                    f.write("Recall: {:.4f}\n".format(binary_recall))
                    f.write("F1-Score: {:.4f}\n\n".format(binary_f1))
                    
                    try:
                        binary_report = classification_report(binary_gt, binary_pred, zero_division=0)
                        f.write("Binary Classification Report:\n")
                        f.write(binary_report)
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Could not generate binary classification report: {e}\n\n")
                    
                    # Multiclass Classification Report
                    f.write(f"--- Multiclass Classification Report (IoU >= {threshold}) ---\n")
                    try:
                        # Calculate multiclass metrics
                        from sklearn.metrics import accuracy_score, balanced_accuracy_score
                        
                        multiclass_accuracy = accuracy_score(threshold_gt_labels, threshold_pred_labels)
                        try:
                            multiclass_balanced_accuracy = balanced_accuracy_score(threshold_gt_labels, threshold_pred_labels)
                        except:
                            multiclass_balanced_accuracy = 0.0
                        
                        f.write("Accuracy: {:.4f}\n".format(multiclass_accuracy))
                        f.write("Balanced Accuracy: {:.4f}\n\n".format(multiclass_balanced_accuracy))
                        
                        multiclass_report = classification_report(threshold_gt_labels, threshold_pred_labels, zero_division=0)
                        f.write("Multiclass Classification Report:\n")
                        f.write(multiclass_report)
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Could not generate multiclass classification report: {e}\n\n")
                    
                    # Confusion Matrix Analysis
                    f.write(f"--- Confusion Matrix Analysis (IoU >= {threshold}) ---\n")
                    f.write(f"IoU Threshold: {threshold}\n")
                    f.write(f"Detections with IoU >= {threshold}: {iou_sufficient_count}/{total_profanity_gt} ({(iou_sufficient_count/total_profanity_gt*100) if total_profanity_gt > 0 else 0:.1f}%)\n")
                    f.write(f"Binary Confusion Matrix:\n")
                    f.write(f"                    Predicted\n")
                    f.write(f"                    No-Prof  Profanity\n")
                    f.write(f"  Actual No-Prof     {binary_tn:6d}     {binary_fp:6d}\n")
                    f.write(f"  Actual Profanity   {binary_fn:6d}     {binary_tp:6d}\n")
                    f.write(f"  \n")
                    f.write(f"Binary Metrics Summary:\n")
                    f.write(f"  Precision: {binary_precision:.4f}\n")
                    f.write(f"  Recall: {binary_recall:.4f}\n") 
                    f.write(f"  F1-Score: {binary_f1:.4f}\n")
                    f.write(f"  Accuracy: {binary_accuracy:.4f}\n")
                    f.write(f"\n")
        f.write("\n")
        
        # Per-Class Ground Truth and Prediction Analysis
        if combined_matches:
            f.write("=== GROUND TRUTH vs PREDICTION BREAKDOWN BY CLASS ===\n")
            f.write("Detailed analysis of each class: ground truth count vs correct/incorrect predictions\n\n")
            
            # Count ground truth occurrences by class
            gt_class_counts = {}
            correct_predictions = {}
            incorrect_predictions = {}
            
            # Initialize counters
            all_classes = set()
            for match in combined_matches:
                all_classes.add(match['gt_label'])
            
            for class_name in all_classes:
                gt_class_counts[class_name] = 0
                correct_predictions[class_name] = 0
                incorrect_predictions[class_name] = 0
            
            # Count statistics for each ground truth word
            for match in combined_matches:
                gt_label = match['gt_label']
                gt_class_counts[gt_label] += 1
                
                if match['has_correct_label']:
                    correct_predictions[gt_label] += 1
                else:
                    incorrect_predictions[gt_label] += 1
            
            # Display the breakdown
            f.write("Class Analysis:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<12} {'GT Count':<10} {'Correct':<10} {'Incorrect':<12} {'Accuracy':<10}\n")
            f.write("-" * 70 + "\n")
            
            total_gt = 0
            total_correct = 0
            total_incorrect = 0
            
            # Sort classes: profanity first, then none
            profanity_classes = [cls for cls in sorted(all_classes) if cls != 'none']
            none_classes = [cls for cls in all_classes if cls == 'none']
            sorted_classes = profanity_classes + none_classes
            
            for class_name in sorted_classes:
                gt_count = gt_class_counts[class_name]
                correct = correct_predictions[class_name]
                incorrect = incorrect_predictions[class_name]
                accuracy = (correct / gt_count) * 100 if gt_count > 0 else 0.0
                
                f.write(f"{class_name:<12} {gt_count:<10} {correct:<10} {incorrect:<12} {accuracy:<9.1f}%\n")
                
                total_gt += gt_count
                total_correct += correct
                total_incorrect += incorrect
            
            f.write("-" * 70 + "\n")
            overall_accuracy = (total_correct / total_gt) * 100 if total_gt > 0 else 0.0
            f.write(f"{'TOTAL':<12} {total_gt:<10} {total_correct:<10} {total_incorrect:<12} {overall_accuracy:<9.1f}%\n")
            f.write("-" * 70 + "\n\n")
            
            # Additional detailed breakdown
            f.write("Detailed Explanation:\n")
            for class_name in sorted_classes:
                gt_count = gt_class_counts[class_name]
                correct = correct_predictions[class_name]
                incorrect = incorrect_predictions[class_name]
                
                if class_name == 'none':
                    f.write(f"• {class_name}: {gt_count} regions in ground truth\n")
                    f.write(f"  - {correct} correctly identified (no profanity predictions overlap)\n")
                    f.write(f"  - {incorrect} incorrectly identified (profanity predictions overlap)\n")
                else:
                    f.write(f"• {class_name}: {gt_count} words in ground truth\n")
                    f.write(f"  - {correct} correctly predicted (sufficient IoU + correct label)\n")
                    f.write(f"  - {incorrect} incorrectly predicted (insufficient IoU or wrong label)\n")
            f.write("\n")
        
        # Classification evaluation based on IoU matches
        if combined_matches:
            f.write("=== CLASSIFICATION EVALUATION (IoU-based) ===\n")
            f.write("Classification performance based on combined IoU predictions\n\n")
            
            # Create comprehensive labels for all ground truth words (including proper none handling)
            iou_true_labels = []
            iou_pred_labels = []
            
            for match in combined_matches:
                gt_label = match['gt_label']
                iou_true_labels.append(gt_label)
                
                if match['num_predictions'] > 0:
                    # If predictions exist, use the most common predicted label or the correct one if available
                    if match['has_correct_label']:
                        iou_pred_labels.append(gt_label)  # Correct prediction
                    else:
                        # Use the first predicted label (could be improved to use most common)
                        if match['predicted_labels']:
                            iou_pred_labels.append(match['predicted_labels'][0])
                        else:
                            iou_pred_labels.append('none')  # No valid prediction
                else:
                    # No predictions for this ground truth word
                    iou_pred_labels.append('none')
            
            # Three-tier classification reporting
            
            # 1. Binary Classification (profane vs none)
            f.write("=== 1. BINARY CLASSIFICATION (Profane vs None) ===\n")
            binary_true = ['profane' if label != 'none' else 'none' for label in iou_true_labels]
            binary_pred = ['profane' if label != 'none' else 'none' for label in iou_pred_labels]
            
            # Calculate binary metrics
            binary_accuracy = sum(1 for t, p in zip(binary_true, binary_pred) if t == p) / len(binary_true) if binary_true else 0.0
            profane_correct = sum(1 for t, p in zip(binary_true, binary_pred) if t == 'profane' and p == 'profane')
            profane_total = sum(1 for t in binary_true if t == 'profane')
            profane_pred_total = sum(1 for p in binary_pred if p == 'profane')
            
            binary_precision = profane_correct / profane_pred_total if profane_pred_total > 0 else 0.0
            binary_recall = profane_correct / profane_total if profane_total > 0 else 0.0
            binary_f1 = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0.0
            
            f.write(f"Accuracy: {binary_accuracy:.4f}\n")
            f.write(f"Precision: {binary_precision:.4f}\n")
            f.write(f"Recall: {binary_recall:.4f}\n")
            f.write(f"F1-Score: {binary_f1:.4f}\n\n")
            
            # Binary classification report
            if len(set(binary_true)) > 1 or len(set(binary_pred)) > 1:
                binary_report = create_consistent_classification_report(binary_true, binary_pred, 
                                                                          target_names=['none', 'profane'])
                f.write("Binary Classification Report:\n")
                f.write(binary_report + "\n\n")
                
                # Generate and save binary confusion matrix
                try:
                    create_profanity_only_confusion_matrix(binary_true, binary_pred, 
                                                         'Binary Classification Confusion Matrix (IoU-based)',
                                                         output_path / 'binary_confusion_matrix.png')
                except Exception as e:
                    print(f"Warning: Could not create binary confusion matrix: {e}")
            else:
                f.write("Insufficient class diversity for binary classification report.\n\n")
            
            # 2. Multiclass Classification (all classes including none)
            f.write("=== 2. MULTICLASS CLASSIFICATION (All Classes Including None) ===\n")
            
            # Calculate multiclass metrics
            multiclass_accuracy = sum(1 for t, p in zip(iou_true_labels, iou_pred_labels) if t == p) / len(iou_true_labels) if iou_true_labels else 0.0
            
            # Calculate precision, recall, F1 for profanity classes only
            profanity_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
            correct_profane = sum(1 for t, p in zip(iou_true_labels, iou_pred_labels) if t == p and t in profanity_classes)
            total_profane_true = sum(1 for t in iou_true_labels if t in profanity_classes)
            total_profane_pred = sum(1 for p in iou_pred_labels if p in profanity_classes)
            
            multiclass_precision = correct_profane / total_profane_pred if total_profane_pred > 0 else 0.0
            multiclass_recall = correct_profane / total_profane_true if total_profane_true > 0 else 0.0
            multiclass_f1 = 2 * (multiclass_precision * multiclass_recall) / (multiclass_precision + multiclass_recall) if (multiclass_precision + multiclass_recall) > 0 else 0.0
            
            f.write(f"Accuracy: {multiclass_accuracy:.4f}\n")
            f.write(f"Precision (profanity): {multiclass_precision:.4f}\n")
            f.write(f"Recall (profanity): {multiclass_recall:.4f}\n")
            f.write(f"F1-Score (profanity): {multiclass_f1:.4f}\n\n")
            
            # Multiclass classification report
            all_labels = ['เย็ด', 'กู', 'มึง', 'เหี้ย', 'none']
            if len(set(iou_true_labels)) > 1 or len(set(iou_pred_labels)) > 1:
                multiclass_report = create_consistent_classification_report(iou_true_labels, iou_pred_labels, 
                                                                              labels=all_labels)
                f.write("Multiclass Classification Report (All Classes Including None):\n")
                f.write(multiclass_report + "\n\n")
                
                # Generate and save multiclass confusion matrix (including none)
                try:
                    create_profanity_only_confusion_matrix(iou_true_labels, iou_pred_labels, 
                                                         'Multiclass Confusion Matrix (IoU-based, All Classes)',
                                                         output_path / 'multiclass_confusion_matrix.png')
                except Exception as e:
                    print(f"Warning: Could not create multiclass confusion matrix: {e}")
            else:
                f.write("Insufficient class diversity for multiclass classification report.\n\n")
            
            # 3. Multiclass Classification (profanity words only - exclude none)
            f.write("=== 3. MULTICLASS CLASSIFICATION (Profanity Words Only) ===\n")
            
            # Filter to only profanity words for profanity-only matrix
            profanity_true = []
            profanity_pred = []
            for i, (true_label, pred_label) in enumerate(zip(iou_true_labels, iou_pred_labels)):
                if true_label != 'none':  # Only include ground truth profanity words
                    profanity_true.append(true_label)
                    profanity_pred.append(pred_label)
            
            if profanity_true and profanity_pred:
                profanity_accuracy = sum(1 for t, p in zip(profanity_true, profanity_pred) if t == p) / len(profanity_true)
                f.write(f"Accuracy (profanity words only): {profanity_accuracy:.4f}\n\n")
                
                profanity_report = create_consistent_classification_report(profanity_true, profanity_pred, 
                                                                         labels=['เย็ด', 'กู', 'มึง', 'เหี้ย'])
                f.write("Multiclass Classification Report (Profanity Words Only - Excluding None):\n")
                f.write(profanity_report + "\n\n")
                
                # Generate profanity-only confusion matrix (excludes none like other methods)
                try:
                    create_profanity_only_confusion_matrix(profanity_true, profanity_pred, 
                                                         'Profanity-Only Confusion Matrix (IoU-based)',
                                                         output_path / 'profanity_only_confusion_matrix.png')
                except Exception as e:
                    print(f"Warning: Could not create profanity-only confusion matrix: {e}")
            else:
                f.write("No profanity words in ground truth for profanity-only classification.\n\n")
            
            # IoU threshold performance
            f.write("=== IoU THRESHOLD ANALYSIS ===\n")
            f.write("Percentage of ground truth words that achieve different IoU thresholds:\n\n")
            
            iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            # Separate analysis for profanity and none
            profanity_matches = [m for m in combined_matches if m['gt_label'] != 'none']
            none_matches = [m for m in combined_matches if m['gt_label'] == 'none']
            
            # Overall IoU distribution
            f.write("Overall IoU Distribution:\n")
            for threshold in iou_thresholds:
                high_iou_matches = len([m for m in combined_matches if m['iou'] >= threshold])
                percentage = (high_iou_matches / len(combined_matches)) * 100 if combined_matches else 0.0
                f.write(f"  IoU >= {threshold}: {high_iou_matches}/{len(combined_matches)} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Profanity-only IoU distribution
            if profanity_matches:
                f.write("Profanity Words IoU Distribution:\n")
                for threshold in iou_thresholds:
                    high_iou_profanity = len([m for m in profanity_matches if m['iou'] >= threshold])
                    percentage = (high_iou_profanity / len(profanity_matches)) * 100 if profanity_matches else 0.0
                    f.write(f"  IoU >= {threshold}: {high_iou_profanity}/{len(profanity_matches)} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # IoU range distribution
            f.write("IoU Range Distribution (Profanity Words Only):\n")
            if profanity_matches:
                iou_ranges = [
                    (0.0, 0.1, "0.0-0.1"),
                    (0.1, 0.3, "0.1-0.3"), 
                    (0.3, 0.5, "0.3-0.5"),
                    (0.5, 0.7, "0.5-0.7"),
                    (0.7, 0.9, "0.7-0.9"),
                    (0.9, 1.0, "0.9-1.0")
                ]
                
                for min_iou, max_iou, range_label in iou_ranges:
                    if max_iou == 1.0:
                        count = len([m for m in profanity_matches if min_iou <= m['iou'] <= max_iou])
                    else:
                        count = len([m for m in profanity_matches if min_iou <= m['iou'] < max_iou])
                    percentage = (count / len(profanity_matches)) * 100 if profanity_matches else 0.0
                    f.write(f"  IoU {range_label}: {count}/{len(profanity_matches)} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Binary detection summary across all thresholds
            f.write("Binary Detection Summary (Profanity vs None):\n")
            for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:  # Key thresholds only
                detected_profanity = len([m for m in profanity_matches if m['iou'] >= threshold])
                missed_profanity = len(profanity_matches) - detected_profanity
                correct_none = len([m for m in none_matches if m['num_predictions'] == 0])
                incorrect_none = len(none_matches) - correct_none
                
                total_correct = detected_profanity + correct_none
                total_samples = len(combined_matches)
                binary_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
                
                f.write(f"  IoU >= {threshold}: Detected {detected_profanity}/{len(profanity_matches)} profanity, ")
                f.write(f"Correct {correct_none}/{len(none_matches)} none (Accuracy: {binary_accuracy:.1f}%)\n")
            f.write("\n")
        
        f.write("=== ANALYSIS FILES ===\n")
        f.write("- combined_iou_analysis.csv: Combined predictions per ground truth word\n")
        f.write("- individual_predictions_iou.csv: Individual prediction IoU values\n")
        f.write("- mean_iou_by_word_class.csv: IoU statistics by profanity word class\n")
        f.write("- window_stride_iou_summary.csv: Summary for finding best window/stride configurations\n\n")
        
        # Statistics
        f.write("=== DATASET STATISTICS ===\n")
        stats = results_dict['statistics']
        f.write(f"Total Ground Truth Words: {stats['total_gt_words']}\n")
        f.write(f"Total Predicted Words (after merging): {stats['total_pred_words']}\n\n")
        
        # Generate ROC and DET curves
        f.write("=== ROC AND DET CURVE ANALYSIS ===\n")
        
        if combined_matches:
            # Convert combined matches to labels for ROC/DET analysis
            # Use the combined IoU evaluation results for overall performance
            combined_true_labels = []
            combined_pred_labels = []
            combined_iou_scores = []
            
            for match in combined_matches:
                combined_true_labels.append(match['gt_label'])
                # Use IoU as confidence score and apply threshold-based prediction
                iou_score = match['iou']
                combined_iou_scores.append(iou_score)
                
                if match['gt_label'] == 'none':
                    # For none regions, any profanity detection is wrong
                    if match['num_predictions'] > 0:
                        combined_pred_labels.append('profane')  # Wrong prediction
                    else:
                        combined_pred_labels.append('none')     # Correct
                else:
                    # For profanity regions, use IoU threshold to determine prediction
                    if iou_score >= 0.1:  # Using 0.1 as representative threshold
                        if match['has_correct_label']:
                            combined_pred_labels.append(match['gt_label'])  # Correct specific prediction
                        else:
                            combined_pred_labels.append('profane')  # Generic profanity detection
                    else:
                        combined_pred_labels.append('none')  # Insufficient IoU
            
            # Binary ROC/DET using IoU scores as confidence
            try:
                binary_true = [1 if label != 'none' else 0 for label in combined_true_labels]
                # Use IoU scores as prediction confidence
                binary_pred_scores = []
                for i, (true_label, iou_score, num_preds) in enumerate(zip(combined_true_labels, combined_iou_scores, [m['num_predictions'] for m in combined_matches])):
                    if true_label == 'none':
                        # For none, confidence is inverse of any profanity detection
                        binary_pred_scores.append(1.0 - min(1.0, num_preds * 0.5))  # More predictions = lower none confidence
                    else:
                        # For profanity, confidence is IoU score
                        binary_pred_scores.append(iou_score)
                
                binary_auc = generate_roc_det_curves(binary_true, binary_pred_scores, output_path, 
                                                   "iou_eval_binary", binary_labels=['none', 'profane'])
                if binary_auc is not None:
                    f.write(f"Binary ROC AUC (IoU-based): {binary_auc:.4f}\n")
                else:
                    f.write("Binary ROC AUC (IoU-based): Could not calculate\n")
            except Exception as e:
                f.write(f"Binary ROC/DET generation failed: {e}\n")
            
            # Multiclass ROC using IoU and label confidence
            try:
                label_to_num = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4}
                multiclass_true = [label_to_num.get(label, 0) for label in combined_true_labels]
                
                # Create multiclass prediction scores based on IoU and label correctness
                n_classes = 5
                multiclass_pred_scores = np.zeros((len(combined_matches), n_classes))
                
                for i, match in enumerate(combined_matches):
                    if match['gt_label'] == 'none':
                        if match['num_predictions'] == 0:
                            multiclass_pred_scores[i, 0] = 1.0  # High confidence for none
                        else:
                            # Distribute among profanity classes based on predictions
                            multiclass_pred_scores[i, 1:] = 0.25  # Equal distribution among profanity
                    else:
                        gt_idx = label_to_num[match['gt_label']]
                        if match['iou'] >= 0.1 and match['has_correct_label']:
                            multiclass_pred_scores[i, gt_idx] = match['iou']  # Confidence based on IoU
                        else:
                            multiclass_pred_scores[i, 0] = 1.0 - match['iou']  # Higher none confidence for low IoU
                
                multiclass_auc = generate_roc_det_curves(multiclass_true, multiclass_pred_scores, output_path, 
                                                       "iou_eval_multiclass")
                if multiclass_auc is not None:
                    f.write(f"Multiclass Average ROC AUC (IoU-based): {multiclass_auc:.4f}\n")
                else:
                    f.write("Multiclass ROC AUC (IoU-based): Could not calculate\n")
            except Exception as e:
                f.write(f"Multiclass ROC generation failed: {e}\n")
        else:
            f.write("No combined matches available for ROC/DET analysis.\n")
            
        f.write("\nROC and DET curve plots saved as PNG files in the output directory.\n")
        f.write("\nNote: Combined IoU provides better temporal overlap measurement by treating\n")
        f.write("multiple predictions within the same ground truth word as unified detection areas.\n")

def process_single_configuration(model, feature_extractor, device, config, ground_truth_df, output_base_dir):
    """Process a single window/stride configuration"""
    
    csv_path = config['csv_path']
    window_name = config['window']
    stride_name = config['stride']
    
    print(f"\nProcessing: {window_name}/{stride_name}")
    print(f"CSV: {csv_path}")
    
    # Create output directories for different evaluation methods
    methods = ['window_eval', 'word_eval', 'word_iou_eval', 'iou_eval']
    output_dirs = {}
    
    for method in methods:
        method_dir = output_base_dir / method / window_name / stride_name
        method_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[method] = method_dir
    
    # Load windowed data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} windows")
    
    # Perform one-time prediction for all windows
    print("Performing model predictions...")
    predictions = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        pred_result = evaluate_window(
            model, feature_extractor, 
            row['file_path'], row['start_time'], row['end_time'], 
            device
        )
        
        predictions.append({
            'file_path': row['file_path'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'true_label': row['label'],
            'predicted_label': pred_result['predicted_label'],
            'predicted_class': pred_result['predicted_class'],
            'max_confidence': pred_result['max_confidence'],
            'confidence_scores': pred_result['confidence_scores']
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Save raw predictions
    predictions_df.to_csv(output_dirs['window_eval'] / 'raw_predictions.csv', index=False)
    
    print("Computing evaluations...")
    
    # 1. Window-level evaluation
    print("  1. Window-level evaluation...")
    window_binary = calculate_binary_metrics(
        predictions_df['true_label'].tolist(),
        predictions_df['predicted_label'].tolist()
    )
    window_multiclass = calculate_multiclass_metrics(
        predictions_df['true_label'].tolist(),
        predictions_df['predicted_label'].tolist()
    )
    
    # Create confusion matrices for window evaluation
    create_binary_confusion_matrix(
        predictions_df['true_label'].tolist(),
        predictions_df['predicted_label'].tolist(),
        'Window Level',
        output_dirs['window_eval'] / 'binary_confusion_matrix.png'
    )
    
    create_multiclass_confusion_matrix(
        predictions_df['true_label'].tolist(),
        predictions_df['predicted_label'].tolist(),
        'Window Level',
        output_dirs['window_eval'] / 'multiclass_confusion_matrix.png'
    )
    
    # Create profanity-only confusion matrix (no 'none' label)
    create_profanity_only_confusion_matrix(
        predictions_df['true_label'].tolist(),
        predictions_df['predicted_label'].tolist(),
        'Window Level',
        output_dirs['window_eval'] / 'profanity_only_confusion_matrix.png'
    )
    
    # 2. Word-level merged evaluation
    print("  2. Word-level merged evaluation...")
    
    # Merge overlapping predictions
    profanity_predictions = predictions_df[predictions_df['predicted_label'] != 'none'].copy()
    merged_predictions = merge_overlapping_predictions(profanity_predictions)
    
    # Save merged predictions
    merged_predictions.to_csv(output_dirs['word_eval'] / 'merged_predictions.csv', index=False)
    
    # Create word-level ground truth mapping
    word_true_labels = []
    word_pred_labels = []
    
    # For each ground truth word, find best matching merged prediction
    gt_profanity = ground_truth_df[ground_truth_df['label'] != 'none'].copy()
    
    for _, gt_row in gt_profanity.iterrows():
        # Find overlapping merged predictions
        file_predictions = merged_predictions[
            merged_predictions['file_path'] == gt_row['file_path']
        ]
        
        best_match = None
        best_overlap = 0.0
        
        for _, pred_row in file_predictions.iterrows():
            # Calculate overlap
            overlap_start = max(gt_row['start_time'], pred_row['start_time'])
            overlap_end = min(gt_row['end_time'], pred_row['end_time'])
            
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = pred_row
        
        word_true_labels.append(gt_row['label'])
        if best_match is not None:
            word_pred_labels.append(best_match['predicted_label'])
        else:
            word_pred_labels.append('none')
    
    # Calculate word-level metrics
    word_binary = calculate_binary_metrics(word_true_labels, word_pred_labels)
    word_multiclass = calculate_multiclass_metrics(word_true_labels, word_pred_labels)
    
    # Create confusion matrices for word evaluation
    create_binary_confusion_matrix(
        word_true_labels, word_pred_labels,
        'Word Level (Merged)',
        output_dirs['word_eval'] / 'binary_confusion_matrix.png'
    )
    
    create_multiclass_confusion_matrix(
        word_true_labels, word_pred_labels,
        'Word Level (Merged)',
        output_dirs['word_eval'] / 'multiclass_confusion_matrix.png'
    )
    
    # Create profanity-only confusion matrix (no 'none' label)
    create_profanity_only_confusion_matrix(
        word_true_labels, word_pred_labels,
        'Word Level (Merged)',
        output_dirs['word_eval'] / 'profanity_only_confusion_matrix.png'
    )
    
    # 3. Word-level IoU evaluation (word_eval + IoU thresholds)
    print("  3. Word-level IoU evaluation...")
    word_iou_results = evaluate_word_iou(merged_predictions, ground_truth_df)
    
    # Create confusion matrices for different IoU thresholds (save best threshold only)
    best_threshold = 0.5  # Default threshold for confusion matrices
    if f'iou_{best_threshold}' in word_iou_results:
        best_result = word_iou_results[f'iou_{best_threshold}']
        
        # Binary confusion matrix
        create_binary_confusion_matrix(
            best_result['word_true_labels'], best_result['word_pred_labels'],
            f'Word IoU Level (threshold={best_threshold})',
            output_dirs['word_iou_eval'] / 'binary_confusion_matrix.png'
        )
        
        # Multiclass confusion matrix
        create_multiclass_confusion_matrix(
            best_result['word_true_labels'], best_result['word_pred_labels'],
            f'Word IoU Level (threshold={best_threshold})',
            output_dirs['word_iou_eval'] / 'multiclass_confusion_matrix.png'
        )
        
        # Profanity-only confusion matrix
        create_profanity_only_confusion_matrix(
            best_result['word_true_labels'], best_result['word_pred_labels'],
            f'Word IoU Level (threshold={best_threshold})',
            output_dirs['word_iou_eval'] / 'profanity_only_confusion_matrix.png'
        )
    
    # Save word IoU threshold results
    word_iou_summary = pd.DataFrame([
        {'threshold': result['threshold'], 
         'binary_f1': result['binary']['f1'],
         'binary_precision': result['binary']['precision'],
         'binary_recall': result['binary']['recall'],
         'binary_accuracy': result['binary']['accuracy'],
         'binary_balanced_accuracy': result['binary']['balanced_accuracy'],
         'multiclass_f1': result['multiclass']['f1'],
         'multiclass_precision': result['multiclass']['precision'],
         'multiclass_recall': result['multiclass']['recall'],
         'multiclass_accuracy': result['multiclass']['accuracy']}
        for result in word_iou_results.values()
    ])
    word_iou_summary.to_csv(output_dirs['word_iou_eval'] / 'word_iou_threshold_results.csv', index=False)
    
    # 4. IoU analysis (pure temporal overlap analysis)
    print("  4. IoU analysis...")
    
    # Combined IoU evaluation (multiple predictions combined per ground truth word)
    combined_matches, combined_mean_iou, word_ious, mean_iou_per_class = evaluate_combined_iou(merged_predictions, ground_truth_df)
    
    # Individual prediction IoU evaluation (for reference)
    iou_results, iou_matches, individual_mean_iou = evaluate_with_iou(merged_predictions, ground_truth_df)
    
    # Save combined IoU analysis (main focus)
    combined_matches_df = pd.DataFrame(combined_matches)
    combined_matches_df.to_csv(output_dirs['iou_eval'] / 'combined_iou_analysis.csv', index=False)
    
    # Save individual prediction IoU analysis (reference)
    individual_matches_df = pd.DataFrame(iou_matches)
    individual_matches_df.to_csv(output_dirs['iou_eval'] / 'individual_predictions_iou.csv', index=False)
    
    # Save IoU threshold analysis (for reference)
    iou_summary = pd.DataFrame([
        {'threshold': result['threshold'], **{k: v for k, v in result.items() if k != 'threshold'}}
        for result in iou_results.values()
    ])
    iou_summary.to_csv(output_dirs['iou_eval'] / 'iou_threshold_reference.csv', index=False)
    
    # Save IoU statistics by word class
    iou_by_class_df = pd.DataFrame([
        {'word_class': word_class, 'mean_iou': mean_iou, 'num_instances': len([m for m in combined_matches if m['gt_label'] == word_class])}
        for word_class, mean_iou in mean_iou_per_class.items()
    ])
    iou_by_class_df.to_csv(output_dirs['iou_eval'] / 'mean_iou_by_word_class.csv', index=False)
    
    # Save window/stride IoU summary for finding best configurations
    window_stride_summary = pd.DataFrame([{
        'window': window_name,
        'stride': stride_name,
        'combined_mean_iou': combined_mean_iou,
        'individual_mean_iou': individual_mean_iou,
        'total_gt_words': len(gt_profanity),
        'total_predictions': len(merged_predictions),
        **{f'mean_iou_{word_class}': mean_iou for word_class, mean_iou in mean_iou_per_class.items()}
    }])
    window_stride_summary.to_csv(output_dirs['iou_eval'] / 'window_stride_iou_summary.csv', index=False)
    
    # Compile all results
    results_dict = {
        'window_evaluation': {
            'binary': window_binary,
            'multiclass': window_multiclass
        },
        'word_evaluation': {
            'binary': word_binary,
            'multiclass': word_multiclass
        },
        'word_iou_evaluation': word_iou_results,
        'iou_evaluation': iou_results,
        'combined_mean_iou': combined_mean_iou,
        'individual_mean_iou': individual_mean_iou,
        'mean_iou_per_class': mean_iou_per_class,
        'statistics': {
            'total_windows': len(predictions_df),
            'total_gt_words': len(gt_profanity),
            'total_pred_words': len(merged_predictions),
            'profanity_windows': len(predictions_df[predictions_df['predicted_label'] != 'none']),
            'nonprofanity_windows': len(predictions_df[predictions_df['predicted_label'] == 'none'])
        }
    }
    
    # Save results to note.txt for each evaluation method
    save_window_eval_note(results_dict, output_dirs['window_eval'], 
                         predictions_df['true_label'].tolist(), 
                         predictions_df['predicted_label'].tolist())
    
    save_word_eval_note(results_dict, output_dirs['word_eval'],
                       word_true_labels, word_pred_labels)
    
    save_word_iou_eval_note(results_dict, output_dirs['word_iou_eval'])
    
    save_iou_eval_note(results_dict, output_dirs['iou_eval'], combined_matches)
    
    # Save consolidated results as JSON
    with open(output_dirs['window_eval'] / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  Completed: {window_name}/{stride_name}")
    
    return results_dict

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation processor - evaluate a single window/stride configuration",
        epilog="""
Examples:
  # Evaluate using window and stride arguments
  python comprehensive_evaluation_processor.py --window 0.3s --stride 0.05s
  
  # Evaluate using direct CSV file path
  python comprehensive_evaluation_processor.py --csv_file csv/eval_by_0.05/window_0.3s/stride_0.05s.csv
  
  # Evaluate with custom model and output directory
  python comprehensive_evaluation_processor.py --window 2.0s --stride 0.25s --model_path models/my_model --output_dir results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv_file", help="Path to CSV file to evaluate (alternative to --window and --stride)")
    parser.add_argument("--window", help="Window size (e.g., '0.3s', '1.0s', '2.0s')")
    parser.add_argument("--stride", help="Stride size (e.g., '0.05s', '0.125s', '0.25s')")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path (recommended: 4_classes_max_steps)")
    parser.add_argument("--ground_truth", default="./csv/eval_5labels.csv", help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="./new_evaluation_results", help="Output directory")
    parser.add_argument("--eval_type", default="eval_by_0.05", help="Evaluation type (eval_by_0.05 or eval_percent)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.csv_file and not (args.window and args.stride):
        parser.error("Either --csv_file OR both --window and --stride must be provided")
    
    if args.csv_file and (args.window or args.stride):
        parser.error("Cannot use --csv_file together with --window/--stride. Choose one approach.")
    
    # Determine CSV file path and window/stride configuration
    if args.csv_file:
        # Use provided CSV file and extract window/stride from path
        csv_path = Path(args.csv_file)
        window_name = csv_path.parent.name  # e.g., "window_0.3s"
        stride_name = csv_path.stem  # e.g., "stride_0.125s"
        csv_file_path = args.csv_file
    else:
        # Build CSV file path from window and stride
        window_name = f"window_{args.window}"
        stride_name = f"stride_{args.stride}"
        
        # Construct the expected CSV file path
        csv_file_path = f"csv/{args.eval_type}/{window_name}/{stride_name}.csv"
        
        # Check if the CSV file exists
        if not Path(csv_file_path).exists():
            print(f"Error: CSV file not found at expected path: {csv_file_path}")
            print(f"Please ensure the CSV file exists or use --csv_file to specify the exact path")
            return
    
    print("=== COMPREHENSIVE EVALUATION PROCESSOR ===")
    print(f"Model: {args.model_path}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Evaluation Type: {args.eval_type}")
    
    if args.csv_file:
        print(f"CSV File: {args.csv_file}")
    else:
        print(f"Window: {args.window}")
        print(f"Stride: {args.stride}")
        print(f"Expected CSV: {csv_file_path}")
    print(f"Window Name: {window_name}")
    print(f"Stride Name: {stride_name}")
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try loading as Hugging Face model (works for models with preprocessor_config.json)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
        print(f"Successfully loaded model and feature extractor from {args.model_path}")
    except Exception as e1:
        print(f"Error loading model from pretrained: {e1}")
        try:
            # Try loading model with safetensors but use base feature extractor
            print("Trying to load model with safetensors and base feature extractor...")
            model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_path)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            print("Successfully loaded model with base feature extractor")
        except Exception as e2:
            print(f"Error with safetensors approach: {e2}")
            try:
                # Manual loading approach
                print("Trying manual model loading...")
                from transformers import Wav2Vec2Config
                
                # Load config
                if os.path.exists(f"{args.model_path}/config.json"):
                    config = Wav2Vec2Config.from_pretrained(args.model_path)
                else:
                    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
                    config.num_labels = 5  # Ensure 5 classes
                
                model = Wav2Vec2ForSequenceClassification(config)
                
                # Load weights manually
                if os.path.exists(f"{args.model_path}/model.safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(f"{args.model_path}/model.safetensors")
                    model.load_state_dict(state_dict)
                    print("Loaded weights from model.safetensors")
                elif os.path.exists(f"{args.model_path}/pytorch_model.bin"):
                    state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device)
                    model.load_state_dict(state_dict)
                    print("Loaded weights from pytorch_model.bin")
                elif os.path.exists(f"{args.model_path}/model.pth"):
                    # For backward compatibility
                    model = torch.load(f"{args.model_path}/model.pth", map_location=device)
                    print("Loaded model from model.pth")
                else:
                    raise Exception("No model weights found")
                
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
                print("Successfully loaded model manually with base feature extractor")
                
            except Exception as e3:
                raise Exception(f"Failed to load model: {e1}, {e2}, {e3}")
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load ground truth
    print("Loading ground truth...")
    ground_truth_df = pd.read_csv(args.ground_truth)
    print(f"Loaded {len(ground_truth_df)} ground truth entries")
    
    # Filter ground truth to only include classes that the 4-class model can predict
    valid_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย', 'none']
    original_count = len(ground_truth_df)
    ground_truth_df = ground_truth_df[ground_truth_df['label'].isin(valid_classes)].copy()
    filtered_count = len(ground_truth_df)
    
    if filtered_count < original_count:
        print(f"Filtered ground truth to 4-class model classes: {filtered_count} entries (removed {original_count - filtered_count} entries)")
        try:
            remaining_classes = ground_truth_df['label'].value_counts().to_dict()
            print("Remaining classes:", remaining_classes)
        except UnicodeEncodeError:
            print("Remaining classes: [Thai characters - encoding issue]")
    else:
        print("All ground truth entries match 4-class model classes")
    
    config = {
        'csv_path': csv_file_path,
        'window': window_name,
        'stride': stride_name
    }
    
    # Create output directory structure
    output_base_dir = Path(args.output_dir) / args.eval_type
    
    # Process the configuration
    results = process_single_configuration(
        model, feature_extractor, device, config, 
        ground_truth_df, output_base_dir
    )
    
    print("\n=== EVALUATION COMPLETED ===")
    print(f"Results saved to: {output_base_dir}")
    print(f"Window-level Binary F1: {results['window_evaluation']['binary']['f1']:.4f}")
    print(f"Word-level Binary F1: {results['word_evaluation']['binary']['f1']:.4f}")
    print(f"Combined Mean IoU: {results['combined_mean_iou']:.4f}")

if __name__ == "__main__":
    main()
