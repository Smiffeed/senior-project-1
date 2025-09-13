#!/usr/bin/env python3
"""
Test IoU evaluation without heavy dependencies
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from collections import defaultdict
import os

def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    """Calculate IoU between prediction and ground truth segments"""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection_duration = intersection_end - intersection_start
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union_duration = union_end - union_start
    
    return intersection_duration / union_duration if union_duration > 0 else 0.0

def evaluate_iou_classification(merged_predictions, ground_truth_df, iou_thresholds=None):
    """
    Evaluate IoU-based classification including none detection
    This addresses the issue where IoU eval binary confusion matrix shows nothing
    """
    if iou_thresholds is None:
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("IoU Evaluation with Classification Metrics:")
    print("=" * 50)
    
    # Use only profanity predictions for IoU calculation
    profanity_predictions = merged_predictions[merged_predictions['predicted_label'] != 'none'].copy()
    
    # Use ALL ground truth (including none) for complete evaluation
    all_ground_truth = ground_truth_df.copy()
    profanity_ground_truth = ground_truth_df[ground_truth_df['label'] != 'none'].copy()
    
    print(f"Ground truth: {len(all_ground_truth)} total ({len(profanity_ground_truth)} profanity, {len(all_ground_truth) - len(profanity_ground_truth)} none)")
    print(f"Predictions: {len(merged_predictions)} total ({len(profanity_predictions)} profanity, {len(merged_predictions) - len(profanity_predictions)} none)")
    
    results = {}
    
    for threshold in iou_thresholds:
        print(f"\n--- IoU Threshold: {threshold} ---")
        
        # Create ground-truth centered evaluation
        y_true = []
        y_pred = []
        
        detailed_matches = []
        
        # Process each ground truth word
        for _, gt_row in all_ground_truth.iterrows():
            gt_label = gt_row['label']
            y_true.append(gt_label)
            
            if gt_label == 'none':
                # For none ground truth: check if any profanity predictions overlap
                file_preds = profanity_predictions[
                    profanity_predictions['file_path'] == gt_row['file_path']
                ]
                
                best_overlap = 0.0
                overlapping_pred = None
                
                for _, pred_row in file_preds.iterrows():
                    iou = calculate_iou(
                        pred_row['start_time'], pred_row['end_time'],
                        gt_row['start_time'], gt_row['end_time']
                    )
                    if iou > best_overlap:
                        best_overlap = iou
                        overlapping_pred = pred_row
                
                # For none: if no overlapping profanity predictions, it's correctly classified as none
                if overlapping_pred is None or best_overlap == 0:
                    y_pred.append('none')  # Correct: no profanity detected
                else:
                    y_pred.append(overlapping_pred['predicted_label'])  # Incorrect: profanity detected in none region
                    
            else:
                # For profanity ground truth: find best matching prediction
                file_preds = profanity_predictions[
                    profanity_predictions['file_path'] == gt_row['file_path']
                ]
                
                best_iou = 0.0
                best_pred = None
                
                for _, pred_row in file_preds.iterrows():
                    iou = calculate_iou(
                        pred_row['start_time'], pred_row['end_time'],
                        gt_row['start_time'], gt_row['end_time']
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_row
                
                # Classification based on IoU threshold
                if best_pred is not None and best_iou >= threshold:
                    y_pred.append(best_pred['predicted_label'])
                else:
                    y_pred.append('none')  # No sufficient overlap = missed detection
        
        # Calculate binary classification metrics (profanity vs none)
        y_true_binary = ['profanity' if label != 'none' else 'none' for label in y_true]
        y_pred_binary = ['profanity' if label != 'none' else 'none' for label in y_pred]
        
        # Binary confusion matrix
        binary_cm = confusion_matrix(y_true_binary, y_pred_binary, labels=['none', 'profanity'])
        print(f"Binary Confusion Matrix (none vs profanity):")
        print(f"                 Predicted")
        print(f"                 none  profanity")
        print(f"Actual none      {binary_cm[0,0]:4d}      {binary_cm[0,1]:4d}")
        print(f"Actual profanity {binary_cm[1,0]:4d}      {binary_cm[1,1]:4d}")
        
        # Multiclass confusion matrix (including none)
        all_labels = sorted(set(y_true + y_pred))
        multiclass_cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        print(f"\nMulticlass Confusion Matrix:")
        print("Labels:", all_labels)
        print(multiclass_cm)
        
        # Calculate metrics
        binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        binary_balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
        
        multiclass_accuracy = accuracy_score(y_true, y_pred)
        multiclass_balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for binary
        binary_metrics = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='weighted')
        
        print(f"\nBinary Classification Metrics:")
        print(f"  Accuracy: {binary_accuracy:.4f}")
        print(f"  Balanced Accuracy: {binary_balanced_acc:.4f}")
        print(f"  Precision: {binary_metrics[0]:.4f}")
        print(f"  Recall: {binary_metrics[1]:.4f}")
        print(f"  F1-Score: {binary_metrics[2]:.4f}")
        
        print(f"\nMulticlass Classification Metrics:")
        print(f"  Accuracy: {multiclass_accuracy:.4f}")
        print(f"  Balanced Accuracy: {multiclass_balanced_acc:.4f}")
        
        # Store results
        results[threshold] = {
            'binary_confusion_matrix': binary_cm,
            'multiclass_confusion_matrix': multiclass_cm,
            'binary_accuracy': binary_accuracy,
            'binary_balanced_accuracy': binary_balanced_acc,
            'multiclass_accuracy': multiclass_accuracy,
            'multiclass_balanced_accuracy': multiclass_balanced_acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_true_binary': y_true_binary,
            'y_pred_binary': y_pred_binary
        }
    
    return results

# Test with sample data
if __name__ == "__main__":
    print("Testing IoU evaluation with classification...")
    
    # Check if we have real data files
    csv_file = "../csv/eval_by_0.05/window_2.0s/stride_2.0s.csv"
    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}")
        
        # Load the data
        df = pd.read_csv(csv_file)
        print(f"Dataset size: {len(df)} windows")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        # Create sample predictions (for testing - replace with real predictions)
        sample_predictions = []
        for i, row in df.head(100).iterrows():  # Test with first 100 rows
            # Simulate some predictions
            pred_label = row['label']  # Perfect prediction for testing
            if np.random.random() < 0.1:  # 10% random errors
                labels = ['none', 'guu', 'kontol', 'anjing', 'bangsat']
                pred_label = np.random.choice([l for l in labels if l != row['label']])
            
            sample_predictions.append({
                'file_path': row['file_path'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'predicted_label': pred_label,
                'max_confidence': np.random.uniform(0.7, 0.99)
            })
        
        pred_df = pd.DataFrame(sample_predictions)
        gt_df = df.head(100)[['file_path', 'start_time', 'end_time', 'label']].copy()
        
        # Run IoU evaluation
        results = evaluate_iou_classification(pred_df, gt_df, iou_thresholds=[0.3, 0.5, 0.7])
        
        print("\n" + "="*50)
        print("IoU Evaluation completed!")
        print("This shows how IoU evaluation should include:")
        print("1. Binary confusion matrix (none vs profanity)")
        print("2. Multiclass confusion matrix (including none)")
        print("3. Classification metrics for different IoU thresholds")
        
    else:
        print(f"Data file not found: {csv_file}")
        print("Please provide the correct path to your evaluation data.")
