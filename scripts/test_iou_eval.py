#!/usr/bin/env python3
"""
Quick test to verify IoU evaluation functionality
"""

import pandas as pd
import numpy as np

def test_iou_basic():
    """Test basic functionality without full model loading"""
    print("Testing basic IoU evaluation...")
    
    # Create sample data
    ground_truth = pd.DataFrame({
        'file_path': ['test.wav', 'test.wav', 'test.wav'],
        'start_time': [0.0, 2.0, 4.0],
        'end_time': [1.0, 3.0, 5.0],
        'label': ['กู', 'none', 'เย็ด']
    })
    
    predictions = pd.DataFrame({
        'file_path': ['test.wav', 'test.wav'],
        'start_time': [0.1, 4.2],
        'end_time': [0.9, 4.8],
        'predicted_label': ['กู', 'เย็ด'],
        'max_confidence': [0.9, 0.8]
    })
    
    print("Ground truth:")
    print(ground_truth)
    print("\nPredictions:")
    print(predictions)
    
    # Test the IoU calculation function
    def calculate_iou(pred_start, pred_end, gt_start, gt_end):
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection_duration = intersection_end - intersection_start
        union_start = min(pred_start, gt_start)
        union_end = max(pred_end, gt_end)
        union_duration = union_end - union_start
        
        return intersection_duration / union_duration if union_duration > 0 else 0.0
    
    # Test IoU calculations
    iou1 = calculate_iou(0.1, 0.9, 0.0, 1.0)  # Should be high IoU
    iou2 = calculate_iou(4.2, 4.8, 4.0, 5.0)  # Should be good IoU
    
    print(f"\nIoU test 1 (prediction 0.1-0.9 vs GT 0.0-1.0): {iou1:.3f}")
    print(f"IoU test 2 (prediction 4.2-4.8 vs GT 4.0-5.0): {iou2:.3f}")
    
    # Basic classification test
    profanity_gt = ground_truth[ground_truth['label'] != 'none']
    print(f"\nProfanity GT count: {len(profanity_gt)}")
    print(f"All GT count: {len(ground_truth)}")
    print(f"Predictions count: {len(predictions)}")
    
    # Test threshold-based evaluation
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        tp_count = 0
        fp_count = 0
        fn_count = 0
        
        # Count TPs and FNs from ground truth perspective
        for _, gt_row in ground_truth.iterrows():
            if gt_row['label'] == 'none':
                # For none GT, check if any profanity predictions overlap
                has_overlap = False
                for _, pred_row in predictions.iterrows():
                    iou = calculate_iou(
                        pred_row['start_time'], pred_row['end_time'],
                        gt_row['start_time'], gt_row['end_time']
                    )
                    if iou > 0:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    tp_count += 1  # Correct none detection
                else:
                    fn_count += 1  # Missed none detection
            else:
                # For profanity GT, find best matching prediction
                best_iou = 0
                best_label = None
                for _, pred_row in predictions.iterrows():
                    iou = calculate_iou(
                        pred_row['start_time'], pred_row['end_time'],
                        gt_row['start_time'], gt_row['end_time']
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_label = pred_row['predicted_label']
                
                if best_iou >= threshold and best_label == gt_row['label']:
                    tp_count += 1
                else:
                    fn_count += 1
        
        # Count FPs - predictions that don't match ground truth well
        for _, pred_row in predictions.iterrows():
            matched = False
            for _, gt_row in ground_truth.iterrows():
                if gt_row['label'] != 'none':  # Only check against profanity GT
                    iou = calculate_iou(
                        pred_row['start_time'], pred_row['end_time'],
                        gt_row['start_time'], gt_row['end_time']
                    )
                    if iou >= threshold and pred_row['predicted_label'] == gt_row['label']:
                        matched = True
                        break
            
            if not matched:
                fp_count += 1
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nThreshold {threshold}:")
        print(f"  TP: {tp_count}, FP: {fp_count}, FN: {fn_count}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    test_iou_basic()
