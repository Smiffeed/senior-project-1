#!/usr/bin/env python3
"""
Advanced evaluation and monitoring for model improvement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from collections import defaultdict

class AdvancedModelEvaluator:
    """
    Comprehensive evaluation system based on your evaluation results
    """
    
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        
        # Optimal configurations from your evaluation results
        self.optimal_configs = {
            'binary_task': {'window': 2.0, 'stride': 1.9, 'expected_f1': 0.85},
            'multiclass_task': {'window': 0.3, 'stride': 0.15, 'expected_f1': 0.75},
            'iou_task': {'window': 0.5, 'stride': 0.25, 'expected_iou': 0.6}
        }
    
    def comprehensive_evaluation(self, test_data, evaluation_types=['window_eval', 'word_eval', 'iou_eval']):
        """
        Run comprehensive evaluation matching your evaluation framework
        """
        results = {}
        
        for eval_type in evaluation_types:
            print(f"\nRunning {eval_type} evaluation...")
            
            if eval_type == 'window_eval':
                results[eval_type] = self.window_level_evaluation(test_data)
            elif eval_type == 'word_eval':
                results[eval_type] = self.word_level_evaluation(test_data)
            elif eval_type == 'iou_eval':
                results[eval_type] = self.iou_evaluation(test_data)
            elif eval_type == 'word_iou_eval':
                results[eval_type] = self.word_iou_evaluation(test_data)
        
        # Generate comparison with optimal results
        performance_comparison = self.compare_with_optimal(results)
        
        return results, performance_comparison
    
    def window_level_evaluation(self, test_data):
        """Window-level evaluation (direct prediction evaluation)"""
        predictions = []
        true_labels = []
        confidence_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for example in test_data:
                # Get model prediction
                inputs = {
                    'input_values': torch.tensor(example['audio']).unsqueeze(0),
                }
                
                outputs = self.model(**inputs)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probs).item()
                
                predictions.append(predicted_class)
                true_labels.append(example['label'])
                confidence_scores.append(confidence)
        
        # Calculate metrics
        binary_metrics = self.calculate_binary_metrics(true_labels, predictions)
        multiclass_metrics = self.calculate_multiclass_metrics(true_labels, predictions)
        
        return {
            'binary_metrics': binary_metrics,
            'multiclass_metrics': multiclass_metrics,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidence_scores': confidence_scores,
            'confusion_matrix': confusion_matrix(true_labels, predictions)
        }
    
    def word_level_evaluation(self, test_data):
        """Word-level evaluation with prediction merging"""
        # Group predictions by original audio file and merge overlapping predictions
        file_predictions = defaultdict(list)
        
        # First, get all predictions
        for example in test_data:
            prediction = self.predict_single_window(example['audio'])
            
            file_predictions[example.get('original_file', 'unknown')].append({
                'start_time': example.get('start_time', 0),
                'end_time': example.get('end_time', 1),
                'prediction': prediction['class'],
                'confidence': prediction['confidence'],
                'true_label': example['label']
            })
        
        # Merge overlapping predictions for each file
        merged_results = []
        for file_path, preds in file_predictions.items():
            merged_preds = self.merge_overlapping_predictions(preds)
            merged_results.extend(merged_preds)
        
        # Calculate word-level metrics
        word_true_labels = [r['true_label'] for r in merged_results]
        word_predictions = [r['merged_prediction'] for r in merged_results]
        
        binary_metrics = self.calculate_binary_metrics(word_true_labels, word_predictions)
        multiclass_metrics = self.calculate_multiclass_metrics(word_true_labels, word_predictions)
        
        return {
            'binary_metrics': binary_metrics,
            'multiclass_metrics': multiclass_metrics,
            'merged_predictions': merged_results,
            'word_true_labels': word_true_labels,
            'word_predictions': word_predictions
        }
    
    def iou_evaluation(self, test_data, iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """IoU-based evaluation matching your framework"""
        results = {}
        
        # Get temporal predictions
        temporal_predictions = self.get_temporal_predictions(test_data)
        
        for threshold in iou_thresholds:
            threshold_results = self.evaluate_iou_threshold(
                temporal_predictions, threshold
            )
            results[f'iou_{threshold}'] = threshold_results
        
        # Calculate mean IoU
        mean_iou = np.mean([r['mean_iou'] for r in results.values()])
        
        return {
            'threshold_results': results,
            'mean_iou': mean_iou,
            'temporal_predictions': temporal_predictions
        }
    
    def predict_single_window(self, audio):
        """Predict single audio window"""
        self.model.eval()
        with torch.no_grad():
            inputs = {'input_values': torch.tensor(audio).unsqueeze(0)}
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = torch.max(probs).item()
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probs.numpy()[0]
            }
    
    def merge_overlapping_predictions(self, predictions, overlap_threshold=0.1):
        """Merge overlapping predictions of the same class"""
        if not predictions:
            return []
        
        # Sort by start time
        sorted_preds = sorted(predictions, key=lambda x: x['start_time'])
        merged = []
        
        current_group = [sorted_preds[0]]
        
        for pred in sorted_preds[1:]:
            last_pred = current_group[-1]
            
            # Check if overlapping and same class
            overlap = (pred['start_time'] < last_pred['end_time'] + overlap_threshold and
                      pred['prediction'] == last_pred['prediction'])
            
            if overlap:
                current_group.append(pred)
            else:
                # Merge current group
                merged_pred = self.merge_prediction_group(current_group)
                merged.append(merged_pred)
                current_group = [pred]
        
        # Don't forget the last group
        if current_group:
            merged_pred = self.merge_prediction_group(current_group)
            merged.append(merged_pred)
        
        return merged
    
    def merge_prediction_group(self, group):
        """Merge a group of overlapping predictions"""
        start_time = min(p['start_time'] for p in group)
        end_time = max(p['end_time'] for p in group)
        
        # Use majority vote or highest confidence
        predictions = [p['prediction'] for p in group]
        confidences = [p['confidence'] for p in group]
        
        # Weighted vote by confidence
        class_votes = defaultdict(float)
        for pred, conf in zip(predictions, confidences):
            class_votes[pred] += conf
        
        merged_prediction = max(class_votes, key=class_votes.get)
        merged_confidence = max(confidences)
        
        # Use true label from any prediction (should be same for same temporal region)
        true_label = group[0]['true_label']
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'merged_prediction': merged_prediction,
            'merged_confidence': merged_confidence,
            'true_label': true_label,
            'num_merged': len(group)
        }
    
    def get_temporal_predictions(self, test_data):
        """Get predictions with temporal information for IoU evaluation"""
        temporal_preds = []
        
        for example in test_data:
            prediction = self.predict_single_window(example['audio'])
            
            temporal_preds.append({
                'start_time': example.get('start_time', 0),
                'end_time': example.get('end_time', 1),
                'prediction': prediction['class'],
                'confidence': prediction['confidence'],
                'true_label': example['label'],
                'true_start': example.get('true_start', 0),
                'true_end': example.get('true_end', 1)
            })
        
        return temporal_preds
    
    def evaluate_iou_threshold(self, temporal_predictions, threshold):
        """Evaluate at specific IoU threshold"""
        tp = fp = fn = 0
        ious = []
        
        for pred in temporal_predictions:
            # Calculate IoU
            iou = self.calculate_temporal_iou(
                pred['start_time'], pred['end_time'],
                pred['true_start'], pred['true_end']
            )
            ious.append(iou)
            
            # Classification based on IoU threshold
            if iou >= threshold:
                if pred['prediction'] == pred['true_label']:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'mean_iou': np.mean(ious) if ious else 0
        }
    
    def calculate_temporal_iou(self, pred_start, pred_end, true_start, true_end):
        """Calculate IoU between predicted and true temporal segments"""
        intersection_start = max(pred_start, true_start)
        intersection_end = min(pred_end, true_end)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start
        union = (pred_end - pred_start) + (true_end - true_start) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_binary_metrics(self, true_labels, predictions):
        """Calculate binary classification metrics"""
        # Convert to binary (profane vs non-profane)
        true_binary = [0 if label == 0 else 1 for label in true_labels]
        pred_binary = [0 if pred == 0 else 1 for pred in predictions]
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average='binary', zero_division='0'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def calculate_multiclass_metrics(self, true_labels, predictions):
        """Calculate multiclass metrics (profanity classes only)"""
        # Filter to only profanity classes
        profanity_indices = [i for i, label in enumerate(true_labels) if label != 0]
        
        if not profanity_indices:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        profanity_true = [true_labels[i] for i in profanity_indices]
        profanity_pred = [predictions[i] for i in profanity_indices]
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(profanity_true, profanity_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            profanity_true, profanity_pred, average='weighted', zero_division='0'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compare_with_optimal(self, current_results):
        """Compare current results with optimal configurations"""
        comparison = {}
        
        # Binary task comparison
        if 'window_eval' in current_results:
            current_binary_f1 = current_results['window_eval']['binary_metrics']['f1']
            optimal_binary_f1 = self.optimal_configs['binary_task']['expected_f1']
            
            comparison['binary_task'] = {
                'current_f1': current_binary_f1,
                'optimal_f1': optimal_binary_f1,
                'performance_ratio': current_binary_f1 / optimal_binary_f1,
                'improvement_needed': max(0, optimal_binary_f1 - current_binary_f1),
                'status': 'good' if current_binary_f1 >= optimal_binary_f1 * 0.9 else 'needs_improvement'
            }
        
        # Multiclass task comparison
        if 'word_eval' in current_results:
            current_multiclass_f1 = current_results['word_eval']['multiclass_metrics']['f1']
            optimal_multiclass_f1 = self.optimal_configs['multiclass_task']['expected_f1']
            
            comparison['multiclass_task'] = {
                'current_f1': current_multiclass_f1,
                'optimal_f1': optimal_multiclass_f1,
                'performance_ratio': current_multiclass_f1 / optimal_multiclass_f1,
                'improvement_needed': max(0, optimal_multiclass_f1 - current_multiclass_f1),
                'status': 'good' if current_multiclass_f1 >= optimal_multiclass_f1 * 0.9 else 'needs_improvement'
            }
        
        # IoU task comparison
        if 'iou_eval' in current_results:
            current_iou = current_results['iou_eval']['mean_iou']
            optimal_iou = self.optimal_configs['iou_task']['expected_iou']
            
            comparison['iou_task'] = {
                'current_iou': current_iou,
                'optimal_iou': optimal_iou,
                'performance_ratio': current_iou / optimal_iou,
                'improvement_needed': max(0, optimal_iou - current_iou),
                'status': 'good' if current_iou >= optimal_iou * 0.9 else 'needs_improvement'
            }
        
        return comparison
    
    def generate_improvement_recommendations(self, performance_comparison):
        """Generate specific recommendations based on performance gaps"""
        recommendations = []
        
        for task, metrics in performance_comparison.items():
            if metrics['status'] == 'needs_improvement':
                if task == 'binary_task':
                    recommendations.extend([
                        f"Binary classification F1 is {metrics['current_f1']:.3f}, target {metrics['optimal_f1']:.3f}",
                        "- Consider using larger window size (2.0s) for binary classification",
                        "- Increase stride overlap (1.9s stride for 2.0s windows)",
                        "- Apply stronger data augmentation for 'none' class balance",
                        "- Use focal loss with higher gamma for hard negative mining"
                    ])
                
                elif task == 'multiclass_task':
                    recommendations.extend([
                        f"Multiclass F1 is {metrics['current_f1']:.3f}, target {metrics['optimal_f1']:.3f}",
                        "- Use smaller window size (0.3s) for multiclass precision",
                        "- Reduce stride to 0.15s for better temporal coverage",
                        "- Implement class-specific data augmentation",
                        "- Consider temporal attention mechanism",
                        "- Add context-aware loss function"
                    ])
                
                elif task == 'iou_task':
                    recommendations.extend([
                        f"IoU performance is {metrics['current_iou']:.3f}, target {metrics['optimal_iou']:.3f}",
                        "- Improve temporal boundary prediction accuracy",
                        "- Use adaptive windowing based on audio content",
                        "- Implement post-processing to merge overlapping predictions",
                        "- Add temporal consistency regularization"
                    ])
        
        return recommendations

def create_monitoring_dashboard(evaluator, results):
    """Create monitoring dashboard for model performance"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Binary Classification Performance
    if 'window_eval' in results:
        binary_metrics = results['window_eval']['binary_metrics']
        metrics_names = list(binary_metrics.keys())
        metrics_values = list(binary_metrics.values())
        
        axes[0, 0].bar(metrics_names, metrics_values)
        axes[0, 0].set_title('Binary Classification Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # Add optimal line
        axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Target F1')
        axes[0, 0].legend()
    
    # Multiclass Classification Performance
    if 'word_eval' in results:
        multiclass_metrics = results['word_eval']['multiclass_metrics']
        metrics_names = list(multiclass_metrics.keys())
        metrics_values = list(multiclass_metrics.values())
        
        axes[0, 1].bar(metrics_names, metrics_values)
        axes[0, 1].set_title('Multiclass Classification Metrics')
        axes[0, 1].set_ylim(0, 1)
        
        # Add optimal line
        axes[0, 1].axhline(y=0.75, color='r', linestyle='--', label='Target F1')
        axes[0, 1].legend()
    
    # IoU Performance
    if 'iou_eval' in results:
        iou_thresholds = []
        iou_f1_scores = []
        
        for threshold_key, threshold_result in results['iou_eval']['threshold_results'].items():
            threshold = threshold_result['threshold']
            f1 = threshold_result['f1']
            iou_thresholds.append(threshold)
            iou_f1_scores.append(f1)
        
        axes[0, 2].plot(iou_thresholds, iou_f1_scores, 'bo-')
        axes[0, 2].set_title('IoU Threshold vs F1 Score')
        axes[0, 2].set_xlabel('IoU Threshold')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].grid(True)
    
    # Confusion Matrix
    if 'window_eval' in results:
        cm = results['window_eval']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('True')
    
    # Confidence Distribution
    if 'window_eval' in results:
        confidences = results['window_eval']['confidence_scores']
        axes[1, 1].hist(confidences, bins=20, alpha=0.7)
        axes[1, 1].set_title('Prediction Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=np.mean(confidences), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(confidences):.3f}')
        axes[1, 1].legend()
    
    # Performance Comparison
    axes[1, 2].text(0.1, 0.9, "Performance Summary:", fontsize=12, fontweight='bold')
    y_pos = 0.8
    
    if 'window_eval' in results:
        binary_f1 = results['window_eval']['binary_metrics']['f1']
        axes[1, 2].text(0.1, y_pos, f"Binary F1: {binary_f1:.3f} (Target: 0.85)", fontsize=10)
        y_pos -= 0.1
    
    if 'word_eval' in results:
        multiclass_f1 = results['word_eval']['multiclass_metrics']['f1']
        axes[1, 2].text(0.1, y_pos, f"Multiclass F1: {multiclass_f1:.3f} (Target: 0.75)", fontsize=10)
        y_pos -= 0.1
    
    if 'iou_eval' in results:
        mean_iou = results['iou_eval']['mean_iou']
        axes[1, 2].text(0.1, y_pos, f"Mean IoU: {mean_iou:.3f} (Target: 0.60)", fontsize=10)
    
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig