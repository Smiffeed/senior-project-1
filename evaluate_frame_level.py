#!/usr/bin/env python3
"""
Evaluate Frame-Level Localization Accuracy (Research Method)

This script implements the frame-level evaluation methodology described in the
research diagram. It provides a more direct and potentially more accurate way
to measure localization performance compared to the two-stage windowing approach.

The process is as follows:
1.  The entire audio file is passed through the Wav2Vec2 encoder to get a
    sequence of frame-level feature representations (ca. 25ms each).
2.  A classification head is applied to *each frame* to get a sequence of
    profanity/non-profanity predictions.
3.  The sequence of frame predictions is compared against the ground-truth
    annotations to calculate metrics.

Usage:
    python evaluate_frame_level.py \
        --model-path "models/advanced_training/final_model" \
        --test-csv "csv/eval.csv" \
        --confidence-threshold 0.9
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple

# --- Configuration ---
SAMPLE_RATE = 16000
# The Wav2Vec2 model outputs a feature vector roughly every 25ms.
# This is a fundamental property of the model architecture.
FRAME_DURATION_S = 0.025 

def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrameLevelEvaluator:
    """
    Implements the research method for frame-level evaluation.
    """
    def __init__(self, model_path: str):
        self.device = get_device()
        self.model, self.feature_extractor, self.class_names = self._load_model(model_path)

    def _load_model(self, model_path: str) -> Tuple[Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, List[str]]:
        """Loads the fine-tuned model, its feature extractor, and class names."""
        print(f"Loading model from: {model_path}")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
        
        model.eval()
        print(f"Model loaded successfully on device: {self.device}")
        return model, feature_extractor, class_names

    def get_frame_level_predictions(self, waveform: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """
        Generates a list of profane detections based on frame-by-frame classification.
        """
        # 1. Get frame-level features from the Wav2Vec2 encoder
        inputs = self.feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden_states = self.model.wav2vec2(inputs.input_values).last_hidden_state
        
        # Squeeze the batch dimension, leaving (num_frames, num_features)
        frame_features = hidden_states.squeeze(0)
        
        # 2. Apply the "Frame Classification Head" one frame at a time
        # This is more robust than vectorization and avoids shape conflicts with custom heads.
        all_probs = []
        # Process in batches to avoid CUDA memory issues with very long files
        batch_size = 64 
        for i in range(0, frame_features.shape[0], batch_size):
            batch = frame_features[i:i+batch_size]
            with torch.no_grad():
                # The classifier expects a pooled output of shape (batch, features).
                # A batch of frames has the correct shape.
                logits = self.model.classifier(batch)
            
            # Append probabilities for this batch to our list
            all_probs.append(torch.softmax(logits, dim=-1))

        # Concatenate probabilities from all batches
        probs = torch.cat(all_probs, dim=0) # Shape: [num_frames, num_classes]
        predictions = torch.argmax(probs, dim=1) # Shape: [num_frames]
        
        # 3. Convert frame predictions into timed detections
        detections = []
        in_detection = False
        start_time = 0

        for i, frame_pred_idx in enumerate(predictions):
            frame_label = self.class_names[frame_pred_idx]
            frame_confidence = probs[i, frame_pred_idx].item()
            
            is_profane = (frame_label != 'none' and frame_confidence >= confidence_threshold)
            
            current_time = i * FRAME_DURATION_S

            if is_profane and not in_detection:
                # Start of a new detection
                in_detection = True
                start_time = current_time
            elif not is_profane and in_detection:
                # End of a detection
                in_detection = False
                detections.append({
                    "start_time": start_time,
                    "end_time": current_time, # The end is the start of the first non-profane frame
                    "label": "profanity" # Generic label for now
                })

        # If the audio ends while in a detection
        if in_detection:
            detections.append({
                "start_time": start_time,
                "end_time": len(predictions) * FRAME_DURATION_S,
                "label": "profanity"
            })
            
        return detections

def calculate_iou(box_a: Dict, box_b: Dict) -> float:
    """Calculates Intersection over Union for two time intervals."""
    start_a, end_a = box_a['start_time'], box_a['end_time']
    start_b, end_b = box_b['start_time'], box_b['end_time']

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)

    intersection = max(0, inter_end - inter_start)
    union = (end_a - start_a) + (end_b - start_b) - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_localization(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float) -> Dict:
    """Calculates Precision, Recall, and F1-Score for localization."""
    if not ground_truths and not predictions:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    if not ground_truths:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'true_positives': 0, 'false_positives': len(predictions), 'false_negatives': 0}
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': len(ground_truths)}

    true_positives = 0
    false_positives = 0
    matched_gt_indices = set()

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices:
            true_positives += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            false_positives += 1
            
    false_negatives = len(ground_truths) - len(matched_gt_indices)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model localization with the frame-level research method.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--test-csv", required=True, help="Path to the test CSV with ground-truth annotations.")
    parser.add_argument("--confidence-threshold", type=float, default=0.9, help="Minimum confidence for a frame to be considered profane.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for a detection to be a True Positive.")
    
    args = parser.parse_args()

    try:
        evaluator = FrameLevelEvaluator(args.model_path)
        test_df = pd.read_csv(args.test_csv)
        
        all_predictions = []
        all_ground_truths = []

        for file_path, group in test_df.groupby('file_path'):
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            
            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
                waveform_np = waveform.squeeze().numpy()
            except Exception as e:
                print(f"  -> Error loading audio file: {e}. Skipping.")
                continue

            predictions = evaluator.get_frame_level_predictions(waveform_np, args.confidence_threshold)
            all_predictions.extend(predictions)
            
            ground_truths = group[['start_time', 'end_time', 'label']].to_dict('records')
            ground_truths = [gt for gt in ground_truths if gt['label'] != 'none']
            all_ground_truths.extend(ground_truths)

            print(f"  -> Found {len(ground_truths)} ground truth(s) and {len(predictions)} prediction(s).")

        print("\n" + "="*60)
        print(f"PERFORMANCE REPORT (IoU Threshold: {args.iou_threshold})")
        print("="*60)
        
        results = evaluate_localization(all_predictions, all_ground_truths, args.iou_threshold)
        
        print(f"Overall Precision: {results['precision']:.4f}")
        print(f"Overall Recall:    {results['recall']:.4f}")
        print(f"Overall F1-Score:  {results['f1_score']:.4f}")
        print("-"*60)
        print(f"Total True Positives:  {results['true_positives']}")
        print(f"Total False Positives: {results['false_positives']}")
        print(f"Total False Negatives: {results['false_negatives']}")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
