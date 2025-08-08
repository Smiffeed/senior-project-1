#!/usr/bin/env python3
"""
Evaluate Localization Accuracy

This script provides a quantitative measure of the profanity detection model's
localization performance. It uses the same robust two-stage detection method
as the censoring script but compares the final predicted timestamps against
ground-truth annotations from a test CSV.

The primary metric used is Intersection over Union (IoU), which leads to
standard classification metrics like Precision, Recall, and F1-Score, providing
a clear, objective measure of how well the model finds profane words.

Usage:
    python evaluate_localization_accuracy.py \
        --model-path "models/advanced_training/final_model" \
        --test-csv "csv/eval.csv" \
        --iou-threshold 0.5
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional

# --- Configuration ---
SAMPLE_RATE = 16000

def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProfanityDetector:
    """
    A two-stage profanity detection system, mirroring frame_level_censor.py.
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
        return model, feature_extractor, class_names

    def _classify_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """Classifies a single chunk of audio."""
        inputs = self.feature_extractor(
            audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True, truncation=True, max_length=len(audio_chunk)
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs).logits
            scores = torch.softmax(logits, dim=1)[0]
            prediction_idx = torch.argmax(scores).item()
            confidence = scores[prediction_idx].item()
        
        return self.class_names[prediction_idx], confidence

    def _coarse_detection(self, waveform: np.ndarray, window_size_s: float, step_size_s: float) -> List[Dict]:
        """Stage 1: Find potential profane regions using a sliding window."""
        window_samples = int(window_size_s * SAMPLE_RATE)
        step_samples = int(step_size_s * SAMPLE_RATE)
        
        detected_windows = []
        for i in range(0, len(waveform) - window_samples, step_samples):
            window = waveform[i : i + window_samples]
            label, _ = self._classify_audio_chunk(window)
            if label != 'none':
                detected_windows.append({"start_time": i / SAMPLE_RATE, "end_time": (i + window_samples) / SAMPLE_RATE})
        
        return self._merge_overlapping_detections(detected_windows)

    def _merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merges overlapping time intervals into continuous regions."""
        if not detections: return []
        detections.sort(key=lambda x: x['start_time'])
        merged = []
        current_detection = detections[0]
        for next_detection in detections[1:]:
            if next_detection['start_time'] < current_detection['end_time']:
                current_detection['end_time'] = max(current_detection['end_time'], next_detection['end_time'])
            else:
                merged.append(current_detection)
                current_detection = next_detection
        merged.append(current_detection)
        return merged

    def _refine_boundaries(self, waveform: np.ndarray, coarse_region: Dict, confidence_threshold: float) -> Optional[Dict]:
        """Stage 2: Refine boundaries using energy-based VAD and re-classification."""
        start_sample = int(coarse_region['start_time'] * SAMPLE_RATE)
        end_sample = int(coarse_region['end_time'] * SAMPLE_RATE)
        region_audio = waveform[start_sample:end_sample]
        
        clips = librosa.effects.split(region_audio, top_db=25, frame_length=512, hop_length=128)
        
        confirmed_clips = []
        if clips.size > 0:
            for clip_start, clip_end in clips:
                if clip_end - clip_start < (SAMPLE_RATE * 0.1): continue
                clip_audio = region_audio[clip_start:clip_end]
                label, confidence = self._classify_audio_chunk(clip_audio)
                if label != 'none' and confidence >= confidence_threshold:
                    confirmed_clips.append({
                        "start_time": coarse_region['start_time'] + (clip_start / SAMPLE_RATE),
                        "end_time": coarse_region['start_time'] + (clip_end / SAMPLE_RATE),
                        "label": label, "confidence": confidence
                    })
        if not confirmed_clips: return None
        
        final_detection = confirmed_clips[0]
        for clip in confirmed_clips[1:]:
            final_detection['end_time'] = max(final_detection['end_time'], clip['end_time'])
            final_detection['confidence'] = max(final_detection['confidence'], clip['confidence'])
        return final_detection

    def detect(self, waveform: np.ndarray, window_size: float, step_size: float, confidence: float) -> List[Dict]:
        """Runs the full two-stage detection process."""
        coarse_detections = self._coarse_detection(waveform, window_size, step_size)
        if not coarse_detections: return []
        
        final_detections = []
        for region in coarse_detections:
            refined = self._refine_boundaries(waveform, region, confidence)
            if refined:
                final_detections.append(refined)
        return final_detections

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
    parser = argparse.ArgumentParser(description="Evaluate model localization accuracy.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--test-csv", required=True, help="Path to the test CSV with ground-truth annotations.")
    parser.add_argument("--window-size", type=float, default=0.5, help="Sliding window size for coarse detection.")
    parser.add_argument("--step-size", type=float, default=0.25, help="Sliding window step size.")
    parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for detection.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for a detection to be a True Positive.")
    
    args = parser.parse_args()

    try:
        detector = ProfanityDetector(args.model_path)
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

            # Get model predictions for this file
            predictions = detector.detect(waveform_np, args.window_size, args.step_size, args.confidence)
            all_predictions.extend(predictions)
            
            # Get ground truths for this file
            ground_truths = group[['start_time', 'end_time', 'label']].to_dict('records')
            # Filter out 'none' labels from ground truth for evaluation
            ground_truths = [gt for gt in ground_truths if gt['label'] != 'none']
            all_ground_truths.extend(ground_truths)

            print(f"  -> Found {len(ground_truths)} ground truth(s) and {len(predictions)} prediction(s).")

        # --- Final Evaluation ---
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
