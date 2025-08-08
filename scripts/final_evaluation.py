import os
import argparse
import torch
import pandas as pd
import numpy as np
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import classification_report
from tqdm import tqdm
from pathlib import Path
import sys
from typing import Tuple, List, Dict

# --- Configuration ---
# This configuration is taken from your working 'evaluate_windowed.py' script
# to ensure it matches your trained model.
CLASS_NAMES = ['none', 'เย็ด', 'กู', 'มึง', 'เหี้ย', 'ควย', 'สวะ', 'หี', 'แตด']
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_LABELS = len(CLASS_NAMES)


# --- Helper Functions & Classes ---

def preprocess_window(audio_np: np.ndarray) -> np.ndarray:
    """
    Applies the same advanced preprocessing to an audio window as the training script.
    This is the key function adapted from evaluate_windowed.py.
    """
    if audio_np.size == 0:
        return audio_np

    # 1. Apply Hamming window
    audio_np = audio_np * np.hamming(len(audio_np))
    
    # 2. Apply pre-emphasis
    audio_np = librosa.effects.preemphasis(audio_np)

    # 3. Normalize
    if np.std(audio_np) > 1e-8:
        audio_np = (audio_np - np.mean(audio_np)) / (np.std(audio_np) + 1e-8)
    else:
        audio_np = np.zeros_like(audio_np)

    return audio_np


def merge_overlapping_regions(regions: List[Dict]) -> List[Dict]:
    """Merges overlapping or adjacent regions with the same label."""
    if not regions:
        return []
    regions.sort(key=lambda r: r['start'])
    merged = []
    current_merge = regions[0]
    for next_region in regions[1:]:
        if next_region['start'] < current_merge['end'] and next_region['label'] == current_merge['label']:
            current_merge['end'] = max(current_merge['end'], next_region['end'])
        else:
            merged.append(current_merge)
            current_merge = next_region
    merged.append(current_merge)
    return merged


class ProfanityDetector:
    """
    A detector class that uses advanced preprocessing before prediction.
    """
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = model.device
        self.sr = feature_extractor.sampling_rate

    def predict(self, audio_np: np.ndarray) -> Tuple[int, float]:
        """Applies preprocessing and then predicts."""
        if audio_np.size == 0:
            return LABEL_MAP['none'], 0.0
        
        processed_audio = preprocess_window(audio_np)

        inputs = self.feature_extractor(processed_audio, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
            scores = torch.nn.functional.softmax(logits, dim=-1)
            pred_class_id = torch.argmax(scores, dim=-1).item()
            confidence = scores[0, pred_class_id].item()
        return pred_class_id, confidence

    def find_profanity(self, file_path: str, coarse_threshold: float, fine_threshold: float) -> Tuple[List[Dict], float]:
        """Main detection logic, returning detections and max observed confidence."""
        max_observed_confidence = 0.0
        try:
            audio_np, _ = librosa.load(file_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Warning: Could not load {file_path}. Skipping. Error: {e}")
            return [], max_observed_confidence

        window_size_samples = int(0.5 * self.sr)
        hop_size_samples = int(0.25 * self.sr)
        coarse_regions = []
        for i in range(0, len(audio_np) - window_size_samples, hop_size_samples):
            window = audio_np[i:i + window_size_samples]
            class_id, confidence = self.predict(window)
            
            if class_id != LABEL_MAP['none']:
                max_observed_confidence = max(max_observed_confidence, confidence)
                if confidence > coarse_threshold:
                    start_time = i / self.sr
                    end_time = (i + window_size_samples) / self.sr
                    coarse_regions.append({'start': start_time, 'end': end_time, 'label': ID2LABEL[class_id]})
        
        if not coarse_regions:
            return [], max_observed_confidence

        merged_regions = merge_overlapping_regions(coarse_regions)

        refined_detections = []
        for region in merged_regions:
            region_audio = audio_np[int(region['start'] * self.sr):int(region['end'] * self.sr)]
            clips = librosa.effects.split(region_audio, top_db=25, frame_length=2048, hop_length=512)
            
            for clip_start, clip_end in clips:
                speech_segment = region_audio[clip_start:clip_end]
                if len(speech_segment) / self.sr < 0.1:
                    continue
                
                class_id, confidence = self.predict(speech_segment)
                if class_id != LABEL_MAP['none'] and confidence > fine_threshold:
                    refined_start = region['start'] + (clip_start / self.sr)
                    refined_end = region['start'] + (clip_end / self.sr)
                    refined_detections.append({
                        'start': refined_start,
                        'end': refined_end,
                        'label': ID2LABEL[class_id]
                    })
        return refined_detections, max_observed_confidence


def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union for 1D segments."""
    start_a, end_a = box_a
    start_b, end_b = box_b
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0, inter_end - inter_start)
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return intersection / union if union > 0 else 0


def evaluate_localization(ground_truth_df, predictions_list, iou_threshold=0.5):
    """Evaluates localization performance using IoU."""
    predictions_df = pd.DataFrame(predictions_list)
    tp, fp, fn = 0, 0, 0
    true_labels_matched, pred_labels_matched = [], []
    matched_preds_indices = set()

    for gt_idx, gt_row in ground_truth_df.iterrows():
        gt_segment = (gt_row['start_time'], gt_row['end_time'])
        gt_label = gt_row['label']
        best_iou = 0
        best_match_pred_idx = -1

        if not predictions_df.empty:
            for pred_idx, pred_row in predictions_df.iterrows():
                if pred_idx in matched_preds_indices:
                    continue
                pred_segment = (pred_row['start'], pred_row['end'])
                iou = calculate_iou(gt_segment, pred_segment)
                if iou > best_iou:
                    best_iou = iou
                    best_match_pred_idx = pred_idx

        if best_match_pred_idx != -1 and best_iou >= iou_threshold and predictions_df.loc[best_match_pred_idx, 'label'] == gt_label:
            tp += 1
            matched_preds_indices.add(best_match_pred_idx)
            true_labels_matched.append(gt_label)
            pred_labels_matched.append(predictions_df.loc[best_match_pred_idx, 'label'])
        else:
            fn += 1

    fp = len(predictions_df) - len(matched_preds_indices)
    return tp, fp, fn, true_labels_matched, pred_labels_matched


def main(args):
    print("--- Final Profanity Localization Evaluation ---")
    
    print(f"Loading model and feature extractor from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_path).to(device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model from {args.model_path}. Please ensure it's a valid Hugging Face model directory.")
        print(f"Details: {e}")
        sys.exit(1)
        
    model.eval()
    detector = ProfanityDetector(model, feature_extractor)
    
    print(f"Loading ground truth from {args.ground_truth_csv}...")
    if not Path(args.ground_truth_csv).exists():
        print(f"Error: Ground truth file not found at {args.ground_truth_csv}")
        sys.exit(1)
    ground_truth_df = pd.read_csv(args.ground_truth_csv)
    # IMPORTANT: Filter out labels from the CSV that the model wasn't trained on.
    ground_truth_df = ground_truth_df[ground_truth_df['label'].isin(CLASS_NAMES)]
    ground_truth_df = ground_truth_df[ground_truth_df['label'] != 'none'].copy()

    all_true_labels, all_pred_labels = [], []
    total_tp, total_fp, total_fn = 0, 0, 0
    max_overall_confidence = 0.0
    
    audio_files = ground_truth_df['file_path'].unique()
    print(f"Running evaluation on {len(audio_files)} unique audio files...")
    
    for file_path_str in tqdm(audio_files, desc="Evaluating Files"):
        # The CSV file_path is relative, so we join it with the audio_dir
        full_path = Path(args.audio_dir).parent / Path(file_path_str)
        if not full_path.exists():
            print(f"Warning: Audio file not found at {full_path}. Skipping.")
            continue
        
        gt_file_df = ground_truth_df[ground_truth_df['file_path'] == file_path_str]
        predictions_list, max_conf = detector.find_profanity(str(full_path), args.coarse_threshold, args.fine_threshold)
        max_overall_confidence = max(max_overall_confidence, max_conf)
        
        tp, fp, fn, true_matched, pred_matched = evaluate_localization(
            gt_file_df, predictions_list, args.iou_threshold
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_true_labels.extend(true_matched)
        all_pred_labels.extend(pred_matched)

    print("\n--- Overall Evaluation Results ---")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nLocalization Performance (IoU Threshold = {args.iou_threshold}):")
    print(f"  - True Positives:  {total_tp}")
    print(f"  - False Positives: {total_fp}")
    print(f"  - False Negatives: {total_fn}")
    print("------------------------------------")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")

    if all_true_labels:
        print("\nClassification Report for Correctly Localized Detections (TPs only):")
        labels = sorted(list(set(all_true_labels) | set(all_pred_labels)))
        print(classification_report(all_true_labels, all_pred_labels, labels=labels, zero_division=0))
    else:
        print("\nNo detections were correctly localized, so no classification report is available.")

    if total_tp == 0 and total_fp == 0 and total_fn > 0:
        print("\n--- DIAGNOSTIC INFORMATION ---")
        print("No profanity was detected. This could be because the confidence thresholds are still too high.")
        print(f"The highest confidence score for any non-'none' word detected across all files was: {max_overall_confidence:.4f}")
        print("\nTry re-running the evaluation with lower thresholds. For example:")
        suggested_coarse = max(0.1, max_overall_confidence - 0.1)
        suggested_fine = max(0.15, max_overall_confidence - 0.05)
        print(f"  --coarse-threshold {suggested_coarse:.2f} --fine-threshold {suggested_fine:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final evaluation script for localization with advanced preprocessing.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained Hugging Face model DIRECTORY (e.g., ./models/CW_ham).")
    parser.add_argument("--ground-truth-csv", type=str, required=True, help="Path to the ground truth CSV with precise timestamps (e.g., ./csv/test.csv).")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing the audio files referenced in the CSV (e.g., ./dataset/test).")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold to consider a detection a True Positive.")
    parser.add_argument("--coarse-threshold", type=float, default=0.5, help="Confidence threshold for the initial sliding window detection.")
    parser.add_argument("--fine-threshold", type=float, default=0.6, help="Confidence threshold for the refined VAD-based detection.")
    
    args = parser.parse_args()
    main(args)
