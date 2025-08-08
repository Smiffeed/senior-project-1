#!/usr/bin/env python3
"""
Detect and Censor: A Practical Profanity Detection Tool

This script provides a robust, two-stage method to find the exact locations
of profane words in an audio file, mirroring the effective logic from
frame_level_censor.py.

Stage 1: Coarse Detection
    A sliding window identifies potential regions of profanity.

Stage 2: Fine-Grained Refinement
    For each potential region, an energy-based Voice Activity Detection (VAD)
    isolates the actual speech, which is then re-classified to confirm
    profanity and find precise start/end times.

This tool is designed to be a reliable, single-purpose utility to
get tangible results from your trained model.

Usage:
    python detect_and_censor.py \
        --model-path "models/advanced_training/final_model" \
        --audio-path "path/to/your/audio.wav"
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
    A two-stage profanity detection system.
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
        print(f"Model class names: {class_names}")
        return model, feature_extractor, class_names

    def _classify_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """Classifies a single chunk of audio."""
        # This is the correct way to use the feature extractor, matching frame_level_censor.py
        inputs = self.feature_extractor(
            audio_chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=len(audio_chunk) # Ensure consistent length
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs).logits
            scores = torch.softmax(logits, dim=1)[0]
            prediction_idx = torch.argmax(scores).item()
            confidence = scores[prediction_idx].item()
        
        predicted_label = self.class_names[prediction_idx]
        return predicted_label, confidence

    def _coarse_detection(self, waveform: np.ndarray, window_size_s: float, step_size_s: float) -> List[Dict]:
        """Stage 1: Find potential profane regions using a sliding window."""
        window_samples = int(window_size_s * SAMPLE_RATE)
        step_samples = int(step_size_s * SAMPLE_RATE)
        
        detected_windows = []
        print(f"\nStage 1: Starting coarse detection with {window_size_s}s window...")
        
        for i in range(0, len(waveform) - window_samples, step_samples):
            start_time = i / SAMPLE_RATE
            end_time = (i + window_samples) / SAMPLE_RATE
            window = waveform[i : i + window_samples]
            
            label, _ = self._classify_audio_chunk(window)
            
            if label != 'none':
                detected_windows.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "label": label
                })
        
        print(f"Found {len(detected_windows)} potentially profane windows.")
        return self._merge_overlapping_detections(detected_windows)

    def _merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merges overlapping time intervals into continuous regions."""
        if not detections:
            return []

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
        print(f"Merged into {len(merged)} distinct coarse regions.")
        return merged

    def _refine_boundaries(self, waveform: np.ndarray, coarse_region: Dict, confidence_threshold: float) -> Optional[Dict]:
        """
        Stage 2: Refine boundaries using energy-based VAD and re-classification.
        """
        start_sample = int(coarse_region['start_time'] * SAMPLE_RATE)
        end_sample = int(coarse_region['end_time'] * SAMPLE_RATE)
        region_audio = waveform[start_sample:end_sample]
        
        # Use librosa to split the region into clips based on silence (VAD)
        clips = librosa.effects.split(region_audio, top_db=25, frame_length=512, hop_length=128)
        
        confirmed_clips = []
        if clips.size > 0:
            for clip_start, clip_end in clips:
                if clip_end - clip_start < (SAMPLE_RATE * 0.1): # Ignore clips shorter than 100ms
                    continue

                clip_audio = region_audio[clip_start:clip_end]
                label, confidence = self._classify_audio_chunk(clip_audio)
                
                if label != 'none' and confidence >= confidence_threshold:
                    abs_start_time = coarse_region['start_time'] + (clip_start / SAMPLE_RATE)
                    abs_end_time = coarse_region['start_time'] + (clip_end / SAMPLE_RATE)
                    confirmed_clips.append({
                        "start_time": abs_start_time,
                        "end_time": abs_end_time,
                        "label": label,
                        "confidence": confidence
                    })

        if not confirmed_clips:
            return None
        
        # Merge all confirmed clips within this region into one final detection
        final_detection = confirmed_clips[0]
        for clip in confirmed_clips[1:]:
            final_detection['end_time'] = max(final_detection['end_time'], clip['end_time'])
            final_detection['confidence'] = max(final_detection['confidence'], clip['confidence'])
        
        return final_detection

    def detect(self, waveform: np.ndarray, window_size: float, step_size: float, confidence: float) -> List[Dict]:
        """Runs the full two-stage detection process."""
        # Stage 1
        coarse_detections = self._coarse_detection(waveform, window_size, step_size)
        if not coarse_detections:
            return []

        # Stage 2
        print("\nStage 2: Starting fine-grained refinement...")
        final_detections = []
        for i, region in enumerate(coarse_detections):
            print(f"Refining region {i+1}/{len(coarse_detections)} [{region['start_time']:.2f}s - {region['end_time']:.2f}s]...")
            refined = self._refine_boundaries(waveform, region, confidence)
            if refined:
                final_detections.append(refined)
                print(f"  -> Confirmed profanity: '{refined['label']}' at {refined['start_time']:.2f}s (Conf: {refined['confidence']:.2f})")
            else:
                print("  -> Discarded region (no profanity confirmed).")

        return final_detections

def main():
    parser = argparse.ArgumentParser(
        description="A two-stage profanity detector for audio files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--audio-path", required=True, help="Path to the audio file to analyze.")
    parser.add_argument("--window-size", type=float, default=0.5, help="Sliding window size for coarse detection in seconds.")
    parser.add_argument("--step-size", type=float, default=0.25, help="Sliding window step size in seconds.")
    parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for final detection.")
    
    args = parser.parse_args()

    try:
        detector = ProfanityDetector(args.model_path)
        
        print(f"\nLoading audio from: {args.audio_path}")
        waveform, sr = torchaudio.load(args.audio_path)
        
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
            
        waveform_np = waveform.squeeze().numpy()
        print(f"Audio loaded. Duration: {len(waveform_np)/SAMPLE_RATE:.2f} seconds.")

        final_detections = detector.detect(waveform_np, args.window_size, args.step_size, args.confidence)

        if not final_detections:
            print("\n✅ Analysis complete. No profanity was detected.")
            return
            
        print("\n✅ Analysis complete. Final Profanity Report:")
        print("="*60)
        
        df = pd.DataFrame(final_detections)
        # Reorder columns for clarity
        df = df[['start_time', 'end_time', 'label', 'confidence']]
        df['duration'] = df['end_time'] - df['start_time']
        print(df.to_string(index=False, float_format="%.3f"))
        
        print("="*60)
        print(f"Found {len(final_detections)} distinct profane segment(s).")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
