import torch
import torchaudio
import numpy as np
import librosa
import json
import argparse
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FrameLevelCensor:
    """
    Implements a frame-level profanity detection and censoring system.
    Uses a two-stage approach:
    1. Coarse detection with a window-based classifier.
    2. Fine-grained boundary refinement using energy-based VAD.
    """

    def __init__(self, model_path: str, window_size: float = 0.5, overlap: float = 0.25, max_merge_gap: float = 0.5):
        self.model_path = model_path
        self.window_size = window_size
        self.overlap = overlap
        self.hop_length = window_size - overlap
        self.sample_rate = 16000
        self.max_merge_gap = max_merge_gap # Max gap in seconds to merge windows

        self.model, self.feature_extractor = self._load_model()
        # Hardcode the correct label mapping to fix the "LABEL_0" issue
        self.id_to_label = {
            0: 'none', 1: 'เย็ด', 2: 'กู', 3: 'มึง', 4: 'เหี้ย',
            5: 'ควย', 6: 'สวะ', 7: 'หี', 8: 'แตด'
        }

    def _load_model(self):
        """Loads the fine-tuned Wav2Vec2 model and feature extractor."""
        try:
            print(f"Loading model from: {self.model_path}")
            model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path).to(device)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
            model.eval()
            print("Model loaded successfully.")
            return model, feature_extractor
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _preprocess_window(self, audio_window: np.ndarray) -> torch.Tensor:
        """Preprocesses a single audio window for the model."""
        inputs = self.feature_extractor(
            audio_window,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.window_size * self.sample_rate)
        )
        return inputs.input_values.to(device)

    def _detect_profane_regions(self, audio_np: np.ndarray) -> List[Tuple[float, float, str]]:
        """
        Stage 1: Perform coarse detection using the sliding window approach.
        Merges overlapping windows flagged as profane.
        """
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        detected_windows = []
        
        print("Starting coarse detection with sliding window...")
        for i in range(0, len(audio_np) - window_samples, hop_samples):
            start_sample = i
            end_sample = i + window_samples
            window = audio_np[start_sample:end_sample]

            # Get model prediction
            with torch.no_grad():
                inputs = self._preprocess_window(window)
                logits = self.model(inputs).logits
                prediction = torch.argmax(logits, dim=-1).item()
            
            label = self.id_to_label.get(prediction, "unknown")

            if label != 'none' and label != 'unknown':
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                detected_windows.append((start_time, end_time, label))
        
        print(f"Found {len(detected_windows)} potentially profane windows.")
        return self._merge_overlapping_windows(detected_windows)

    def _merge_overlapping_windows(self, windows: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """Merges overlapping time intervals into continuous regions."""
        if not windows:
            return []

        # Sort by start time
        windows.sort(key=lambda x: x[0])
        
        merged = []
        current_start, current_end, current_label = windows[0]

        for next_start, next_end, next_label in windows[1:]:
            # Merge if overlapping or the gap is smaller than max_merge_gap
            if next_start < (current_end + self.max_merge_gap):
                # Overlap detected, merge them
                current_end = max(current_end, next_end)
                # Simple label merge: take the first one or the more specific one
                if current_label == 'profanity' and next_label != 'profanity':
                    current_label = next_label
            else:
                # No overlap, start a new region
                merged.append((current_start, current_end, current_label))
                current_start, current_end, current_label = next_start, next_end, next_label
        
        merged.append((current_start, current_end, current_label))
        print(f"Merged into {len(merged)} distinct profane regions.")
        return merged

    def _refine_boundaries(self, audio_np: np.ndarray, region: Tuple[float, float, str]) -> Tuple[float, float, str]:
        """
        Stage 2: Refine the boundaries of a detected region using energy-based VAD.
        This finds the actual speech within the coarse window.
        """
        start_time, end_time, label = region
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        region_audio = audio_np[start_sample:end_sample]
        
        # Use librosa's split to find non-silent parts
        # top_db is the threshold in dB below the peak for what is considered silence
        clips = librosa.effects.split(region_audio, top_db=20, frame_length=256, hop_length=64)
        
        if clips.size > 0:
            # Get the start of the first clip and end of the last clip
            refined_start_sample = start_sample + clips[0][0]
            refined_end_sample = start_sample + clips[-1][1]
            
            refined_start_time = refined_start_sample / self.sample_rate
            refined_end_time = refined_end_sample / self.sample_rate
            
            print(f"Refined region [{start_time:.2f}-{end_time:.2f}] to [{refined_start_time:.2f}-{refined_end_time:.2f}]")
            return (refined_start_time, refined_end_time, label)
        
        # If no speech is found, return the original region
        print(f"Could not refine region [{start_time:.2f}-{end_time:.2f}], using original.")
        return region

    def censor_audio_file(self, input_path: str, output_path: str):
        """
        Loads an audio file, detects profanity, censors it, and saves the output.
        """
        print(f"Processing file: {input_path}")
        try:
            audio, sr = torchaudio.load(input_path)
            # Ensure mono and correct sample rate
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            audio_np = audio.squeeze().numpy()
            censored_audio_np = audio_np.copy()
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return

        # Stage 1: Coarse Detection
        profane_regions = self._detect_profane_regions(audio_np)
        
        if not profane_regions:
            print("No profanity detected. File is clean.")
            # Save the original audio if no profanity is found
            torchaudio.save(output_path, torch.from_numpy(censored_audio_np).unsqueeze(0), self.sample_rate)
            return

        # Stage 2: Refinement and Censoring
        censor_report = []
        print("\nStarting refinement and censoring...")
        for region in profane_regions:
            refined_region = self._refine_boundaries(audio_np, region)
            start_time, end_time, label = refined_region
            
            # Censor by silencing the refined region
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            censored_audio_np[start_sample:end_sample] = 0.0
            
            report_entry = {
                "label": label,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "censoring_method": "silence"
            }
            censor_report.append(report_entry)

        # Save the censored audio
        torchaudio.save(output_path, torch.from_numpy(censored_audio_np).unsqueeze(0), self.sample_rate)
        print(f"\nCensored audio saved to: {output_path}")

        # Save the report
        report_path = Path(output_path).with_suffix('.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(censor_report, f, indent=4, ensure_ascii=False)
        print(f"Censoring report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Frame-level profanity censoring PoC.")
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to the input audio file (.wav, .mp3, etc.)"
    )
    parser.add_argument(
        "-m", "--model-path", type=str, required=True,
        help="Path to the directory containing the fine-tuned model."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to save the censored output .wav file."
    )
    parser.add_argument(
        "--max-gap", type=float, default=0.5,
        help="The maximum gap in seconds between detected windows to merge them into a single region."
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    censor_system = FrameLevelCensor(model_path=args.model_path, max_merge_gap=args.max_gap)
    censor_system.censor_audio_file(input_path=args.input, output_path=args.output)

if __name__ == "__main__":
    main()
