#!/usr/bin/env python3
"""
🎯 OPTIMIZED FRAME-LEVEL PROFANITY CENSORING
Censoring system optimized for models trained with train_optimized.py

Features:
- Compatibility with fast binary/multiclass models from train_optimized.py
- Two-stage detection: Binary screening + Multiclass classification
- Optimized preprocessing matching training pipeline
- Multiple censoring methods (silence, beep, fade)
- Enhanced audio preprocessing from train_optimized.py
- Comprehensive reporting and analysis
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants from train_optimized.py
BINARY_LABEL_MAP = {
    'none': 0,
    'เย็ด': 1, 'กู': 1, 'มึง': 1, 'เหี้ย': 1
}

MULTICLASS_LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4
}

BINARY_CLASSES = ['none', 'profanity']
MULTICLASS_CLASSES = ['none', 'เย็ด', 'กู', 'มึง', 'เหี้ย']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class OptimizedAudioPreprocessor:
    """Enhanced audio preprocessor matching train_optimized.py"""
    
    def __init__(self):
        self.sample_rate = 16000
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def enhance_audio(self, audio):
        """Enhanced audio preprocessing from train_optimized.py"""
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Apply slight noise reduction using spectral gating
        # Compute short-time energy
        frame_length = 512
        hop_length = 256
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2) 
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        # Noise gate threshold (remove very quiet sections)
        threshold = np.percentile(energy, 10)  # Bottom 10% considered noise
        
        # Create mask for non-noise sections
        mask = energy > threshold
        
        # Expand mask to original audio length
        expanded_mask = np.repeat(mask, hop_length)
        if len(expanded_mask) > len(audio):
            expanded_mask = expanded_mask[:len(audio)]
        elif len(expanded_mask) < len(audio):
            expanded_mask = np.pad(expanded_mask, (0, len(audio) - len(expanded_mask)), mode='edge')
        
        # Apply noise gate
        audio = audio * expanded_mask
        
        return audio
    
    def apply_voice_activity_detection(self, audio, top_db=25):
        """Apply VAD to remove silence"""
        # Split audio by silence detection
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return audio
        
        # Keep only voiced segments
        voiced_audio = []
        for start, end in intervals:
            voiced_audio.append(audio[start:end])
        
        if voiced_audio:
            return np.concatenate(voiced_audio)
        return audio
    
    def preprocess(self, audio):
        """Main preprocessing function matching train_optimized.py"""
        # Apply enhanced audio preprocessing
        audio = self.enhance_audio(audio)
        
        # Apply VAD for better feature extraction
        audio = self.apply_voice_activity_detection(audio)
        
        return audio

class OptimizedDataProcessor:
    """Data processor matching train_optimized.py pipeline"""
    
    def __init__(self, feature_extractor, preprocessor):
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
    
    def process_audio_window(self, audio_window):
        """Process audio window matching train_optimized.py"""
        try:
            # Preprocess audio using optimized pipeline
            processed_audio = self.preprocessor.preprocess(audio_window)
            
            # Extract features using train_optimized.py approach
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'labels': torch.tensor(0, dtype=torch.long)  # Dummy label for inference
            }
        except Exception as e:
            print(f"Error processing audio window: {e}")
            return None

class OptimizedFrameLevelCensor:
    """
    Optimized frame-level censoring system for train_optimized.py models
    Uses two-stage detection: Binary screening + Multiclass classification
    """

    def __init__(self, binary_model_path: str, multiclass_model_path: str, 
                 binary_window_size: float = 2.0, multiclass_window_size: float = 0.3,
                 overlap_ratio: float = 0.5, confidence_threshold: float = 0.7):
        """
        Initialize the optimized censoring system
        
        Args:
            binary_model_path: Path to binary classifier model (fast screening)
            multiclass_model_path: Path to multiclass classifier model (detailed classification)
            binary_window_size: Window size for binary detection (from train_optimized: 2.0s)
            multiclass_window_size: Window size for multiclass detection (from train_optimized: 0.3s)
            overlap_ratio: Overlap ratio between windows
            confidence_threshold: Minimum confidence for detection
        """
        self.binary_model_path = binary_model_path
        self.multiclass_model_path = multiclass_model_path
        self.binary_window_size = binary_window_size
        self.multiclass_window_size = multiclass_window_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.sample_rate = 16000
        
        # Initialize preprocessor
        self.preprocessor = OptimizedAudioPreprocessor()
        
        # Load models and feature extractors
        self.binary_model, self.binary_feature_extractor = self._load_model_and_extractor(
            binary_model_path, 2, "binary"
        )
        self.multiclass_model, self.multiclass_feature_extractor = self._load_model_and_extractor(
            multiclass_model_path, 5, "multiclass"
        )
        
        # Initialize data processors
        self.binary_processor = OptimizedDataProcessor(self.binary_feature_extractor, self.preprocessor)
        self.multiclass_processor = OptimizedDataProcessor(self.multiclass_feature_extractor, self.preprocessor)
        
        print("✅ Optimized censoring system initialized")
        print(f"   Binary model: {binary_model_path}")
        print(f"   Multiclass model: {multiclass_model_path}")
        print(f"   Binary window: {binary_window_size}s")
        print(f"   Multiclass window: {multiclass_window_size}s")
    
    def _load_model_and_extractor(self, model_path: str, num_labels: int, stage: str):
        """Load model and feature extractor"""
        try:
            # Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                local_files_only=True
            ).to(device)
            model.eval()
            
            # Load feature extractor (try model-specific first, then fallback)
            try:
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    model_path,
                    local_files_only=True
                )
            except:
                # Fallback to base model feature extractor
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-base"
                )
            
            print(f"✅ Loaded {stage} model with {num_labels} classes")
            return model, feature_extractor
            
        except Exception as e:
            print(f"❌ Failed to load {stage} model from {model_path}: {e}")
            raise
    
    def _detect_binary_regions(self, audio_np: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Stage 1: Binary detection to find potential profanity regions
        Uses larger windows (2.0s) for fast screening
        """
        window_samples = int(self.binary_window_size * self.sample_rate)
        hop_samples = int(window_samples * (1 - self.overlap_ratio))
        
        potential_regions = []
        
        print("🔍 Stage 1: Binary screening for potential profanity...")
        
        for i in tqdm(range(0, len(audio_np) - window_samples, hop_samples), desc="Binary detection"):
            window = audio_np[i:i + window_samples]
            
            # Process window
            processed_sample = self.binary_processor.process_audio_window(window)
            if processed_sample is None:
                continue
            
            # Prepare inputs
            inputs = {
                'input_values': processed_sample['input_values'].unsqueeze(0).to(device)
            }
            
            # Get binary prediction
            with torch.no_grad():
                outputs = self.binary_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities.max().item()
            
            # If profanity detected with sufficient confidence
            if prediction == 1 and confidence >= self.confidence_threshold:
                start_time = i / self.sample_rate
                end_time = (i + window_samples) / self.sample_rate
                potential_regions.append((start_time, end_time, confidence))
        
        print(f"🎯 Found {len(potential_regions)} potential profanity regions")
        return self._merge_overlapping_regions(potential_regions)
    
    def _detect_multiclass_details(self, audio_np: np.ndarray, binary_regions: List[Tuple[float, float, float]]) -> List[Tuple[float, float, str, float]]:
        """
        Stage 2: Multiclass detection for detailed classification
        Uses smaller windows (0.3s) for precise classification within binary-detected regions
        """
        detailed_detections = []
        
        if not binary_regions:
            return detailed_detections
        
        print("🔍 Stage 2: Detailed multiclass classification...")
        
        window_samples = int(self.multiclass_window_size * self.sample_rate)
        hop_samples = int(window_samples * (1 - self.overlap_ratio))
        
        for region_start, region_end, binary_conf in tqdm(binary_regions, desc="Multiclass analysis"):
            # Extract region audio with some padding
            padding = 0.1  # 100ms padding
            start_sample = max(0, int((region_start - padding) * self.sample_rate))
            end_sample = min(len(audio_np), int((region_end + padding) * self.sample_rate))
            region_audio = audio_np[start_sample:end_sample]
            
            # Scan region with multiclass windows
            region_detections = []
            
            for i in range(0, len(region_audio) - window_samples, hop_samples):
                window = region_audio[i:i + window_samples]
                
                # Process window
                processed_sample = self.multiclass_processor.process_audio_window(window)
                if processed_sample is None:
                    continue
                
                # Prepare inputs
                inputs = {
                    'input_values': processed_sample['input_values'].unsqueeze(0).to(device)
                }
                
                # Get multiclass prediction
                with torch.no_grad():
                    outputs = self.multiclass_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    prediction = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities.max().item()
                
                # If specific profanity detected
                if prediction != 0 and confidence >= self.confidence_threshold:
                    abs_start_time = (start_sample + i) / self.sample_rate
                    abs_end_time = (start_sample + i + window_samples) / self.sample_rate
                    label = MULTICLASS_CLASSES[prediction] if prediction < len(MULTICLASS_CLASSES) else f"class_{prediction}"
                    
                    region_detections.append((abs_start_time, abs_end_time, label, confidence))
            
            # Merge overlapping detections within this region
            merged_region_detections = self._merge_overlapping_detections(region_detections)
            detailed_detections.extend(merged_region_detections)
        
        print(f"🎯 Found {len(detailed_detections)} detailed profanity detections")
        return detailed_detections
    
    def _merge_overlapping_regions(self, regions: List[Tuple[float, float, float]], max_gap: float = 0.5) -> List[Tuple[float, float, float]]:
        """Merge overlapping binary regions"""
        if not regions:
            return []
        
        regions.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end, current_conf = regions[0]
        
        for next_start, next_end, next_conf in regions[1:]:
            if next_start <= current_end + max_gap:
                current_end = max(current_end, next_end)
                current_conf = max(current_conf, next_conf)
            else:
                merged.append((current_start, current_end, current_conf))
                current_start, current_end, current_conf = next_start, next_end, next_conf
        
        merged.append((current_start, current_end, current_conf))
        return merged
    
    def _merge_overlapping_detections(self, detections: List[Tuple[float, float, str, float]], max_gap: float = 0.2) -> List[Tuple[float, float, str, float]]:
        """Merge overlapping multiclass detections"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end, current_label, current_conf = detections[0]
        
        for next_start, next_end, next_label, next_conf in detections[1:]:
            if next_start <= current_end + max_gap and next_label == current_label:
                current_end = max(current_end, next_end)
                current_conf = max(current_conf, next_conf)
            else:
                merged.append((current_start, current_end, current_label, current_conf))
                current_start, current_end, current_label, current_conf = next_start, next_end, next_label, next_conf
        
        merged.append((current_start, current_end, current_label, current_conf))
        return merged
    
    def _refine_boundaries_with_vad(self, audio_np: np.ndarray, region: Tuple[float, float, str, float]) -> Optional[Tuple[float, float, str, float]]:
        """Refine boundaries using Voice Activity Detection"""
        start_time, end_time, label, confidence = region
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        region_audio = audio_np[start_sample:end_sample]
        
        try:
            speech_intervals = librosa.effects.split(region_audio, top_db=20)
            
            if len(speech_intervals) == 0:
                return None
            
            # Take the largest speech segment
            largest_interval = max(speech_intervals, key=lambda x: x[1] - x[0])
            speech_start, speech_end = largest_interval
            
            # Convert back to absolute time
            refined_start = start_time + (speech_start / self.sample_rate)
            refined_end = start_time + (speech_end / self.sample_rate)
            
            # Ensure minimum duration
            if refined_end - refined_start < 0.1:  # Minimum 100ms
                return None
            
            return (refined_start, refined_end, label, confidence)
            
        except Exception:
            return region
    
    def censor_audio_file(self, input_path: str, output_path: str, method: str = "silence", 
                         beep_freq: float = 1000.0, fade_duration: float = 0.05):
        """
        Censor profanity using two-stage optimized detection
        
        Args:
            input_path: Input audio file path
            output_path: Output censored audio file path
            method: Censoring method ('silence', 'beep', 'fade')
            beep_freq: Frequency for beep censoring
            fade_duration: Fade duration in seconds
        """
        print(f"🎵 Processing audio file: {input_path}")
        print(f"📊 Using two-stage detection with method: {method}")
        
        # Load audio
        try:
            audio, sr = torchaudio.load(input_path)
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            audio_np = audio.squeeze().numpy()
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return
        
        # Stage 1: Binary detection for fast screening
        binary_regions = self._detect_binary_regions(audio_np)
        
        if not binary_regions:
            print("✅ No profanity detected - audio is clean!")
            torchaudio.save(output_path, audio, self.sample_rate)
            self._save_report([], output_path, method)
            return
        
        # Stage 2: Detailed multiclass classification
        detailed_detections = self._detect_multiclass_details(audio_np, binary_regions)
        
        if not detailed_detections:
            print("✅ Binary screening found candidates, but detailed analysis shows audio is clean!")
            torchaudio.save(output_path, audio, self.sample_rate)
            self._save_report([], output_path, method)
            return
        
        print(f"🚨 Found {len(detailed_detections)} profanity regions to censor")
        
        # Refine boundaries
        refined_regions = []
        for region in detailed_detections:
            refined = self._refine_boundaries_with_vad(audio_np, region)
            if refined:
                refined_regions.append(refined)
        
        print(f"✨ Refined to {len(refined_regions)} precise regions")
        
        # Apply censoring
        censored_audio = self._apply_censoring(audio_np, refined_regions, method, beep_freq, fade_duration)
        
        # Save censored audio
        torchaudio.save(output_path, torch.from_numpy(censored_audio).unsqueeze(0), self.sample_rate)
        print(f"💾 Censored audio saved to: {output_path}")
        
        # Save report
        self._save_report(refined_regions, output_path, method, binary_regions)
    
    def _apply_censoring(self, audio_np: np.ndarray, regions: List[Tuple[float, float, str, float]], 
                        method: str, beep_freq: float, fade_duration: float) -> np.ndarray:
        """Apply censoring to detected regions"""
        censored_audio = audio_np.copy()
        
        for start_time, end_time, label, confidence in regions:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if method == "silence":
                censored_audio[start_sample:end_sample] = 0.0
                
            elif method == "beep":
                duration = end_time - start_time
                t = np.linspace(0, duration, end_sample - start_sample, False)
                beep = 0.3 * np.sin(2 * np.pi * beep_freq * t)
                
                # Apply fade in/out
                fade_samples = int(fade_duration * self.sample_rate)
                if len(beep) > 2 * fade_samples:
                    beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                censored_audio[start_sample:end_sample] = beep
                
            elif method == "fade":
                fade_samples = int(fade_duration * self.sample_rate)
                region_length = end_sample - start_sample
                
                if region_length > 2 * fade_samples:
                    # Fade out
                    censored_audio[start_sample:start_sample + fade_samples] *= np.linspace(1, 0, fade_samples)
                    # Silence middle
                    censored_audio[start_sample + fade_samples:end_sample - fade_samples] = 0.0
                    # Fade in
                    censored_audio[end_sample - fade_samples:end_sample] *= np.linspace(0, 1, fade_samples)
                else:
                    # Short region, just fade out/in
                    censored_audio[start_sample:end_sample] *= np.linspace(1, 0, region_length)
            
            print(f"  🔇 Censored {label} at {start_time:.2f}-{end_time:.2f}s (confidence: {confidence:.3f})")
        
        return censored_audio
    
    def _save_report(self, regions: List[Tuple[float, float, str, float]], output_path: str, method: str, binary_regions: List[Tuple[float, float, float]] = None):
        """Save detailed censoring report"""
        report_path = Path(output_path).with_suffix('.json')
        
        report_data = {
            "censoring_method": method,
            "model_info": {
                "binary_model": self.binary_model_path,
                "multiclass_model": self.multiclass_model_path,
                "binary_window_size": self.binary_window_size,
                "multiclass_window_size": self.multiclass_window_size,
                "confidence_threshold": self.confidence_threshold,
                "overlap_ratio": self.overlap_ratio
            },
            "detection_summary": {
                "binary_regions_found": len(binary_regions) if binary_regions else 0,
                "detailed_detections": len(regions),
                "two_stage_filtering_efficiency": f"{len(regions)}/{len(binary_regions) if binary_regions else 0}" if binary_regions else "N/A"
            },
            "detections": [],
            "summary": {
                "total_detections": len(regions),
                "total_censored_duration": sum(end - start for start, end, _, _ in regions),
                "detections_by_class": {}
            }
        }
        
        # Process each detection
        for start_time, end_time, label, confidence in regions:
            detection = {
                "label": label,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "confidence": round(confidence, 3)
            }
            report_data["detections"].append(detection)
            
            # Update class summary
            if label not in report_data["summary"]["detections_by_class"]:
                report_data["summary"]["detections_by_class"][label] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_confidence": 0.0
                }
            
            class_info = report_data["summary"]["detections_by_class"][label]
            class_info["count"] += 1
            class_info["total_duration"] += detection["duration"]
            class_info["avg_confidence"] = (class_info["avg_confidence"] * (class_info["count"] - 1) + confidence) / class_info["count"]
        
        # Round averages
        for label, info in report_data["summary"]["detections_by_class"].items():
            info["total_duration"] = round(info["total_duration"], 3)
            info["avg_confidence"] = round(info["avg_confidence"], 3)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Censoring report saved to: {report_path}")
        
        # Print summary
        print("\n📊 CENSORING SUMMARY:")
        print(f"   Binary screening: {len(binary_regions) if binary_regions else 0} potential regions")
        print(f"   Detailed analysis: {len(regions)} confirmed profanity regions")
        print(f"   Total censored duration: {report_data['summary']['total_censored_duration']:.3f}s")
        if report_data["summary"]["detections_by_class"]:
            print("   Detections by class:")
            for label, info in report_data["summary"]["detections_by_class"].items():
                print(f"     {label}: {info['count']} ({info['total_duration']:.3f}s, avg conf: {info['avg_confidence']:.3f})")

def main():
    """Main function for optimized censoring"""
    parser = argparse.ArgumentParser(description="Optimized frame-level profanity censoring for train_optimized.py models")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input audio file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output censored audio file path")
    parser.add_argument("-b", "--binary-model", type=str, default="models/binary_classifier_fast", help="Binary classifier model path")
    parser.add_argument("-m", "--multiclass-model", type=str, default="models/multiclass_classifier_fast", help="Multiclass classifier model path")
    parser.add_argument("--method", choices=["silence", "beep", "fade"], default="silence", help="Censoring method")
    parser.add_argument("--binary-window", type=float, default=2.0, help="Binary detection window size in seconds")
    parser.add_argument("--multiclass-window", type=float, default=0.3, help="Multiclass detection window size in seconds")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Window overlap ratio")
    parser.add_argument("--confidence-threshold", type=float, default=0.1, help="Minimum confidence for detection")
    parser.add_argument("--beep-freq", type=float, default=1000.0, help="Beep frequency for beep method")
    parser.add_argument("--fade-duration", type=float, default=0.05, help="Fade duration for fade method")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize optimized censoring system
    print("🚀 Initializing Optimized Frame-Level Censoring System")
    print("=" * 60)
    
    try:
        censor = OptimizedFrameLevelCensor(
            binary_model_path=args.binary_model,
            multiclass_model_path=args.multiclass_model,
            binary_window_size=args.binary_window,
            multiclass_window_size=args.multiclass_window,
            overlap_ratio=args.overlap_ratio,
            confidence_threshold=args.confidence_threshold
        )
    except Exception as e:
        print(f"❌ Failed to initialize censoring system: {e}")
        print("\n💡 Make sure you have trained models using train_optimized.py:")
        print("   python train_optimized.py --mode fast")
        print("   or")
        print("   python train_optimized.py --mode full")
        return
    
    # Process the audio file
    censor.censor_audio_file(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        beep_freq=args.beep_freq,
        fade_duration=args.fade_duration
    )
    
    print("\n✅ Optimized censoring completed successfully!")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Method: {args.method}")
    print(f"   Report: {Path(args.output).with_suffix('.json')}")
    print(f"   Binary model: {args.binary_model}")
    print(f"   Multiclass model: {args.multiclass_model}")

if __name__ == "__main__":
    main()